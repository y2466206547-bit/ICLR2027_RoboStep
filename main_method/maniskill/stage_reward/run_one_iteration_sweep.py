#!/usr/bin/env python3
"""Run formal PPO training for RuleGate and Qwen3-VL-8B gate on ManiSkill-19."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path, PosixPath
import random
import time
import traceback
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MS_ASSET_DIR", str(REPO_ROOT.parent / "maniskill_assets"))

import mani_skill.envs  # noqa: E402,F401 - registers environments
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper  # noqa: E402
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # noqa: E402

from stage_reward.maniskill_wrapper import wrap_stage_aware_env  # noqa: E402
from stage_reward.qwen3_client import Qwen3VL8BClient  # noqa: E402
from stage_reward.rgb_dino import FrozenDinoFeatureEncoder  # noqa: E402
from stage_reward.rule_specs import get_rule_stage_spec  # noqa: E402
from stage_reward.task_suite import load_task_specs  # noqa: E402
from stage_reward.visual_observer import SynchronousVisualStageObserver  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "results" / "ours" / "maniskill19_formal_seed1"
DEFAULT_FRAME_VLM = REPO_ROOT / "results" / "ours" / "frame_vlm_iteration19"
DEFAULT_PYTHON = Path(os.environ.get("ROBOSTEP_VLM_PYTHON", "python"))
DEFAULT_MODEL = Path(os.environ.get("ROBOSTEP_VLM_MODEL", "models/Qwen3-VL-8B-Instruct"))
DEFAULT_WORKER = Path(__file__).with_name("qwen3_vl_worker.py")
DEFAULT_DINO_REPO = Path(os.environ.get("ROBOSTEP_DINO_REPO", "models/dinov2"))



def load_agent_class() -> Any:
    path = REPO_ROOT / "examples" / "baselines" / "ppo" / "ppo.py"
    spec = importlib.util.spec_from_file_location("maniskill_official_ppo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import PPO Agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


Agent = load_agent_class()


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AsymmetricStageAgent(nn.Module):
    """PPO actor/critic split for asymmetric input ablations."""

    def __init__(
        self,
        envs: ManiSkillVectorEnv,
        critic_observation_shape: tuple[int, ...],
        actor_observation_shape: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if actor_observation_shape is None:
            actor_observation_shape = envs.single_observation_space.shape
        actor_dim = int(np.array(actor_observation_shape).prod())
        critic_dim = int(np.array(critic_observation_shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(actor_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)

    def get_action(self, actor_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action_mean = self.actor_mean(actor_obs)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ):
        if critic_obs is None:
            critic_obs = actor_obs
        action_mean = self.actor_mean(actor_obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(critic_obs),
        )


def stage_augmented_observation_shape(env_id: str, actor_observation_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(actor_observation_shape) != 1:
        raise ValueError(f"expected flat actor observation, got {actor_observation_shape}")
    stage_width = len(get_rule_stage_spec(env_id).stage_names) + 1
    return (int(actor_observation_shape[0]) + stage_width,)


def actor_critic_are_asymmetric(args: argparse.Namespace) -> bool:
    return bool(args.actor_blind_stage or args.rgb_actor)


def require_info_tensor(
    info: dict[str, Any], key: str, reference: torch.Tensor
) -> torch.Tensor:
    value = info.get(key)
    if value is None:
        raise KeyError(f"missing info[{key!r}] from StageAwareRewardWrapper")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=reference.device)
    return value.to(device=reference.device, dtype=reference.dtype)


def make_rgb_encoder(
    args: argparse.Namespace, device: torch.device
) -> FrozenDinoFeatureEncoder | None:
    if not args.rgb_actor:
        return None
    if args.rgb_dino_checkpoint is None:
        raise ValueError("--rgb-actor requires --rgb-dino-checkpoint")
    return FrozenDinoFeatureEncoder(
        repo_path=args.rgb_dino_repo,
        checkpoint_path=args.rgb_dino_checkpoint,
        arch=args.rgb_dino_arch,
        device=device,
        image_size=args.rgb_dino_image_size,
        batch_size=args.rgb_dino_batch_size,
    )


def build_actor_observation(
    envs: ManiSkillVectorEnv,
    info: dict[str, Any],
    policy_obs: torch.Tensor,
    *,
    args: argparse.Namespace,
    rgb_encoder: FrozenDinoFeatureEncoder | None,
) -> torch.Tensor:
    if not args.rgb_actor:
        return policy_obs
    if rgb_encoder is None:
        raise RuntimeError("RGB actor requested without a DINO encoder")
    rgb_feature = rgb_encoder.encode_env(envs).to(
        device=policy_obs.device, dtype=policy_obs.dtype
    )
    proprio = require_info_tensor(info, "robot_proprio_observation", policy_obs)
    parts = [rgb_feature, proprio]
    if not args.actor_blind_stage:
        parts.append(require_info_tensor(info, "active_stage_one_hot", policy_obs))
    return torch.cat(parts, dim=-1)


def build_critic_observation(
    info: dict[str, Any],
    policy_obs: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.no_stage_obs:
        return policy_obs
    if args.actor_blind_stage or args.rgb_actor:
        return require_info_tensor(info, "stage_critic_observation", policy_obs)
    return policy_obs


def observation_labels(args: argparse.Namespace) -> tuple[str, str]:
    if args.no_stage_obs:
        return "state", "state"
    if args.rgb_actor:
        actor = f"{args.rgb_dino_arch}_rgb_plus_robot_proprio"
        if not args.actor_blind_stage:
            actor += "_plus_active_stage_one_hot"
        return actor, "state_plus_active_stage_one_hot"
    if args.actor_blind_stage:
        return "state", "state_plus_active_stage_one_hot"
    return "state_plus_active_stage_one_hot", "state_plus_active_stage_one_hot"


def input_ablation_label(args: argparse.Namespace) -> str:
    if args.no_stage_obs:
        return "Ours-NoStageObs"
    if args.rgb_actor:
        return "Ours-RGB-ActorBlindStage" if args.actor_blind_stage else "Ours-RGB"
    if args.actor_blind_stage:
        return "Ours-ActorBlindStage"
    return "Ours-ZeroShotGate"


def rgb_dino_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.rgb_actor:
        return None
    return {
        "repo": str(args.rgb_dino_repo),
        "checkpoint": None if args.rgb_dino_checkpoint is None else str(args.rgb_dino_checkpoint),
        "arch": args.rgb_dino_arch,
        "image_size": args.rgb_dino_image_size,
        "feature_batch_size": args.rgb_dino_batch_size,
    }


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_frame_vlm_stage_instructions(frame_vlm_dir: Path, env_id: str) -> dict[str, Any]:
    path = frame_vlm_dir / "parsed_designs.json"
    if not path.exists():
        return {"status": "missing", "instructions": None}
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("task_id") != env_id:
            continue
        instructions = []
        for stage in row.get("stages", []):
            visible = stage.get("visible_completion", [])
            if isinstance(visible, list):
                visible_text = "; ".join(str(item) for item in visible)
            else:
                visible_text = str(visible)
            instructions.append(
                f"{stage.get('name', 'stage')}: {stage.get('objective', '')}. "
                f"Visible completion: {visible_text}"
            )
        return {"status": "loaded", "instructions": tuple(instructions), "raw": row}
    return {"status": "not_found", "instructions": None}


def native_reward_mode(env_id: str) -> str:
    # Some sparse-only ManiSkill tasks reject reward_mode="none" at construction.
    # The wrapper still returns ours_rule reward; this native mode is only an
    # ignored placeholder required by the official environment API.
    if env_id == "DrawTriangle-v1":
        return "sparse"
    return "none"


def make_env(
    *,
    env_id: str,
    num_envs: int,
    reconfiguration_freq: int | None,
    stage_observer: str,
    reward_backend: str,
    append_stage_obs: bool,
    qwen_client: Qwen3VL8BClient | None,
    frame_vlm_dir: Path,
    query_interval_steps: int,
    qwen_query_mode: str,
    transition_authority_mode: str | None,
    ignore_terminations: bool,
    eval_env: bool,
) -> gym.Env:
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        reward_mode=native_reward_mode(env_id),
        control_mode="pd_joint_delta_pos",
        render_mode=(None if os.environ.get("MANISKILL_RENDER_MODE", "rgb_array") == "none" else os.environ.get("MANISKILL_RENDER_MODE", "rgb_array")),
        sim_backend=os.environ.get("MANISKILL_SIM_BACKEND", "physx_cuda"),
        reconfiguration_freq=reconfiguration_freq,
    )
    spec = get_rule_stage_spec(env_id)
    visual_observer = None
    if stage_observer.startswith("qwen"):
        if qwen_client is None:
            raise ValueError(f"{stage_observer} observer requires a Qwen client")
        loaded = load_frame_vlm_stage_instructions(frame_vlm_dir, env_id)
        stage_instructions = loaded.get("instructions")
        if stage_instructions is not None and len(stage_instructions) != len(spec.stage_names):
            stage_instructions = None
        visual_observer = SynchronousVisualStageObserver(
            env_id=env_id,
            stage_names=spec.stage_names,
            task_instruction=f"{env_id}: execute the ManiSkill manipulation task.",
            num_envs=num_envs,
            device=env.unwrapped.device,
            predictor=qwen_client,
            query_interval_steps=query_interval_steps,
            stage_instructions=stage_instructions,
            close_predictor=False,
            query_mode=qwen_query_mode,
        )
    if transition_authority_mode is None:
        transition_authority_mode = (
            "external_authorization"
            if qwen_query_mode == "fixed_frequency"
            else "rule_candidate_and_external_authorization"
        )
    wrapped = wrap_stage_aware_env(
        env,
        env_id=env_id,
        stage_observer="qwen" if stage_observer.startswith("qwen") else "rule",
        reward_backend=reward_backend,
        append_stage_obs=append_stage_obs,
        visual_observer=visual_observer,
        transition_authority_mode=(
            "rule_candidate" if not stage_observer.startswith("qwen") else transition_authority_mode
        ),
    )
    if isinstance(wrapped.action_space, gym.spaces.Dict):
        wrapped = FlattenActionSpaceWrapper(wrapped)
    return ManiSkillVectorEnv(
        wrapped,
        num_envs,
        ignore_terminations=ignore_terminations,
        record_metrics=True,
    )


def clip_action_fn(envs: ManiSkillVectorEnv, device: torch.device):
    low = torch.from_numpy(envs.single_action_space.low).to(device)
    high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action.detach(), low, high)

    return clip_action


def empty_stage_stats() -> dict[str, Any]:
    return {
        "transition_count": 0,
        "rule_candidate_count": 0,
        "stage_after_hist": defaultdict(int),
        "event_stage_hist": defaultdict(int),
    }


def collect_stage_stats(info: dict[str, Any], stats: dict[str, Any]) -> None:
    stage_reward = info.get("stage_reward")
    if not stage_reward:
        return
    transition = stage_reward["transition"]
    rule_candidate = stage_reward["rule_candidate"]
    stage_after = stage_reward["stage_after"]
    event_stage = stage_reward["event_stage"]
    stats["transition_count"] += int(transition.sum().item())
    stats["rule_candidate_count"] += int(rule_candidate.sum().item())
    for value in stage_after.detach().cpu().tolist():
        stats["stage_after_hist"][str(int(value))] += 1
    for value in event_stage.detach().cpu().tolist():
        if int(value) >= 0:
            stats["event_stage_hist"][str(int(value))] += 1


def collect_episode_metrics(info: dict[str, Any], metrics: dict[str, list[float]]) -> None:
    if "final_info" not in info:
        return
    mask = info["_final_info"]
    final_info = info["final_info"]
    if "episode" in final_info:
        for key, value in final_info["episode"].items():
            if mask.any():
                metrics[key].append(float(value[mask].float().mean().item()))
    if "success" in final_info and mask.any():
        metrics["success"].append(float(final_info["success"][mask].float().mean().item()))


def summarize_episode_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(np.mean(values))
        for key, values in metrics.items()
        if values
    }


def run_eval(
    *,
    agent: nn.Module,
    eval_envs: ManiSkillVectorEnv,
    eval_steps: int,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
    rgb_encoder: FrozenDinoFeatureEncoder | None = None,
) -> dict[str, Any]:
    policy_obs, reset_info = eval_envs.reset(seed=seed)
    obs = build_actor_observation(
        eval_envs, reset_info, policy_obs, args=args, rgb_encoder=rgb_encoder
    )
    clip_action = clip_action_fn(eval_envs, device)
    rewards = []
    metrics: dict[str, list[float]] = defaultdict(list)
    stage_stats = empty_stage_stats()
    success_any = torch.zeros(eval_envs.num_envs, dtype=torch.bool, device=device)
    agent.eval()
    for _ in range(eval_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
            policy_obs, reward, _, _, info = eval_envs.step(clip_action(action))
            obs = build_actor_observation(
                eval_envs, info, policy_obs, args=args, rgb_encoder=rgb_encoder
            )
        rewards.append(reward.detach().cpu())
        if "success" in info:
            success_any |= info["success"].bool().view(-1).to(device)
        if "final_info" in info and "success" in info["final_info"]:
            final_mask = info.get("_final_info")
            if final_mask is not None and final_mask.any():
                success_any[final_mask] |= info["final_info"]["success"][final_mask].bool().to(device)
        collect_stage_stats(info, stage_stats)
        collect_episode_metrics(info, metrics)
    stacked_rewards = torch.stack(rewards)
    episode_metrics = summarize_episode_metrics(metrics)
    return {
        "success_rate": float(success_any.float().mean().item()),
        "reward_mean": float(stacked_rewards.mean().item()),
        "reward_sum_mean_per_env": float(stacked_rewards.sum(dim=0).mean().item()),
        "episode_metrics": episode_metrics,
        "stage_stats": jsonable(stage_stats),
    }


def run_one_task(
    *,
    env_id: str,
    horizon: int,
    stage_observer: str,
    output_dir: Path,
    qwen_client: Qwen3VL8BClient | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed_everything(args.seed, args.torch_deterministic)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    actor_observation_label, critic_observation_label = observation_labels(args)
    actor_blind_stage = bool(args.actor_blind_stage)
    asymmetric_actor_critic = actor_critic_are_asymmetric(args)
    envs = make_env(
        env_id=env_id,
        num_envs=args.num_envs,
        reconfiguration_freq=args.reconfiguration_freq,
        stage_observer=stage_observer,
        reward_backend=args.reward_backend,
        append_stage_obs=not args.no_stage_obs and not args.actor_blind_stage and not args.rgb_actor,
        qwen_client=qwen_client,
        frame_vlm_dir=args.frame_vlm_dir,
        query_interval_steps=args.query_interval_steps,
        qwen_query_mode=args.qwen_query_mode,
        transition_authority_mode=args.transition_authority_mode,
        ignore_terminations=args.ignore_terminations,
        eval_env=False,
    )
    eval_envs = make_env(
        env_id=env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        stage_observer=stage_observer,
        reward_backend=args.reward_backend,
        append_stage_obs=not args.no_stage_obs and not args.actor_blind_stage and not args.rgb_actor,
        qwen_client=qwen_client,
        frame_vlm_dir=args.frame_vlm_dir,
        query_interval_steps=args.query_interval_steps,
        qwen_query_mode=args.qwen_query_mode,
        transition_authority_mode=args.transition_authority_mode,
        ignore_terminations=args.eval_ignore_terminations,
        eval_env=True,
    )
    try:
        rgb_encoder = make_rgb_encoder(args, device)
        next_policy_obs, reset_info = envs.reset(seed=args.seed)
        next_obs = build_actor_observation(
            envs, reset_info, next_policy_obs, args=args, rgb_encoder=rgb_encoder
        )
        next_critic_obs = build_critic_observation(
            reset_info, next_policy_obs, args=args
        )
        actor_observation_shape = tuple(next_obs.shape[1:])
        critic_observation_shape = tuple(next_critic_obs.shape[1:])
        agent = (
            AsymmetricStageAgent(
                envs,
                critic_observation_shape,
                actor_observation_shape=actor_observation_shape,
            )
            if asymmetric_actor_critic
            else Agent(envs)
        ).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        batch_size = args.num_envs * args.num_steps
        minibatch_size = batch_size // args.num_minibatches
        if minibatch_size <= 0:
            raise ValueError("num_minibatches is larger than rollout batch size")
        if args.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        num_iterations = max(1, math.ceil(args.total_timesteps / batch_size))

        obs = torch.zeros(
            (args.num_steps, args.num_envs) + actor_observation_shape,
            device=device,
        )
        critic_obs = torch.zeros(
            (args.num_steps, args.num_envs) + critic_observation_shape,
            device=device,
        )
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape,
            device=device,
        )
        logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        dones = torch.zeros((args.num_steps, args.num_envs), device=device)
        values = torch.zeros((args.num_steps, args.num_envs), device=device)
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

        next_done = torch.zeros(args.num_envs, device=device)
        clip_action = clip_action_fn(envs, device)
        eval_steps = args.num_eval_steps or horizon
        global_step = 0
        iteration_summaries: list[dict[str, Any]] = []
        eval_history: list[dict[str, Any]] = []
        best_eval_success = -1.0
        best_eval_iteration: int | None = None
        task_started = time.time()
        last_summary: dict[str, Any] | None = None

        # Resume the policy/optimizer state and metric history. The vectorized
        # environments are intentionally reset above; replaying an exact
        # simulator state is neither required by PPO nor portable across
        # cluster jobs.
        resume_iteration = 0
        if args.resume_checkpoint is not None:
            resume_path = args.resume_checkpoint.expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
            # Checkpoints contain only tensors plus the JSON-like summary and
            # pathlib paths from ``vars(args)``. Keep PyTorch's safe
            # ``weights_only`` loader enabled and explicitly allow that one
            # benign path type (PyTorch 2.6 changed the default to safe mode).
            with torch.serialization.safe_globals([PosixPath]):
                checkpoint = torch.load(
                    resume_path,
                    map_location=device,
                    weights_only=True,
                )
            if not isinstance(checkpoint, dict):
                raise ValueError(f"invalid resume checkpoint (expected dict): {resume_path}")
            checkpoint_summary = checkpoint.get("summary")
            if not isinstance(checkpoint_summary, dict):
                raise ValueError(f"resume checkpoint has no summary: {resume_path}")
            checkpoint_env_id = checkpoint_summary.get("env_id")
            if checkpoint_env_id is not None and checkpoint_env_id != env_id:
                raise ValueError(
                    f"resume checkpoint env_id={checkpoint_env_id!r} does not match {env_id!r}"
                )
            checkpoint_gate = checkpoint_summary.get("stage_observer")
            if checkpoint_gate is not None and checkpoint_gate != stage_observer:
                raise ValueError(
                    f"resume checkpoint gate={checkpoint_gate!r} does not match {stage_observer!r}"
                )
            agent.load_state_dict(checkpoint["agent_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            resume_last_iteration = checkpoint_summary.get("last_iteration") or {}
            resume_iteration = int(resume_last_iteration.get("iteration", 0))
            if resume_iteration < 0 or resume_iteration > num_iterations:
                raise ValueError(
                    f"resume iteration {resume_iteration} is outside 0..{num_iterations}"
                )
            global_step = int(
                checkpoint_summary.get(
                    "total_timesteps_collected",
                    resume_last_iteration.get("global_step", 0),
                )
            )
            iteration_summaries = list(checkpoint_summary.get("iteration_summaries") or [])
            eval_history = list(checkpoint_summary.get("eval_history") or [])
            best_eval_success = float(checkpoint_summary.get("best_eval_success_rate", -1.0))
            best_eval_iteration = checkpoint_summary.get("best_eval_iteration")
            last_summary = checkpoint_summary
            print(
                "resume_checkpoint "
                f"path={resume_path} iteration={resume_iteration}/{num_iterations} "
                f"global_step={global_step}",
                flush=True,
            )

        for iteration in range(resume_iteration + 1, num_iterations + 1):
            train_metrics: dict[str, list[float]] = defaultdict(list)
            stage_stats = empty_stage_stats()
            final_values.zero_()
            rollout_start = time.time()
            agent.eval()
            for step in range(args.num_steps):
                obs[step] = next_obs
                critic_obs[step] = next_critic_obs
                dones[step] = next_done
                with torch.no_grad():
                    if asymmetric_actor_critic:
                        action, logprob, _, value = agent.get_action_and_value(
                            next_obs, next_critic_obs
                        )
                    else:
                        action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                next_policy_obs, reward, terminated, truncated, info = envs.step(clip_action(action))
                next_done = torch.logical_or(terminated, truncated).to(torch.float32)
                next_obs = build_actor_observation(
                    envs, info, next_policy_obs, args=args, rgb_encoder=rgb_encoder
                )
                next_critic_obs = build_critic_observation(
                    info, next_policy_obs, args=args
                )
                rewards[step] = reward.view(-1) * args.reward_scale
                global_step += args.num_envs
                collect_stage_stats(info, stage_stats)
                collect_episode_metrics(info, train_metrics)
                if "final_info" in info:
                    done_mask = info["_final_info"]
                    if done_mask.any():
                        with torch.no_grad():
                            final_value_obs = (
                                next_critic_obs[done_mask]
                                if asymmetric_actor_critic
                                else info["final_observation"][done_mask]
                            )
                            final_values[
                                step,
                                torch.arange(args.num_envs, device=device)[done_mask],
                            ] = agent.get_value(final_value_obs).view(-1)
            rollout_walltime_s = time.time() - rollout_start

            with torch.no_grad():
                next_value_obs = next_critic_obs if asymmetric_actor_critic else next_obs
                next_value = agent.get_value(next_value_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    real_next_values = next_not_done * nextvalues + final_values[t]
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    )
                returns = advantages + values

            b_obs = obs.reshape((-1,) + actor_observation_shape)
            b_critic_obs = critic_obs.reshape((-1,) + critic_observation_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_inds = np.arange(batch_size)
            clipfracs = []
            update_start = time.time()
            agent.train()
            approx_kl = torch.tensor(0.0, device=device)
            old_approx_kl = torch.tensor(0.0, device=device)
            pg_loss = torch.tensor(0.0, device=device)
            v_loss = torch.tensor(0.0, device=device)
            entropy_loss = torch.tensor(0.0, device=device)
            for _ in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = b_inds[start : start + minibatch_size]
                    if asymmetric_actor_critic:
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs[mb_inds], b_critic_obs[mb_inds], b_actions[mb_inds]
                        )
                    else:
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )
                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (
                            mb_advantages - mb_advantages.mean()
                        ) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            update_walltime_s = time.time() - update_start

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
            iteration_summary = {
                "iteration": iteration,
                "num_iterations": num_iterations,
                "global_step": global_step,
                "rollout_reward_mean": float(rewards.mean().item()),
                "rollout_reward_sum_mean_per_env": float(rewards.sum(dim=0).mean().item()),
                "train_episode_metrics": summarize_episode_metrics(train_metrics),
                "train_stage_stats": jsonable(stage_stats),
                "losses": {
                    "policy_loss": float(pg_loss.item()),
                    "value_loss": float(v_loss.item()),
                    "entropy": float(entropy_loss.item()),
                    "old_approx_kl": float(old_approx_kl.item()),
                    "approx_kl": float(approx_kl.item()),
                    "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "explained_variance": explained_var,
                },
                "walltime_s": {
                    "rollout": rollout_walltime_s,
                    "update": update_walltime_s,
                },
            }
            iteration_summaries.append(iteration_summary)

            should_eval = (
                iteration == num_iterations
                or iteration % args.eval_freq == 0
            )
            eval_record = None
            if should_eval:
                eval_summary = run_eval(
                    agent=agent,
                    eval_envs=eval_envs,
                    eval_steps=eval_steps,
                    device=device,
                    seed=args.seed + 10000 + iteration,
                    args=args,
                    rgb_encoder=rgb_encoder,
                )
                eval_record = {
                    "iteration": iteration,
                    "global_step": global_step,
                    **eval_summary,
                }
                eval_history.append(eval_record)
                success_rate = float(eval_summary.get("success_rate", 0.0))
                if success_rate > best_eval_success:
                    best_eval_success = success_rate
                    best_eval_iteration = iteration
                    save_checkpoint_atomic(
                        {
                            "agent_state_dict": agent.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "summary": eval_record,
                            "config": vars(args),
                        },
                        output_dir / "checkpoint_best.pt",
                    )

            last_summary = {
                "status": "ok",
                "env_id": env_id,
                "stage_observer": stage_observer,
                "reward_backend": args.reward_backend,
                "append_stage_obs": not args.no_stage_obs and not args.actor_blind_stage and not args.rgb_actor,
                "input_ablation": input_ablation_label(args),
                "actor_observation": actor_observation_label,
                "critic_observation": critic_observation_label,
                "actor_receives_stage_id": actor_observation_label.endswith("stage_one_hot"),
                "critic_receives_stage_id": critic_observation_label.endswith("stage_one_hot"),
                "actor_observation_shape": list(actor_observation_shape),
                "critic_observation_shape": list(critic_observation_shape),
                "stage_fsm_unchanged": True,
                "rgb_dino": rgb_dino_config(args),
                "qwen_query_mode": args.qwen_query_mode,
                "transition_authority_mode": args.transition_authority_mode,
                "seed": args.seed,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "num_eval_envs": args.num_eval_envs,
                "num_eval_steps": eval_steps,
                "total_timesteps_requested": args.total_timesteps,
                "total_timesteps_collected": global_step,
                "num_iterations": num_iterations,
                "resumed_from_checkpoint": (
                    str(args.resume_checkpoint) if args.resume_checkpoint is not None else None
                ),
                "resume_iteration_start": resume_iteration,
                "ignore_terminations": args.ignore_terminations,
                "eval_ignore_terminations": args.eval_ignore_terminations,
                "batch_size": batch_size,
                "best_eval_success_rate": best_eval_success,
                "best_eval_iteration": best_eval_iteration,
                "final_eval": eval_history[-1] if eval_history else None,
                "eval_history": eval_history,
                "last_iteration": iteration_summary,
                "iteration_summaries": iteration_summaries,
                "walltime_s": {
                    "total_so_far": time.time() - task_started,
                },
            }
            early_stop_triggered = (
                should_eval
                and args.early_stop_success is not None
                and iteration >= args.early_stop_min_iterations
                and eval_record is not None
                and float(eval_record.get("success_rate", 0.0)) >= args.early_stop_success
            )
            last_summary["early_stop_success"] = args.early_stop_success
            last_summary["early_stop_min_iterations"] = args.early_stop_min_iterations
            last_summary["early_stop_triggered"] = early_stop_triggered
            if qwen_client is not None:
                last_summary["qwen_statistics"] = qwen_client.statistics()

            if (
                iteration == 1
                or iteration == num_iterations
                or iteration % args.save_every == 0
                or should_eval
            ):
                checkpoint = {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "summary": last_summary,
                    "config": vars(args),
                }
                save_checkpoint_atomic(checkpoint, output_dir / "checkpoint_latest.pt")
                if should_eval:
                    checkpoint_dir = output_dir / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_checkpoint_atomic(
                        checkpoint,
                        checkpoint_dir / f"checkpoint_{iteration:05d}.pt",
                    )
                write_json(output_dir / "progress.json", last_summary)

            if should_eval:
                print(
                    "run_progress "
                    f"gate={stage_observer} env_id={env_id} "
                    f"iteration={iteration}/{num_iterations} "
                    f"global_step={global_step} "
                    f"eval_success={eval_record['success_rate']:.4f} "
                    f"best_success={best_eval_success:.4f}",
                    flush=True,
                )
            elif iteration == 1 or iteration % max(1, args.save_every) == 0:
                print(
                    "run_progress "
                    f"gate={stage_observer} env_id={env_id} "
                    f"iteration={iteration}/{num_iterations} "
                    f"global_step={global_step}",
                    flush=True,
                )

            if early_stop_triggered:
                print(
                    "run_early_stop "
                    f"gate={stage_observer} env_id={env_id} "
                    f"iteration={iteration}/{num_iterations} "
                    f"global_step={global_step} "
                    f"eval_success={eval_record['success_rate']:.4f} "
                    f"threshold={args.early_stop_success:.4f}",
                    flush=True,
                )
                break

        if last_summary is None:
            raise RuntimeError("training loop produced no summary")
        last_summary["walltime_s"]["total"] = time.time() - task_started
        save_checkpoint_atomic(
            {
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "summary": last_summary,
                "config": vars(args),
            },
            output_dir / "checkpoint.pt",
        )
        return last_summary
    finally:
        envs.close()
        eval_envs.close()


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def save_checkpoint_atomic(checkpoint: dict[str, Any], path: Path) -> None:
    """Atomically replace a checkpoint, including when resuming in-place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-vlm-dir", type=Path, default=DEFAULT_FRAME_VLM)
    parser.add_argument("--task-manifest", type=Path, default=None)
    parser.add_argument("--gates", nargs="+", default=["rule", "qwen3_vl8b"], choices=["rule", "qwen3_vl8b", "qwen3_5_9b"])
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument(
        "--reward-backend",
        choices=[
            "ours_rule",
            "per_subtask_sparse",
            "completed_cumulative",
            "all_dense_with_stage_actor",
            "environment",
            "code_as_reward",
        ],
        default="ours_rule",
    )
    parser.add_argument("--no-stage-obs", action="store_true")
    parser.add_argument("--actor-blind-stage", action="store_true")
    parser.add_argument("--rgb-actor", action="store_true")
    parser.add_argument("--rgb-dino-repo", type=Path, default=DEFAULT_DINO_REPO)
    parser.add_argument("--rgb-dino-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--rgb-dino-arch",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
            "dinov2_vits14_reg",
            "dinov2_vitb14_reg",
            "dinov2_vitl14_reg",
            "dinov2_vitg14_reg",
        ],
        default="dinov2_vits14",
    )
    parser.add_argument("--rgb-dino-image-size", type=int, default=224)
    parser.add_argument("--rgb-dino-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-eval-envs", type=int, default=20)
    parser.add_argument("--num-eval-steps", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--early-stop-success", type=float, default=None)
    parser.add_argument("--early-stop-min-iterations", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--reconfiguration-freq", type=int, default=None)
    parser.add_argument("--eval-reconfiguration-freq", type=int, default=1)
    parser.add_argument(
        "--ignore-terminations",
        action="store_true",
        default=False,
        help="Keep training envs running after success/truncation; default False matches official PPO partial reset.",
    )
    parser.add_argument(
        "--eval-ignore-terminations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep eval envs running for the full horizon; default True matches existing success_once evaluation.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--query-interval-steps", type=int, default=10)
    parser.add_argument(
        "--qwen-query-mode",
        choices=["candidate_retry", "fixed_frequency"],
        default="candidate_retry",
    )
    parser.add_argument(
        "--transition-authority-mode",
        choices=[
            "rule_candidate_and_external_authorization",
            "external_authorization",
        ],
        default=None,
    )
    parser.add_argument("--qwen-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--qwen-worker-script", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--qwen-device-index", type=int, default=1)
    parser.add_argument("--qwen-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume policy/optimizer and metric history from checkpoint_latest.pt.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_manifest is not None:
        args.task_manifest = args.task_manifest.expanduser().resolve()
        os.environ["OURS_TASK_MANIFEST"] = str(args.task_manifest)
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
    if args.reward_backend == "code_as_reward" and args.output_root == DEFAULT_OUTPUT:
        args.output_root = REPO_ROOT / "results" / "code_as_reward"
    if args.eval_freq <= 0:
        raise ValueError("eval_freq must be positive")
    if args.save_every <= 0:
        raise ValueError("save_every must be positive")
    if args.early_stop_success is not None and not (0.0 <= args.early_stop_success <= 1.0):
        raise ValueError("early_stop_success must be in [0, 1]")
    if args.early_stop_min_iterations < 0:
        raise ValueError("early_stop_min_iterations must be non-negative")
    if args.no_stage_obs and args.actor_blind_stage:
        raise ValueError("--no-stage-obs and --actor-blind-stage are mutually exclusive")
    if args.no_stage_obs and args.rgb_actor:
        raise ValueError("--no-stage-obs and --rgb-actor are mutually exclusive")
    actor_observation_label, critic_observation_label = observation_labels(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = list(load_task_specs())
    if args.task_ids:
        specs_by_id = {spec.env_id: spec for spec in specs}
        requested = list(dict.fromkeys(args.task_ids))
        missing = set(requested) - set(specs_by_id)
        if missing:
            raise ValueError(f"unknown task ids: {sorted(missing)}")
        specs = [specs_by_id[env_id] for env_id in requested]
    suite_manifest = {
        "schema": "maniskill19-formal-sweep-v1",
        "rule_variant": os.environ.get("OURS_RULE_VARIANT", "rule_v1"),
        "task_ids": [spec.env_id for spec in specs],
        "gates": args.gates,
        "ppo": {
            "total_timesteps": args.total_timesteps,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "num_iterations": max(1, math.ceil(args.total_timesteps / (args.num_envs * args.num_steps))),
            "num_eval_envs": args.num_eval_envs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "num_minibatches": args.num_minibatches,
            "update_epochs": args.update_epochs,
            "clip_coef": args.clip_coef,
            "vf_coef": args.vf_coef,
            "ent_coef": args.ent_coef,
            "target_kl": args.target_kl,
            "eval_freq": args.eval_freq,
            "save_every": args.save_every,
            "early_stop_success": args.early_stop_success,
            "early_stop_min_iterations": args.early_stop_min_iterations,
            "ignore_terminations": args.ignore_terminations,
            "eval_ignore_terminations": args.eval_ignore_terminations,
        },
        "input_ablation": input_ablation_label(args),
        "actor_observation": actor_observation_label,
        "critic_observation": critic_observation_label,
        "actor_receives_stage_id": actor_observation_label.endswith("stage_one_hot"),
        "critic_receives_stage_id": critic_observation_label.endswith("stage_one_hot"),
        "stage_fsm_unchanged": True,
        "rgb_dino": rgb_dino_config(args),
        "reward_backend": args.reward_backend,
        "qwen_query_mode": args.qwen_query_mode,
        "transition_authority_mode": args.transition_authority_mode,
        "qwen_model": str(args.qwen_model_path),
        "frame_vlm_dir": str(args.frame_vlm_dir),
        "task_manifest": str(args.task_manifest) if args.task_manifest else None,
    }
    write_json(args.output_root / "sweep_manifest.json", suite_manifest)

    failures = []
    for gate in args.gates:
        qwen_client = None
        if gate.startswith("qwen"):
            qwen_client = Qwen3VL8BClient(
                python_executable=args.qwen_python,
                worker_script=args.qwen_worker_script,
                model_path=args.qwen_model_path,
                output_dir=args.output_root / gate / "_qwen_worker",
                device_index=args.qwen_device_index,
                timeout_s=args.qwen_timeout_s,
                image_size=320,
            )
        try:
            for spec in specs:
                task_dir = args.output_root / gate / spec.env_id
                task_dir.mkdir(parents=True, exist_ok=True)
                summary_path = task_dir / "summary.json"
                if summary_path.exists() and not args.overwrite:
                    existing = json.loads(summary_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "ok":
                        print(f"skip_existing gate={gate} env_id={spec.env_id}", flush=True)
                        continue
                config = {
                    "env_id": spec.env_id,
                    "rule_variant": os.environ.get("OURS_RULE_VARIANT", "rule_v1"),
                    "horizon": spec.horizon,
                    "family": spec.family,
                    "gate": gate,
                    "args": jsonable(vars(args)),
                }
                write_json(task_dir / "config.json", config)
                print(f"run_start gate={gate} env_id={spec.env_id}", flush=True)
                started = time.time()
                try:
                    summary = run_one_task(
                        env_id=spec.env_id,
                        horizon=spec.horizon,
                        stage_observer=gate,
                        output_dir=task_dir,
                        qwen_client=qwen_client,
                        args=args,
                    )
                    summary["total_walltime_s"] = time.time() - started
                    write_json(summary_path, summary)
                    stale_failure_path = task_dir / "failure.json"
                    if stale_failure_path.exists():
                        stale_failure_path.unlink()
                    print(
                        f"run_ok gate={gate} env_id={spec.env_id} "
                        f"best_success={summary.get('best_eval_success_rate', -1):.4f} "
                        f"walltime_s={summary['total_walltime_s']:.3f}",
                        flush=True,
                    )
                except Exception as error:
                    failure = {
                        "status": "failed",
                        "gate": gate,
                        "env_id": spec.env_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    write_json(task_dir / "failure.json", failure)
                    print(f"run_failed gate={gate} env_id={spec.env_id} error={type(error).__name__}: {error}", flush=True)
        finally:
            if qwen_client is not None:
                qwen_client.close()
    write_json(args.output_root / "failures.json", failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

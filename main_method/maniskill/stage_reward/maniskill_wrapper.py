"""ManiSkill wrappers that replace the environment reward with ours."""

from __future__ import annotations

from typing import Any
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from mani_skill.utils import common

from stage_reward.core import BatchedStageReward
from stage_reward.rule_specs import get_rule_stage_spec


class StageAwareRewardWrapper(gym.Wrapper):
    """Shared stage observer with independently selectable scalar reward.

    RuleStage is a privileged upper bound. QwenStage receives transitions from
    a separate visual observer. Both can be shared by ours and comparison
    rewards without changing actor/critic observations.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        env_id: str,
        reward_backend: str,
        stage_observer: str = "rule",
        visual_observer: Any | None = None,
        append_stage_obs: bool = True,
        transition_authority_mode: str = "rule_candidate",
    ):
        super().__init__(env)
        base = self.unwrapped
        if reward_backend not in {
            "ours_rule",
            "per_subtask_sparse",
            "completed_cumulative",
            "all_dense_with_stage_actor",
            "environment",
            "code_as_reward",
        }:
            raise ValueError(f"Unsupported reward backend: {reward_backend}")
        self.reward_backend = reward_backend
        self.code_expression = None
        if reward_backend == "code_as_reward":
            expression_path = os.environ.get("CODE_REWARD_EXPRESSION_FILE")
            if not expression_path:
                raise ValueError("code_as_reward requires CODE_REWARD_EXPRESSION_FILE")
            payload = json.loads(Path(expression_path).read_text(encoding="utf-8"))
            self.code_expression = str(payload.get("expression", "")).strip()
            if not self.code_expression:
                raise ValueError(f"empty code-as-reward expression: {expression_path}")
        if stage_observer not in {"rule", "qwen"}:
            raise ValueError(f"Unsupported stage observer: {stage_observer}")
        if stage_observer == "qwen" and visual_observer is None:
            raise ValueError(
                "stage_observer='qwen' requires a visual_observer instance"
            )
        if stage_observer == "rule" and visual_observer is not None:
            raise ValueError("RuleStage must not receive a visual observer")
        if transition_authority_mode not in {
            "rule_candidate",
            "rule_candidate_and_external_authorization",
            "external_authorization",
        }:
            raise ValueError(
                f"Unsupported transition_authority_mode: {transition_authority_mode}"
            )
        if stage_observer == "rule" and transition_authority_mode != "rule_candidate":
            raise ValueError("RuleStage requires transition_authority_mode='rule_candidate'")
        if stage_observer == "qwen" and transition_authority_mode == "rule_candidate":
            raise ValueError("QwenStage requires an external transition authority mode")
        self.stage_observer = stage_observer
        self.visual_observer = visual_observer
        self.append_stage_obs = bool(append_stage_obs)
        self.transition_authority_mode = transition_authority_mode
        self.rule_spec = get_rule_stage_spec(env_id)
        self.stage_names = self.rule_spec.stage_names
        self.num_stages = len(self.stage_names)
        base_single_space = env.get_wrapper_attr("single_observation_space")
        if not isinstance(base_single_space, gym.spaces.Box) or len(
            base_single_space.shape
        ) != 1:
            raise TypeError(
                "Stage observations require a flat Box state space; "
                f"found {base_single_space}"
            )
        self.base_single_observation_space = base_single_space
        stage_width = self.num_stages + 1
        self.single_stage_critic_observation_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    base_single_space.low,
                    np.zeros(stage_width, dtype=base_single_space.dtype),
                )
            ),
            high=np.concatenate(
                (
                    base_single_space.high,
                    np.ones(stage_width, dtype=base_single_space.dtype),
                )
            ),
            dtype=np.float32,
        )
        self.machine = BatchedStageReward(
            num_envs=base.num_envs,
            num_stages=self.num_stages,
            device=base.device,
            config=self.rule_spec.config,
        )
        if self.append_stage_obs:
            self.single_observation_space = self.single_stage_critic_observation_space
            self.observation_space = batch_space(
                self.single_observation_space, n=base.num_envs
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        potentials, _, _, _ = self._reward_inputs(info)
        options = kwargs.get("options")
        env_idx = None if options is None else options.get("env_idx")
        self.machine.reset(potentials, env_idx=env_idx)
        if self.stage_observer == "qwen":
            self.visual_observer.reset(self.env, env_idx=env_idx)
        info.update(self._observation_info(obs))
        return self._policy_observation(obs), info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        potentials, gates, maintenance, safety = self._reward_inputs(info)
        transition_authorization = None
        if self.stage_observer == "qwen":
            rule_candidate = self.machine.rule_candidate(gates)
            active = self.machine.stage < self.machine.num_stages
            gather_stage = self.machine.stage.clamp(max=self.machine.num_stages - 1)
            rows = torch.arange(self.machine.num_envs, device=self.machine.device)
            # Audit against the persistent transition candidate used by the
            # training protocol, rather than the instantaneous raw gate bit.
            ground_truth_complete = rule_candidate.bool() & active
            transition_authorization = self.visual_observer.authorize(
                self.env,
                self.machine.stage,
                rule_candidate,
                ground_truth_complete=ground_truth_complete,
            )
        reward_form_mode = (
            "staged_reset" if self.reward_backend in {"ours_rule", "per_subtask_sparse", "environment", "code_as_reward"}
            else self.reward_backend
        )
        reward, diagnostics = self.machine.step(
            potentials=potentials,
            gates=gates,
            maintenance=maintenance,
            safety_penalty=safety,
            transition_authorization=transition_authorization,
            reward_form_mode=reward_form_mode,
            transition_authority_mode=self.transition_authority_mode,
        )
        diagnostics["env_reward"] = env_reward
        diagnostics["ours_reward"] = reward
        diagnostics["reward_backend"] = self.reward_backend
        diagnostics["stage_observer"] = self.stage_observer
        diagnostics["transition_authority_mode"] = self.transition_authority_mode
        if self.reward_backend == "code_as_reward":
            # Expressions are frozen artifacts, not model-generated at runtime.
            # The stage machine and observations above remain unchanged.
            local_values = {
                "reward": reward,
                "env_reward": env_reward,
                "potentials": potentials,
                "gates": gates,
                "maintenance": maintenance,
                "safety": safety,
                "stage": self.machine.stage,
            }
            selected_reward = eval(self.code_expression, {"__builtins__": {}, "torch": torch}, local_values)
            if not isinstance(selected_reward, torch.Tensor):
                selected_reward = torch.as_tensor(selected_reward, device=reward.device, dtype=reward.dtype)
            selected_reward = selected_reward.reshape(reward.shape)
            diagnostics["code_expression"] = self.code_expression
        else:
            if self.reward_backend == "environment":
                selected_reward = env_reward
            elif self.reward_backend == "per_subtask_sparse":
                selected_reward = diagnostics["transition"].to(dtype=reward.dtype)
            else:
                selected_reward = reward
        info["stage_reward"] = diagnostics
        info.update(self._observation_info(obs))
        return (
            self._policy_observation(obs),
            selected_reward,
            terminated,
            truncated,
            info,
        )

    def _policy_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.append_stage_obs:
            return obs
        return self._stage_observation(obs)

    def _stage_observation(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat((obs, self._stage_one_hot(obs)), dim=-1)

    def _stage_one_hot(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            self.machine.stage, num_classes=self.num_stages + 1
        ).to(dtype=obs.dtype)

    def _robot_proprio_observation(self, obs: torch.Tensor) -> torch.Tensor:
        proprio = self.unwrapped.agent.get_proprioception()
        return common.flatten_state_dict(
            proprio, use_torch=True, device=obs.device
        ).to(dtype=obs.dtype)

    def _observation_info(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "stage_critic_observation": self._stage_observation(obs),
            "active_stage_one_hot": self._stage_one_hot(obs),
            "robot_proprio_observation": self._robot_proprio_observation(obs),
        }

    def _reward_inputs(self, info):
        result = self.rule_spec.compute(self.unwrapped, info)
        return (
            result.potentials,
            result.gates,
            result.maintenance,
            result.safety_penalty,
        )

    def close(self):
        if self.visual_observer is not None:
            self.visual_observer.close()
            self.visual_observer = None
        return super().close()


def wrap_stage_reward_env(
    env: gym.Env, env_id: str, backend: str
) -> gym.Env:
    """Backward-compatible entry point for the ours_rule reward."""
    if backend != "ours_rule":
        raise ValueError(f"Unknown stage reward backend: {backend}")
    return wrap_stage_aware_env(
        env,
        env_id=env_id,
        stage_observer="rule",
        reward_backend="ours_rule",
        append_stage_obs=True,
    )


def wrap_stage_aware_env(
    env: gym.Env,
    *,
    env_id: str,
    stage_observer: str,
    reward_backend: str,
    append_stage_obs: bool,
    visual_observer: Any | None = None,
    transition_authority_mode: str | None = None,
) -> gym.Env:
    if stage_observer not in {"rule", "qwen"}:
        raise ValueError(f"Unknown stage observer: {stage_observer}")
    if transition_authority_mode is None:
        transition_authority_mode = (
            "rule_candidate"
            if stage_observer == "rule"
            else "rule_candidate_and_external_authorization"
        )
    return StageAwareRewardWrapper(
        env,
        env_id=env_id,
        reward_backend=reward_backend,
        stage_observer=stage_observer,
        visual_observer=visual_observer,
        append_stage_obs=append_stage_obs,
        transition_authority_mode=transition_authority_mode,
    )

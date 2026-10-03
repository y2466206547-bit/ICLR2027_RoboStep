"""Train a single shared PredStageActor on an observable MT50 reach target."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from .suite_qwen_gate import MandatoryQwenStageGate
from .target_reach_vec_env import MetaWorldTargetReachVecEnv
from .target_reach_core import TargetReachConfig
from .train_drawer import (
    ISAAC_PYTHON,
    METAWORLD_ROOT,
    QWEN_MODEL,
    QWEN_WORKER,
)


QWEN_PROMPT = (
    METAWORLD_ROOT / "stage_reward" / "prompts" / "qwen_target_reach_gate.md"
)


def stage_instruction(task: str) -> str:
    if task == "hand-insert-v3":
        return (
            "State has found a visually plausible completed hand insertion at "
            "the visible aperture. Audit that the TCP is centered inside the "
            "aperture rather than merely near its outside surface. Do not "
            "rejudge sub-centimeter metric uncertainty from pixels."
        )
    return (
        f"State has verified that the robot TCP is within 5 cm of the visible "
        f"task target for MetaWorld {task}. Audit that the target and TCP are "
        "visibly present in the correct relation, with no clearly wrong scene. "
        "Do not rejudge sub-centimeter metric uncertainty from pixels."
    )


def task_reward_config(task: str) -> TargetReachConfig:
    if task == "hand-insert-v3":
        # The task marker is recessed below the visible aperture. A read-only
        # reference-frame audit using the same tcp_center as this VecEnv places
        # the successful TCP approximately 4.7 cm above and 1.3 cm behind that
        # marker. The offset is a task-semantic target definition; scripted
        # actions and native reward remain excluded from PPO and the gate.
        return TargetReachConfig(
            dense_scale=18.0,
            transition_bonus=40.0,
            required_hold=2,
            enter_distance_m=0.018,
            exit_distance_m=0.024,
            target_offset_xyz=(-0.0048, -0.0125, 0.0468),
        )
    return TargetReachConfig()


def runner_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "device": args.train_device,
        "num_steps_per_env": args.steps_per_env,
        "max_iterations": args.iterations,
        "save_interval": args.save_interval,
        "experiment_name": "metaworld_target_reach_core",
        "run_name": args.run_name,
        "logger": "tensorboard",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": args.init_noise_std,
            "noise_std_type": "scalar",
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": args.entropy_coef,
            "num_learning_epochs": 8,
            "num_mini_batches": 8,
            "learning_rate": args.learning_rate,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("reach-v3", "reach-wall-v3", "hand-insert-v3"))
    parser.add_argument("--gate", choices=("rule", "qwen"), required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--max-episode-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--qwen-gpu", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--init-noise-std", type=float, default=0.35)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=METAWORLD_ROOT / "results" / "mt50_target_reach",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.run_name = args.run_name or (
        f"{args.task}_predstage_{args.gate}_scratch_seed{args.seed}_{timestamp}"
    )
    run_dir = args.output_root.expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    instruction = stage_instruction(args.task)
    gate = None
    if args.gate == "qwen":
        gate = MandatoryQwenStageGate(
            output_dir=run_dir / "mandatory_qwen",
            python_executable=ISAAC_PYTHON,
            worker_script=QWEN_WORKER,
            model_path=QWEN_MODEL,
            prompt_template=QWEN_PROMPT,
            stage_names=("reach_target",),
            stage_instructions=(instruction,),
            qwen_gpu=args.qwen_gpu,
        )
    env = MetaWorldTargetReachVecEnv(
        task=args.task,
        stage_instruction=instruction,
        num_envs=args.num_envs,
        seed=args.seed,
        gate_mode=args.gate,
        qwen_gate=gate,
        config=task_reward_config(args.task),
        device=args.train_device,
        max_episode_length=args.max_episode_length,
    )
    config = runner_config(args)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args)
                | {
                    "output_root": str(args.output_root),
                    "resume": str(args.resume.resolve()) if args.resume else None,
                },
                "runner": config,
                "environment": env.cfg,
                "stage_instruction": instruction,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        runner = OnPolicyRunner(env, config, log_dir=str(run_dir), device=args.train_device)
        if args.resume is not None:
            runner.load(
                str(args.resume.expanduser().resolve()),
                load_optimizer=True,
                map_location=args.train_device,
            )
            runner.current_learning_iteration += 1
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        summary = env.statistics() | {
            "checkpoint": str(run_dir / f"model_{runner.current_learning_iteration}.pt"),
            "resumed_from": str(args.resume) if args.resume else None,
        }
        (run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()

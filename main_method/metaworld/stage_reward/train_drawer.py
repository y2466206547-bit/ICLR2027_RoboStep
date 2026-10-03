"""Train a shared PredStageActor on staged MetaWorld drawer tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from .drawer_open_core import (
    STAGE_INSTRUCTIONS,
    STAGE_NAMES,
    DrawerRewardConfig,
)
from .drawer_vec_env import MetaWorldDrawerVecEnv
from .qwen_gate import MandatoryQwenGate


ROOT = Path(__file__).resolve().parents[3]
METAWORLD_ROOT = Path(__file__).resolve().parents[1]
STAGE_POLICY_ROOT = ROOT / "main_method" / "stage_policy"
ISAAC_PYTHON = Path(os.environ.get("STAGE_TRANSITION_VLM_PYTHON", "python"))
QWEN_MODEL = Path(
    os.environ.get("STAGE_TRANSITION_VLM_MODEL_PATH", "models/Qwen3-VL-8B-Instruct")
)
QWEN_WORKER = Path(
    os.environ.get(
        "STAGE_TRANSITION_VLM_WORKER",
        str(STAGE_POLICY_ROOT / "run_qwen_stage_live_worker.py"),
    )
)
QWEN_PROMPT = (
    METAWORLD_ROOT / "stage_reward" / "prompts" / "qwen_drawer_stage_gate.md"
)
QWEN_CLOSE_PROMPT = (
    METAWORLD_ROOT
    / "stage_reward"
    / "prompts"
    / "qwen_drawer_close_stage_gate.md"
)
CLOSE_STAGE_NAMES = (
    "align_above_handle",
    "descend_and_contact_handle",
    "push_drawer_closed",
)
CLOSE_STAGE_INSTRUCTIONS = (
    "State has verified exact above-handle pregrasp geometry. Audit only the correct drawer scene and that the gripper is not obviously at a wrong part. Do not rejudge centimeter-scale alignment from pixels.",
    "State has verified exact drawer-handle contact geometry. Audit only that there is no obvious empty relation, wrong-part contact, or lost drawer handle. Do not rejudge centimeter-scale contact from pixels.",
    "State has verified the drawer handle is within 5.5 cm of the closed target, matching MetaWorld's official close-success tolerance. Audit only that the drawer visibly moved inward relative to the reference, with no open drawer, robot-only motion, or re-opening.",
)


def drawer_task_spec(task_name: str) -> tuple[Path, tuple[str, ...], tuple[str, ...], float]:
    if task_name == "drawer-open-v3":
        return QWEN_PROMPT, STAGE_NAMES, STAGE_INSTRUCTIONS, -1.0
    if task_name == "drawer-close-v3":
        return QWEN_CLOSE_PROMPT, CLOSE_STAGE_NAMES, CLOSE_STAGE_INSTRUCTIONS, 1.0
    raise ValueError(f"unsupported drawer task: {task_name}")


def drawer_reward_config(task_name: str) -> DrawerRewardConfig:
    """Return only the task-specific terminal geometry of the shared CORE.

    ``drawer-close-v3`` defines official success at 0.04 + 0.015 m. Its
    final *candidate* must therefore use 5.5 cm before a frozen-frame Qwen
    audit; this does not authorize the transition directly.
    """
    if task_name == "drawer-close-v3":
        return DrawerRewardConfig(
            dense_scales=(6.0, 22.0, 160.0),
            transition_bonuses=(8.0, 36.0, 210.0),
            required_holds=(2, 3, 1),
            affordance_offset_xyz=(0.0, 0.060, -0.020),
            action_rate_weight=0.015,
            action_curve_weight=0.008,
            boundary_weight=0.020,
            open_distance_enter_m=0.055,
            open_distance_exit_m=0.070,
        )
    if task_name == "drawer-open-v3":
        return DrawerRewardConfig(
            dense_scales=(6.0, 24.0, 160.0),
            transition_bonuses=(8.0, 36.0, 210.0),
            required_holds=(2, 2, 1),
            hook_distance_enter_m=0.035,
            hook_distance_exit_m=0.050,
            hook_height_abs_max_m=0.035,
            action_rate_weight=0.015,
            action_curve_weight=0.008,
            boundary_weight=0.020,
        )
    raise ValueError(f"unsupported drawer task: {task_name}")


def runner_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "device": args.train_device,
        "num_steps_per_env": args.steps_per_env,
        "max_iterations": args.iterations,
        "save_interval": args.save_interval,
        "experiment_name": "metaworld_drawer_stage_reward",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("rule", "qwen"), required=True)
    parser.add_argument(
        "--task-name",
        choices=("drawer-open-v3", "drawer-close-v3"),
        default="drawer-open-v3",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=500)
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
        default=METAWORLD_ROOT / "results" / "drawer_stage_reward",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"{args.task_name.removesuffix('-v3').replace('-', '_')}_v1_predstage_{args.gate}_scratch_seed{args.seed}_{timestamp}"
    )
    run_dir = args.output_root.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    args.run_name = run_name

    qwen_prompt, stage_names, stage_instructions, gripper_command = drawer_task_spec(
        args.task_name
    )
    qwen_gate = None
    if args.gate == "qwen":
        qwen_dir = run_dir / "mandatory_qwen"
        qwen_dir.mkdir()
        qwen_gate = MandatoryQwenGate(
            output_dir=qwen_dir,
            python_executable=ISAAC_PYTHON,
            worker_script=QWEN_WORKER,
            model_path=QWEN_MODEL,
            prompt_template=qwen_prompt,
            qwen_gpu=args.qwen_gpu,
            stage_names=stage_names,
            stage_instructions=stage_instructions,
        )

    reward_config = drawer_reward_config(args.task_name)
    env = MetaWorldDrawerVecEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        gate_mode=args.gate,
        reward_config=reward_config,
        qwen_gate=qwen_gate,
        task_name=args.task_name,
        gripper_command=gripper_command,
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
                    "resume": (
                        str(args.resume.resolve()) if args.resume else None
                    ),
                },
                "runner": config,
                "environment": env.cfg,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        runner = OnPolicyRunner(
            env,
            config,
            log_dir=str(run_dir),
            device=args.train_device,
        )
        if args.resume is not None:
            runner.load(
                str(args.resume.expanduser().resolve()),
                load_optimizer=True,
                map_location=args.train_device,
            )
            runner.current_learning_iteration += 1
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=False,
        )
        summary = env.statistics()
        summary["resumed_from"] = (
            str(args.resume.expanduser().resolve())
            if args.resume is not None
            else None
        )
        summary["checkpoint"] = str(
            run_dir / f"model_{runner.current_learning_iteration}.pt"
        )
        (run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()

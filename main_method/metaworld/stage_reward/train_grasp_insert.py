"""Train the generic PredStageActor grasp-and-insert CORE on MT50."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from .grasp_insert_core import GraspInsertConfig
from .grasp_insert_vec_env import MetaWorldGraspInsertVecEnv
from .suite_qwen_gate import MandatoryQwenStageGate
from .train_drawer import ISAAC_PYTHON, METAWORLD_ROOT, QWEN_MODEL, QWEN_WORKER
from .train_target_reach import runner_config


QWEN_PROMPT = METAWORLD_ROOT / "stage_reward" / "prompts" / "qwen_target_reach_gate.md"
STAGE_NAMES = ("pregrasp", "capture_move", "transport_insert")
STAGE_INSTRUCTIONS = (
    "State has already verified exact pregrasp geometry. Do not judge gripper aperture or centimeter alignment from pixels. Reply success whenever the frozen scene visibly contains the expected robot, tabletop puck, and target; reply failure only for a clearly wrong or missing scene/object.",
    "State has already verified capture and object displacement. Do not rejudge contact or movement magnitude from pixels. Reply success when the frozen scene visibly contains the expected robot and puck, and failure only for a clearly wrong or missing scene/object.",
    "State has already verified object-at-aperture geometry. Do not rejudge centimeter placement from pixels. Reply success when the frozen scene visibly contains the expected robot, puck, and target, and failure only for a clearly wrong or missing scene/object.",
)


def task_reward_config(task: str) -> GraspInsertConfig:
    if task == "pick-place-v3":
        return GraspInsertConfig(
            dense_scales=(4.0, 10.0, 32.0),
            transition_bonuses=(5.0, 16.0, 70.0),
            pregrasp_xy_enter_m=0.040,
            pregrasp_xy_exit_m=0.060,
            pregrasp_height_low_m=0.045,
            pregrasp_height_high_m=0.140,
            capture_lift_enter_m=0.035,
            capture_move_exit_m=0.020,
            insert_enter_m=0.075,
            insert_exit_m=0.090,
            stage2_tcp_goal_weight=0.30,
            stage2_tcp_obj_weight=0.25,
            stage2_tcp_goal_z_offset_m=0.02,
        )
    if task == "basketball-v3":
        return GraspInsertConfig(
            dense_scales=(4.0, 14.0, 42.0),
            transition_bonuses=(5.0, 22.0, 90.0),
            pregrasp_xy_enter_m=0.045,
            pregrasp_xy_exit_m=0.065,
            pregrasp_height_low_m=0.050,
            pregrasp_height_high_m=0.190,
            capture_move_enter_m=0.200,
            capture_lift_enter_m=0.180,
            capture_move_exit_m=0.140,
            insert_enter_m=0.090,
            insert_exit_m=0.115,
            stage2_tcp_goal_weight=0.35,
            stage2_tcp_obj_weight=0.20,
            stage2_tcp_goal_z_offset_m=0.04,
            target_z_override_m=0.300,
        )
    if task in {"bin-picking-v3", "pick-place-wall-v3", "shelf-place-v3"}:
        return GraspInsertConfig(
            dense_scales=(4.0, 10.0, 32.0),
            transition_bonuses=(5.0, 16.0, 70.0),
            pregrasp_xy_enter_m=0.040,
            pregrasp_xy_exit_m=0.060,
            pregrasp_height_low_m=0.045,
            pregrasp_height_high_m=0.160,
            capture_lift_enter_m=0.040,
            capture_move_exit_m=0.025,
            insert_enter_m=0.070,
            insert_exit_m=0.090,
            stage2_tcp_goal_weight=0.30,
            stage2_tcp_obj_weight=0.25,
            stage2_tcp_goal_z_offset_m=0.02,
        )
    if task in {"assembly-v3", "peg-insert-side-v3"}:
        return GraspInsertConfig(
            dense_scales=(4.0, 10.0, 34.0),
            transition_bonuses=(5.0, 16.0, 75.0),
            pregrasp_xy_enter_m=0.045,
            pregrasp_xy_exit_m=0.065,
            pregrasp_height_low_m=0.040,
            pregrasp_height_high_m=0.160,
            capture_move_enter_m=0.045,
            capture_move_exit_m=0.030,
            insert_enter_m=0.070,
            insert_exit_m=0.090,
            stage2_tcp_goal_weight=0.35,
            stage2_tcp_obj_weight=0.25,
            stage2_tcp_goal_z_offset_m=0.02,
        )
    if task in {"disassemble-v3", "peg-unplug-side-v3", "pick-out-of-hole-v3"}:
        return GraspInsertConfig(
            dense_scales=(4.0, 10.0, 36.0),
            transition_bonuses=(5.0, 16.0, 80.0),
            pregrasp_xy_enter_m=0.045,
            pregrasp_xy_exit_m=0.065,
            pregrasp_height_low_m=0.035,
            pregrasp_height_high_m=0.170,
            capture_lift_enter_m=0.040,
            capture_move_exit_m=0.025,
            insert_enter_m=0.060,
            insert_exit_m=0.080,
            stage2_tcp_goal_weight=0.35,
            stage2_tcp_obj_weight=0.25,
            stage2_tcp_goal_z_offset_m=0.02,
        )
    if task in {"hammer-v3", "stick-pull-v3", "stick-push-v3"}:
        return GraspInsertConfig(
            dense_scales=(4.0, 10.0, 30.0),
            transition_bonuses=(5.0, 16.0, 65.0),
            pregrasp_xy_enter_m=0.050,
            pregrasp_xy_exit_m=0.075,
            pregrasp_height_low_m=0.035,
            pregrasp_height_high_m=0.180,
            capture_lift_enter_m=0.040,
            capture_move_exit_m=0.025,
            insert_enter_m=0.090,
            insert_exit_m=0.120,
            stage2_tcp_goal_weight=0.40,
            stage2_tcp_obj_weight=0.20,
            stage2_tcp_goal_z_offset_m=0.02,
        )
    if task == "box-close-v3":
        return GraspInsertConfig(
            dense_scales=(4.0, 8.0, 18.0),
            transition_bonuses=(5.0, 14.0, 45.0),
            pregrasp_xy_enter_m=0.040,
            pregrasp_xy_exit_m=0.060,
            pregrasp_height_low_m=0.035,
            pregrasp_height_high_m=0.130,
            capture_move_enter_m=0.050,
            capture_move_exit_m=0.035,
            insert_enter_m=0.070,
            insert_exit_m=0.085,
            stage2_tcp_goal_weight=0.35,
            stage2_tcp_obj_weight=0.25,
            stage2_tcp_goal_z_offset_m=0.06,
        )
    if task == "coffee-pull-v3":
        return GraspInsertConfig(
            dense_scales=(4.0, 8.0, 20.0),
            transition_bonuses=(5.0, 14.0, 50.0),
            pregrasp_xy_enter_m=0.040,
            pregrasp_xy_exit_m=0.060,
            pregrasp_height_low_m=0.040,
            pregrasp_height_high_m=0.130,
            capture_move_enter_m=0.050,
            capture_move_exit_m=0.035,
            insert_enter_m=0.070,
            insert_exit_m=0.085,
        )
    return GraspInsertConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("hand-insert-v3", "coffee-pull-v3", "box-close-v3", "pick-place-v3", "basketball-v3", "bin-picking-v3", "pick-place-wall-v3", "shelf-place-v3", "assembly-v3", "peg-insert-side-v3", "disassemble-v3", "peg-unplug-side-v3", "pick-out-of-hole-v3", "hammer-v3", "stick-pull-v3", "stick-push-v3"))
    parser.add_argument("--gate", choices=("rule", "qwen", "synthetic"), required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--max-episode-length", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--qwen-gpu", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--init-noise-std", type=float, default=0.35)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", type=Path, default=METAWORLD_ROOT / "results" / "mt50_grasp_insert")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.run_name = args.run_name or f"{args.task}_predstage_{args.gate}_scratch_seed{args.seed}_{timestamp}"
    run_dir = args.output_root.expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    gate = None
    if args.gate == "qwen":
        gate = MandatoryQwenStageGate(
            output_dir=run_dir / "mandatory_qwen",
            python_executable=ISAAC_PYTHON,
            worker_script=QWEN_WORKER,
            model_path=QWEN_MODEL,
            prompt_template=QWEN_PROMPT,
            stage_names=STAGE_NAMES,
            stage_instructions=STAGE_INSTRUCTIONS,
            qwen_gpu=args.qwen_gpu,
        )
    env = MetaWorldGraspInsertVecEnv(
        task=args.task,
        stage_instructions=STAGE_INSTRUCTIONS,
        num_envs=args.num_envs,
        seed=args.seed,
        gate_mode=args.gate,
        qwen_gate=gate,
        config=task_reward_config(args.task),
        device=args.train_device,
        max_episode_length=args.max_episode_length,
    )
    config = runner_config(args)
    config["experiment_name"] = "metaworld_grasp_insert_core"
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args) | {
                    "output_root": str(args.output_root),
                    "resume": str(args.resume.resolve()) if args.resume else None,
                },
                "runner": config,
                "environment": env.cfg,
                "stage_names": STAGE_NAMES,
                "stage_instructions": STAGE_INSTRUCTIONS,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    try:
        runner = OnPolicyRunner(env, config, log_dir=str(run_dir), device=args.train_device)
        if args.resume is not None:
            runner.load(str(args.resume.expanduser().resolve()), load_optimizer=True, map_location=args.train_device)
            runner.current_learning_iteration += 1
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        summary = env.statistics() | {
            "checkpoint": str(run_dir / f"model_{runner.current_learning_iteration}.pt"),
            "resumed_from": str(args.resume) if args.resume else None,
        }
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()


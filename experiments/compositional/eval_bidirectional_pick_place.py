"""Evaluate a bidirectional same-object composition checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from .grasp_insert_core import GraspInsertConfig
from .grasp_insert_vec_env import MetaWorldGraspInsertVecEnv
from .train_bidirectional_pick_place import (
    APPEND_STAGE_CONTEXT_PROTOCOL,
    RAW_STAGE_CONTEXT_PROTOCOL,
    STAGE_ID_SIZE,
    _install_state_only_normalizers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gate", choices=("rule", "synthetic"), default="rule")
    parser.add_argument("--source-macro-index", type=int, default=None)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0 or args.episodes % args.num_envs:
        raise SystemExit("episodes must be positive and divisible by num-envs")
    checkpoint = args.checkpoint.expanduser().resolve()
    saved = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    task_definition = saved["task_definition"]
    env_cfg = saved["environment"]
    runner_cfg = json.loads(json.dumps(saved["runner"]))
    runner_cfg["device"] = "cpu"
    output = args.output or (checkpoint.parent / (
        f"eval_{args.gate}_{checkpoint.stem}_seed{args.seed}_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ))
    env = MetaWorldGraspInsertVecEnv(
        task=str(env_cfg["task"]),
        stage_instructions=tuple(saved["stage_instructions"]),
        num_envs=args.num_envs,
        seed=args.seed,
        gate_mode=args.gate,
        config=GraspInsertConfig(**env_cfg["reward_config"]),
        device="cpu",
        max_episode_length=int(env_cfg["max_episode_length"]),
        max_macro_length=int(
            env_cfg.get("max_macro_length", env_cfg["max_episode_length"])
        ),
        camera_name=str(env_cfg.get("qwen_camera") or "corner3"),
        macro_count=int(task_definition["macro_count"]),
        stage_id_size=int(task_definition.get("stage_id_size", STAGE_ID_SIZE)),
        source_macro_index=int(
            args.source_macro_index
            if args.source_macro_index is not None
            else task_definition.get("source_macro_index", 0)
        ),
        geometry_protocol=str(
            task_definition.get("geometry_protocol", "native_forward_v0")
        ),
        reset_between_macros=bool(task_definition.get("episode_reset_between_macros", False)),
        reset_arm_between_macros=bool(task_definition.get("arm_reset_between_macros", False)),
    )
    try:
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device="cpu")
        runner.load(str(checkpoint), load_optimizer=False, map_location="cpu")
        if (
            task_definition.get("stage_context_normalization")
            in (APPEND_STAGE_CONTEXT_PROTOCOL, RAW_STAGE_CONTEXT_PROTOCOL)
        ):
            _install_state_only_normalizers(runner.alg.policy)
        policy = runner.get_inference_policy(device="cpu")
        obs = env.get_observations()
        with torch.inference_mode():
            while env.completed_episodes < args.episodes:
                obs, _, _, _ = env.step(policy(obs))
        summary = env.statistics() | {
            "checkpoint": str(checkpoint),
            "evaluation": f"deterministic_{args.gate}_gate",
            "requested_episodes": args.episodes,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "macro_count": int(task_definition["macro_count"]),
            "active_atomic_stage_count": int(task_definition["active_atomic_stage_count"]),
            "source_macro_index": int(
                args.source_macro_index
                if args.source_macro_index is not None
                else task_definition.get("source_macro_index", 0)
            ),
            "geometry_protocol": str(
                task_definition.get("geometry_protocol", "native_forward_v0")
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()

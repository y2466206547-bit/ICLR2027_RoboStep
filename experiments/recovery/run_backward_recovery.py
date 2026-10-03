#!/usr/bin/env python3
"""Run one controlled backward-regression or stagnation condition.

The actor checkpoint is frozen.  Official task success is used only as the
final outcome metric; rollback and invalid-credit decisions use the frozen
Frame-VLM-authored stage predicates in ``rule_specs.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from mani_skill.utils.structs.pose import Pose

from stage_reward.bidirectional import (
    BidirectionalBatchedStageReward,
    make_task_prerequisite_fn,
)
from stage_reward.maniskill_wrapper import StageAwareRewardWrapper
from stage_reward import run_one_iteration_sweep as sweep


@dataclass(frozen=True)
class TaskProtocol:
    failure_mode: str
    regression_trigger_stage: int
    stagnation_trigger_stage: int
    prerequisite_stage: int
    stagnation_prerequisite_stage: int
    max_steps: int
    trigger_delay: int
    severity_unit: str


PROTOCOLS = {
    "PickCube-v1": TaskProtocol(
        failure_mode="Object drop",
        regression_trigger_stage=2,
        stagnation_trigger_stage=2,
        prerequisite_stage=2,
        stagnation_prerequisite_stage=2,
        max_steps=200,
        trigger_delay=2,
        severity_unit="m",
    ),
    "RotateValveLevel2-v1": TaskProtocol(
        failure_mode="Re-close",
        regression_trigger_stage=2,
        stagnation_trigger_stage=1,
        prerequisite_stage=2,
        stagnation_prerequisite_stage=1,
        max_steps=500,
        trigger_delay=2,
        severity_unit="rad",
    ),
    "PlaceSphere-v1": TaskProtocol(
        failure_mode="Object displacement",
        regression_trigger_stage=4,
        stagnation_trigger_stage=3,
        prerequisite_stage=4,
        stagnation_prerequisite_stage=3,
        max_steps=200,
        trigger_delay=0,
        severity_unit="m",
    ),
}
nSTAGNATION_FAILURE_MODES = {
    "PickCube-v1": "Motion blocked",
    "RotateValveLevel2-v1": "Motion blocked",
    "PlaceSphere-v1": "Progress blocked",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("forward_only", "bidirectional"), required=True
    )
    parser.add_argument(
        "--intervention", choices=("regression", "stagnation"), required=True
    )
    parser.add_argument("--severity", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--rollback-dwell", type=int, default=2)
    parser.add_argument("--rollback-window", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def find_stage_wrapper(envs: Any) -> StageAwareRewardWrapper:
    current = envs._env
    while current is not None:
        if isinstance(current, StageAwareRewardWrapper):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("StageAwareRewardWrapper not found")


def make_eval_args() -> SimpleNamespace:
    return SimpleNamespace(
        no_stage_obs=False,
        actor_blind_stage=False,
        rgb_actor=False,
        rgb_dino_arch="dinov2_vits14",
        rgb_dino_checkpoint=None,
        rgb_dino_repo=Path("."),
        rgb_dino_image_size=224,
        rgb_dino_batch_size=32,
    )


def task_trigger_ready(
    task: str,
    intervention: str,
    stage: torch.Tensor,
    base: Any,
    protocol: TaskProtocol,
) -> torch.Tensor:
    target = (
        protocol.regression_trigger_stage
        if intervention == "regression"
        else protocol.stagnation_trigger_stage
    )
    ready = stage >= target
    if task == "RotateValveLevel2-v1":
        signed_rotation = (
            (base.valve.qpos - base.rest_qpos)[:, 0]
            * base.rotate_direction
        )
        progress = signed_rotation / float(base.success_threshold)
        ready &= progress >= 0.55
    return ready


def zero_actor_motion(actor: Any, mask: torch.Tensor) -> None:
    linear = actor.linear_velocity.clone()
    angular = actor.angular_velocity.clone()
    linear[mask] = 0
    angular[mask] = 0
    actor.set_linear_velocity(linear)
    actor.set_angular_velocity(angular)


def inject_regression(
    task: str, base: Any, mask: torch.Tensor, severity: float
) -> None:
    if not mask.any():
        return
    if task == "PickCube-v1":
        actor = base.cube
        p = actor.pose.p.clone()
        q = actor.pose.q.clone()
        tcp = base.agent.tcp_pose.p
        away = tcp[:, :2] - base.goal_site.pose.p[:, :2]
        norm = torch.linalg.norm(away, dim=1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0], device=p.device).expand_as(away)
        direction = torch.where(norm > 1e-5, away / norm.clamp_min(1e-5), fallback)
        p[mask, :2] = torch.clamp(
            tcp[mask, :2] + float(severity) * direction[mask], -0.25, 0.25
        )
        p[mask, 2] = float(base.cube_half_size)
        actor.set_pose(Pose.create_from_pq(p, q))
        zero_actor_motion(actor, mask)
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
        return

    if task == "PlaceSphere-v1":
        actor = base.obj
        p = actor.pose.p.clone()
        q = actor.pose.q.clone()
        direction = torch.where(
            base.bin.pose.p[:, 0] <= 0.05,
            torch.ones_like(base.bin.pose.p[:, 0]),
            -torch.ones_like(base.bin.pose.p[:, 0]),
        )
        p[mask, 0] = base.bin.pose.p[mask, 0] + direction[mask] * float(severity)
        p[mask, 1] = base.bin.pose.p[mask, 1]
        p[mask, 2] = float(base.radius)
        actor.set_pose(Pose.create_from_pq(p, q))
        zero_actor_motion(actor, mask)
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
        return

    if task == "RotateValveLevel2-v1":
        qpos = base.valve.qpos.clone()
        qvel = base.valve.qvel.clone()
        qpos[mask, 0] = (
            base.rest_qpos[mask, 0]
            - base.rotate_direction[mask] * float(severity)
        )
        qvel[mask] = 0
        base.valve.set_qpos(qpos)
        base.valve.set_qvel(qvel)
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
        return

    raise KeyError(task)


def bool_list(value: torch.Tensor) -> list[bool]:
    return [bool(item) for item in value.detach().cpu().tolist()]


def int_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().tolist()]


def run_batch(
    *,
    args: argparse.Namespace,
    batch_index: int,
    agent: torch.nn.Module,
    envs: Any,
    wrapper: StageAwareRewardWrapper,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = PROTOCOLS[args.task]
    base = envs.base_env
    machine = wrapper.machine
    prerequisite_fn = make_task_prerequisite_fn(args.task, base)
    num_envs = envs.num_envs
    device = base.device
    policy_obs, _ = envs.reset(seed=args.seed + batch_index * 1009)
    if isinstance(machine, BidirectionalBatchedStageReward):
        machine.rollback_enabled.zero_()
    clip_action = sweep.clip_action_fn(envs, device)

    trigger_seen = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    disturbed_at = torch.full_like(trigger_seen, -1)
    released_at = torch.full_like(trigger_seen, -1)
    first_invalid = torch.full_like(trigger_seen, -1)
    first_rollback = torch.full_like(trigger_seen, -1)
    first_prerequisite_recovery = torch.full_like(trigger_seen, -1)
    first_task_recovery = torch.full_like(trigger_seen, -1)
    invalid_steps = torch.zeros_like(trigger_seen)
    invalid_positive_steps = torch.zeros_like(trigger_seen)
    invalid_positive_mass = torch.zeros(num_envs, device=device)
    invalid_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    any_rollback = torch.zeros_like(invalid_seen)
    stagnation_prerequisite_broken = torch.zeros_like(invalid_seen)
    traces: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(min(3, num_envs))}

    block_steps = max(1, int(round(args.severity)))
    agent.eval()
    for step in range(protocol.max_steps):
        stage_before = machine.stage.clone()
        ready = task_trigger_ready(
            args.task, args.intervention, stage_before, base, protocol
        )
        new_ready = (trigger_seen < 0) & ready
        trigger_seen[new_ready] = step
        due = (
            (disturbed_at < 0)
            & (trigger_seen >= 0)
            & (step - trigger_seen >= protocol.trigger_delay)
        )

        if args.intervention == "regression" and due.any():
            inject_regression(args.task, base, due, args.severity)
            disturbed_at[due] = step
        elif args.intervention == "stagnation" and due.any():
            disturbed_at[due] = step
            released_at[due] = step + block_steps

        if isinstance(machine, BidirectionalBatchedStageReward):
            machine.rollback_enabled[due] = True

        with torch.no_grad():
            action = agent.get_action(policy_obs, deterministic=True)
        action = clip_action(action)
        regression_settle = due if args.intervention == "regression" else torch.zeros_like(due)
        if args.intervention == "stagnation":
            blocked = (
                (disturbed_at >= 0)
                & (step >= disturbed_at)
                & (step < released_at)
            )
        else:
            blocked = regression_settle
        if args.intervention == "stagnation" and args.task != "RotateValveLevel2-v1":
            action[blocked, :-1] = 0
            action[blocked, -1] = -1
        else:
            action[blocked] = 0

        policy_obs, reward, _, _, info = envs.step(action)
        diagnostics = info["stage_reward"]
        rollback = diagnostics.get("rollback")
        if rollback is None:
            rollback = torch.zeros(num_envs, dtype=torch.bool, device=device)
        rollback = rollback.bool()
        if args.intervention == "stagnation":
            any_rollback |= rollback & blocked
        else:
            any_rollback |= rollback & (disturbed_at >= 0)
        new_rollback = rollback & (first_rollback < 0)
        first_rollback[new_rollback] = step

        inputs = wrapper.rule_spec.compute(base, info)
        prerequisites = prerequisite_fn(inputs.potentials, inputs.gates)
        prerequisite_stage = (
            protocol.prerequisite_stage
            if args.intervention == "regression"
            else protocol.stagnation_prerequisite_stage
        )
        target_valid = prerequisites[:, prerequisite_stage]
        after_disturbance = disturbed_at >= 0

        if args.intervention == "regression":
            currently_invalid = after_disturbance & (~target_valid)
            new_invalid = currently_invalid & (first_invalid < 0)
            first_invalid[new_invalid] = step
            invalid_seen |= currently_invalid
            invalid_credit_exposure = currently_invalid & (
                diagnostics["stage_before"] >= protocol.prerequisite_stage
            )
            invalid_steps += invalid_credit_exposure.long()
            positive = reward > 1e-8
            invalid_positive_steps += (invalid_credit_exposure & positive).long()
            invalid_positive_mass += torch.where(
                invalid_credit_exposure, torch.clamp(reward, min=0.0), 0.0
            )
            recovered = (
                invalid_seen
                & target_valid
                & (first_prerequisite_recovery < 0)
                & (step > first_invalid)
            )
            first_prerequisite_recovery[recovered] = step
            success = info.get("success", torch.zeros_like(target_valid)).bool()
            task_recovered = (
                invalid_seen
                & success
                & (first_task_recovery < 0)
                & (step > first_invalid)
            )
            first_task_recovery[task_recovered] = step
        else:
            stagnation_prerequisite_broken |= blocked & (~target_valid)
            success = info.get("success", torch.zeros_like(target_valid)).bool()
            task_recovered = (
                (released_at >= 0)
                & (step >= released_at)
                & success
                & (first_task_recovery < 0)
            )
            first_task_recovery[task_recovered] = step

        for idx in traces:
            traces[idx].append(
                {
                    "step": step,
                    "stage_before": int(diagnostics["stage_before"][idx].item()),
                    "stage_after": int(diagnostics["stage_after"][idx].item()),
                    "reward": float(reward[idx].item()),
                    "gate_values": bool_list(inputs.gates[idx]),
                    "potential_values": [float(x) for x in inputs.potentials[idx].detach().cpu().tolist()],
                    "prerequisite_valid": bool(target_valid[idx].item()),
                    "rollback": bool(rollback[idx].item()),
                    "success": bool(info.get("success", target_valid)[idx].item()),
                    "disturbed": bool(disturbed_at[idx].item() == step),
                    "blocked": bool(blocked[idx].item()),
                }
            )

    records = []
    for idx in range(num_envs):
        if disturbed_at[idx] < 0:
            continue
        td = int(disturbed_at[idx].item())
        fi = int(first_invalid[idx].item())
        fr = int(first_rollback[idx].item())
        fpr = int(first_prerequisite_recovery[idx].item())
        ftr = int(first_task_recovery[idx].item())
        effective = (
            fi >= 0
            if args.intervention == "regression"
            else not bool(stagnation_prerequisite_broken[idx].item())
        )
        rollback_correct = (
            effective
            and fr >= fi
            and fr - fi <= int(args.rollback_window)
        )
        if args.intervention == "regression":
            recovery_time = None if fpr < 0 else fpr - td
            task_recovery_time = None if ftr < 0 else ftr - td
        else:
            release = int(released_at[idx].item())
            recovery_time = None if ftr < 0 else ftr - release
            task_recovery_time = recovery_time
        records.append(
            {
                "batch_index": batch_index,
                "env_index": idx,
                "seed": args.seed + batch_index * 1009,
                "intervention_effective": effective,
                "disturbance_step": td,
                "release_step": int(released_at[idx].item()),
                "first_invalid_step": fi,
                "first_rollback_step": fr,
                "rollback_correct": rollback_correct,
                "false_rollback": bool(any_rollback[idx].item())
                if args.intervention == "stagnation"
                else None,
                "prerequisite_recovered": fpr >= 0,
                "task_recovered": ftr >= 0,
                "recovery_time_steps": recovery_time,
                "task_recovery_time_steps": task_recovery_time,
                "invalid_steps": int(invalid_steps[idx].item()),
                "invalid_positive_steps": int(
                    invalid_positive_steps[idx].item()
                ),
                "invalid_positive_reward_mass": float(
                    invalid_positive_mass[idx].item()
                ),
                "final_stage": int(machine.stage[idx].item()),
            }
        )
    metadata = {
        "attempted_envs": num_envs,
        "triggered_envs": len(records),
        "trace": traces,
    }
    return records, metadata


def summarize_records(
    args: argparse.Namespace, records: list[dict[str, Any]], attempts: int
) -> dict[str, Any]:
    if args.intervention == "regression":
        eligible = [row for row in records if row["intervention_effective"]]
    else:
        eligible = records
    recovered = [
        row["recovery_time_steps"]
        for row in eligible
        if row["recovery_time_steps"] is not None
    ]
    invalid_steps = sum(row["invalid_steps"] for row in eligible)
    positive_steps = sum(row["invalid_positive_steps"] for row in eligible)
    positive_mass = sum(
        row["invalid_positive_reward_mass"] for row in eligible
    )
    denominator = max(1, len(eligible))
    return {
        "attempted_episodes": attempts,
        "triggered_episodes": len(records),
        "effective_episodes": len(eligible),
        "intervention_coverage": len(records) / max(1, attempts),
        "rollback_accuracy": (
            sum(bool(row["rollback_correct"]) for row in eligible) / denominator
            if args.intervention == "regression"
            else None
        ),
        "false_rollback_rate": (
            sum(bool(row["false_rollback"]) for row in eligible) / denominator
            if args.intervention == "stagnation"
            else None
        ),
        "recovery_success_rate": sum(
            bool(row["task_recovered"]) for row in eligible
        )
        / denominator,
        "prerequisite_recovery_rate": sum(
            bool(row["prerequisite_recovered"]) for row in eligible
        )
        / denominator,
        "mean_recovery_time_steps": (
            float(np.mean(recovered)) if recovered else None
        ),
        "invalid_reward_rate": positive_steps / max(1, invalid_steps),
        "invalid_positive_reward_per_step": positive_mass
        / max(1, invalid_steps),
        "invalid_steps": invalid_steps,
        "invalid_positive_steps": positive_steps,
    }


def main() -> None:
    args = parse_args()
    if args.task == "RotateValveLevel2-v1":
        os.environ["OURS_TASK_MANIFEST"] = str(
            Path(__file__).with_name("tasks_19_replacement_v1.json")
        )
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("episodes and batch-size must be positive")
    protocol = PROTOCOLS[args.task]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sweep.seed_everything(args.seed, deterministic=True)
    envs = sweep.make_env(
        env_id=args.task,
        num_envs=args.batch_size,
        reconfiguration_freq=1,
        stage_observer="rule",
        reward_backend="ours_rule",
        append_stage_obs=True,
        qwen_client=None,
        frame_vlm_dir=Path("."),
        query_interval_steps=10,
        qwen_query_mode="candidate_retry",
        transition_authority_mode=None,
        ignore_terminations=True,
        eval_env=True,
    )
    envs.auto_reset = False
    wrapper = find_stage_wrapper(envs)
    if args.method == "bidirectional":
        wrapper.machine = BidirectionalBatchedStageReward(
            num_envs=args.batch_size,
            num_stages=wrapper.num_stages,
            device=envs.base_env.device,
            config=wrapper.rule_spec.config,
            prerequisite_fn=make_task_prerequisite_fn(args.task, envs.base_env),
            rollback_dwell_steps=args.rollback_dwell,
        )

    try:
        agent = sweep.Agent(envs).to(device)
        checkpoint_path = args.checkpoint.expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        all_records: list[dict[str, Any]] = []
        all_traces: list[dict[str, Any]] = []
        attempts = 0
        for batch_index in range(args.max_batches):
            records, metadata = run_batch(
                args=args,
                batch_index=batch_index,
                agent=agent,
                envs=envs,
                wrapper=wrapper,
            )
            attempts += int(metadata["attempted_envs"])
            all_records.extend(records)
            all_traces.append(
                {"batch_index": batch_index, "traces": metadata["trace"]}
            )
            effective = [
                row for row in all_records if row["intervention_effective"]
            ]
            if len(effective) >= args.episodes:
                break
        effective_records = [
            row for row in all_records if row["intervention_effective"]
        ]
        selected = effective_records[: args.episodes]
    finally:
        envs.close()

    summary = summarize_records(args, selected, attempts)
    payload = {
        "schema_version": 1,
        "task": args.task,
        "failure_mode": (
            protocol.failure_mode
            if args.intervention == "regression"
            else STAGNATION_FAILURE_MODES[args.task]
        ),
        "method": args.method,
        "intervention": args.intervention,
        "severity": args.severity,
        "severity_unit": (
            "policy_steps"
            if args.intervention == "stagnation"
            else protocol.severity_unit
        ),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "evaluation_seed": args.seed,
        "actor_receives_stage_id": True,
        "official_dense_reward_used": False,
        "official_success_used_for_outcome_only": True,
        "rollback_dwell_steps": args.rollback_dwell,
        "rollback_window_steps": args.rollback_window,
        "protocol": {
            "regression_trigger_stage": protocol.regression_trigger_stage,
            "stagnation_trigger_stage": protocol.stagnation_trigger_stage,
            "prerequisite_stage": protocol.prerequisite_stage,
            "stagnation_prerequisite_stage": protocol.stagnation_prerequisite_stage,
            "trigger_delay_steps": protocol.trigger_delay,
            "max_steps": protocol.max_steps,
        },
        "summary": summary,
        "episodes": selected,
        "diagnostic_traces": all_traces[:1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

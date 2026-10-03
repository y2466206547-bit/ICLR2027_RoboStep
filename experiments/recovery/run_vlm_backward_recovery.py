#!/usr/bin/env python3
"""Evaluate zero-shot recovery from visually inferred task-stage regression.

Policies are frozen checkpoints trained on clean tasks.  Forward transitions
remain the original RuleGate transitions.  After a controlled disturbance,
Qwen sees RGB images plus semantic stage descriptions and directly predicts
the earlier stage index that should condition the same actor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

from stage_reward import run_one_iteration_sweep as sweep
from stage_reward.maniskill_wrapper import StageAwareRewardWrapper
from stage_reward.visual_observer import SynchronousVisualStageObserver
from stage_reward.vlm_rollback_client import QwenRollbackClient, RollbackStageRequest


@dataclass(frozen=True)
class RecoveryProtocol:
    instruction: str
    stages: tuple[str, ...]
    trigger_stage: int
    oracle_target_stage: int
    severity_unit: str
    default_severity: float
    max_pre_steps: int
    recovery_steps: int


PROTOCOLS = {
    "PushCube-v1": RecoveryProtocol(
        instruction="Push the blue cube into the red-white circular target.",
        stages=(
            "reach a pre-push pose immediately behind the blue cube",
            "establish pushing contact with the blue cube",
            "push the blue cube into the red-white target region",
        ),
        trigger_stage=3,
        oracle_target_stage=2,
        severity_unit="m",
        default_severity=0.11,
        max_pre_steps=120,
        recovery_steps=120,
    ),
    "PullCube-v1": RecoveryProtocol(
        instruction="Pull the blue cube backward into the red-white circular target.",
        stages=(
            "reach a pre-pull pose immediately in front of the blue cube",
            "establish pulling contact with the blue cube",
            "pull the blue cube into the red-white target region",
        ),
        trigger_stage=3,
        oracle_target_stage=2,
        severity_unit="m",
        default_severity=0.11,
        max_pre_steps=120,
        recovery_steps=120,
    ),
    "RotateValveLevel2-v1": RecoveryProtocol(
        instruction="Rotate the valve through the commanded half-turn.",
        stages=(
            "place the fingers around the valve ring and establish contact",
            "rotate the contacted valve in the commanded direction until the half-turn goal is reached",
        ),
        trigger_stage=2,
        oracle_target_stage=1,
        severity_unit="rad",
        default_severity=0.30,
        max_pre_steps=300,
        recovery_steps=300,
    ),
    "PickCube-v1": RecoveryProtocol(
        instruction="Pick up the red cube, carry it to the green goal, and place it there.",
        stages=(
            "persistent approach prerequisite: the red cube remains centered within the gripper capture region closely enough that no further reaching motion is needed; a cube currently held between closed fingers satisfies this predicate",
            "persistent grasp prerequisite: the red cube remains stably secured between the gripper fingers",
            "transport is complete only after the stably held red cube has been carried to and placed at the green goal",
        ),
        trigger_stage=2,
        oracle_target_stage=0,
        severity_unit="m",
        default_severity=0.05,
        max_pre_steps=120,
        recovery_steps=120,
    ),
    "PickSingleYCB-v1": RecoveryProtocol(
        instruction="Pick up the visible YCB object and carry it to the green goal.",
        stages=(
            "persistent approach prerequisite: the YCB object remains centered within the gripper capture region closely enough that no further reaching motion is needed; an object currently held between closed fingers satisfies this predicate",
            "persistent grasp prerequisite: the YCB object remains stably secured between the gripper fingers",
            "transport is complete only while the stably held object has been carried to the green goal",
            "settling is complete only while the object remains at the green goal and the robot is static",
        ),
        trigger_stage=2,
        oracle_target_stage=0,
        severity_unit="m",
        default_severity=0.05,
        max_pre_steps=120,
        recovery_steps=200,
    ),
    "PlaceSphere-v1": RecoveryProtocol(
        instruction="Pick up the red sphere, carry it to the green bin, place it inside, and release it.",
        stages=(
            "reach the red sphere with the gripper",
            "close the gripper and establish a stable grasp on the sphere",
            "while maintaining the grasp, carry the sphere above the green bin",
            "lower the sphere into the bin and release it stably",
        ),
        trigger_stage=4,
        oracle_target_stage=0,
        severity_unit="m",
        default_severity=0.04,
        max_pre_steps=160,
        recovery_steps=160,
    ),
    "StackCube-v1": RecoveryProtocol(
        instruction="Pick up the red cube, stack it on the green cube, and release it stably.",
        stages=(
            "move the open gripper until the red cube is centered between the fingers and can be closed without further reaching",
            "with the red cube already centered between the fingers, close the gripper and establish a stable grasp",
            "move the grasped red cube above the green cube",
            "lower the red cube onto the green cube",
            "release the red cube and leave a stable two-cube stack",
        ),
        trigger_stage=5,
        oracle_target_stage=0,
        severity_unit="m",
        default_severity=0.08,
        max_pre_steps=160,
        recovery_steps=160,
    ),
    "PokeCube-v1": RecoveryProtocol(
        instruction="Use the blue peg to push the red cube into the red-white target.",
        stages=(
            "reach the blue peg with the gripper",
            "close the gripper and establish a stable grasp on the peg",
            "align the peg head with the red cube while maintaining the grasp",
            "push the red cube into the red-white target region",
        ),
        trigger_stage=4,
        oracle_target_stage=2,
        severity_unit="m",
        default_severity=0.06,
        max_pre_steps=50,
        recovery_steps=100,
    ),
    "LiftPegUpright-v1": RecoveryProtocol(
        instruction="Pick up the red-blue peg and leave it standing upright on the table.",
        stages=(
            "reach the red-blue peg with the gripper",
            "close the gripper and establish a stable grasp on the peg",
            "rotate the grasped peg until its long axis is vertical",
            "set the peg upright on the table and stabilize it",
        ),
        trigger_stage=4,
        oracle_target_stage=0,
        severity_unit="m",
        default_severity=0.08,
        max_pre_steps=160,
        recovery_steps=160,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("forward_only", "oracle", "vlm"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3042)
    parser.add_argument("--severity", type=float, default=None)
    parser.add_argument(
        "--recovery-steps",
        type=int,
        default=None,
        help="Post-disturbance policy-step budget; defaults to the task protocol.",
    )
    parser.add_argument("--trigger-delay", type=int, default=2)
    parser.add_argument(
        "--trigger-delay-jitter", type=int, default=0,
        help="Uniform per-episode extra delay in [0, jitter] policy steps.",
    )
    parser.add_argument("--query-interval", type=int, default=4)
    parser.add_argument("--rollback-window", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rule-variant", default=None)
    parser.add_argument(
        "--refresh-anchor-at-disturbance",
        action="store_true",
        help="Use the last validated pre-disturbance RGB as the comparison anchor.",
    )
    parser.add_argument(
        "--qwen-python",
        type=Path,
        default=Path(os.environ.get("ROBOSTEP_QWEN_PYTHON", "python")),
    )
    parser.add_argument(
        "--qwen-worker",
        type=Path,
        default=Path(__file__).with_name("qwen3_5_worker_v5_rollback.py"),
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=Path(os.environ.get("ROBOSTEP_QWEN_MODEL", "models/Qwen3.5-9B")),
    )
    parser.add_argument("--qwen-device-index", type=int, default=1)
    parser.add_argument("--qwen-timeout", type=float, default=3600.0)
    parser.add_argument("--qwen-gate-max-new-tokens", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=448)
    return parser.parse_args()


def find_stage_wrapper(envs: Any) -> StageAwareRewardWrapper:
    current = envs._env
    while current is not None:
        if isinstance(current, StageAwareRewardWrapper):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("StageAwareRewardWrapper not found")


def freeze_rgb(value: Any) -> np.ndarray:
    return SynchronousVisualStageObserver._freeze_rgb_batch(value)


def zero_actor_motion(actor: Any, mask: torch.Tensor) -> None:
    linear = actor.linear_velocity.clone()
    angular = actor.angular_velocity.clone()
    linear[mask] = 0
    angular[mask] = 0
    actor.set_linear_velocity(linear)
    actor.set_angular_velocity(angular)


def away_direction(source_xy: torch.Tensor, reference_xy: torch.Tensor) -> torch.Tensor:
    delta = source_xy - reference_xy
    norm = torch.linalg.norm(delta, dim=1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0], device=source_xy.device).expand_as(delta)
    return torch.where(norm > 1e-5, delta / norm.clamp_min(1e-5), fallback)


def inject_regression(
    task: str, base: Any, mask: torch.Tensor, severity: float
) -> None:
    if not mask.any():
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
    if task == "PickCube-v1":
        actor = base.cube
        p = actor.pose.p.clone()
        q = actor.pose.q.clone()
        tcp = base.agent.tcp_pose.p
        direction = away_direction(tcp[:, :2], base.goal_site.pose.p[:, :2])
        p[mask, :2] = torch.clamp(
            tcp[mask, :2] + float(severity) * direction[mask], -0.24, 0.24
        )
        p[mask, 2] = float(base.cube_half_size)
    elif task == "StackCube-v1":
        actor = base.cubeA
        p = actor.pose.p.clone()
        q = actor.pose.q.clone()
        tcp = base.agent.tcp.pose.p
        direction = away_direction(base.cubeB.pose.p[:, :2], tcp[:, :2])
        p[mask, :2] = torch.clamp(
            base.cubeB.pose.p[mask, :2]
            + float(severity) * direction[mask],
            -0.24,
            0.24,
        )
        p[mask, 2] = float(base.cube_half_size[2])
    elif task == "PickSingleYCB-v1":
        actor = base.obj
        p = actor.pose.p.clone()
        q = actor.pose.q.clone()
        tcp = base.agent.tcp.pose.p
        direction = away_direction(tcp[:, :2], base.goal_site.pose.p[:, :2])
        p[mask, :2] = torch.clamp(
            tcp[mask, :2] + float(severity) * direction[mask], -0.24, 0.24
        )
        p[mask, 2] = base.object_zs[mask]
    elif task == "PlaceSphere-v1":
        actor = base.obj
        p = actor.pose.p.clone()
        q = actor.pose.q.clone()
        direction = torch.where(
            base.bin.pose.p[:, 0] <= 0.05,
            torch.ones_like(base.bin.pose.p[:, 0]),
            -torch.ones_like(base.bin.pose.p[:, 0]),
        )
        p[mask, 0] = torch.clamp(
            base.bin.pose.p[mask, 0] + direction[mask] * float(severity),
            -0.24,
            0.24,
        )
        p[mask, 1] = base.bin.pose.p[mask, 1]
        p[mask, 2] = float(base.radius)
    elif task == "PokeCube-v1":
        actor = base.cube
        p = actor.pose.p.clone()
        q = actor.pose.q.clone()
        direction = away_direction(
            base.cube.pose.p[:, :2], base.goal_region.pose.p[:, :2]
        )
        p[mask, :2] = torch.clamp(
            base.cube.pose.p[mask, :2] + float(severity) * direction[mask],
            -0.24,
            0.24,
        )
        p[mask, 2] = float(base.cube_half_size)

    elif task in {"PushCube-v1", "PullCube-v1"}:
        target = base.goal_region
        p = target.pose.p.clone()
        q = target.pose.q.clone()
        direction = 1.0 if task == "PushCube-v1" else -1.0
        p[mask, 0] = torch.clamp(
            base.obj.pose.p[mask, 0] + direction * float(severity),
            -0.30,
            0.30,
        )
        p[mask, 1] = base.obj.pose.p[mask, 1]
        target.set_pose(Pose.create_from_pq(p, q))
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
        return
    elif task == "LiftPegUpright-v1":
        actor = base.peg
        p = actor.pose.p.clone()
        tcp = base.agent.tcp.pose.p
        direction = away_direction(p[:, :2], tcp[:, :2])
        p[mask, :2] = torch.clamp(
            p[mask, :2] + float(severity) * direction[mask], -0.24, 0.24
        )
        p[mask, 2] = float(base.peg_half_width)
        q = actor.pose.q.clone()
        flat_q = torch.tensor(
            [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
            device=q.device,
            dtype=q.dtype,
        )
        q[mask] = flat_q
    else:
        raise KeyError(task)
    actor.set_pose(Pose.create_from_pq(p, q))
    zero_actor_motion(actor, mask)
    base.scene._gpu_apply_all()
    base.scene._gpu_fetch_all()


def rewrite_policy_stage(
    policy_obs: torch.Tensor,
    wrapper: StageAwareRewardWrapper,
    targets: torch.Tensor,
    mask: torch.Tensor,
    info: dict[str, Any],
) -> torch.Tensor:
    if not mask.any():
        return policy_obs
    machine = wrapper.machine
    machine.stage[mask] = targets[mask]
    machine.dwell[mask] = 0
    inputs = wrapper.rule_spec.compute(wrapper.unwrapped, info)
    rows = torch.arange(machine.num_envs, device=machine.device)
    cols = machine.stage.clamp(max=machine.num_stages - 1)
    machine.prev_potential[mask] = inputs.potentials[rows[mask], cols[mask]]
    width = wrapper.num_stages + 1
    one_hot = torch.nn.functional.one_hot(
        machine.stage, num_classes=width
    ).to(dtype=policy_obs.dtype)
    return torch.cat((policy_obs[:, :-width], one_hot), dim=-1)


def parse_stage_label(label: str, num_stages: int) -> int | None:
    if not label.startswith("stage_"):
        return None
    try:
        value = int(label.split("_", 1)[1])
    except ValueError:
        return None
    return value if 0 <= value < num_stages else None


def recovery_eligible(
    disturbed_at: torch.Tensor, deadline: torch.Tensor, step: int
) -> torch.Tensor:
    return (disturbed_at >= 0) & (step > disturbed_at) & (step <= deadline)


def sample_trigger_delays(
    seed: int, batch_index: int, num_envs: int, base: int, jitter: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + batch_index * 1009 + 0xA17)
    offsets = torch.randint(jitter + 1, (num_envs,), generator=generator)
    return base + offsets.to(device)


def run_batch(
    *,
    args: argparse.Namespace,
    protocol: RecoveryProtocol,
    severity: float,
    recovery_steps: int,
    batch_index: int,
    agent: torch.nn.Module,
    envs: Any,
    wrapper: StageAwareRewardWrapper,
    qwen: QwenRollbackClient | None,
) -> list[dict[str, Any]]:
    base = envs.base_env
    machine = wrapper.machine
    device = base.device
    num_envs = envs.num_envs
    trigger_delays = sample_trigger_delays(
        args.seed, batch_index, num_envs, args.trigger_delay, args.trigger_delay_jitter, device
    )
    policy_obs, info = envs.reset(seed=args.seed + batch_index * 1009)
    clip_action = sweep.clip_action_fn(envs, device)

    trigger_seen = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    disturbed_at = torch.full_like(trigger_seen, -1)
    deadline = torch.full_like(trigger_seen, -1)
    first_success = torch.full_like(trigger_seen, -1)
    first_rollback = torch.full_like(trigger_seen, -1)
    rollback_target = torch.full_like(trigger_seen, -1)
    false_rollback = torch.zeros(num_envs, dtype=torch.bool, device=device)
    effective = torch.zeros_like(false_rollback)
    anchor_frames: list[np.ndarray | None] = [None] * num_envs
    predictions: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    query_number = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Terminal-stage protocols must physically invalidate official success.
    # This prevents small displacements inside a success tolerance from being
    # mislabeled as regression episodes.
    require_success_invalidation = protocol.trigger_stage >= len(protocol.stages)
    total_steps = (
        protocol.max_pre_steps + args.trigger_delay + args.trigger_delay_jitter
        + recovery_steps + 1
    )
    agent.eval()
    for step in range(total_steps):
        stage = machine.stage.clone()
        ready = (
            (trigger_seen < 0)
            & (stage >= protocol.trigger_stage)
            & (step < protocol.max_pre_steps)
        )
        if ready.any():
            frames = freeze_rgb(envs.render())
            for index in torch.nonzero(ready).flatten().tolist():
                anchor_frames[index] = frames[index].copy()
            trigger_seen[ready] = step

        due_disturbance = (
            (trigger_seen >= 0)
            & (disturbed_at < 0)
            & (step - trigger_seen >= trigger_delays)
        )
        if due_disturbance.any():
            if args.refresh_anchor_at_disturbance:
                frames = freeze_rgb(envs.render())
                for index in torch.nonzero(due_disturbance).flatten().tolist():
                    anchor_frames[index] = frames[index].copy()
            inject_regression(args.task, base, due_disturbance, severity)
            disturbed_at[due_disturbance] = step
            deadline[due_disturbance] = step + recovery_steps
            effective[due_disturbance] = True

        if args.method == "oracle" and due_disturbance.any():
            targets = torch.full_like(machine.stage, protocol.oracle_target_stage)
            policy_obs = rewrite_policy_stage(
                policy_obs, wrapper, targets, due_disturbance, info
            )
            first_rollback[due_disturbance] = step
            rollback_target[due_disturbance] = protocol.oracle_target_stage

        query_due = (
            (args.method == "vlm")
            and qwen is not None
            and bool((trigger_seen >= 0).any().item())
        )
        if query_due:
            scheduled = (
                (trigger_seen >= 0)
                & (first_rollback < 0)
                & ((step - trigger_seen) % int(args.query_interval) == 0)
                & ((disturbed_at < 0) | (step <= deadline))
            )
            indices = torch.nonzero(scheduled).flatten()
            if len(indices):
                frames = freeze_rgb(envs.render())
                requests = []
                request_indices = indices.tolist()
                for index in request_indices:
                    anchor = anchor_frames[index]
                    if anchor is None:
                        raise RuntimeError(f"missing anchor for env {index}")
                    requests.append(
                        RollbackStageRequest(
                            env_index=index,
                            episode_id=batch_index * num_envs + index,
                            request_step=step,
                            task_id=args.task,
                            task_instruction=protocol.instruction,
                            stage_id=int(machine.stage[index].item()),
                            stage_descriptions=protocol.stages,
                            reference_rgb=anchor.copy(),
                            current_rgb=frames[index].copy(),
                        )
                    )
                labels = qwen.predict_rollback_stage(requests)
                targets = machine.stage.clone()
                apply_mask = torch.zeros_like(effective)
                for index, label in zip(request_indices, labels, strict=True):
                    predicted = parse_stage_label(label, wrapper.num_stages)
                    after_disturbance = disturbed_at[index] >= 0
                    predictions[index].append(
                        {
                            "step": step,
                            "query_number": int(query_number[index].item()),
                            "after_disturbance": bool(after_disturbance.item()),
                            "stage_before": int(machine.stage[index].item()),
                            "label": label,
                            "predicted_stage": predicted,
                        }
                    )
                    query_number[index] += 1
                    if predicted is None or predicted >= int(machine.stage[index].item()):
                        continue
                    if not bool(after_disturbance.item()):
                        false_rollback[index] = True
                        # Keep the matched clean query diagnostic-only so it cannot
                        # alter the state presented to the subsequent disturbance.
                        continue
                    targets[index] = predicted
                    apply_mask[index] = True
                    first_rollback[index] = step
                    rollback_target[index] = predicted
                policy_obs = rewrite_policy_stage(
                    policy_obs, wrapper, targets, apply_mask, info
                )

        with torch.no_grad():
            action = clip_action(agent.get_action(policy_obs, deterministic=True))
        action[due_disturbance] = 0
        policy_obs, _, _, _, info = envs.step(action)
        success = info.get(
            "success", torch.zeros(num_envs, dtype=torch.bool, device=device)
        ).bool()
        if require_success_invalidation and due_disturbance.any():
            effective[due_disturbance] = ~success[due_disturbance]
        recovered = recovery_eligible(disturbed_at, deadline, step) & success & (
            first_success < 0
        )
        first_success[recovered] = step

        active_deadline = deadline >= 0
        finished = active_deadline & ((first_success >= 0) | (step >= deadline))
        triggered = trigger_seen >= 0
        if bool(triggered.all().item()) and bool(finished.all().item()):
            break

    records = []
    for index in range(num_envs):
        if not bool(effective[index].item()):
            continue
        td = int(disturbed_at[index].item())
        fr = int(first_rollback[index].item())
        fs = int(first_success[index].item())
        post_predictions = [
            item for item in predictions[index] if item["after_disturbance"]
        ]
        detected_in_window = fr >= td and fr - td <= int(args.rollback_window)
        records.append(
            {
                "batch_index": batch_index,
                "env_index": index,
                "seed": args.seed + batch_index * 1009,
                "trigger_step": int(trigger_seen[index].item()),
                "planned_trigger_delay_steps": int(trigger_delays[index].item()),
                "disturbance_step": td,
                "deadline_step": int(deadline[index].item()),
                "first_rollback_step": fr,
                "rollback_target": int(rollback_target[index].item()),
                "oracle_target": protocol.oracle_target_stage,
                "rollback_detected_in_window": detected_in_window,
                "exact_stage_correct": bool(
                    detected_in_window
                    and int(rollback_target[index].item())
                    == protocol.oracle_target_stage
                ),
                "false_rollback_before_disturbance": bool(
                    false_rollback[index].item()
                ),
                "task_recovered": fs >= 0,
                "first_success_step": fs,
                "recovery_time_steps": None if fs < 0 else fs - td,
                "predictions": predictions[index],
                "first_post_disturbance_prediction": (
                    post_predictions[0] if post_predictions else None
                ),
                "final_stage": int(machine.stage[index].item()),
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = max(1, len(records))
    recovery_times = [
        row["recovery_time_steps"]
        for row in records
        if row["recovery_time_steps"] is not None
    ]
    return {
        "effective_episodes": len(records),
        "rollback_detection_rate": sum(
            bool(row["rollback_detected_in_window"]) for row in records
        )
        / denominator,
        "exact_stage_accuracy": sum(
            bool(row["exact_stage_correct"]) for row in records
        )
        / denominator,
        "false_rollback_rate": sum(
            bool(row["false_rollback_before_disturbance"]) for row in records
        )
        / denominator,
        "recovery_success_rate": sum(
            bool(row["task_recovered"]) for row in records
        )
        / denominator,
        "mean_recovery_time_steps": (
            float(np.mean(recovery_times)) if recovery_times else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("episodes and batch-size must be positive")
    if args.query_interval <= args.trigger_delay:
        raise ValueError("query interval must exceed trigger delay")
    if args.trigger_delay_jitter < 0:
        raise ValueError("trigger-delay-jitter must be nonnegative")
    if args.rule_variant:
        os.environ["OURS_RULE_VARIANT"] = args.rule_variant
    protocol = PROTOCOLS[args.task]
    recovery_steps = (
        protocol.recovery_steps if args.recovery_steps is None else args.recovery_steps
    )
    if recovery_steps <= 0:
        raise ValueError("recovery-steps must be positive")
    severity = (
        protocol.default_severity if args.severity is None else float(args.severity)
    )
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
    qwen = None
    if args.method == "vlm":
        qwen = QwenRollbackClient(
            python_executable=args.qwen_python,
            worker_script=args.qwen_worker,
            model_path=args.qwen_model,
            output_dir=args.output.parent / f"{args.output.stem}_qwen_worker",
            device_index=args.qwen_device_index,
            timeout_s=args.qwen_timeout,
            image_size=args.image_size,
            gate_max_new_tokens=args.qwen_gate_max_new_tokens,
        )
    try:
        agent = sweep.Agent(envs).to(device)
        checkpoint = torch.load(
            args.checkpoint.expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        agent.load_state_dict(checkpoint["agent_state_dict"])
        records: list[dict[str, Any]] = []
        for batch_index in range(args.max_batches):
            records.extend(
                run_batch(
                    args=args,
                    protocol=protocol,
                    severity=severity,
                    recovery_steps=recovery_steps,
                    batch_index=batch_index,
                    agent=agent,
                    envs=envs,
                    wrapper=wrapper,
                    qwen=qwen,
                )
            )
            if len(records) >= args.episodes:
                break
        records = records[: args.episodes]
    finally:
        envs.close()
        if qwen is not None:
            qwen.close()

    summary = summarize(records)
    payload = {
        "schema": "robostep-vlm-backward-recovery-v1",
        "task": args.task,
        "method": args.method,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_frozen": True,
        "disturbance_training_used": False,
        "actor_receives_stage_id": True,
        "vlm_receives_simulator_state": False,
        "official_dense_reward_used": False,
        "official_success_used_for_outcome_only": True,
        "rule_variant": args.rule_variant or os.environ.get("OURS_RULE_VARIANT", "rule_v1"),
        "evaluation_seed": args.seed,
        "severity": severity,
        "severity_unit": protocol.severity_unit,
        "query_interval_steps": args.query_interval,
        "trigger_delay_steps": args.trigger_delay,
        "trigger_delay_jitter_steps": args.trigger_delay_jitter,
        "rollback_window_steps": args.rollback_window,
        "anchor_refreshed_immediately_before_disturbance": args.refresh_anchor_at_disturbance,
        "clean_query_diagnostic_only": True,
        "fixed_post_disturbance_budget": recovery_steps,
        "protocol": {
            "instruction": protocol.instruction,
            "stage_descriptions": protocol.stages,
            "trigger_stage": protocol.trigger_stage,
            "oracle_target_stage": protocol.oracle_target_stage,
            "max_pre_steps": protocol.max_pre_steps,
        },
        "summary": summary,
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**summary, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

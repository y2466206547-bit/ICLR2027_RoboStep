"""Privileged RuleStage specifications for the current ManiSkill-19 suite.

These rules are an auditable simulator-state upper bound. They are deliberately
separate from the visual/Qwen observer and must never be described as visual.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

from mani_skill.utils.geometry import rotation_conversions

from stage_reward.core import StageRewardConfig
from stage_reward.task_suite import load_task_specs


@dataclass(frozen=True)
class RuleInputs:
    potentials: torch.Tensor
    gates: torch.Tensor
    maintenance: torch.Tensor
    safety_penalty: torch.Tensor


Extractor = Callable[[Any, dict[str, Any]], RuleInputs]


@dataclass(frozen=True)
class RuleStageSpec:
    env_id: str
    stage_names: tuple[str, ...]
    config: StageRewardConfig
    extractor: Extractor
    version: str = "rule_v1"

    def compute(self, env: Any, info: dict[str, Any]) -> RuleInputs:
        result = self.extractor(env, info)
        expected = (env.num_envs, len(self.stage_names))
        for name in ("potentials", "gates", "maintenance"):
            value = getattr(result, name)
            if value.shape != expected:
                raise ValueError(
                    f"{self.env_id} {name} must have shape {expected}, "
                    f"found {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{self.env_id} {name} contains non-finite values")
        if (
            (result.potentials < -1e-5).any()
            or (result.potentials > 1.0 + 1e-5).any()
        ):
            raise ValueError(
                f"{self.env_id} potentials must remain in [0, 1]"
            )
        return result


def _config(num_stages: int) -> StageRewardConfig:
    if num_stages == 1:
        bonuses = (1.0,)
    else:
        bonuses = tuple(
            0.2 + 0.8 * index / (num_stages - 1)
            for index in range(num_stages)
        )
    return StageRewardConfig(
        gamma=0.99,
        delta_max=0.10,
        dense_scales=(1.0,) * num_stages,
        transition_bonuses=bonuses,
        dwell_steps=(2,) * num_stages,
    )


def _tcp(env: Any) -> torch.Tensor:
    return env.agent.tcp.pose.p


def _tip_positions(agent: Any) -> torch.Tensor:
    """Return fingertip xyz positions with shape [B, 3, 3].

    ManiSkill dexterous robots expose fingertip poses in two layouts:
    DClaw uses [B, 3 fingers, 7 pose fields], while TriFinger uses
    [B, 7 pose fields, 3 fingers].
    """
    tips = agent.tip_poses
    if tips.ndim != 3:
        raise ValueError(f"unsupported tip_poses rank: {tuple(tips.shape)}")
    if tips.shape[-1] == 7:
        return tips[..., :3]
    if tips.shape[1] >= 3 and tips.shape[-1] == 3:
        return tips[:, :3, :].transpose(1, 2)
    raise ValueError(f"unsupported tip_poses layout: {tuple(tips.shape)}")


def _distance(a: torch.Tensor, b: torch.Tensor, xy: bool = False) -> torch.Tensor:
    if xy:
        a = a[..., :2]
        b = b[..., :2]
    return torch.linalg.norm(a - b, dim=-1)


def _phi(distance: torch.Tensor, gain: float = 5.0) -> torch.Tensor:
    return 1.0 - torch.tanh(gain * distance)


def _quat_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    relative = rotation_conversions.quaternion_multiply(
        rotation_conversions.quaternion_invert(b), a
    )
    axis_angle = rotation_conversions.quaternion_to_axis_angle(relative)
    angle = torch.linalg.norm(axis_angle, dim=-1)
    return torch.minimum(angle, 2 * torch.pi - angle)


def _yaw_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_yaw = rotation_conversions.matrix_to_euler_angles(
        rotation_conversions.quaternion_to_matrix(a), "XYZ"
    )[:, 2]
    b_yaw = rotation_conversions.matrix_to_euler_angles(
        rotation_conversions.quaternion_to_matrix(b), "XYZ"
    )[:, 2]
    return torch.acos(torch.cos(a_yaw - b_yaw).clamp(-1.0, 1.0))


def _grasp(env: Any, actor: Any, max_angle: float | None = None) -> torch.Tensor:
    if max_angle is None:
        return env.agent.is_grasping(actor).bool()
    return env.agent.is_grasping(actor, max_angle=max_angle).bool()


def _success(env: Any, info: dict[str, Any]) -> torch.Tensor:
    if "success" in info:
        value = info["success"].bool()
        if value.shape == (env.num_envs,):
            return value
    return env.evaluate()["success"].bool()


def _static(env: Any, threshold: float = 0.2) -> torch.Tensor:
    return env.agent.is_static(threshold).bool()


def _safety(actor: Any, speed_limit: float = 3.0) -> torch.Tensor:
    speed = torch.linalg.norm(actor.linear_velocity, dim=-1)
    too_fast = torch.clamp((speed - speed_limit) / speed_limit, min=0.0, max=1.0)
    below_table = (actor.pose.p[:, 2] < -0.01).float()
    return too_fast + below_table


def _pack(
    potentials: tuple[torch.Tensor, ...],
    gates: tuple[torch.Tensor, ...],
    *,
    maintenance: tuple[torch.Tensor, ...] | None = None,
    safety: torch.Tensor | None = None,
) -> RuleInputs:
    p = torch.stack(potentials, dim=1).to(dtype=torch.float32)
    g = torch.stack(tuple(value.bool() for value in gates), dim=1)
    if maintenance is None:
        m = torch.zeros_like(p)
    else:
        m = torch.stack(maintenance, dim=1).to(dtype=p.dtype)
    if safety is None:
        safety = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
    else:
        safety = safety.to(dtype=p.dtype)
    return RuleInputs(p, g, m, safety)


def _loss_penalty(condition: torch.Tensor, value: float = -0.05) -> torch.Tensor:
    return torch.where(
        condition,
        torch.zeros_like(condition, dtype=torch.float32),
        torch.full_like(condition, value, dtype=torch.float32),
    )


def _pick_cube(env: Any, info: dict[str, Any]) -> RuleInputs:
    grasped = info.get("is_grasped", _grasp(env, env.cube)).bool()
    obj_goal = _distance(env.cube.pose.p, env.goal_site.pose.p)
    tcp_obj = _distance(_tcp(env), env.cube.pose.p)
    placed = info.get("is_obj_placed", obj_goal <= env.goal_thresh).bool()
    success = _success(env, info)
    return _pack(
        (_phi(tcp_obj), grasped.float(), _phi(obj_goal)),
        (tcp_obj <= 0.05, grasped, success),
        maintenance=(
            torch.zeros_like(tcp_obj),
            torch.zeros_like(tcp_obj),
            _loss_penalty(grasped | placed),
        ),
        safety=_safety(env.cube),
    )


def _push_cube(env: Any, info: dict[str, Any]) -> RuleInputs:
    push_pos = env.obj.pose.p + torch.tensor(
        [-env.cube_half_size - 0.005, 0.0, 0.0], device=env.device
    )
    tcp_push = _distance(_tcp(env), push_pos)
    obj_goal = _distance(env.obj.pose.p, env.goal_region.pose.p, xy=True)
    contact = tcp_push <= 0.025
    return _pack(
        (_phi(tcp_push), _phi(tcp_push, 12.0), _phi(obj_goal)),
        (tcp_push <= 0.05, contact, _success(env, info)),
        maintenance=(
            torch.zeros_like(tcp_push),
            torch.zeros_like(tcp_push),
            _loss_penalty(contact),
        ),
        safety=_safety(env.obj),
    )


def _pull_cube(env: Any, info: dict[str, Any]) -> RuleInputs:
    pull_pos = env.obj.pose.p + torch.tensor(
        [env.cube_half_size + 0.01, 0.0, 0.0], device=env.device
    )
    tcp_pull = _distance(_tcp(env), pull_pos)
    obj_goal = _distance(env.obj.pose.p, env.goal_region.pose.p, xy=True)
    contact = tcp_pull <= 0.025
    return _pack(
        (_phi(tcp_pull), _phi(tcp_pull, 12.0), _phi(obj_goal)),
        (tcp_pull <= 0.05, contact, _success(env, info)),
        maintenance=(
            torch.zeros_like(tcp_pull),
            torch.zeros_like(tcp_pull),
            _loss_penalty(contact),
        ),
        safety=_safety(env.obj),
    )


def _lift_peg(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_peg = _distance(_tcp(env), env.peg.pose.p)
    grasped = _grasp(env, env.peg)
    matrix = rotation_conversions.quaternion_to_matrix(env.peg.pose.q)
    upright = matrix[:, 2, 0].abs()
    z_error = torch.abs(env.peg.pose.p[:, 2] - env.peg_half_length)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_peg)
    return _pack(
        (_phi(tcp_peg), grasped.float(), upright, 0.5 * upright + 0.5 * _phi(z_error, 20.0)),
        (
            tcp_peg <= 0.05,
            grasped,
            upright >= 0.98,
            success & env.peg.is_static(lin_thresh=1e-2, ang_thresh=0.5),
        ),
        maintenance=(zeros, zeros, _loss_penalty(grasped), zeros),
        safety=_safety(env.peg),
    )


def _peg_insertion(env: Any, info: dict[str, Any]) -> RuleInputs:
    rotation = rotation_conversions.quaternion_to_matrix(env.peg.pose.q)
    tail_offset = torch.tensor([-0.06, 0.0, 0.0], device=env.device)
    tail_offset = tail_offset.expand(env.num_envs, -1)
    target = env.peg.pose.p + torch.bmm(
        rotation, tail_offset.unsqueeze(-1)
    ).squeeze(-1)
    tcp_peg = _distance(_tcp(env), target)
    grasped = _grasp(env, env.peg, max_angle=20)
    head_goal = env.goal_pose.inv() * env.peg_head_pose
    peg_goal = env.goal_pose.inv() * env.peg.pose
    head_yz = torch.linalg.norm(head_goal.p[:, 1:], dim=1)
    peg_yz = torch.linalg.norm(peg_goal.p[:, 1:], dim=1)
    align_error = 0.5 * (head_yz + peg_yz) + 4.5 * torch.maximum(head_yz, peg_yz)
    hole_error = torch.linalg.norm(
        (env.box_hole_pose.inv() * env.peg_head_pose).p, dim=1
    )
    aligned = (head_yz < 0.01) & (peg_yz < 0.01) & grasped
    zeros = torch.zeros_like(tcp_peg)
    return _pack(
        (_phi(tcp_peg, 4.0), grasped.float(), _phi(align_error), _phi(hole_error)),
        (tcp_peg <= 0.05, grasped, aligned, _success(env, info)),
        maintenance=(zeros, zeros, _loss_penalty(grasped), _loss_penalty(grasped)),
        safety=_safety(env.peg),
    )


def _plug_charger(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    oriented = angle <= 0.35
    aligned = grasped & oriented & (obj_goal <= 0.025)
    zeros = torch.zeros_like(tcp_obj)
    keep = _loss_penalty(grasped)
    return _pack(
        (
            _phi(tcp_obj),
            grasped.float(),
            1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0),
            0.5 * _phi(obj_goal, 10.0) + 0.5 * (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)),
            _phi(obj_goal, 40.0),
        ),
        (tcp_obj <= 0.05, grasped, oriented & grasped, aligned, _success(env, info)),
        maintenance=(zeros, zeros, keep, keep, keep),
        safety=_safety(env.charger),
    )


def _peg_insertion_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PegInsertionSide v2: official-dense geometry inside the staged recipe."""
    rotation = rotation_conversions.quaternion_to_matrix(env.peg.pose.q)
    tail_offset = torch.tensor([-0.06, 0.0, 0.0], device=env.device)
    tail_offset = tail_offset.expand(env.num_envs, -1)
    target = env.peg.pose.p + torch.bmm(
        rotation, tail_offset.unsqueeze(-1)
    ).squeeze(-1)
    tcp_peg = _distance(_tcp(env), target)
    grasped = _grasp(env, env.peg, max_angle=20)

    head_goal = env.goal_pose.inv() * env.peg_head_pose
    peg_goal = env.goal_pose.inv() * env.peg.pose
    head_yz = torch.linalg.norm(head_goal.p[:, 1:], dim=1)
    peg_yz = torch.linalg.norm(peg_goal.p[:, 1:], dim=1)
    align_error = 0.5 * (head_yz + peg_yz) + 4.5 * torch.maximum(head_yz, peg_yz)
    align_phi = (1.0 - torch.tanh(align_error)).clamp(0.0, 1.0)
    hole_error = torch.linalg.norm(
        (env.box_hole_pose.inv() * env.peg_head_pose).p, dim=1
    )
    insert_phi = (1.0 - torch.tanh(5.0 * hole_error)).clamp(0.0, 1.0)
    reach_phi = (1.0 - torch.tanh(4.0 * tcp_peg)).clamp(0.0, 1.0)
    aligned_loose = (head_yz < 0.018) & (peg_yz < 0.018) & grasped
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_peg)
    keep_grasp = _loss_penalty(grasped, -0.02)

    return _pack(
        (
            reach_phi,
            (0.65 * grasped.float() + 0.35 * reach_phi).clamp(0.0, 1.0),
            (0.75 * align_phi + 0.25 * grasped.float()).clamp(0.0, 1.0),
            (0.80 * insert_phi + 0.15 * align_phi + 0.05 * success.float()).clamp(0.0, 1.0),
        ),
        (
            tcp_peg <= 0.06,
            grasped,
            aligned_loose | (grasped & (align_phi >= 0.90)),
            success,
        ),
        maintenance=(
            0.04 * reach_phi,
            0.06 * reach_phi + 0.08 * grasped.float(),
            keep_grasp + 0.12 * align_phi,
            keep_grasp + 0.16 * insert_phi,
        ),
        safety=zeros,
    )


def _plug_charger_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v2: continuous grasp/orient/insert shaping for sparse task."""
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    reach_phi = _phi(tcp_obj, 5.0)
    angle_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    goal_phi = _phi(obj_goal, 10.0)
    insert_phi = _phi(obj_goal, 35.0)
    pose_phi = (0.55 * goal_phi + 0.45 * angle_phi).clamp(0.0, 1.0)
    oriented_loose = grasped & (angle <= 0.60)
    aligned_loose = grasped & (angle <= 0.45) & (obj_goal <= 0.060)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.02)

    return _pack(
        (
            reach_phi,
            (0.65 * grasped.float() + 0.35 * reach_phi).clamp(0.0, 1.0),
            (0.75 * angle_phi + 0.25 * grasped.float()).clamp(0.0, 1.0),
            (0.70 * pose_phi + 0.20 * insert_phi + 0.10 * grasped.float()).clamp(0.0, 1.0),
            (0.80 * insert_phi + 0.15 * pose_phi + 0.05 * success.float()).clamp(0.0, 1.0),
        ),
        (
            tcp_obj <= 0.06,
            grasped,
            oriented_loose,
            aligned_loose,
            success,
        ),
        maintenance=(
            0.04 * reach_phi,
            0.06 * reach_phi + 0.08 * grasped.float(),
            keep_grasp + 0.10 * angle_phi,
            keep_grasp + 0.12 * pose_phi,
            keep_grasp + 0.16 * insert_phi,
        ),
        safety=zeros,
    )


def _peg_insertion_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PegInsertionSide v3: add an official hole-axis insertion proxy.

    v2 reaches the insert stage but can optimize distance to the hole center
    without exploiting the evaluator's actual success condition: the peg head
    only needs to pass the x threshold while staying inside the hole radius in
    y/z. This keeps the same four stages and replaces the terminal funnel with
    that success-compatible geometry.
    """
    rotation = rotation_conversions.quaternion_to_matrix(env.peg.pose.q)
    tail_offset = torch.tensor([-0.06, 0.0, 0.0], device=env.device)
    tail_offset = tail_offset.expand(env.num_envs, -1)
    target = env.peg.pose.p + torch.bmm(
        rotation, tail_offset.unsqueeze(-1)
    ).squeeze(-1)
    tcp_peg = _distance(_tcp(env), target)
    grasped = _grasp(env, env.peg, max_angle=20)

    head_goal = env.goal_pose.inv() * env.peg_head_pose
    peg_goal = env.goal_pose.inv() * env.peg.pose
    head_yz = torch.linalg.norm(head_goal.p[:, 1:], dim=1)
    peg_yz = torch.linalg.norm(peg_goal.p[:, 1:], dim=1)
    align_error = 0.5 * (head_yz + peg_yz) + 4.5 * torch.maximum(head_yz, peg_yz)
    align_phi = (1.0 - torch.tanh(align_error)).clamp(0.0, 1.0)

    head_hole = (env.box_hole_pose.inv() * env.peg_head_pose).p
    hole_yz = torch.linalg.norm(head_hole[:, 1:], dim=1)
    x_shortfall = torch.clamp(-0.015 - head_hole[:, 0], min=0.0)
    yz_phi = (1.0 - torch.tanh(18.0 * hole_yz)).clamp(0.0, 1.0)
    x_phi = (1.0 - torch.tanh(35.0 * x_shortfall)).clamp(0.0, 1.0)
    insert_phi = (0.65 * yz_phi + 0.35 * x_phi).clamp(0.0, 1.0)
    reach_phi = (1.0 - torch.tanh(4.0 * tcp_peg)).clamp(0.0, 1.0)
    preinsert = grasped & (head_yz < 0.014) & (peg_yz < 0.014)
    hole_ready = grasped & (hole_yz <= env.box_hole_radii + 0.006) & (x_shortfall <= 0.030)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_peg)
    keep_grasp = _loss_penalty(grasped, -0.02)

    return _pack(
        (
            reach_phi,
            (0.65 * grasped.float() + 0.35 * reach_phi).clamp(0.0, 1.0),
            (0.70 * align_phi + 0.20 * grasped.float() + 0.10 * yz_phi).clamp(0.0, 1.0),
            (0.72 * insert_phi + 0.18 * align_phi + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            tcp_peg <= 0.06,
            grasped,
            preinsert | hole_ready | (grasped & (align_phi >= 0.92)),
            success,
        ),
        maintenance=(
            0.04 * reach_phi,
            0.06 * reach_phi + 0.08 * grasped.float(),
            keep_grasp + 0.10 * align_phi + 0.04 * yz_phi,
            keep_grasp + 0.18 * insert_phi + 0.04 * hole_ready.float(),
        ),
        safety=zeros,
    )


def _plug_charger_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v3: pre-insert/lateral funnel for the 5 mm sparse goal."""
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis_error = torch.abs(rel[:, 0])

    reach_phi = _phi(tcp_obj, 5.0)
    angle_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.60, 0.0, 1.0)).clamp(0.0, 1.0)
    lateral_phi = _phi(lateral, 18.0)
    axis_phi = _phi(axis_error, 14.0)
    goal_phi = _phi(obj_goal, 12.0)
    tight_goal_phi = _phi(obj_goal, 80.0)
    pose_phi = (0.35 * lateral_phi + 0.25 * axis_phi + 0.25 * tight_angle_phi + 0.15 * goal_phi).clamp(0.0, 1.0)

    oriented_loose = grasped & (angle <= 0.75)
    aligned_loose = grasped & (angle <= 0.50) & (lateral <= 0.055)
    insert_funnel = grasped & (angle <= 0.35) & (obj_goal <= 0.045)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.02)

    return _pack(
        (
            reach_phi,
            (0.65 * grasped.float() + 0.35 * reach_phi).clamp(0.0, 1.0),
            (0.65 * angle_phi + 0.20 * grasped.float() + 0.15 * lateral_phi).clamp(0.0, 1.0),
            (0.65 * pose_phi + 0.20 * goal_phi + 0.15 * grasped.float()).clamp(0.0, 1.0),
            (0.65 * tight_goal_phi + 0.20 * pose_phi + 0.10 * tight_angle_phi + 0.05 * success.float()).clamp(0.0, 1.0),
        ),
        (
            tcp_obj <= 0.06,
            grasped,
            oriented_loose,
            aligned_loose | insert_funnel,
            success,
        ),
        maintenance=(
            0.04 * reach_phi,
            0.06 * reach_phi + 0.08 * grasped.float(),
            keep_grasp + 0.08 * angle_phi + 0.04 * lateral_phi,
            keep_grasp + 0.14 * pose_phi + 0.04 * insert_funnel.float(),
            keep_grasp + 0.18 * tight_goal_phi + 0.04 * success.float(),
        ),
        safety=zeros,
    )


def _plug_charger_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v4: keep v2 grasp/orient behavior, add tight insertion funnel."""
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis_error = torch.abs(rel[:, 0])

    reach_phi = _phi(tcp_obj, 5.0)
    angle_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.55, 0.0, 1.0)).clamp(0.0, 1.0)
    lateral_phi = _phi(lateral, 22.0)
    axis_phi = _phi(axis_error, 14.0)
    goal_phi = _phi(obj_goal, 10.0)
    tight_goal_phi = _phi(obj_goal, 70.0)
    pose_phi = (
        0.35 * lateral_phi
        + 0.25 * axis_phi
        + 0.25 * tight_angle_phi
        + 0.15 * goal_phi
    ).clamp(0.0, 1.0)

    oriented_loose = grasped & (angle <= 0.65)
    aligned_loose = grasped & (angle <= 0.50) & (lateral <= 0.070)
    insert_funnel = grasped & (angle <= 0.35) & (obj_goal <= 0.050)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.02)

    return _pack(
        (
            reach_phi,
            (0.65 * grasped.float() + 0.35 * reach_phi).clamp(0.0, 1.0),
            (0.75 * angle_phi + 0.25 * grasped.float()).clamp(0.0, 1.0),
            (0.60 * pose_phi + 0.25 * goal_phi + 0.15 * grasped.float()).clamp(0.0, 1.0),
            (0.65 * tight_goal_phi + 0.20 * pose_phi + 0.10 * tight_angle_phi + 0.05 * success.float()).clamp(0.0, 1.0),
        ),
        (
            tcp_obj <= 0.06,
            grasped,
            oriented_loose,
            aligned_loose | insert_funnel,
            success,
        ),
        maintenance=(
            0.04 * reach_phi,
            0.06 * reach_phi + 0.10 * grasped.float(),
            keep_grasp + 0.10 * angle_phi,
            keep_grasp + 0.14 * pose_phi + 0.05 * insert_funnel.float(),
            keep_grasp + 0.20 * tight_goal_phi + 0.05 * success.float(),
        ),
        safety=zeros,
    )


def _plug_charger_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v5: looser align entry with stronger 3D pose funnel."""
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis_error = torch.abs(rel[:, 0])

    reach_phi = _phi(tcp_obj, 5.0)
    angle_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    mid_angle_phi = (1.0 - torch.clamp(angle / 0.90, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.35, 0.0, 1.0)).clamp(0.0, 1.0)
    lateral_phi = _phi(lateral, 18.0)
    axis_phi = _phi(axis_error, 12.0)
    goal_phi = _phi(obj_goal, 12.0)
    tight_goal_phi = _phi(obj_goal, 60.0)
    pose_phi = (
        0.38 * goal_phi
        + 0.24 * lateral_phi
        + 0.18 * axis_phi
        + 0.20 * mid_angle_phi
    ).clamp(0.0, 1.0)

    oriented_loose = grasped & (angle <= 1.00)
    aligned_loose = grasped & (angle <= 0.80) & ((obj_goal <= 0.12) | (pose_phi >= 0.42))
    insert_funnel = grasped & (angle <= 0.55) & (obj_goal <= 0.075)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.02)

    return _pack(
        (
            reach_phi,
            (0.55 * grasped.float() + 0.45 * reach_phi).clamp(0.0, 1.0),
            (0.55 * angle_phi + 0.25 * goal_phi + 0.20 * grasped.float()).clamp(0.0, 1.0),
            (0.62 * pose_phi + 0.23 * goal_phi + 0.15 * grasped.float()).clamp(0.0, 1.0),
            (0.55 * tight_goal_phi + 0.25 * tight_angle_phi + 0.15 * pose_phi + 0.05 * success.float()).clamp(0.0, 1.0),
        ),
        (
            tcp_obj <= 0.06,
            grasped,
            oriented_loose,
            aligned_loose | insert_funnel,
            success,
        ),
        maintenance=(
            0.04 * reach_phi,
            0.06 * reach_phi + 0.10 * grasped.float(),
            keep_grasp + 0.08 * angle_phi + 0.06 * goal_phi,
            keep_grasp + 0.18 * pose_phi + 0.06 * insert_funnel.float(),
            keep_grasp + 0.22 * tight_goal_phi + 0.08 * tight_angle_phi + 0.05 * success.float(),
        ),
        safety=zeros,
    )



def _plug_charger_v6(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v6: learnable grasp acquisition plus tight insertion funnel."""
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis_error = torch.abs(rel[:, 0])

    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    reach_phi = _phi(tcp_obj, 8.0)
    near_grasp_phi = _phi(tcp_obj, 18.0)
    close_near = (near_grasp_phi * closed).clamp(0.0, 1.0)

    angle_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    mid_angle_phi = (1.0 - torch.clamp(angle / 0.90, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.20, 0.0, 1.0)).clamp(0.0, 1.0)
    lateral_phi = _phi(lateral, 24.0)
    axis_phi = _phi(axis_error, 16.0)
    goal_phi = _phi(obj_goal, 14.0)
    tight_goal_phi = _phi(obj_goal, 120.0)
    pose_phi = (0.34 * goal_phi + 0.24 * lateral_phi + 0.18 * axis_phi + 0.24 * mid_angle_phi).clamp(0.0, 1.0)
    tight_pose_phi = (0.50 * tight_goal_phi + 0.35 * tight_angle_phi + 0.15 * pose_phi).clamp(0.0, 1.0)

    close_attempt = (tcp_obj <= 0.045) & (closed >= 0.55)
    oriented_loose = grasped & (angle <= 1.00)
    aligned_loose = grasped & (angle <= 0.60) & ((obj_goal <= 0.10) | (pose_phi >= 0.48))
    insert_funnel = grasped & (angle <= 0.35) & (obj_goal <= 0.040)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.02)

    return _pack(
        (
            (0.65 * reach_phi + 0.25 * close_near + 0.10 * grasped.float()).clamp(0.0, 1.0),
            (0.32 * near_grasp_phi + 0.30 * close_near + 0.28 * grasped.float() + 0.10 * closed).clamp(0.0, 1.0),
            (0.54 * angle_phi + 0.26 * goal_phi + 0.20 * grasped.float()).clamp(0.0, 1.0),
            (0.62 * pose_phi + 0.23 * goal_phi + 0.15 * grasped.float()).clamp(0.0, 1.0),
            (0.68 * tight_pose_phi + 0.22 * pose_phi + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            (tcp_obj <= 0.055) | grasped,
            grasped | close_attempt,
            oriented_loose,
            aligned_loose | insert_funnel,
            success,
        ),
        maintenance=(
            0.06 * reach_phi + 0.04 * close_near,
            0.10 * near_grasp_phi + 0.14 * close_near + 0.16 * grasped.float(),
            keep_grasp + 0.10 * angle_phi + 0.06 * goal_phi,
            keep_grasp + 0.18 * pose_phi + 0.08 * insert_funnel.float(),
            keep_grasp + 0.24 * tight_goal_phi + 0.12 * tight_angle_phi + 0.08 * success.float(),
        ),
        safety=zeros,
    )


def _plug_charger_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v7: open pose/insert shaping after contact, not grasp bit."""
    base = _plug_charger_v6(env, info)
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis = rel[:, 0].abs()
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    contact_like = tcp_obj <= 0.070
    close_contact = contact_like & (closed >= 0.30)
    carrying = grasped | close_contact
    angle_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.35, 0.0, 1.0)).clamp(0.0, 1.0)
    goal_phi = _phi(obj_goal, 12.0)
    tight_goal_phi = _phi(obj_goal, 90.0)
    lateral_phi = _phi(lateral, 35.0)
    axis_phi = _phi(axis, 25.0)
    pose_phi = (0.36 * goal_phi + 0.24 * lateral_phi + 0.18 * axis_phi + 0.22 * angle_phi).clamp(0.0, 1.0)
    tight_pose_phi = (0.50 * tight_goal_phi + 0.30 * tight_angle_phi + 0.20 * pose_phi).clamp(0.0, 1.0)
    aligned_loose = (angle <= 0.75) & ((obj_goal <= 0.12) | (pose_phi >= 0.45))
    insert_funnel = (angle <= 0.45) & (obj_goal <= 0.060)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 1] = torch.ones_like(gates[:, 1], dtype=torch.bool)
    gates[:, 2] = aligned_loose | carrying | (pose_phi >= 0.42)
    gates[:, 3] = insert_funnel | ((obj_goal <= 0.075) & (angle <= 0.55))
    gates[:, 4] = success
    potentials[:, 2] = (0.54 * pose_phi + 0.22 * goal_phi + 0.14 * carrying.float() + 0.10 * closed).clamp(0.0, 1.0)
    potentials[:, 3] = (0.62 * tight_pose_phi + 0.20 * pose_phi + 0.10 * carrying.float() + 0.08 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 2] = 0.16 * pose_phi + 0.10 * goal_phi + 0.06 * carrying.float()
    maintenance[:, 3] = 0.20 * tight_pose_phi + 0.08 * tight_goal_phi + 0.06 * tight_angle_phi
    maintenance[:, 4] = maintenance[:, 4] + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _plug_charger_v8(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v8: keep late insertion practice active after coarse align."""
    base = _plug_charger_v7(env, info)
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis = rel[:, 0].abs()
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    goal_vec = env.goal_pose.p - env.charger.pose.p
    goal_dir = goal_vec / torch.linalg.norm(goal_vec, dim=1, keepdim=True).clamp_min(1e-6)
    directed_speed = (env.charger.linear_velocity * goal_dir).sum(dim=1)
    positive_motion = torch.clamp(torch.tanh(8.0 * directed_speed), 0.0, 1.0)

    reach_phi = _phi(tcp_obj, 10.0)
    goal_phi = _phi(obj_goal, 12.0)
    tight_goal_phi = _phi(obj_goal, 90.0)
    lateral_phi = _phi(lateral, 45.0)
    axis_phi = _phi(axis, 35.0)
    angle_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.30, 0.0, 1.0)).clamp(0.0, 1.0)
    carrying = grasped | ((tcp_obj <= 0.090) & (closed >= 0.25))
    coarse_pose = (0.36 * goal_phi + 0.24 * lateral_phi + 0.18 * axis_phi + 0.14 * angle_phi + 0.08 * carrying.float()).clamp(0.0, 1.0)
    tight_pose = (0.42 * tight_goal_phi + 0.24 * lateral_phi + 0.18 * tight_angle_phi + 0.10 * axis_phi + 0.06 * positive_motion).clamp(0.0, 1.0)
    aligned_loose = (coarse_pose >= 0.38) | ((obj_goal <= 0.16) & (angle <= 1.10))
    preinsert = (tight_pose >= 0.42) | ((obj_goal <= 0.085) & (angle <= 0.70))
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 1] = (tcp_obj <= 0.10) | grasped
    gates[:, 2] = carrying & aligned_loose
    gates[:, 3] = carrying & preinsert
    gates[:, 4] = success
    potentials[:, 1] = (0.45 * reach_phi + 0.25 * closed + 0.20 * grasped.float() + 0.10 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 2] = coarse_pose
    potentials[:, 3] = tight_pose
    potentials[:, 4] = (0.70 * tight_pose + 0.20 * success.float() + 0.10 * positive_motion).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.10 * reach_phi + 0.10 * closed + 0.08 * grasped.float()
    maintenance[:, 2] = 0.18 * coarse_pose + 0.08 * carrying.float() + 0.06 * positive_motion
    maintenance[:, 3] = 0.24 * tight_pose + 0.08 * positive_motion
    maintenance[:, 4] = 0.26 * tight_pose + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _plug_charger_v9(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v9: direct official-world insertion funnel."""
    base = _plug_charger_v8(env, info)
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis = rel[:, 0].abs()
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    carrying = grasped | ((tcp_obj <= 0.10) & (closed >= 0.20))

    reach_phi = _phi(tcp_obj, 10.0)
    coarse_dist = (1.0 - obj_goal / 0.22).clamp(0.0, 1.0)
    mid_dist = (1.0 - obj_goal / 0.10).clamp(0.0, 1.0)
    tight_dist = _phi(obj_goal, 120.0)
    lateral_phi = _phi(lateral, 55.0)
    axis_phi = _phi(axis, 45.0)
    angle_phi = (1.0 - torch.clamp(angle / 1.40, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.25, 0.0, 1.0)).clamp(0.0, 1.0)
    coarse_pose = (0.32 * coarse_dist + 0.24 * lateral_phi + 0.18 * axis_phi + 0.18 * angle_phi + 0.08 * carrying.float()).clamp(0.0, 1.0)
    preinsert = (0.34 * mid_dist + 0.28 * lateral_phi + 0.22 * tight_angle_phi + 0.16 * axis_phi).clamp(0.0, 1.0)
    terminal = (0.42 * tight_dist + 0.24 * lateral_phi + 0.24 * tight_angle_phi + 0.10 * axis_phi).clamp(0.0, 1.0)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 1] = (tcp_obj <= 0.11) | grasped
    gates[:, 2] = carrying | (coarse_pose >= 0.30)
    gates[:, 3] = (obj_goal <= 0.075) | (preinsert >= 0.42)
    gates[:, 4] = success
    potentials[:, 1] = (0.46 * reach_phi + 0.24 * closed + 0.20 * grasped.float() + 0.10 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 2] = coarse_pose
    potentials[:, 3] = preinsert
    potentials[:, 4] = (0.62 * terminal + 0.26 * preinsert + 0.12 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.12 * reach_phi + 0.10 * closed + 0.08 * grasped.float()
    maintenance[:, 2] = 0.24 * coarse_pose + 0.10 * carrying.float()
    maintenance[:, 3] = 0.30 * preinsert + 0.10 * mid_dist
    maintenance[:, 4] = 0.34 * terminal + 0.12 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)

def _poke_cube(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_peg = _distance(_tcp(env), env.peg.pose.p)
    grasped = info.get("is_peg_grasped", _grasp(env, env.peg)).bool()
    angle = info.get("angle_diff", _yaw_error(env.peg.pose.q, env.cube.pose.q))
    head_cube = info.get(
        "head_to_cube_dist",
        _distance(env.peg_head_pos, env.cube.pose.p, xy=True),
    )
    fit = info.get(
        "is_peg_cube_fit",
        (angle < 0.05) & (head_cube <= env.cube_half_size + 0.005),
    ).bool()
    cube_goal = _distance(env.cube.pose.p, env.goal_region.pose.p, xy=True)
    zeros = torch.zeros_like(tcp_peg)
    keep = _loss_penalty(grasped)
    return _pack(
        (
            _phi(tcp_peg),
            grasped.float(),
            0.5 * _phi(angle) + 0.5 * _phi(head_cube),
            _phi(cube_goal),
        ),
        (tcp_peg <= 0.05, grasped, fit & grasped, _success(env, info)),
        maintenance=(zeros, zeros, keep, keep),
        safety=_safety(env.peg) + _safety(env.cube),
    )


def _poke_cube_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Add terminal pose-and-settle credit while preserving the four stages."""
    tcp_peg = _distance(_tcp(env), env.peg.pose.p)
    grasped = info.get("is_peg_grasped", _grasp(env, env.peg)).bool()
    angle = info.get("angle_diff", _yaw_error(env.peg.pose.q, env.cube.pose.q))
    head_cube = info.get(
        "head_to_cube_dist",
        _distance(env.peg_head_pos, env.cube.pose.p, xy=True),
    )
    fit = info.get(
        "is_peg_cube_fit",
        (angle < 0.05) & (head_cube <= env.cube_half_size + 0.005),
    ).bool()
    cube_goal = _distance(env.cube.pose.p, env.goal_region.pose.p, xy=True)
    robot_speed = torch.linalg.norm(env.agent.robot.get_qvel()[..., :-2], dim=1)
    static_phi = _phi(robot_speed, 5.0)
    fit_phi = (
        0.5 * (1.0 - angle / torch.pi).clamp(0.0, 1.0)
        + 0.5 * _phi(head_cube, 8.0)
    )
    zeros = torch.zeros_like(tcp_peg)
    keep = _loss_penalty(grasped, -0.03)
    return _pack(
        (
            _phi(tcp_peg),
            0.7 * grasped.float() + 0.3 * _phi(tcp_peg),
            0.8 * fit_phi + 0.2 * grasped.float(),
            0.70 * _phi(cube_goal, 8.0) + 0.20 * static_phi + 0.10 * fit_phi,
        ),
        (tcp_peg <= 0.05, grasped, fit & grasped, _success(env, info)),
        maintenance=(zeros, zeros, keep, keep),
        safety=_safety(env.peg) + _safety(env.cube),
    )


def _poke_cube_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Reward poking progress without rewarding premature robot stillness.

    V2 consistently reached the final active stage, but its unconditional
    static term made stopping before the cube entered the goal a local optimum.
    V3 makes stillness conditional on placement and normalizes the short
    remaining push into a broad, monotonic potential.
    """
    tcp_peg = _distance(_tcp(env), env.peg.pose.p)
    grasped = info.get("is_peg_grasped", _grasp(env, env.peg)).bool()
    angle = info.get("angle_diff", _yaw_error(env.peg.pose.q, env.cube.pose.q))
    head_cube = info.get(
        "head_to_cube_dist",
        _distance(env.peg_head_pos, env.cube.pose.p, xy=True),
    )
    fit = info.get(
        "is_peg_cube_fit",
        (angle < 0.05) & (head_cube <= env.cube_half_size + 0.005),
    ).bool()
    cube_goal = _distance(env.cube.pose.p, env.goal_region.pose.p, xy=True)
    placed = info.get("is_cube_placed", cube_goal < env.goal_radius).bool()
    robot_speed = torch.linalg.norm(env.agent.robot.get_qvel()[..., :-2], dim=1)
    placed_and_static = placed.float() * _phi(robot_speed, 5.0)
    fit_phi = (
        0.5 * (1.0 - angle / torch.pi).clamp(0.0, 1.0)
        + 0.5 * _phi(head_cube, 8.0)
    )
    # Initial cube-goal distance is about 10 cm and success starts at 5 cm.
    # The loose 12 cm outer radius keeps useful gradients after disturbances.
    push_progress = (
        1.0 - (cube_goal - env.goal_radius) / (0.12 - env.goal_radius)
    ).clamp(0.0, 1.0)
    zeros = torch.zeros_like(tcp_peg)
    keep_grasp = _loss_penalty(grasped, -0.03)
    keep_tool_near_cube = _loss_penalty(
        grasped & (head_cube <= env.cube_half_size + 0.08), -0.02
    )
    return _pack(
        (
            _phi(tcp_peg),
            0.7 * grasped.float() + 0.3 * _phi(tcp_peg),
            0.8 * fit_phi + 0.2 * grasped.float(),
            0.85 * push_progress + 0.10 * fit_phi + 0.05 * placed_and_static,
        ),
        (tcp_peg <= 0.05, grasped, fit & grasped, _success(env, info)),
        maintenance=(
            zeros,
            zeros,
            keep_grasp,
            keep_grasp + keep_tool_near_cube,
        ),
        safety=_safety(env.peg) + _safety(env.cube),
    )



def _poke_cube_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Use the official poke geometry with a broader terminal funnel."""
    tcp_peg = _distance(_tcp(env), env.peg.pose.p)
    grasped = info.get("is_peg_grasped", _grasp(env, env.peg)).bool()
    angle = info.get("angle_diff", _yaw_error(env.peg.pose.q, env.cube.pose.q))
    head_cube = info.get(
        "head_to_cube_dist",
        _distance(env.peg_head_pos, env.cube.pose.p, xy=True),
    )
    cube_goal_vec = env.goal_region.pose.p - env.cube.pose.p
    cube_goal = torch.linalg.norm(cube_goal_vec[:, :2], dim=1)
    goal_dir = cube_goal_vec / torch.linalg.norm(
        cube_goal_vec, dim=1, keepdim=True
    ).clamp_min(1e-6)
    directed_speed = (env.cube.linear_velocity[:, :2] * goal_dir[:, :2]).sum(dim=1)
    forward_credit = 0.03 * torch.clamp(torch.tanh(8.0 * directed_speed), 0.0, 1.0)
    placed = info.get("is_cube_placed", cube_goal < env.goal_radius).bool()
    robot_speed = torch.linalg.norm(env.agent.robot.get_qvel()[..., :-2], dim=1)
    placed_and_static = placed.float() * _phi(robot_speed, 5.0)
    align_phi = _phi(angle, 5.0)
    close_phi = _phi(head_cube, 5.0)
    fit_phi = 0.5 * align_phi + 0.5 * close_phi
    push_progress = (
        1.0 - (cube_goal - env.goal_radius) / (0.12 - env.goal_radius)
    ).clamp(0.0, 1.0)
    loose_fit = (
        grasped
        & (angle < 0.20)
        & (head_cube <= env.cube_half_size + 0.040)
    )
    tool_on_cube = (
        grasped
        & (angle < 0.35)
        & (head_cube <= env.cube_half_size + 0.060)
    )
    zeros = torch.zeros_like(tcp_peg)
    keep_grasp = _loss_penalty(grasped, -0.03)
    keep_tool = _loss_penalty(tool_on_cube, -0.02)
    return _pack(
        (
            _phi(tcp_peg),
            0.75 * grasped.float() + 0.25 * _phi(tcp_peg),
            0.70 * fit_phi + 0.20 * grasped.float() + 0.10 * push_progress,
            0.88 * push_progress + 0.07 * fit_phi + 0.05 * placed_and_static,
        ),
        (tcp_peg <= 0.05, grasped, loose_fit, _success(env, info)),
        maintenance=(
            zeros,
            0.02 * grasped.float(),
            keep_grasp + 0.04 * fit_phi * grasped.float(),
            keep_grasp + keep_tool + 0.06 * push_progress * tool_on_cube.float()
            + forward_credit,
        ),
        safety=_safety(env.peg) + _safety(env.cube),
    )

def _pull_cube_tool(env: Any, info: dict[str, Any]) -> RuleInputs:
    tool_pos = env.l_shape_tool.pose.p
    grasp_pos = tool_pos + torch.tensor([0.02, 0.0, 0.0], device=env.device)
    tcp_tool = _distance(_tcp(env), grasp_pos)
    grasped = _grasp(env, env.l_shape_tool, max_angle=20)
    ideal_hook = env.cube.pose.p + torch.tensor(
        [-(env.hook_length + env.cube_half_size), -0.067, 0.0],
        device=env.device,
    )
    hook_error = _distance(tool_pos, ideal_hook)
    base_pos = env.agent.robot.get_links()[0].pose.p
    cube_base = _distance(env.cube.pose.p, base_pos, xy=True)
    hooked = grasped & (hook_error < 0.06)
    zeros = torch.zeros_like(tcp_tool)
    keep = _loss_penalty(grasped)
    return _pack(
        (_phi(tcp_tool), grasped.float(), _phi(hook_error), _phi(cube_base, 2.0)),
        (tcp_tool <= 0.05, grasped, hooked, _success(env, info)),
        maintenance=(zeros, zeros, keep, keep),
        safety=_safety(env.l_shape_tool) + _safety(env.cube),
    )


def _pull_cube_tool_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Use normalized long-range pull progress and retain the tool hook."""
    tool_pos = env.l_shape_tool.pose.p
    grasp_pos = tool_pos + torch.tensor([0.02, 0.0, 0.0], device=env.device)
    tcp_tool = _distance(_tcp(env), grasp_pos)
    grasped = _grasp(env, env.l_shape_tool, max_angle=20)
    ideal_hook = env.cube.pose.p + torch.tensor(
        [-(env.hook_length + env.cube_half_size), -0.067, 0.0],
        device=env.device,
    )
    hook_error = _distance(tool_pos, ideal_hook)
    identity = torch.zeros_like(env.l_shape_tool.pose.q)
    identity[:, 0] = 1.0
    yaw_error = _yaw_error(env.l_shape_tool.pose.q, identity)
    yaw_phi = (1.0 - yaw_error / torch.pi).clamp(0.0, 1.0)
    base_pos = env.agent.robot.get_links()[0].pose.p
    cube_base = _distance(env.cube.pose.p, base_pos, xy=True)
    pull_progress = 1.0 - ((cube_base - 0.55) / 0.45).clamp(0.0, 1.0)
    hooked = grasped & (hook_error < 0.065) & (yaw_error < 0.45)
    hook_retained = grasped & (hook_error < 0.14)
    zeros = torch.zeros_like(tcp_tool)
    keep_grasp = _loss_penalty(grasped, -0.05)
    keep_hook = _loss_penalty(hook_retained, -0.03)
    return _pack(
        (
            _phi(tcp_tool),
            0.7 * grasped.float() + 0.3 * _phi(tcp_tool),
            0.65 * _phi(hook_error, 4.0) + 0.20 * yaw_phi + 0.15 * grasped.float(),
            0.80 * pull_progress + 0.20 * _phi(hook_error, 4.0),
        ),
        (tcp_tool <= 0.05, grasped, hooked, _success(env, info)),
        maintenance=(zeros, zeros, keep_grasp, keep_grasp + keep_hook),
        safety=_safety(env.l_shape_tool) + _safety(env.cube),
    )


def _push_t(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    near_goal = xy_error <= 0.08
    aligned = near_goal & (angle <= 0.25)
    contact = tcp_obj <= 0.06
    zeros = torch.zeros_like(tcp_obj)
    return _pack(
        (_phi(tcp_obj), _phi(xy_error), 1.0 - angle / torch.pi, 0.5 * _phi(xy_error, 10.0) + 0.5 * (1.0 - angle / torch.pi)),
        (
            contact,
            near_goal,
            aligned,
            _success(env, info)
            & env.tee.is_static(lin_thresh=1e-2, ang_thresh=0.5),
        ),
        maintenance=(zeros, _loss_penalty(contact), _loss_penalty(contact), zeros),
        safety=_safety(env.tee),
    )


def _push_t_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Denser PushT reward for the final align-and-settle phase.

    V1 reached intermediate alignment stages but rarely converted them into
    official success. V2 keeps the same four-stage policy input shape and adds
    smooth final pose/yaw/velocity progress; official success is still used only
    for evaluation and as one sufficient final gate condition.
    """

    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    lin_speed = torch.linalg.norm(env.tee.linear_velocity, dim=-1)
    ang_speed = torch.linalg.norm(env.tee.angular_velocity, dim=-1)
    static = (lin_speed <= 0.05) & (ang_speed <= 0.5)
    contact = tcp_obj <= 0.065
    near_goal_loose = xy_error <= 0.12
    aligned_loose = (xy_error <= 0.08) & (angle <= 0.45)
    final_pose = (xy_error <= 0.045) & (angle <= 0.18) & static
    xy_phi = _phi(xy_error, 8.0)
    xy_tight_phi = _phi(xy_error, 16.0)
    yaw_phi = (1.0 - angle / torch.pi).clamp(0.0, 1.0)
    still_phi = (0.7 * _phi(lin_speed, 10.0) + 0.3 * _phi(ang_speed, 1.0)).clamp(0.0, 1.0)
    zeros = torch.zeros_like(tcp_obj)
    return _pack(
        (
            _phi(tcp_obj),
            0.7 * xy_phi + 0.3 * _phi(tcp_obj),
            0.55 * xy_phi + 0.45 * yaw_phi,
            0.45 * xy_tight_phi + 0.35 * yaw_phi + 0.20 * still_phi,
        ),
        (
            contact,
            near_goal_loose,
            aligned_loose,
            _success(env, info) | final_pose,
        ),
        maintenance=(zeros, _loss_penalty(contact), _loss_penalty(contact), zeros),
        safety=_safety(env.tee),
    )


def _push_t_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Couple translation and rotation before the final settle stage.

    V2 first optimized translation alone, which commonly delivered the T to
    the goal with an unrecoverable orientation. V3 exposes joint pose progress
    in both manipulation stages and only rewards stillness after success.
    """
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    lin_speed = torch.linalg.norm(env.tee.linear_velocity, dim=-1)
    ang_speed = torch.linalg.norm(env.tee.angular_velocity, dim=-1)
    contact = tcp_obj <= 0.07
    xy_phi = _phi(xy_error, 5.0)
    xy_tight_phi = _phi(xy_error, 12.0)
    yaw_phi = 0.5 * (1.0 + torch.cos(angle))
    pose_phi = 0.55 * xy_phi + 0.45 * yaw_phi
    tight_pose_phi = 0.55 * xy_tight_phi + 0.45 * yaw_phi
    loose_pose = (xy_error <= 0.15) & (angle <= 1.0)
    aligned = (xy_error <= 0.075) & (angle <= 0.40)
    success = _success(env, info)
    success_and_static = success.float() * (
        0.7 * _phi(lin_speed, 10.0) + 0.3 * _phi(ang_speed, 1.0)
    )
    zeros = torch.zeros_like(tcp_obj)
    keep_contact = _loss_penalty(contact, -0.03)
    return _pack(
        (
            _phi(tcp_obj),
            0.65 * pose_phi + 0.25 * _phi(tcp_obj) + 0.10 * contact.float(),
            0.85 * pose_phi + 0.15 * contact.float(),
            0.90 * tight_pose_phi + 0.10 * success_and_static,
        ),
        (
            contact,
            loose_pose & contact,
            aligned,
            success,
        ),
        maintenance=(zeros, keep_contact, keep_contact, zeros),
        safety=_safety(env.tee),
    )



def _push_t_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Track official pose progress and keep contact as a shaping term."""
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    yaw_phi = ((1.0 + torch.cos(angle)) / 2.0).clamp(0.0, 1.0) ** 2
    xy_phi = _phi(xy_error, 5.0) ** 2
    pose_phi = (0.50 * xy_phi + 0.50 * yaw_phi).clamp(0.0, 1.0)
    contact = tcp_obj <= 0.10
    loose_pose = contact & ((pose_phi >= 0.35) | (xy_error <= 0.18))
    aligned = (pose_phi >= 0.72) | ((xy_error <= 0.09) & (angle <= 0.70))
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_contact = _loss_penalty(contact, -0.025)
    return _pack(
        (
            _phi(tcp_obj, 5.0),
            0.60 * pose_phi + 0.30 * _phi(tcp_obj, 5.0) + 0.10 * contact.float(),
            0.88 * pose_phi + 0.12 * contact.float(),
            0.90 * pose_phi + 0.10 * success.float(),
        ),
        (
            contact,
            loose_pose,
            aligned,
            success,
        ),
        maintenance=(
            zeros,
            keep_contact + 0.03 * pose_phi,
            keep_contact + 0.07 * pose_phi,
            0.10 * pose_phi,
        ),
        safety=_safety(env.tee),
    )


def _push_t_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Use the official overlap signal to avoid pose-proxy local optima."""
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    yaw_phi = ((1.0 + torch.cos(angle)) / 2.0).clamp(0.0, 1.0) ** 2
    xy_phi = _phi(xy_error, 5.0) ** 2
    pose_phi = (0.45 * xy_phi + 0.35 * yaw_phi).clamp(0.0, 1.0)
    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    joint_phi = (pose_phi + 0.20 * overlap_phi).clamp(0.0, 1.0)
    contact = tcp_obj <= 0.10
    loose_pose = contact & ((joint_phi >= 0.30) | (xy_error <= 0.20))
    aligned = contact & ((overlap >= 0.70) | ((xy_error <= 0.08) & (angle <= 0.60)))
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_contact = _loss_penalty(contact, -0.025)
    return _pack(
        (
            _phi(tcp_obj, 5.0),
            0.55 * joint_phi + 0.35 * _phi(tcp_obj, 5.0) + 0.10 * contact.float(),
            0.55 * overlap_phi + 0.35 * joint_phi + 0.10 * contact.float(),
            0.85 * overlap_phi + 0.15 * success.float(),
        ),
        (
            contact,
            loose_pose,
            aligned,
            success,
        ),
        maintenance=(
            zeros,
            keep_contact + 0.03 * joint_phi,
            keep_contact + 0.08 * overlap_phi,
            0.10 * overlap_phi,
        ),
        safety=_safety(env.tee),
    )


def _push_t_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Stronger official-style PushT shaping for reward-iteration v7."""
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    tcp_phi = torch.sqrt(_phi(tcp_obj, 5.0).clamp(0.0, 1.0))
    yaw_phi = (((1.0 + torch.cos(angle)) / 2.0).clamp(0.0, 1.0)) ** 2
    xy_phi = (_phi(xy_error, 5.0).clamp(0.0, 1.0)) ** 2
    pose_phi = (0.50 * xy_phi + 0.50 * yaw_phi).clamp(0.0, 1.0)
    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    joint_phi = (0.40 * xy_phi + 0.35 * yaw_phi + 0.25 * overlap_phi).clamp(0.0, 1.0)
    contact = tcp_obj <= 0.12
    loose_pose = contact & ((joint_phi >= 0.25) | (xy_error <= 0.22))
    aligned = contact & (
        (overlap_phi >= 0.72)
        | (joint_phi >= 0.68)
        | ((xy_error <= 0.10) & (angle <= 0.75))
    )
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_contact = _loss_penalty(contact, -0.012)
    return _pack(
        (
            tcp_phi,
            0.55 * joint_phi + 0.35 * tcp_phi + 0.10 * contact.float(),
            0.60 * overlap_phi + 0.35 * pose_phi + 0.05 * contact.float(),
            0.90 * overlap_phi + 0.10 * success.float(),
        ),
        (
            tcp_obj <= 0.10,
            loose_pose,
            aligned,
            success,
        ),
        maintenance=(
            zeros,
            keep_contact + 0.10 * joint_phi + 0.02 * contact.float(),
            keep_contact + 0.16 * overlap_phi + 0.07 * pose_phi,
            0.22 * overlap_phi + 0.05 * success.float(),
        ),
        safety=_safety(env.tee),
    )


def _push_t_v8(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PushT v8: high-overlap terminal funnel with official pose shaping."""
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    tcp_phi = torch.sqrt(_phi(tcp_obj, 5.0).clamp(0.0, 1.0))
    yaw_phi = (((1.0 + torch.cos(angle)) / 2.0).clamp(0.0, 1.0)) ** 2
    xy_phi = (_phi(xy_error, 5.0).clamp(0.0, 1.0)) ** 2
    official_pose = (0.50 * yaw_phi + 0.50 * xy_phi).clamp(0.0, 1.0)
    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    mid_overlap = torch.clamp((overlap_phi - 0.45) / 0.35, 0.0, 1.0)
    high_overlap = torch.clamp((overlap_phi - 0.70) / 0.30, 0.0, 1.0)
    high_overlap = high_overlap * high_overlap
    contact = tcp_obj <= 0.12
    loose_pose = contact & ((official_pose >= 0.30) | (xy_error <= 0.22))
    final_funnel = contact & ((official_pose >= 0.62) | (mid_overlap >= 0.45))
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep_contact = _loss_penalty(contact, -0.010)
    return _pack(
        (
            tcp_phi,
            0.70 * official_pose + 0.25 * tcp_phi + 0.05 * contact.float(),
            0.55 * official_pose + 0.35 * mid_overlap + 0.10 * contact.float(),
            0.75 * high_overlap + 0.20 * official_pose + 0.05 * success.float(),
        ),
        (
            contact,
            loose_pose,
            final_funnel,
            success,
        ),
        maintenance=(
            zeros,
            keep_contact + 0.07 * official_pose + 0.02 * contact.float(),
            keep_contact + 0.09 * official_pose + 0.08 * mid_overlap,
            0.34 * high_overlap + 0.08 * success.float(),
        ),
        safety=_safety(env.tee),
    )


def _push_t_v9(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PushT v9: native-dense-equivalent progress maintenance inside stages."""
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    tcp_phi = torch.sqrt(_phi(tcp_obj, 5.0).clamp(0.0, 1.0))
    yaw_phi = (((1.0 + torch.cos(angle)) / 2.0).clamp(0.0, 1.0)) ** 2
    xy_phi = (_phi(xy_error, 5.0).clamp(0.0, 1.0)) ** 2
    native_pose = (0.50 * yaw_phi + 0.50 * xy_phi + 0.05 * tcp_phi).clamp(0.0, 1.05)
    success = _success(env, info)
    native_normalized = torch.where(
        success,
        torch.ones_like(native_pose),
        (native_pose / 3.0).clamp(0.0, 1.0),
    )
    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    high_overlap = torch.clamp((overlap_phi - 0.70) / 0.30, 0.0, 1.0)
    contact = tcp_obj <= 0.12
    loose_pose = contact & ((native_pose >= 0.25) | (xy_error <= 0.25))
    final_funnel = contact & ((native_pose >= 0.55) | (overlap_phi >= 0.55))
    return _pack(
        (
            tcp_phi,
            0.75 * native_pose.clamp(0.0, 1.0) + 0.25 * tcp_phi,
            0.65 * native_pose.clamp(0.0, 1.0) + 0.35 * overlap_phi,
            0.60 * native_pose.clamp(0.0, 1.0) + 0.35 * high_overlap + 0.05 * success.float(),
        ),
        (
            contact,
            loose_pose,
            final_funnel,
            success,
        ),
        maintenance=(
            native_normalized,
            native_normalized,
            native_normalized,
            native_normalized,
        ),
        safety=_safety(env.tee),
    )


def _push_t_v10(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PushT v10: official-dense progress with stricter stage gates.

    v9 proved that a native-like absolute progress term is not sufficient when
    easy gates move every episode into the final stage. v10 keeps the same
    4-stage blueprint but removes the PushT-specific safety penalty and delays
    later transitions until the pose/overlap evidence is stronger.
    """
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)

    tcp_phi = torch.sqrt(_phi(tcp_obj, 5.0).clamp(0.0, 1.0))
    yaw_phi = (((1.0 + torch.cos(angle)) / 2.0).clamp(0.0, 1.0)) ** 2
    xy_phi = (_phi(xy_error, 5.0).clamp(0.0, 1.0)) ** 2
    official_dense = (0.50 * yaw_phi + 0.50 * xy_phi + 0.05 * tcp_phi).clamp(
        0.0, 1.05
    )
    success = _success(env, info)
    official_normalized = torch.where(
        success,
        torch.ones_like(official_dense),
        (official_dense / 3.0).clamp(0.0, 1.0),
    )

    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    mid_overlap = torch.clamp((overlap_phi - 0.40) / 0.35, 0.0, 1.0)
    high_overlap = torch.clamp((overlap_phi - 0.65) / 0.35, 0.0, 1.0)
    pose_phi = (0.50 * yaw_phi + 0.50 * xy_phi).clamp(0.0, 1.0)

    contact = tcp_obj <= 0.12
    translate_progress = contact & (
        (official_dense >= 0.45) | (xy_error <= 0.18) | (overlap_phi >= 0.35)
    )
    align_progress = contact & (
        (official_dense >= 0.72)
        | (overlap_phi >= 0.62)
        | ((xy_error <= 0.08) & (angle <= 0.65))
    )
    zeros = torch.zeros_like(tcp_obj)

    return _pack(
        (
            tcp_phi,
            (0.45 * tcp_phi + 0.45 * pose_phi + 0.10 * contact.float()).clamp(0.0, 1.0),
            (0.45 * pose_phi + 0.35 * mid_overlap + 0.20 * contact.float()).clamp(0.0, 1.0),
            (0.55 * high_overlap + 0.35 * pose_phi + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            contact,
            translate_progress,
            align_progress,
            success,
        ),
        maintenance=(
            0.50 * official_normalized,
            official_normalized,
            official_normalized,
            official_normalized,
        ),
        safety=zeros,
    )



def _push_t_v11(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PushT v11: geometry-only overlap, pose, and settle shaping."""
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    lin_speed = torch.linalg.norm(env.tee.linear_velocity, dim=-1)
    ang_speed = torch.linalg.norm(env.tee.angular_velocity, dim=-1)

    contact = tcp_obj <= 0.13
    tcp_phi = _phi(tcp_obj, 7.0)
    xy_phi = (1.0 - xy_error / 0.30).clamp(0.0, 1.0)
    tight_xy_phi = (1.0 - xy_error / 0.07).clamp(0.0, 1.0)
    yaw_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_yaw_phi = (1.0 - torch.clamp(angle / 0.45, 0.0, 1.0)).clamp(0.0, 1.0)
    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    mid_overlap = torch.clamp((overlap_phi - 0.25) / 0.45, 0.0, 1.0)
    high_overlap = torch.clamp((overlap_phi - 0.55) / 0.40, 0.0, 1.0)
    still = (_phi(lin_speed, 8.0) * _phi(ang_speed, 3.0)).clamp(0.0, 1.0)
    pose_phi = (0.44 * xy_phi + 0.34 * yaw_phi + 0.22 * overlap_phi).clamp(0.0, 1.0)
    tight_pose = (0.38 * tight_xy_phi + 0.34 * tight_yaw_phi + 0.28 * high_overlap).clamp(0.0, 1.0)
    translate_ready = contact & ((xy_phi >= 0.35) | (mid_overlap >= 0.20))
    align_ready = contact & ((pose_phi >= 0.55) | (mid_overlap >= 0.55) | ((xy_error <= 0.13) & (angle <= 1.10)))
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)

    return _pack(
        (
            tcp_phi,
            (0.42 * tcp_phi + 0.40 * pose_phi + 0.18 * contact.float()).clamp(0.0, 1.0),
            (0.50 * pose_phi + 0.30 * mid_overlap + 0.20 * contact.float()).clamp(0.0, 1.0),
            (0.54 * tight_pose + 0.24 * high_overlap + 0.12 * still + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            contact,
            translate_ready,
            align_ready,
            success,
        ),
        maintenance=(
            0.06 * tcp_phi,
            0.14 * pose_phi + 0.06 * contact.float(),
            0.20 * pose_phi + 0.12 * mid_overlap + 0.05 * contact.float(),
            0.22 * tight_pose + 0.12 * high_overlap + 0.08 * still + 0.08 * success.float(),
        ),
        safety=zeros,
    )


def _push_t_v12(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PushT v12: stronger terminal geometry shaping after v11 stalls in stage 3."""
    base = _push_t_v11(env, info)
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    lin_speed = torch.linalg.norm(env.tee.linear_velocity, dim=-1)
    ang_speed = torch.linalg.norm(env.tee.angular_velocity, dim=-1)

    contact = tcp_obj <= 0.14
    tcp_phi = _phi(tcp_obj, 8.0)
    xy_phi = (1.0 - xy_error / 0.32).clamp(0.0, 1.0)
    tight_xy_phi = (1.0 - xy_error / 0.055).clamp(0.0, 1.0)
    yaw_phi = (1.0 - torch.clamp(angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_yaw_phi = (1.0 - torch.clamp(angle / 0.32, 0.0, 1.0)).clamp(0.0, 1.0)
    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    mid_overlap = torch.clamp((overlap_phi - 0.20) / 0.50, 0.0, 1.0)
    high_overlap = torch.clamp((overlap_phi - 0.58) / 0.35, 0.0, 1.0)
    still = (_phi(lin_speed, 10.0) * _phi(ang_speed, 4.0)).clamp(0.0, 1.0)
    pose_phi = (0.42 * xy_phi + 0.34 * yaw_phi + 0.24 * overlap_phi).clamp(0.0, 1.0)
    terminal_pose = (
        0.32 * tight_xy_phi
        + 0.30 * tight_yaw_phi
        + 0.28 * high_overlap
        + 0.10 * still
    ).clamp(0.0, 1.0)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 1] = contact & ((xy_phi >= 0.25) | (mid_overlap >= 0.12) | (pose_phi >= 0.35))
    gates[:, 2] = contact & (
        (pose_phi >= 0.48)
        | (mid_overlap >= 0.45)
        | ((xy_error <= 0.16) & (angle <= 1.25))
    )
    gates[:, 3] = success
    potentials[:, 1] = (0.34 * tcp_phi + 0.42 * pose_phi + 0.24 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 2] = (0.42 * pose_phi + 0.36 * mid_overlap + 0.14 * high_overlap + 0.08 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.58 * terminal_pose + 0.26 * high_overlap + 0.10 * still + 0.06 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.18 * pose_phi + 0.08 * mid_overlap + 0.06 * contact.float()
    maintenance[:, 2] = 0.28 * pose_phi + 0.24 * mid_overlap + 0.10 * high_overlap + 0.05 * contact.float()
    maintenance[:, 3] = (
        0.40 * terminal_pose
        + 0.24 * high_overlap
        + 0.10 * tight_xy_phi
        + 0.10 * tight_yaw_phi
        + 0.10 * still
        + 0.08 * success.float()
    )
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)

def _roll_ball(env: Any, info: dict[str, Any]) -> RuleInputs:
    direction = env.ball.pose.p - env.goal_region.pose.p
    direction = direction / torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    hit_pos = env.ball.pose.p + direction * (env.ball_radius + 0.05)
    tcp_hit = _distance(_tcp(env), hit_pos)
    tcp_ball = _distance(_tcp(env), env.ball.pose.p)
    contact = tcp_ball <= env.ball_radius + 0.06
    goal = _distance(env.ball.pose.p, env.goal_region.pose.p, xy=True)
    placed = goal < env.goal_radius
    settled = placed & (torch.linalg.norm(env.ball.linear_velocity, dim=1) < 0.05)
    zeros = torch.zeros_like(tcp_hit)
    return _pack(
        (_phi(tcp_hit, 2.0), _phi(tcp_ball), _phi(goal), _phi(torch.linalg.norm(env.ball.linear_velocity, dim=1), 10.0)),
        (tcp_hit <= 0.06, contact, placed, settled),
        maintenance=(zeros, zeros, _loss_penalty(contact), zeros),
        safety=_safety(env.ball),
    )


def _roll_ball_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Replace saturated long-range tanh shaping with normalized progress."""
    ball_to_goal = env.goal_region.pose.p - env.ball.pose.p
    push_direction = ball_to_goal / torch.linalg.norm(
        ball_to_goal, dim=1, keepdim=True
    ).clamp_min(1e-6)
    hit_pos = env.ball.pose.p - push_direction * (env.ball_radius + 0.05)
    tcp_hit = _distance(_tcp(env), hit_pos)
    tcp_ball = _distance(_tcp(env), env.ball.pose.p)
    contact = tcp_ball <= env.ball_radius + 0.06
    tracking = tcp_ball <= 0.20
    goal = _distance(env.ball.pose.p, env.goal_region.pose.p, xy=True)
    long_progress = (1.0 - goal / 1.6).clamp(0.0, 1.0)
    placed = goal < env.goal_radius
    speed = torch.linalg.norm(env.ball.linear_velocity, dim=1)
    settled = placed & (speed < 0.05)
    zeros = torch.zeros_like(tcp_hit)
    return _pack(
        (
            _phi(tcp_hit, 2.0),
            0.7 * _phi(tcp_hit, 3.0) + 0.3 * contact.float(),
            0.8 * long_progress + 0.2 * _phi(tcp_hit, 2.0),
            0.7 * (1.0 - goal / env.goal_radius).clamp(0.0, 1.0)
            + 0.3 * _phi(speed, 10.0),
        ),
        (tcp_hit <= 0.06, contact, placed, settled),
        maintenance=(zeros, zeros, _loss_penalty(tracking, -0.02), zeros),
        safety=_safety(env.ball),
    )


def _roll_ball_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Retain the behind-ball geometry and reward directed rolling motion."""
    ball_to_goal = env.goal_region.pose.p - env.ball.pose.p
    goal = torch.linalg.norm(ball_to_goal[:, :2], dim=1)
    push_direction = ball_to_goal / torch.linalg.norm(
        ball_to_goal, dim=1, keepdim=True
    ).clamp_min(1e-6)
    hit_pos = env.ball.pose.p - push_direction * (env.ball_radius + 0.05)
    tcp_hit = _distance(_tcp(env), hit_pos)
    tcp_ball = _distance(_tcp(env), env.ball.pose.p)
    contact = tcp_ball <= env.ball_radius + 0.06
    placed = goal < env.goal_radius
    speed = torch.linalg.norm(env.ball.linear_velocity, dim=1)
    settled = placed & (speed < 0.05)
    directed_speed = (
        env.ball.linear_velocity[:, :2] * push_direction[:, :2]
    ).sum(dim=1)
    directed_motion = torch.tanh(3.0 * directed_speed)
    # The randomized initial distance is roughly 1.2--1.6 m. Normalize the
    # full route while retaining a small behind-ball term for closed-loop push.
    roll_progress = (
        1.0 - (goal - env.goal_radius) / (1.60 - env.goal_radius)
    ).clamp(0.0, 1.0)
    behind_phi = _phi(tcp_hit, 2.0)
    zeros = torch.zeros_like(tcp_hit)
    forward_credit = 0.02 * directed_motion
    return _pack(
        (
            behind_phi,
            0.75 * _phi(tcp_hit, 3.0) + 0.25 * contact.float(),
            0.85 * roll_progress
            + 0.10 * behind_phi
            + 0.05 * ((directed_motion + 1.0) / 2.0),
            0.75 * placed.float() + 0.25 * _phi(speed, 10.0),
        ),
        (tcp_hit <= 0.06, contact, placed, settled),
        maintenance=(zeros, zeros, forward_credit, zeros),
        safety=_safety(env.ball),
    )



def _roll_ball_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Add absolute in-stage progress credit for the long rolling segment."""
    ball_to_goal = env.goal_region.pose.p - env.ball.pose.p
    goal = torch.linalg.norm(ball_to_goal[:, :2], dim=1)
    push_direction = ball_to_goal / torch.linalg.norm(
        ball_to_goal, dim=1, keepdim=True
    ).clamp_min(1e-6)
    hit_pos = env.ball.pose.p - push_direction * (env.ball_radius + 0.05)
    tcp_hit = _distance(_tcp(env), hit_pos)
    tcp_ball = _distance(_tcp(env), env.ball.pose.p)
    contact = tcp_ball <= env.ball_radius + 0.06
    placed = goal < env.goal_radius
    speed = torch.linalg.norm(env.ball.linear_velocity, dim=1)
    directed_speed = (
        env.ball.linear_velocity[:, :2] * push_direction[:, :2]
    ).sum(dim=1)
    directed_motion = torch.clamp(torch.tanh(4.0 * directed_speed), 0.0, 1.0)
    roll_progress = (
        1.0 - (goal - env.goal_radius) / (1.60 - env.goal_radius)
    ).clamp(0.0, 1.0)
    behind_phi = _phi(tcp_hit, 2.0)
    reached_hit = tcp_hit <= 0.06
    zeros = torch.zeros_like(tcp_hit)
    return _pack(
        (
            behind_phi,
            0.80 * behind_phi + 0.20 * contact.float(),
            0.82 * roll_progress + 0.10 * behind_phi + 0.08 * directed_motion,
            0.85 * placed.float() + 0.15 * _phi(speed, 10.0),
        ),
        (reached_hit, contact | reached_hit, placed, _success(env, info)),
        maintenance=(
            zeros,
            0.02 * contact.float(),
            0.08 * roll_progress * (contact | reached_hit).float()
            + 0.03 * directed_motion,
            0.03 * placed.float(),
        ),
        safety=_safety(env.ball),
    )


def _roll_ball_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Emphasize reaching the hit pose, then official-style goal progress."""
    ball_to_goal = env.goal_region.pose.p - env.ball.pose.p
    goal = torch.linalg.norm(ball_to_goal[:, :2], dim=1)
    push_direction = ball_to_goal / torch.linalg.norm(
        ball_to_goal, dim=1, keepdim=True
    ).clamp_min(1e-6)
    hit_pos = env.ball.pose.p - push_direction * (env.ball_radius + 0.05)
    tcp_hit = _distance(_tcp(env), hit_pos)
    tcp_ball = _distance(_tcp(env), env.ball.pose.p)
    contact = tcp_ball <= env.ball_radius + 0.06
    reached_hit = tcp_hit <= 0.055
    placed = goal < env.goal_radius
    speed = torch.linalg.norm(env.ball.linear_velocity, dim=1)
    directed_speed = (
        env.ball.linear_velocity[:, :2] * push_direction[:, :2]
    ).sum(dim=1)
    directed_motion = torch.clamp(torch.tanh(5.0 * directed_speed), 0.0, 1.0)
    official_goal_phi = (1.0 - torch.tanh(goal)).clamp(0.0, 1.0)
    near_goal_phi = (
        1.0 - (goal - env.goal_radius) / (0.55 - env.goal_radius)
    ).clamp(0.0, 1.0)
    behind_phi = _phi(tcp_hit, 2.0)
    zeros = torch.zeros_like(tcp_hit)
    return _pack(
        (
            behind_phi,
            0.85 * behind_phi + 0.15 * contact.float(),
            0.75 * official_goal_phi + 0.15 * near_goal_phi + 0.10 * directed_motion,
            0.80 * placed.float() + 0.20 * _phi(speed, 10.0),
        ),
        (reached_hit, contact | reached_hit, near_goal_phi >= 0.88, _success(env, info)),
        maintenance=(
            zeros,
            0.02 * contact.float(),
            0.10 * official_goal_phi + 0.05 * directed_motion,
            0.04 * placed.float(),
        ),
        safety=_safety(env.ball),
    )


def _roll_ball_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """C2-oriented RollBall shaping with stronger contact-gated rolling credit."""
    ball_to_goal = env.goal_region.pose.p - env.ball.pose.p
    goal = torch.linalg.norm(ball_to_goal[:, :2], dim=1)
    push_direction = ball_to_goal / torch.linalg.norm(
        ball_to_goal, dim=1, keepdim=True
    ).clamp_min(1e-6)
    hit_pos = env.ball.pose.p - push_direction * (env.ball_radius + 0.05)
    tcp_hit = _distance(_tcp(env), hit_pos)
    tcp_ball = _distance(_tcp(env), env.ball.pose.p)
    contact = tcp_ball <= env.ball_radius + 0.075
    tracking = contact | (tcp_ball <= env.ball_radius + 0.18)
    reached_hit = tcp_hit <= 0.07
    placed = goal < env.goal_radius
    speed = torch.linalg.norm(env.ball.linear_velocity, dim=1)
    directed_speed = (
        env.ball.linear_velocity[:, :2] * push_direction[:, :2]
    ).sum(dim=1)
    directed_motion = torch.clamp(torch.tanh(5.0 * directed_speed), 0.0, 1.0)
    long_progress = (
        1.0 - (goal - env.goal_radius) / (1.60 - env.goal_radius)
    ).clamp(0.0, 1.0)
    mid_progress = (
        1.0 - (goal - env.goal_radius) / (0.85 - env.goal_radius)
    ).clamp(0.0, 1.0)
    near_goal_phi = (
        1.0 - (goal - env.goal_radius) / (0.45 - env.goal_radius)
    ).clamp(0.0, 1.0)
    behind_phi = _phi(tcp_hit, 2.0)
    contact_phi = _phi(tcp_ball, 4.0)
    zeros = torch.zeros_like(tcp_hit)
    tracking_weight = tracking.float()
    return _pack(
        (
            behind_phi,
            0.70 * behind_phi + 0.30 * contact_phi,
            0.60 * long_progress + 0.25 * mid_progress + 0.15 * directed_motion,
            0.70 * near_goal_phi + 0.20 * placed.float() + 0.10 * _phi(speed, 10.0),
        ),
        (
            reached_hit,
            contact | reached_hit,
            (goal <= 0.45) | (mid_progress >= 0.72),
            _success(env, info),
        ),
        maintenance=(
            zeros,
            0.025 * contact.float(),
            tracking_weight * (0.08 * long_progress + 0.06 * directed_motion),
            0.12 * near_goal_phi + 0.04 * placed.float(),
        ),
        safety=_safety(env.ball),
    )



def _push_t_v13(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PushT v13: official-pose funnel with permissive stage entry."""
    base = _push_t_v12(env, info)
    tcp_obj = _distance(_tcp(env), env.tee.pose.p)
    xy_error = _distance(env.tee.pose.p, env.goal_tee.pose.p, xy=True)
    angle = _yaw_error(env.tee.pose.q, env.goal_tee.pose.q)
    lin_speed = torch.linalg.norm(env.tee.linear_velocity, dim=-1)
    ang_speed = torch.linalg.norm(env.tee.angular_velocity, dim=-1)
    contact = tcp_obj <= 0.16
    tcp_phi = _phi(tcp_obj, 7.0)
    xy_phi = ((1.0 - torch.tanh(5.0 * xy_error)) ** 2).clamp(0.0, 1.0)
    yaw_phi = (((torch.cos(angle) + 1.0) / 2.0) ** 2).clamp(0.0, 1.0)
    overlap = env.pseudo_render_intersection().clamp(0.0, 1.0)
    overlap_phi = torch.clamp(overlap / env.intersection_thresh, 0.0, 1.0)
    mid_overlap = torch.clamp((overlap_phi - 0.25) / 0.50, 0.0, 1.0)
    high_overlap = torch.clamp((overlap_phi - 0.58) / 0.35, 0.0, 1.0)
    still = (_phi(lin_speed, 8.0) * _phi(ang_speed, 3.0)).clamp(0.0, 1.0)
    pose_phi = (0.46 * xy_phi + 0.36 * yaw_phi + 0.18 * overlap_phi).clamp(0.0, 1.0)
    terminal = (0.34 * xy_phi + 0.30 * yaw_phi + 0.26 * high_overlap + 0.10 * still).clamp(0.0, 1.0)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = contact | (tcp_phi >= 0.32)
    gates[:, 1] = contact | (pose_phi >= 0.30) | (mid_overlap >= 0.15)
    gates[:, 2] = (pose_phi >= 0.48) | (mid_overlap >= 0.38) | ((xy_error <= 0.18) & (angle <= 1.35))
    gates[:, 3] = success
    potentials[:, 0] = tcp_phi
    potentials[:, 1] = (0.32 * tcp_phi + 0.48 * pose_phi + 0.20 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 2] = (0.48 * pose_phi + 0.30 * mid_overlap + 0.14 * high_overlap + 0.08 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.60 * terminal + 0.24 * high_overlap + 0.10 * still + 0.06 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.08 * tcp_phi
    maintenance[:, 1] = 0.22 * pose_phi + 0.08 * contact.float()
    maintenance[:, 2] = 0.32 * pose_phi + 0.22 * mid_overlap + 0.10 * high_overlap + 0.06 * contact.float()
    maintenance[:, 3] = 0.42 * terminal + 0.22 * high_overlap + 0.08 * still + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)

def _stack_cube(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_obj = _distance(_tcp(env), env.cubeA.pose.p)
    grasped = info.get("is_cubeA_grasped", _grasp(env, env.cubeA)).bool()
    target = env.cubeB.pose.p.clone()
    target[:, 2] += 2 * env.cube_half_size[2]
    goal_error = _distance(env.cubeA.pose.p, target)
    on_base = info.get("is_cubeA_on_cubeB", goal_error <= 0.01).bool()
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_obj)
    keep = _loss_penalty(grasped)
    return _pack(
        (_phi(tcp_obj), grasped.float(), _phi(goal_error), _phi(goal_error, 15.0), success.float()),
        (tcp_obj <= 0.05, grasped, grasped & (goal_error <= 0.04), on_base, success),
        maintenance=(zeros, zeros, keep, keep, zeros),
        safety=_safety(env.cubeA),
    )


def _stack_pyramid(env: Any, info: dict[str, Any]) -> RuleInputs:
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    red_green = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    beside = red_green <= beside_threshold
    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2]) + 2 * env.cube_half_size[2]
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_above = (blue_goal <= 0.04) & grasp_c
    on_red = (_distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True) <= beside_threshold) & ((env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2]) > 0.02)
    on_green = (_distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True) <= beside_threshold) & ((env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2]) > 0.02)
    stacked = on_red & on_green
    success = _success(env, info)
    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    zeros = torch.zeros_like(tcp_a)
    return _pack(
        (
            0.5 * _phi(tcp_a) + 0.5 * grasp_a.float(),
            _phi(red_green),
            0.5 * beside.float() + 0.5 * (~grasp_a).float(),
            0.5 * _phi(tcp_c) + 0.5 * grasp_c.float(),
            _phi(blue_goal),
            stacked.float(),
            success.float(),
        ),
        (
            grasp_a,
            beside & grasp_a,
            beside & (~grasp_a) & env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5),
            grasp_c,
            blue_above,
            stacked,
            success,
        ),
        maintenance=(zeros, _loss_penalty(grasp_a), zeros, zeros, _loss_penalty(grasp_c), _loss_penalty(grasp_c), zeros),
        safety=_safety(env.cubeA) + _safety(env.cubeC),
    )


def _stack_pyramid_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Iteration-2 rule reward for StackPyramid-v1.

    Keep the one-shot Frame-VLM's seven-stage decomposition, but correct three
    failure modes observed in the v1 PPO runs: targeting zero base separation,
    forgetting the completed base while reaching for the blue cube, and giving
    no dense release/settling signal after the blue cube is positioned.
    """

    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    red_green_z = torch.abs(env.cubeA.pose.p[:, 2] - env.cubeB.pose.p[:, 2])
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    # Side-by-side 4 cm cubes should have about 4 cm center separation. The v1
    # potential minimized this distance to zero and rewarded cube collisions.
    base_separation_error = torch.abs(red_green_xy - 2 * env.cube_half_size[0])
    base_score = (
        0.7 * _phi(base_separation_error, 25.0)
        + 0.3 * _phi(red_green_z, 30.0)
    )
    base_valid = (red_green_xy <= beside_threshold) & (red_green_z <= 0.012)

    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    blue_a_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeA.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    blue_b_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeB.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    supported_geometry = (
        base_valid
        & (blue_a_xy <= beside_threshold)
        & (blue_b_xy <= beside_threshold)
        & (blue_a_z_error <= 0.015)
        & (blue_b_z_error <= 0.015)
    )
    placement_score = (
        0.65 * _phi(blue_goal, 15.0)
        + 0.20 * _phi(torch.maximum(blue_a_z_error, blue_b_z_error), 25.0)
        + 0.15 * supported_geometry.float()
    )

    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_a)
    keep_base = _loss_penalty(base_valid, -0.10)
    keep_blue_grasped = _loss_penalty(grasp_c, -0.05)
    keep_stack = _loss_penalty(supported_geometry, -0.10)

    return _pack(
        (
            0.7 * _phi(tcp_a) + 0.3 * grasp_a.float(),
            base_score,
            0.45 * base_score + 0.30 * (~grasp_a).float() + 0.25 * static_a.float(),
            0.45 * _phi(tcp_c) + 0.35 * grasp_c.float() + 0.20 * base_score,
            0.75 * _phi(blue_goal, 12.0) + 0.15 * base_score + 0.10 * grasp_c.float(),
            placement_score,
            (
                0.35 * placement_score
                + 0.30 * (~grasp_c).float()
                + 0.20 * static_c.float()
                + 0.15 * static_a.float()
            ),
        ),
        (
            grasp_a,
            base_valid & grasp_a,
            base_valid & (~grasp_a) & static_a,
            base_valid & grasp_c,
            base_valid & grasp_c & (blue_goal <= 0.035),
            supported_geometry & grasp_c & (blue_goal <= 0.025),
            success,
        ),
        maintenance=(
            zeros,
            _loss_penalty(grasp_a),
            zeros,
            keep_base,
            keep_base + keep_blue_grasped,
            keep_base + keep_blue_grasped,
            keep_base + keep_stack,
        ),
        safety=_safety(env.cubeA) + _safety(env.cubeC),
    )


def _stack_pyramid_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v3 with official-compatible base and stronger blue placement."""
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    red_green_z = torch.abs(env.cubeA.pose.p[:, 2] - env.cubeB.pose.p[:, 2])
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    target_sep = 2 * env.cube_half_size[0]
    base_sep_score = (1.0 - torch.abs(red_green_xy - target_sep) / 0.08).clamp(0.0, 1.0)
    official_base_close = red_green_xy <= beside_threshold
    base_loose = (red_green_xy <= beside_threshold + 0.025) & (red_green_z <= 0.030)
    base_score = (
        0.45 * _phi(red_green_xy, 8.0)
        + 0.35 * base_sep_score
        + 0.10 * _phi(red_green_z, 30.0)
        + 0.10 * official_base_close.float()
    ).clamp(0.0, 1.0)

    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_xy = _distance(env.cubeC.pose.p, target, xy=True)
    blue_z_error = torch.abs(env.cubeC.pose.p[:, 2] - target[:, 2])
    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    blue_a_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeA.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    blue_b_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeB.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    support_loose = (
        base_loose
        & (blue_a_xy <= beside_threshold + 0.020)
        & (blue_b_xy <= beside_threshold + 0.020)
        & (torch.maximum(blue_a_z_error, blue_b_z_error) <= 0.030)
    )
    support_tight = (
        official_base_close
        & (blue_a_xy <= beside_threshold)
        & (blue_b_xy <= beside_threshold)
        & (torch.maximum(blue_a_z_error, blue_b_z_error) <= 0.018)
    )
    blue_place_score = (
        0.50 * _phi(blue_goal, 12.0)
        + 0.20 * _phi(blue_xy, 16.0)
        + 0.15 * _phi(blue_z_error, 30.0)
        + 0.15 * support_loose.float()
    ).clamp(0.0, 1.0)
    release_score = (
        0.50 * support_tight.float()
        + 0.25 * (~grasp_c).float()
        + 0.15 * static_c.float()
        + 0.10 * static_a.float()
    ).clamp(0.0, 1.0)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_a)
    keep_base = _loss_penalty(base_loose, -0.03)
    keep_blue = _loss_penalty(grasp_c | support_loose, -0.03)

    return _pack(
        (
            0.55 * _phi(tcp_a) + 0.45 * grasp_a.float(),
            0.75 * base_score + 0.25 * grasp_a.float(),
            0.65 * base_score + 0.20 * (~grasp_a).float() + 0.15 * static_a.float(),
            0.50 * _phi(tcp_c) + 0.30 * grasp_c.float() + 0.20 * base_score,
            0.65 * blue_place_score + 0.20 * grasp_c.float() + 0.15 * base_score,
            0.80 * blue_place_score + 0.20 * support_loose.float(),
            0.70 * release_score + 0.30 * success.float(),
        ),
        (
            grasp_a,
            base_loose & grasp_a,
            base_loose & (~grasp_a) & static_a,
            base_loose & grasp_c,
            base_loose & grasp_c & (blue_goal <= 0.060),
            support_loose & (blue_goal <= 0.045),
            success,
        ),
        maintenance=(
            zeros,
            0.04 * base_score,
            0.05 * base_score,
            keep_base + 0.04 * _phi(tcp_c) + 0.03 * grasp_c.float(),
            keep_base + keep_blue + 0.08 * blue_place_score,
            keep_base + keep_blue + 0.12 * blue_place_score + 0.03 * support_loose.float(),
            0.10 * release_score,
        ),
        safety=_safety(env.cubeA) + _safety(env.cubeC),
    )


def _stack_pyramid_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v4: open early stages and allow base-ready skip.

    v3 often stayed in grasp_red. The official planner skips moving the red cube
    when the red/green base is already close, so v4 lets a stable base satisfy
    the early red stages and adds small absolute reach/grasp/base maintenance.
    """
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_a = _phi(tcp_a, 5.0)
    reach_c = _phi(tcp_c, 5.0)

    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    red_green_z = torch.abs(env.cubeA.pose.p[:, 2] - env.cubeB.pose.p[:, 2])
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    target_sep = 2 * env.cube_half_size[0]
    base_sep_score = (1.0 - torch.abs(red_green_xy - target_sep) / 0.10).clamp(0.0, 1.0)
    base_distance_score = _phi(red_green_xy, 6.0)
    base_z_score = _phi(red_green_z, 25.0)
    base_score = (
        0.45 * base_sep_score
        + 0.35 * base_distance_score
        + 0.10 * base_z_score
        + 0.10 * (red_green_xy <= beside_threshold).float()
    ).clamp(0.0, 1.0)
    base_loose = (red_green_xy <= beside_threshold + 0.035) & (red_green_z <= 0.035)
    base_tight = (red_green_xy <= beside_threshold) & (red_green_z <= 0.020)
    base_ready = base_loose & static_a & (~grasp_a)

    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_xy = _distance(env.cubeC.pose.p, target, xy=True)
    blue_z_error = torch.abs(env.cubeC.pose.p[:, 2] - target[:, 2])
    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    blue_a_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeA.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    blue_b_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeB.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    support_loose = (
        base_loose
        & (blue_a_xy <= beside_threshold + 0.025)
        & (blue_b_xy <= beside_threshold + 0.025)
        & (torch.maximum(blue_a_z_error, blue_b_z_error) <= 0.035)
    )
    support_tight = (
        base_tight
        & (blue_a_xy <= beside_threshold)
        & (blue_b_xy <= beside_threshold)
        & (torch.maximum(blue_a_z_error, blue_b_z_error) <= 0.020)
    )
    blue_height_score = _phi(blue_z_error, 20.0)
    blue_place_score = (
        0.45 * _phi(blue_goal, 10.0)
        + 0.25 * _phi(blue_xy, 14.0)
        + 0.15 * blue_height_score
        + 0.15 * support_loose.float()
    ).clamp(0.0, 1.0)
    release_score = (
        0.45 * support_tight.float()
        + 0.25 * (~grasp_c).float()
        + 0.15 * static_c.float()
        + 0.15 * static_a.float()
    ).clamp(0.0, 1.0)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_a)
    keep_base = _loss_penalty(base_loose, -0.02)
    keep_blue = _loss_penalty(grasp_c | support_loose, -0.02)

    return _pack(
        (
            (0.55 * reach_a + 0.45 * grasp_a.float()).clamp(0.0, 1.0),
            (0.45 * base_score + 0.35 * grasp_a.float() + 0.20 * reach_a).clamp(0.0, 1.0),
            (0.70 * base_score + 0.15 * (~grasp_a).float() + 0.15 * static_a.float()).clamp(0.0, 1.0),
            (0.35 * base_score + 0.35 * reach_c + 0.30 * grasp_c.float()).clamp(0.0, 1.0),
            (0.45 * blue_place_score + 0.25 * grasp_c.float() + 0.20 * base_score + 0.10 * blue_height_score).clamp(0.0, 1.0),
            (0.70 * blue_place_score + 0.30 * support_loose.float()).clamp(0.0, 1.0),
            (0.60 * release_score + 0.40 * success.float()).clamp(0.0, 1.0),
        ),
        (
            grasp_a | base_ready,
            base_loose & (grasp_a | base_ready),
            base_ready,
            base_ready & grasp_c,
            base_ready & grasp_c & (blue_goal <= 0.075),
            support_loose & (blue_goal <= 0.055),
            success,
        ),
        maintenance=(
            0.04 * reach_a + 0.06 * grasp_a.float(),
            0.04 * grasp_a.float() + 0.08 * base_score,
            0.10 * base_score + 0.03 * base_ready.float(),
            keep_base + 0.04 * reach_c + 0.06 * grasp_c.float(),
            keep_base + keep_blue + 0.10 * blue_place_score,
            keep_base + keep_blue + 0.12 * blue_place_score + 0.03 * support_loose.float(),
            0.10 * release_score,
        ),
        safety=zeros,
    )


def _stack_pyramid_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v5: require tighter blue support before release.

    v4 opens the early red/base stages but can enter release with loose support
    and then reward letting go without satisfying official success. v5 keeps the
    early-stage fixes and makes the stack/release stages focus on tight support.
    """
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_a = _phi(tcp_a, 5.0)
    reach_c = _phi(tcp_c, 5.0)

    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    red_green_z = torch.abs(env.cubeA.pose.p[:, 2] - env.cubeB.pose.p[:, 2])
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    target_sep = 2 * env.cube_half_size[0]
    base_sep_score = (1.0 - torch.abs(red_green_xy - target_sep) / 0.10).clamp(0.0, 1.0)
    base_distance_score = _phi(red_green_xy, 6.0)
    base_z_score = _phi(red_green_z, 25.0)
    base_score = (
        0.45 * base_sep_score
        + 0.35 * base_distance_score
        + 0.10 * base_z_score
        + 0.10 * (red_green_xy <= beside_threshold).float()
    ).clamp(0.0, 1.0)
    base_loose = (red_green_xy <= beside_threshold + 0.035) & (red_green_z <= 0.035)
    base_tight = (red_green_xy <= beside_threshold) & (red_green_z <= 0.020)
    base_ready = base_loose & static_a & (~grasp_a)

    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_xy = _distance(env.cubeC.pose.p, target, xy=True)
    blue_z_error = torch.abs(env.cubeC.pose.p[:, 2] - target[:, 2])
    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    blue_a_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeA.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    blue_b_z_error = torch.abs(
        env.cubeC.pose.p[:, 2]
        - env.cubeB.pose.p[:, 2]
        - 2 * env.cube_half_size[2]
    )
    support_loose = (
        base_loose
        & (blue_a_xy <= beside_threshold + 0.025)
        & (blue_b_xy <= beside_threshold + 0.025)
        & (torch.maximum(blue_a_z_error, blue_b_z_error) <= 0.035)
    )
    support_tight = (
        base_tight
        & (blue_a_xy <= beside_threshold)
        & (blue_b_xy <= beside_threshold)
        & (torch.maximum(blue_a_z_error, blue_b_z_error) <= 0.020)
    )
    blue_height_score = _phi(blue_z_error, 20.0)
    blue_place_score = (
        0.45 * _phi(blue_goal, 10.0)
        + 0.25 * _phi(blue_xy, 14.0)
        + 0.15 * blue_height_score
        + 0.15 * support_loose.float()
    ).clamp(0.0, 1.0)
    final_support = support_tight | (
        support_loose & (blue_goal <= 0.035) & (blue_z_error <= 0.018)
    )
    success = _success(env, info)
    release_score = (
        0.65 * final_support.float()
        + 0.15 * (~grasp_c).float() * final_support.float()
        + 0.10 * static_c.float() * final_support.float()
        + 0.10 * success.float()
    ).clamp(0.0, 1.0)
    zeros = torch.zeros_like(tcp_a)
    keep_base = _loss_penalty(base_loose, -0.02)
    keep_blue = _loss_penalty(grasp_c | support_loose, -0.02)

    return _pack(
        (
            (0.55 * reach_a + 0.45 * grasp_a.float()).clamp(0.0, 1.0),
            (0.45 * base_score + 0.35 * grasp_a.float() + 0.20 * reach_a).clamp(0.0, 1.0),
            (0.70 * base_score + 0.15 * (~grasp_a).float() + 0.15 * static_a.float()).clamp(0.0, 1.0),
            (0.35 * base_score + 0.35 * reach_c + 0.30 * grasp_c.float()).clamp(0.0, 1.0),
            (0.45 * blue_place_score + 0.25 * grasp_c.float() + 0.20 * base_score + 0.10 * blue_height_score).clamp(0.0, 1.0),
            (0.55 * blue_place_score + 0.45 * final_support.float()).clamp(0.0, 1.0),
            (0.75 * final_support.float() + 0.25 * success.float()).clamp(0.0, 1.0),
        ),
        (
            grasp_a | base_ready,
            base_loose & (grasp_a | base_ready),
            base_ready,
            base_ready & grasp_c,
            base_ready & grasp_c & (blue_goal <= 0.075),
            final_support,
            success,
        ),
        maintenance=(
            0.04 * reach_a + 0.06 * grasp_a.float(),
            0.04 * grasp_a.float() + 0.08 * base_score,
            0.10 * base_score + 0.03 * base_ready.float(),
            keep_base + 0.05 * reach_c + 0.08 * grasp_c.float(),
            keep_base + keep_blue + 0.12 * blue_place_score,
            keep_base + keep_blue + 0.10 * blue_place_score + 0.08 * final_support.float(),
            0.14 * release_score,
        ),
        safety=zeros,
    )


def _stack_pyramid_v6(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v6: match official success geometry and isolate release.

    v5 can reach the blue-placement stages but almost never satisfies official
    success. The main mismatch is that v5 uses a precise midpoint/height target,
    while the official evaluator only requires: cube A next to cube B in XY,
    cube C above both A and B in XY, cube C static, and cube C not grasped.
    Keep the same reusable staged recipe, but use the official-compatible
    support predicate for the late stages and only reward release when support
    is already present.
    """
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_a = _phi(tcp_a, 5.0)
    reach_c = _phi(tcp_c, 5.0)

    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_xy_phi = (1.0 - red_green_xy / (beside_threshold + 0.060)).clamp(0.0, 1.0)
    target_sep = 2 * env.cube_half_size[0]
    base_sep_phi = (1.0 - torch.abs(red_green_xy - target_sep) / 0.12).clamp(0.0, 1.0)
    base_official = red_green_xy <= beside_threshold
    base_loose = red_green_xy <= beside_threshold + 0.045
    base_score = (
        0.55 * base_xy_phi
        + 0.25 * base_sep_phi
        + 0.20 * base_official.float()
    ).clamp(0.0, 1.0)
    base_ready = base_official & static_a & (~grasp_a)
    base_usable = base_loose & (~grasp_a)

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(
        blue_a_xy - beside_threshold,
        blue_b_xy - beside_threshold,
    ).clamp_min(0.0)
    support_xy_phi = (1.0 - support_xy_error / 0.070).clamp(0.0, 1.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    top_height_phi = ((min_top_gap - 0.005) / 0.040).clamp(0.0, 1.0)

    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_goal_phi = _phi(blue_goal, 8.0)
    support_score = (
        0.45 * support_xy_phi
        + 0.25 * top_height_phi
        + 0.15 * blue_goal_phi
        + 0.15 * base_official.float()
    ).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.040) & (min_top_gap > 0.010)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)

    zeros = torch.zeros_like(tcp_a)
    keep_base = _loss_penalty(base_loose, -0.02)
    keep_blue_grasp = _loss_penalty(grasp_c | support_loose, -0.02)
    release_score = (
        0.45 * support_official.float()
        + 0.25 * release_good.float()
        + 0.20 * (release_good & static_c).float()
        + 0.10 * success.float()
    ).clamp(0.0, 1.0)

    return _pack(
        (
            (0.55 * reach_a + 0.45 * grasp_a.float()).clamp(0.0, 1.0),
            (0.55 * base_score + 0.30 * grasp_a.float() + 0.15 * reach_a).clamp(0.0, 1.0),
            (0.75 * base_score + 0.15 * base_ready.float() + 0.10 * static_a.float()).clamp(0.0, 1.0),
            (0.40 * reach_c + 0.35 * grasp_c.float() + 0.25 * base_score).clamp(0.0, 1.0),
            (0.55 * support_score + 0.25 * grasp_c.float() + 0.20 * base_score).clamp(0.0, 1.0),
            (0.70 * support_score + 0.30 * support_official.float()).clamp(0.0, 1.0),
            release_score,
        ),
        (
            grasp_a | base_ready,
            base_loose & (grasp_a | base_ready),
            base_ready | (base_official & (~grasp_a)),
            base_usable & grasp_c,
            base_usable & grasp_c & (support_score >= 0.55),
            support_official & grasp_c,
            success,
        ),
        maintenance=(
            0.04 * reach_a + 0.06 * grasp_a.float(),
            0.06 * grasp_a.float() + 0.08 * base_score,
            0.12 * base_score + 0.04 * base_ready.float(),
            keep_base + 0.05 * reach_c + 0.08 * grasp_c.float(),
            keep_base + keep_blue_grasp + 0.14 * support_score,
            keep_base + keep_blue_grasp + 0.12 * support_score + 0.08 * support_official.float(),
            0.16 * release_score,
        ),
        safety=zeros,
    )


def _stack_pyramid_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v7: loosen early base gate, keep official final support.

    v6 made the late-stage geometry official-compatible but narrowed the
    early base gate too much. v7 re-opens the early transition to blue-cube
    manipulation using a loose side-by-side base, while keeping the final
    support/release predicate aligned to the official evaluator.
    """
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_a = _phi(tcp_a, 5.0)
    reach_c = _phi(tcp_c, 5.0)

    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_xy_phi = (1.0 - red_green_xy / (beside_threshold + 0.060)).clamp(0.0, 1.0)
    target_sep = 2 * env.cube_half_size[0]
    base_sep_phi = (1.0 - torch.abs(red_green_xy - target_sep) / 0.12).clamp(0.0, 1.0)
    base_official = red_green_xy <= beside_threshold
    base_loose = red_green_xy <= beside_threshold + 0.045
    base_score = (
        0.55 * base_xy_phi
        + 0.25 * base_sep_phi
        + 0.20 * base_official.float()
    ).clamp(0.0, 1.0)
    base_ready = base_loose & (~grasp_a)
    base_usable = base_loose & (~grasp_a)

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(
        blue_a_xy - beside_threshold,
        blue_b_xy - beside_threshold,
    ).clamp_min(0.0)
    support_xy_phi = (1.0 - support_xy_error / 0.070).clamp(0.0, 1.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    top_height_phi = ((min_top_gap - 0.005) / 0.040).clamp(0.0, 1.0)

    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_goal_phi = _phi(blue_goal, 8.0)
    support_score = (
        0.45 * support_xy_phi
        + 0.25 * top_height_phi
        + 0.15 * blue_goal_phi
        + 0.15 * base_official.float()
    ).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.040) & (min_top_gap > 0.010)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)

    zeros = torch.zeros_like(tcp_a)
    keep_base = _loss_penalty(base_loose, -0.02)
    keep_blue_grasp = _loss_penalty(grasp_c | support_loose, -0.02)
    release_score = (
        0.45 * support_official.float()
        + 0.25 * release_good.float()
        + 0.20 * (release_good & static_c).float()
        + 0.10 * success.float()
    ).clamp(0.0, 1.0)

    return _pack(
        (
            (0.55 * reach_a + 0.45 * grasp_a.float()).clamp(0.0, 1.0),
            (0.55 * base_score + 0.30 * grasp_a.float() + 0.15 * reach_a).clamp(0.0, 1.0),
            (0.75 * base_score + 0.15 * base_ready.float() + 0.10 * static_a.float()).clamp(0.0, 1.0),
            (0.40 * reach_c + 0.35 * grasp_c.float() + 0.25 * base_score).clamp(0.0, 1.0),
            (0.55 * support_score + 0.25 * grasp_c.float() + 0.20 * base_score).clamp(0.0, 1.0),
            (0.70 * support_score + 0.30 * support_official.float()).clamp(0.0, 1.0),
            release_score,
        ),
        (
            grasp_a | base_ready,
            base_loose & (grasp_a | base_ready),
            base_ready,
            base_usable & grasp_c,
            base_usable & grasp_c & (support_score >= 0.55),
            support_official & grasp_c,
            success,
        ),
        maintenance=(
            0.04 * reach_a + 0.06 * grasp_a.float(),
            0.06 * grasp_a.float() + 0.08 * base_score,
            0.12 * base_score + 0.04 * base_ready.float(),
            keep_base + 0.05 * reach_c + 0.08 * grasp_c.float(),
            keep_base + keep_blue_grasp + 0.14 * support_score,
            keep_base + keep_blue_grasp + 0.12 * support_score + 0.08 * support_official.float(),
            0.16 * release_score,
        ),
        safety=zeros,
    )


def _stack_pyramid_v8(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v8: make release-red an explicit objective.

    v7 still spent most evaluation steps in release_red while holding cube A.
    v8 keeps the official-compatible final support, but makes releasing A after
    forming a loose base the dominant stage-2 objective.
    """
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_a = _phi(tcp_a, 5.0)
    reach_c = _phi(tcp_c, 5.0)

    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_xy_phi = (1.0 - red_green_xy / (beside_threshold + 0.060)).clamp(0.0, 1.0)
    target_sep = 2 * env.cube_half_size[0]
    base_sep_phi = (1.0 - torch.abs(red_green_xy - target_sep) / 0.12).clamp(0.0, 1.0)
    base_official = red_green_xy <= beside_threshold
    base_loose = red_green_xy <= beside_threshold + 0.045
    base_score = (
        0.55 * base_xy_phi
        + 0.25 * base_sep_phi
        + 0.20 * base_official.float()
    ).clamp(0.0, 1.0)
    base_ready = base_loose & (~grasp_a)
    base_usable = base_loose & (~grasp_a)

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(
        blue_a_xy - beside_threshold,
        blue_b_xy - beside_threshold,
    ).clamp_min(0.0)
    support_xy_phi = (1.0 - support_xy_error / 0.070).clamp(0.0, 1.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    top_height_phi = ((min_top_gap - 0.005) / 0.040).clamp(0.0, 1.0)

    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_goal_phi = _phi(blue_goal, 8.0)
    support_score = (
        0.45 * support_xy_phi
        + 0.25 * top_height_phi
        + 0.15 * blue_goal_phi
        + 0.15 * base_official.float()
    ).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.040) & (min_top_gap > 0.010)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)

    zeros = torch.zeros_like(tcp_a)
    keep_base = _loss_penalty(base_loose, -0.02)
    keep_blue_grasp = _loss_penalty(grasp_c | support_loose, -0.02)
    release_score = (
        0.45 * support_official.float()
        + 0.25 * release_good.float()
        + 0.20 * (release_good & static_c).float()
        + 0.10 * success.float()
    ).clamp(0.0, 1.0)

    return _pack(
        (
            (0.55 * reach_a + 0.45 * grasp_a.float()).clamp(0.0, 1.0),
            (0.55 * base_score + 0.30 * grasp_a.float() + 0.15 * reach_a).clamp(0.0, 1.0),
            (0.45 * base_score + 0.45 * (~grasp_a).float() + 0.10 * static_a.float()).clamp(0.0, 1.0),
            (0.40 * reach_c + 0.35 * grasp_c.float() + 0.25 * base_score).clamp(0.0, 1.0),
            (0.55 * support_score + 0.25 * grasp_c.float() + 0.20 * base_score).clamp(0.0, 1.0),
            (0.70 * support_score + 0.30 * support_official.float()).clamp(0.0, 1.0),
            release_score,
        ),
        (
            grasp_a | base_ready,
            base_loose & (grasp_a | base_ready),
            base_ready,
            base_usable & grasp_c,
            base_usable & grasp_c & (support_score >= 0.55),
            support_official & grasp_c,
            success,
        ),
        maintenance=(
            0.04 * reach_a + 0.06 * grasp_a.float(),
            0.06 * grasp_a.float() + 0.08 * base_score,
            0.12 * base_score + 0.12 * (~grasp_a).float() - 0.04 * grasp_a.float(),
            keep_base + 0.05 * reach_c + 0.08 * grasp_c.float(),
            keep_base + keep_blue_grasp + 0.14 * support_score,
            keep_base + keep_blue_grasp + 0.12 * support_score + 0.08 * support_official.float(),
            0.16 * release_score,
        ),
        safety=zeros,
    )


def _stack_pyramid_v9(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v9: enter release on loose support, score final by official success."""
    base = _stack_pyramid_v8(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_official = red_green_xy <= beside_threshold
    base_loose = red_green_xy <= beside_threshold + 0.050
    base_usable = base_loose & (~grasp_a)

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(
        blue_a_xy - beside_threshold,
        blue_b_xy - beside_threshold,
    ).clamp_min(0.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    support_xy_phi = (1.0 - support_xy_error / 0.080).clamp(0.0, 1.0)
    top_height_phi = ((min_top_gap - 0.000) / 0.045).clamp(0.0, 1.0)
    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = (
        torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2])
        + 2 * env.cube_half_size[2]
    )
    blue_goal = _distance(env.cubeC.pose.p, target)
    support_score = (
        0.45 * support_xy_phi
        + 0.25 * top_height_phi
        + 0.15 * _phi(blue_goal, 8.0)
        + 0.15 * base_official.float()
    ).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.050) & (min_top_gap > 0.005)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 4] = (0.62 * support_score + 0.22 * grasp_c.float() + 0.16 * base_official.float()).clamp(0.0, 1.0)
    potentials[:, 5] = (0.78 * support_score + 0.22 * support_loose.float()).clamp(0.0, 1.0)
    potentials[:, 6] = (
        0.45 * support_score
        + 0.25 * release_good.float()
        + 0.15 * static_c.float() * release_good.float()
        + 0.15 * success.float()
    ).clamp(0.0, 1.0)
    gates[:, 4] = base_usable & grasp_c & (support_score >= 0.42)
    gates[:, 5] = support_loose & grasp_c
    gates[:, 6] = success
    maintenance[:, 4] = maintenance[:, 4] + 0.05 * support_score
    maintenance[:, 5] = maintenance[:, 5] + 0.08 * support_loose.float() + 0.06 * support_score
    maintenance[:, 6] = maintenance[:, 6] + 0.10 * release_good.float() + 0.06 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)



def _stack_pyramid_v10(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v10: boost blue-cube grasp acquisition after red base is ready."""
    base = _stack_pyramid_v9(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_c = _phi(tcp_c, 10.0)
    near_c = _phi(tcp_c, 20.0)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    close_near = (near_c * closed).clamp(0.0, 1.0)

    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_official = red_green_xy <= beside_threshold
    base_loose = red_green_xy <= beside_threshold + 0.050
    base_usable = base_loose & (~grasp_a)

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(blue_a_xy - beside_threshold, blue_b_xy - beside_threshold).clamp_min(0.0)
    min_top_gap = torch.minimum(env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2], env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2])
    support_xy_phi = (1.0 - support_xy_error / 0.070).clamp(0.0, 1.0)
    top_height_phi = ((min_top_gap - 0.005) / 0.040).clamp(0.0, 1.0)
    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2]) + 2 * env.cube_half_size[2]
    blue_goal = _distance(env.cubeC.pose.p, target)
    support_score = (0.42 * support_xy_phi + 0.26 * top_height_phi + 0.20 * _phi(blue_goal, 10.0) + 0.12 * base_official.float()).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.045) & (min_top_gap > 0.008)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 3] = (0.36 * reach_c + 0.28 * close_near + 0.26 * grasp_c.float() + 0.10 * base_usable.float()).clamp(0.0, 1.0)
    potentials[:, 4] = (0.54 * support_score + 0.26 * grasp_c.float() + 0.20 * base_official.float()).clamp(0.0, 1.0)
    potentials[:, 5] = (0.78 * support_score + 0.22 * support_loose.float()).clamp(0.0, 1.0)
    potentials[:, 6] = (0.45 * support_score + 0.25 * release_good.float() + 0.15 * success.float() + 0.15 * support_official.float()).clamp(0.0, 1.0)
    gates[:, 3] = base_usable & grasp_c
    gates[:, 4] = base_usable & grasp_c & (support_score >= 0.45)
    gates[:, 5] = support_loose & grasp_c
    gates[:, 6] = success
    maintenance[:, 3] = 0.16 * reach_c + 0.20 * close_near + 0.20 * grasp_c.float() + 0.06 * base_usable.float()
    maintenance[:, 4] = maintenance[:, 4] + 0.10 * support_score + 0.06 * grasp_c.float()
    maintenance[:, 5] = maintenance[:, 5] + 0.12 * support_score + 0.08 * support_loose.float()
    maintenance[:, 6] = maintenance[:, 6] + 0.12 * release_good.float() + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _stack_pyramid_v11(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v11: make blue-grasp stage closer to PickCube shaping."""
    base = _stack_pyramid_v10(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_c = _phi(tcp_c, 8.0)
    near_c = _phi(tcp_c, 18.0)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    close_near = (near_c * closed).clamp(0.0, 1.0)
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_loose = red_green_xy <= beside_threshold + 0.050
    base_usable = base_loose & (~grasp_a)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 3] = (0.50 * reach_c + 0.42 * grasp_c.float() + 0.08 * close_near).clamp(0.0, 1.0)
    gates[:, 3] = base_usable & grasp_c
    maintenance[:, 3] = 0.08 * reach_c + 0.08 * close_near + 0.30 * grasp_c.float() - 0.03 * (~base_usable).float()
    maintenance[:, 4] = maintenance[:, 4] + 0.08 * grasp_c.float()
    maintenance[:, 5] = maintenance[:, 5] + 0.08 * grasp_c.float()
    maintenance[:, 6] = maintenance[:, 6] + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _stack_pyramid_v12(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v12: blue-carry funnel without requiring grasp bit only."""
    base = _stack_pyramid_v11(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    reach_c = _phi(tcp_c, 10.0)
    near_c = _phi(tcp_c, 22.0)
    close_attempt = (tcp_c <= 0.055) & (closed >= 0.35)
    lift_c = torch.clamp((env.cubeC.pose.p[:, 2] - 0.02) / 0.060, 0.0, 1.0)
    carrying_c = grasp_c | close_attempt | ((tcp_c <= 0.075) & (lift_c >= 0.15))

    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_official = red_green_xy <= beside_threshold
    base_loose = red_green_xy <= beside_threshold + 0.050
    base_usable = base_loose & (~grasp_a)

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(blue_a_xy - beside_threshold, blue_b_xy - beside_threshold).clamp_min(0.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    support_xy_phi = (1.0 - support_xy_error / 0.080).clamp(0.0, 1.0)
    top_height_phi = ((min_top_gap - 0.000) / 0.045).clamp(0.0, 1.0)
    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2]) + 2 * env.cube_half_size[2]
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_goal_phi = _phi(blue_goal, 10.0)
    support_score = (
        0.42 * support_xy_phi
        + 0.24 * top_height_phi
        + 0.18 * blue_goal_phi
        + 0.10 * carrying_c.float()
        + 0.06 * base_official.float()
    ).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.050) & (min_top_gap > 0.005)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 3] = (0.38 * near_c + 0.24 * closed + 0.18 * carrying_c.float() + 0.14 * lift_c + 0.06 * base_usable.float()).clamp(0.0, 1.0)
    potentials[:, 4] = (0.60 * support_score + 0.22 * carrying_c.float() + 0.10 * lift_c + 0.08 * base_official.float()).clamp(0.0, 1.0)
    potentials[:, 5] = (0.70 * support_score + 0.20 * support_loose.float() + 0.10 * carrying_c.float()).clamp(0.0, 1.0)
    potentials[:, 6] = (0.42 * support_score + 0.22 * release_good.float() + 0.18 * static_c.float() * release_good.float() + 0.18 * success.float()).clamp(0.0, 1.0)
    gates[:, 3] = base_usable & carrying_c
    gates[:, 4] = base_usable & carrying_c & (support_score >= 0.38)
    gates[:, 5] = support_loose & (carrying_c | grasp_c)
    gates[:, 6] = success
    maintenance[:, 3] = 0.14 * reach_c + 0.16 * near_c + 0.16 * closed + 0.18 * carrying_c.float() + 0.06 * lift_c
    maintenance[:, 4] = 0.10 * carrying_c.float() + 0.16 * support_score + 0.06 * lift_c
    maintenance[:, 5] = 0.16 * support_score + 0.08 * support_loose.float() + 0.05 * carrying_c.float()
    maintenance[:, 6] = 0.12 * release_good.float() + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _stack_pyramid_v13(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v13: open blue-placement shaping after blue reach stage."""
    base = _stack_pyramid_v12(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    contact_c = tcp_c <= 0.09
    lift_c = torch.clamp((env.cubeC.pose.p[:, 2] - 0.02) / 0.060, 0.0, 1.0)
    carrying_c = grasp_c | (contact_c & (closed >= 0.25)) | ((tcp_c <= 0.10) & (lift_c >= 0.08))

    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_official = red_green_xy <= beside_threshold
    base_loose = red_green_xy <= beside_threshold + 0.055
    base_usable = base_loose & (~grasp_a)
    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(blue_a_xy - beside_threshold, blue_b_xy - beside_threshold).clamp_min(0.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2]) + 2 * env.cube_half_size[2]
    blue_goal = _distance(env.cubeC.pose.p, target)
    support_xy_phi = (1.0 - support_xy_error / 0.090).clamp(0.0, 1.0)
    top_height_phi = ((min_top_gap - 0.000) / 0.045).clamp(0.0, 1.0)
    blue_goal_phi = _phi(blue_goal, 9.0)
    support_score = (
        0.38 * support_xy_phi
        + 0.24 * top_height_phi
        + 0.18 * blue_goal_phi
        + 0.12 * carrying_c.float()
        + 0.08 * base_official.float()
    ).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.055) & (min_top_gap > 0.000)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 3] = torch.ones_like(gates[:, 3], dtype=torch.bool)
    gates[:, 4] = (support_score >= 0.36) | support_loose | (lift_c >= 0.35)
    gates[:, 5] = support_loose | support_official
    gates[:, 6] = success
    potentials[:, 4] = (
        0.42 * support_score
        + 0.18 * _phi(tcp_c, 10.0)
        + 0.14 * closed
        + 0.14 * lift_c
        + 0.12 * carrying_c.float()
    ).clamp(0.0, 1.0)
    potentials[:, 5] = (0.66 * support_score + 0.22 * support_loose.float() + 0.12 * support_official.float()).clamp(0.0, 1.0)
    potentials[:, 6] = (0.40 * support_score + 0.22 * release_good.float() + 0.18 * static_c.float() * release_good.float() + 0.20 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 4] = 0.16 * support_score + 0.10 * _phi(tcp_c, 10.0) + 0.10 * closed + 0.10 * lift_c + 0.08 * carrying_c.float()
    maintenance[:, 5] = 0.18 * support_score + 0.10 * support_loose.float() + 0.06 * support_official.float()
    maintenance[:, 6] = 0.12 * release_good.float() + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _stack_pyramid_v14(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v14: tighten final stage to official support/release."""
    base = _stack_pyramid_v13(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_error = (red_green_xy - beside_threshold).clamp_min(0.0)
    base_tight_phi = (1.0 - base_error / 0.030).clamp(0.0, 1.0)
    base_official = red_green_xy <= beside_threshold

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(blue_a_xy - beside_threshold, blue_b_xy - beside_threshold).clamp_min(0.0)
    support_tight_phi = (1.0 - support_xy_error / 0.030).clamp(0.0, 1.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    height_tight_phi = ((min_top_gap - 0.015) / 0.030).clamp(0.0, 1.0)
    support_official = base_official & (support_xy_error <= 0.0) & (min_top_gap > 0.020)
    release_good = support_official & (~grasp_c)
    success = _success(env, info)
    final_tight = (
        0.28 * base_tight_phi
        + 0.32 * support_tight_phi
        + 0.22 * height_tight_phi
        + 0.10 * release_good.float()
        + 0.08 * success.float()
    ).clamp(0.0, 1.0)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 5] = (0.68 * final_tight + 0.22 * support_official.float() + 0.10 * (~grasp_c).float()).clamp(0.0, 1.0)
    potentials[:, 6] = (0.62 * final_tight + 0.16 * static_c.float() * release_good.float() + 0.12 * static_a.float() * base_official.float() + 0.10 * success.float()).clamp(0.0, 1.0)
    gates[:, 5] = support_official | (final_tight >= 0.80)
    gates[:, 6] = success
    maintenance[:, 5] = 0.20 * final_tight + 0.08 * support_official.float() - 0.04 * grasp_c.float()
    maintenance[:, 6] = 0.22 * final_tight + 0.08 * release_good.float() + 0.08 * success.float() - 0.04 * grasp_c.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _stack_pyramid_v15(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v15: reopen blue transport and score continuous support."""
    base = _stack_pyramid_v13(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    reach_c = _phi(tcp_c, 10.0)
    lift_c = torch.clamp((env.cubeC.pose.p[:, 2] - 0.02) / 0.065, 0.0, 1.0)
    carrying_c = grasp_c | ((tcp_c <= 0.10) & (closed >= 0.20)) | ((tcp_c <= 0.12) & (lift_c >= 0.10))

    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_error = (red_green_xy - beside_threshold).clamp_min(0.0)
    base_phi = (1.0 - base_error / 0.070).clamp(0.0, 1.0)
    base_tight = red_green_xy <= beside_threshold + 0.020
    base_loose = red_green_xy <= beside_threshold + 0.070
    base_ready = base_loose & ((~grasp_a) | static_a | base_tight)

    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(blue_a_xy - beside_threshold, blue_b_xy - beside_threshold).clamp_min(0.0)
    support_xy_phi = (1.0 - support_xy_error / 0.090).clamp(0.0, 1.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    height_phi = ((min_top_gap + 0.005) / 0.055).clamp(0.0, 1.0)
    target = 0.5 * (env.cubeA.pose.p + env.cubeB.pose.p)
    target[:, 2] = torch.maximum(env.cubeA.pose.p[:, 2], env.cubeB.pose.p[:, 2]) + 2 * env.cube_half_size[2]
    blue_goal = _distance(env.cubeC.pose.p, target)
    blue_goal_phi = _phi(blue_goal, 8.0)
    support_score = (0.34 * support_xy_phi + 0.24 * height_phi + 0.18 * blue_goal_phi + 0.14 * base_phi + 0.10 * carrying_c.float()).clamp(0.0, 1.0)
    support_soft = base_loose & (support_xy_error <= 0.060) & (min_top_gap > -0.002)
    support_tight = base_tight & (support_xy_error <= 0.020) & (min_top_gap > 0.012)
    release_good = support_tight & (~grasp_c)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 2] = base_ready
    gates[:, 3] = base_ready & ((tcp_c <= 0.12) | (reach_c >= 0.35) | carrying_c)
    gates[:, 4] = base_ready & ((support_score >= 0.30) | (blue_goal <= 0.22) | (lift_c >= 0.22))
    gates[:, 5] = support_soft | support_tight
    gates[:, 6] = success
    potentials[:, 3] = (0.38 * reach_c + 0.26 * closed + 0.20 * carrying_c.float() + 0.16 * lift_c).clamp(0.0, 1.0)
    potentials[:, 4] = (0.58 * support_score + 0.18 * blue_goal_phi + 0.14 * carrying_c.float() + 0.10 * lift_c).clamp(0.0, 1.0)
    potentials[:, 5] = (0.64 * support_score + 0.22 * support_soft.float() + 0.14 * support_tight.float()).clamp(0.0, 1.0)
    potentials[:, 6] = (0.48 * support_score + 0.18 * release_good.float() + 0.16 * static_c.float() * release_good.float() + 0.10 * success.float() + 0.08 * base_tight.float()).clamp(0.0, 1.0)
    maintenance[:, 3] = 0.12 * reach_c + 0.12 * closed + 0.12 * carrying_c.float() + 0.08 * lift_c
    maintenance[:, 4] = 0.18 * support_score + 0.08 * blue_goal_phi + 0.06 * carrying_c.float()
    maintenance[:, 5] = 0.20 * support_score + 0.08 * support_soft.float() + 0.06 * support_tight.float()
    maintenance[:, 6] = 0.16 * support_score + 0.10 * release_good.float() + 0.06 * success.float() - 0.03 * grasp_c.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _stack_pyramid_v16(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v16: official-evaluate geometry with simple transport gates."""
    base = _stack_pyramid_v15(env, info)
    tcp = _tcp(env)
    pos_a = env.cubeA.pose.p
    pos_b = env.cubeB.pose.p
    pos_c = env.cubeC.pose.p
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    static_a = env.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    static_c = env.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)

    side_thresh = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    ab_xy = _distance(pos_a, pos_b, xy=True)
    base_error = (ab_xy - side_thresh).clamp_min(0.0)
    base_phi = (1.0 - base_error / 0.10).clamp(0.0, 1.0)
    base_tight = ab_xy <= side_thresh + 0.020
    base_loose = ab_xy <= side_thresh + 0.075

    tcp_a = _distance(tcp, pos_a)
    tcp_c = _distance(tcp, pos_c)
    reach_a = _phi(tcp_a, 9.0)
    reach_c = _phi(tcp_c, 9.0)
    lift_a = torch.clamp((pos_a[:, 2] - 0.02) / 0.060, 0.0, 1.0)
    lift_c = torch.clamp((pos_c[:, 2] - 0.02) / 0.070, 0.0, 1.0)
    carry_a = grasp_a | ((tcp_a <= 0.10) & (closed >= 0.18)) | ((tcp_a <= 0.12) & (lift_a >= 0.08))
    carry_c = grasp_c | ((tcp_c <= 0.10) & (closed >= 0.18)) | ((tcp_c <= 0.12) & (lift_c >= 0.08))

    target_c = 0.5 * (pos_a + pos_b)
    target_c[:, 2] = torch.maximum(pos_a[:, 2], pos_b[:, 2]) + 2 * env.cube_half_size[2]
    c_goal = _distance(pos_c, target_c)
    c_goal_phi = _phi(c_goal, 8.0)
    ca_xy = _distance(pos_c, pos_a, xy=True)
    cb_xy = _distance(pos_c, pos_b, xy=True)
    support_xy_error = torch.maximum(ca_xy - side_thresh, cb_xy - side_thresh).clamp_min(0.0)
    support_xy_phi = (1.0 - support_xy_error / 0.10).clamp(0.0, 1.0)
    min_top_gap = torch.minimum(pos_c[:, 2] - pos_a[:, 2], pos_c[:, 2] - pos_b[:, 2])
    height_phi = ((min_top_gap + 0.005) / 0.060).clamp(0.0, 1.0)
    support_phi = (0.34 * support_xy_phi + 0.26 * height_phi + 0.22 * c_goal_phi + 0.10 * base_phi + 0.08 * carry_c.float()).clamp(0.0, 1.0)
    support_loose = base_loose & (support_xy_error <= 0.070) & (min_top_gap > -0.005)
    support_tight = base_tight & (support_xy_error <= 0.026) & (min_top_gap > 0.010)
    release_good = support_tight & (~grasp_c) & static_c
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = (tcp_a <= 0.12) | carry_a | (base_phi >= 0.35)
    gates[:, 1] = carry_a | (base_phi >= 0.40)
    gates[:, 2] = base_loose | base_tight
    gates[:, 3] = base_loose & ((tcp_c <= 0.13) | carry_c | (reach_c >= 0.35))
    gates[:, 4] = base_loose & ((support_phi >= 0.28) | (c_goal <= 0.22) | (lift_c >= 0.16))
    gates[:, 5] = support_loose | support_tight
    gates[:, 6] = success
    potentials[:, 0] = (0.58 * reach_a + 0.20 * closed + 0.14 * carry_a.float() + 0.08 * base_phi).clamp(0.0, 1.0)
    potentials[:, 1] = (0.42 * base_phi + 0.26 * carry_a.float() + 0.18 * lift_a + 0.14 * reach_a).clamp(0.0, 1.0)
    potentials[:, 2] = (0.66 * base_phi + 0.16 * base_tight.float() + 0.10 * static_a.float() + 0.08 * (~grasp_a).float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.42 * reach_c + 0.24 * closed + 0.20 * carry_c.float() + 0.14 * lift_c).clamp(0.0, 1.0)
    potentials[:, 4] = (0.58 * support_phi + 0.20 * c_goal_phi + 0.12 * lift_c + 0.10 * carry_c.float()).clamp(0.0, 1.0)
    potentials[:, 5] = (0.66 * support_phi + 0.22 * support_loose.float() + 0.12 * support_tight.float()).clamp(0.0, 1.0)
    potentials[:, 6] = (0.48 * support_phi + 0.22 * release_good.float() + 0.18 * support_tight.float() + 0.12 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.10 * reach_a + 0.06 * closed + 0.05 * carry_a.float()
    maintenance[:, 1] = 0.16 * base_phi + 0.08 * lift_a + 0.06 * carry_a.float()
    maintenance[:, 2] = 0.24 * base_phi + 0.08 * base_tight.float() + 0.05 * static_a.float()
    maintenance[:, 3] = 0.12 * reach_c + 0.10 * closed + 0.08 * carry_c.float() + 0.06 * lift_c
    maintenance[:, 4] = 0.20 * support_phi + 0.08 * c_goal_phi + 0.06 * carry_c.float()
    maintenance[:, 5] = 0.24 * support_phi + 0.08 * support_loose.float() + 0.06 * support_tight.float()
    maintenance[:, 6] = 0.18 * support_phi + 0.12 * release_good.float() + 0.08 * success.float() - 0.03 * grasp_c.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)

def _assembling_kits(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_obj = _distance(_tcp(env), env.obj.pose.p)
    grasped = _grasp(env, env.obj)
    pos_error = info.get("pos_diff_norm", env._check_pos_diff()[1])
    rot_error = info.get("rot_diff", env._check_rot_diff()[0])
    height = torch.clamp(env.obj.pose.p[:, 2], min=0.0)
    pos_ok = info.get("pos_correct", pos_error < 0.02).bool()
    rot_ok = info.get("rot_correct", rot_error < torch.deg2rad(torch.tensor(4.0, device=env.device))).bool()
    aligned = grasped & pos_ok & rot_ok
    zeros = torch.zeros_like(tcp_obj)
    keep = _loss_penalty(grasped)
    return _pack(
        (_phi(tcp_obj), grasped.float(), 1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0), _phi(pos_error, 10.0), _phi(height, 20.0)),
        (
            tcp_obj <= 0.05,
            grasped,
            grasped & (rot_error <= 0.15),
            aligned,
            _success(env, info) & (~grasped),
        ),
        maintenance=(zeros, zeros, keep, keep, zeros),
        safety=_safety(env.obj),
    )


def _pick_single_ycb(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_obj = _distance(_tcp(env), env.obj.pose.p)
    grasped = info.get("is_grasped", _grasp(env, env.obj)).bool()
    goal = _distance(env.obj.pose.p, env.goal_site.pose.p)
    placed = info.get("is_obj_placed", goal <= env.goal_thresh).bool()
    zeros = torch.zeros_like(tcp_obj)
    keep = _loss_penalty(grasped)
    return _pack(
        (_phi(tcp_obj), grasped.float(), _phi(goal), 0.5 * placed.float() + 0.5 * _static(env).float()),
        (tcp_obj <= 0.06, grasped, placed, _success(env, info)),
        maintenance=(zeros, zeros, keep, zeros),
        safety=_safety(env.obj),
    )


def _pick_single_ycb_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Enter the terminal stage early enough to learn tight placement and settling."""
    tcp_obj = _distance(_tcp(env), env.obj.pose.p)
    grasped = info.get("is_grasped", _grasp(env, env.obj)).bool()
    goal = _distance(env.obj.pose.p, env.goal_site.pose.p)
    placed = info.get("is_obj_placed", goal <= env.goal_thresh).bool()
    near_goal = grasped & (goal <= 2.0 * env.goal_thresh)
    robot_speed = torch.linalg.norm(env.agent.robot.get_qvel()[..., :-2], dim=1)
    static_phi = _phi(robot_speed, 5.0)
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.05)
    keep_object = _loss_penalty(grasped | placed, -0.05)
    return _pack(
        (
            _phi(tcp_obj),
            0.7 * grasped.float() + 0.3 * _phi(tcp_obj),
            0.8 * _phi(goal, 5.0) + 0.2 * grasped.float(),
            0.75 * _phi(goal, 12.0) + 0.25 * static_phi,
        ),
        (tcp_obj <= 0.06, grasped, near_goal, _success(env, info)),
        maintenance=(zeros, zeros, keep_grasp, keep_object),
        safety=_safety(env.obj),
    )


def _pick_single_ycb_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Make settling credit conditional on actually placing the YCB object."""
    tcp_obj = _distance(_tcp(env), env.obj.pose.p)
    grasped = info.get("is_grasped", _grasp(env, env.obj)).bool()
    goal = _distance(env.obj.pose.p, env.goal_site.pose.p)
    placed = info.get("is_obj_placed", goal <= env.goal_thresh).bool()
    near_goal = grasped & (goal <= 2.5 * env.goal_thresh)
    robot_speed = torch.linalg.norm(env.agent.robot.get_qvel()[..., :-2], dim=1)
    placed_and_static = placed.float() * _phi(robot_speed, 5.0)
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.05)
    keep_object = _loss_penalty(grasped | placed, -0.05)
    return _pack(
        (
            _phi(tcp_obj),
            0.7 * grasped.float() + 0.3 * _phi(tcp_obj),
            0.85 * _phi(goal, 5.0) + 0.15 * grasped.float(),
            0.85 * _phi(goal, 12.0) + 0.15 * placed_and_static,
        ),
        (tcp_obj <= 0.06, grasped, near_goal, _success(env, info)),
        maintenance=(zeros, zeros, keep_grasp, keep_object),
        safety=_safety(env.obj),
    )



def _pick_single_ycb_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Widen the placement funnel and reward holding useful goal progress."""
    tcp_obj = _distance(_tcp(env), env.obj.pose.p)
    grasped = info.get("is_grasped", _grasp(env, env.obj)).bool()
    goal = _distance(env.obj.pose.p, env.goal_site.pose.p)
    placed = info.get("is_obj_placed", goal <= env.goal_thresh).bool()
    near_goal = grasped & (goal <= 4.0 * env.goal_thresh)
    robot_speed = torch.linalg.norm(env.agent.robot.get_qvel()[..., :-2], dim=1)
    placed_and_static = placed.float() * _phi(robot_speed, 5.0)
    goal_phi = _phi(goal, 5.0)
    tight_goal_phi = _phi(goal, 20.0)
    carrying_or_placed = grasped | placed
    zeros = torch.zeros_like(tcp_obj)
    keep_grasp = _loss_penalty(grasped, -0.04)
    keep_object = _loss_penalty(carrying_or_placed, -0.04)
    return _pack(
        (
            _phi(tcp_obj),
            0.75 * grasped.float() + 0.25 * _phi(tcp_obj),
            0.78 * goal_phi + 0.22 * grasped.float(),
            0.70 * tight_goal_phi + 0.20 * placed.float() + 0.10 * placed_and_static,
        ),
        (tcp_obj <= 0.06, grasped, near_goal, _success(env, info)),
        maintenance=(
            zeros,
            0.02 * grasped.float(),
            keep_grasp + 0.08 * goal_phi * grasped.float(),
            keep_object + 0.08 * tight_goal_phi * carrying_or_placed.float()
            + 0.03 * placed_and_static,
        ),
        safety=_safety(env.obj),
    )

def _place_sphere(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp_obj = _distance(_tcp(env), env.obj.pose.p)
    grasped = info.get("is_obj_grasped", _grasp(env, env.obj)).bool()
    target = env.bin.pose.p.clone()
    target[:, 2] += env.block_half_size[0] + env.radius
    goal = _distance(env.obj.pose.p, target)
    on_bin = info.get("is_obj_on_bin", goal <= 0.01).bool()
    zeros = torch.zeros_like(tcp_obj)
    keep = _loss_penalty(grasped)
    return _pack(
        (_phi(tcp_obj), grasped.float(), _phi(goal), 0.5 * on_bin.float() + 0.5 * (~grasped).float()),
        (tcp_obj <= 0.05, grasped, grasped & (goal <= 0.04), _success(env, info)),
        maintenance=(zeros, zeros, keep, zeros),
        safety=_safety(env.obj),
    )


def _turn_faucet(env: Any, info: dict[str, Any]) -> RuleInputs:
    handle_pose = env.target_switch_link.pose * env.target_switch_link.cmass_local_pose
    tcp_handle = _distance(_tcp(env), handle_pose.p)
    contact_like = tcp_handle <= 0.04
    angle_dist = (env.target_angle - env.current_angle).reshape(env.num_envs, -1)[:, 0]
    total = env.target_angle_diff.reshape(env.num_envs, -1)[:, 0].abs().clamp_min(1e-6)
    turn_progress = 1.0 - torch.clamp(angle_dist / total, 0.0, 1.0)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_handle)
    return _pack(
        (_phi(tcp_handle), _phi(tcp_handle, 15.0), turn_progress, 0.5 * success.float() + 0.5 * _static(env).float()),
        (tcp_handle <= 0.07, contact_like, success, success & _static(env)),
        maintenance=(zeros, zeros, _loss_penalty(contact_like), zeros),
    )


def _fmb(env: Any, info: dict[str, Any]) -> RuleInputs:
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp_bridge = _distance(_tcp(env), grasp_pose.p)
    grasped = _grasp(env, env.bridge)
    fixture_error = _distance(env.bridge.pose.p, env.reorienting_fixture.pose.p)
    goal_pos = _distance(env.bridge.pose.p, env.goal_bridge_pose.p)
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    reoriented = (~grasped) & (fixture_error <= 0.10) & (goal_angle <= 0.50)
    aligned = grasped & (goal_pos <= 0.04) & (goal_angle <= 0.35)
    zeros = torch.zeros_like(tcp_bridge)
    keep = _loss_penalty(grasped)
    return _pack(
        (
            _phi(tcp_bridge),
            grasped.float(),
            0.5 * _phi(fixture_error) + 0.5 * (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)),
            grasped.float(),
            0.5 * _phi(goal_pos, 10.0) + 0.5 * (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)),
            _phi(goal_pos, 30.0),
        ),
        (
            tcp_bridge <= 0.05,
            grasped,
            reoriented,
            grasped,
            aligned,
            _success(env, info)
            & (~grasped)
            & env.bridge.is_static(lin_thresh=1e-2, ang_thresh=0.5),
        ),
        maintenance=(zeros, zeros, zeros, zeros, keep, zeros),
        safety=_safety(env.bridge),
    )


def _fmb_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v2: position-first bridge placement with dense grasp credit."""
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp_bridge = _distance(_tcp(env), grasp_pose.p)
    grasped = _grasp(env, env.bridge)
    fixture_error = _distance(env.bridge.pose.p, env.reorienting_fixture.pose.p)
    goal_pos = _distance(env.bridge.pose.p, env.goal_bridge_pose.p)
    goal_xy = _distance(env.bridge.pose.p, env.goal_bridge_pose.p, xy=True)
    goal_z = torch.abs(env.bridge.pose.p[:, 2] - env.goal_bridge_pose.p[:, 2])
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    lift = torch.clamp((env.bridge.pose.p[:, 2] - 0.024) / 0.08, 0.0, 1.0)

    reach_phi = _phi(tcp_bridge, 6.0)
    fixture_phi = _phi(fixture_error, 6.0)
    goal_phi = _phi(goal_pos, 10.0)
    tight_goal_phi = _phi(goal_pos, 70.0)
    xy_phi = _phi(goal_xy, 12.0)
    z_phi = _phi(goal_z, 35.0)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    place_phi = (0.45 * goal_phi + 0.25 * xy_phi + 0.20 * z_phi + 0.10 * angle_phi).clamp(0.0, 1.0)

    near_fixture = grasped & ((fixture_error <= 0.12) | (goal_pos <= 0.16))
    aligned = grasped & (goal_pos <= 0.030)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_bridge)
    keep = _loss_penalty(grasped, -0.02)
    return _pack(
        (
            reach_phi,
            (0.45 * reach_phi + 0.45 * grasped.float() + 0.10 * lift).clamp(0.0, 1.0),
            (0.45 * fixture_phi + 0.25 * goal_phi + 0.20 * grasped.float() + 0.10 * lift).clamp(0.0, 1.0),
            (0.50 * grasped.float() + 0.35 * goal_phi + 0.15 * reach_phi).clamp(0.0, 1.0),
            (0.70 * place_phi + 0.20 * grasped.float() + 0.10 * tight_goal_phi).clamp(0.0, 1.0),
            (0.65 * tight_goal_phi + 0.20 * place_phi + 0.15 * success.float()).clamp(0.0, 1.0),
        ),
        (
            tcp_bridge <= 0.06,
            grasped,
            near_fixture,
            grasped,
            aligned,
            success,
        ),
        maintenance=(
            0.04 * reach_phi,
            0.06 * reach_phi + 0.10 * grasped.float(),
            keep + 0.08 * fixture_phi + 0.05 * goal_phi,
            keep + 0.10 * goal_phi,
            keep + 0.16 * place_phi,
            0.20 * tight_goal_phi + 0.08 * success.float(),
        ),
        safety=zeros,
    )



def _fmb_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v3: stronger bridge grasp before long-horizon placement."""
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp_bridge = _distance(_tcp(env), grasp_pose.p)
    grasped = _grasp(env, env.bridge)
    fixture_error = _distance(env.bridge.pose.p, env.reorienting_fixture.pose.p)
    goal_pos = _distance(env.bridge.pose.p, env.goal_bridge_pose.p)
    goal_xy = _distance(env.bridge.pose.p, env.goal_bridge_pose.p, xy=True)
    goal_z = torch.abs(env.bridge.pose.p[:, 2] - env.goal_bridge_pose.p[:, 2])
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    lift = torch.clamp((env.bridge.pose.p[:, 2] - 0.024) / 0.08, 0.0, 1.0)

    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    reach_phi = _phi(tcp_bridge, 8.0)
    near_grasp_phi = _phi(tcp_bridge, 20.0)
    close_near = (near_grasp_phi * closed).clamp(0.0, 1.0)

    fixture_phi = _phi(fixture_error, 6.0)
    goal_phi = _phi(goal_pos, 10.0)
    tight_goal_phi = _phi(goal_pos, 120.0)
    xy_phi = _phi(goal_xy, 14.0)
    z_phi = _phi(goal_z, 40.0)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    place_phi = (0.42 * goal_phi + 0.25 * xy_phi + 0.23 * z_phi + 0.10 * angle_phi).clamp(0.0, 1.0)

    near_fixture = grasped & ((fixture_error <= 0.12) | (goal_pos <= 0.16))
    aligned = grasped & (goal_pos <= 0.025)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_bridge)
    keep = _loss_penalty(grasped, -0.02)
    return _pack(
        (
            (0.72 * reach_phi + 0.18 * close_near + 0.10 * grasped.float()).clamp(0.0, 1.0),
            (0.32 * near_grasp_phi + 0.32 * close_near + 0.28 * grasped.float() + 0.08 * lift).clamp(0.0, 1.0),
            (0.44 * fixture_phi + 0.26 * goal_phi + 0.20 * grasped.float() + 0.10 * lift).clamp(0.0, 1.0),
            (0.50 * grasped.float() + 0.34 * goal_phi + 0.16 * reach_phi).clamp(0.0, 1.0),
            (0.70 * place_phi + 0.20 * grasped.float() + 0.10 * tight_goal_phi).clamp(0.0, 1.0),
            (0.72 * tight_goal_phi + 0.18 * place_phi + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            (tcp_bridge <= 0.060) | grasped,
            grasped,
            near_fixture,
            grasped,
            aligned,
            success,
        ),
        maintenance=(
            0.06 * reach_phi + 0.04 * close_near,
            0.10 * near_grasp_phi + 0.16 * close_near + 0.18 * grasped.float(),
            keep + 0.08 * fixture_phi + 0.05 * goal_phi,
            keep + 0.10 * goal_phi,
            keep + 0.18 * place_phi,
            0.24 * tight_goal_phi + 0.08 * success.float(),
        ),
        safety=zeros,
    )


def _fmb_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v4: include TCP orientation for bridge grasping."""
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp = env.agent.tcp.pose
    tcp_bridge = _distance(tcp.p, grasp_pose.p)
    tcp_orient_error = _quat_error(tcp.q, grasp_pose.q)
    grasped = _grasp(env, env.bridge)
    fixture_error = _distance(env.bridge.pose.p, env.reorienting_fixture.pose.p)
    goal_pos = _distance(env.bridge.pose.p, env.goal_bridge_pose.p)
    goal_xy = _distance(env.bridge.pose.p, env.goal_bridge_pose.p, xy=True)
    goal_z = torch.abs(env.bridge.pose.p[:, 2] - env.goal_bridge_pose.p[:, 2])
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    lift = torch.clamp((env.bridge.pose.p[:, 2] - 0.024) / 0.08, 0.0, 1.0)

    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    reach_phi = _phi(tcp_bridge, 8.0)
    near_grasp_phi = _phi(tcp_bridge, 22.0)
    orient_phi = (1.0 - torch.clamp(tcp_orient_error / 1.20, 0.0, 1.0)).clamp(0.0, 1.0)
    close_near = (near_grasp_phi * closed * (0.35 + 0.65 * orient_phi)).clamp(0.0, 1.0)

    fixture_phi = _phi(fixture_error, 6.0)
    goal_phi = _phi(goal_pos, 10.0)
    tight_goal_phi = _phi(goal_pos, 120.0)
    xy_phi = _phi(goal_xy, 14.0)
    z_phi = _phi(goal_z, 40.0)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    place_phi = (0.42 * goal_phi + 0.25 * xy_phi + 0.23 * z_phi + 0.10 * angle_phi).clamp(0.0, 1.0)

    near_fixture = grasped & ((fixture_error <= 0.12) | (goal_pos <= 0.16))
    aligned = grasped & (goal_pos <= 0.025)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_bridge)
    keep = _loss_penalty(grasped, -0.02)
    return _pack(
        (
            (0.56 * reach_phi + 0.30 * orient_phi + 0.14 * grasped.float()).clamp(0.0, 1.0),
            (0.26 * near_grasp_phi + 0.22 * orient_phi + 0.24 * close_near + 0.24 * grasped.float() + 0.04 * lift).clamp(0.0, 1.0),
            (0.44 * fixture_phi + 0.26 * goal_phi + 0.20 * grasped.float() + 0.10 * lift).clamp(0.0, 1.0),
            (0.50 * grasped.float() + 0.34 * goal_phi + 0.16 * reach_phi).clamp(0.0, 1.0),
            (0.70 * place_phi + 0.20 * grasped.float() + 0.10 * tight_goal_phi).clamp(0.0, 1.0),
            (0.72 * tight_goal_phi + 0.18 * place_phi + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            ((tcp_bridge <= 0.065) & (tcp_orient_error <= 1.30)) | grasped,
            grasped,
            near_fixture,
            grasped,
            aligned,
            success,
        ),
        maintenance=(
            0.05 * reach_phi + 0.06 * orient_phi,
            0.10 * near_grasp_phi + 0.10 * orient_phi + 0.16 * close_near + 0.18 * grasped.float(),
            keep + 0.08 * fixture_phi + 0.05 * goal_phi,
            keep + 0.10 * goal_phi,
            keep + 0.18 * place_phi,
            0.24 * tight_goal_phi + 0.08 * success.float(),
        ),
        safety=zeros,
    )


def _fmb_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v5: v4 grasp shaping with position-only reach gate."""
    base = _fmb_v4(env, info)
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp_bridge = _distance(_tcp(env), grasp_pose.p)
    grasped = _grasp(env, env.bridge)
    reach_phi = _phi(tcp_bridge, 8.0)
    near_grasp_phi = _phi(tcp_bridge, 22.0)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 0] = (0.82 * reach_phi + 0.18 * grasped.float()).clamp(0.0, 1.0)
    gates[:, 0] = (tcp_bridge <= 0.060) | grasped
    maintenance[:, 0] = 0.10 * reach_phi + 0.04 * near_grasp_phi
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _fmb_v6(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v6: use bridge center-or-offset distance for grasp."""
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp = _tcp(env)
    tcp_grasp = _distance(tcp, grasp_pose.p)
    tcp_center = _distance(tcp, env.bridge.pose.p)
    tcp_bridge = torch.minimum(tcp_grasp, tcp_center)
    grasped = _grasp(env, env.bridge)
    fixture_error = _distance(env.bridge.pose.p, env.reorienting_fixture.pose.p)
    goal_pos = _distance(env.bridge.pose.p, env.goal_bridge_pose.p)
    goal_xy = _distance(env.bridge.pose.p, env.goal_bridge_pose.p, xy=True)
    goal_z = torch.abs(env.bridge.pose.p[:, 2] - env.goal_bridge_pose.p[:, 2])
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    lift = torch.clamp((env.bridge.pose.p[:, 2] - 0.024) / 0.08, 0.0, 1.0)

    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    reach_phi = _phi(tcp_bridge, 8.0)
    near_grasp_phi = _phi(tcp_bridge, 22.0)
    close_near = (near_grasp_phi * closed).clamp(0.0, 1.0)

    fixture_phi = _phi(fixture_error, 6.0)
    goal_phi = _phi(goal_pos, 10.0)
    tight_goal_phi = _phi(goal_pos, 120.0)
    xy_phi = _phi(goal_xy, 14.0)
    z_phi = _phi(goal_z, 40.0)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    place_phi = (0.42 * goal_phi + 0.25 * xy_phi + 0.23 * z_phi + 0.10 * angle_phi).clamp(0.0, 1.0)

    near_fixture = grasped & ((fixture_error <= 0.12) | (goal_pos <= 0.16))
    aligned = grasped & (goal_pos <= 0.025)
    success = _success(env, info)
    zeros = torch.zeros_like(tcp_bridge)
    keep = _loss_penalty(grasped, -0.02)
    return _pack(
        (
            (0.78 * reach_phi + 0.12 * close_near + 0.10 * grasped.float()).clamp(0.0, 1.0),
            (0.32 * near_grasp_phi + 0.30 * close_near + 0.30 * grasped.float() + 0.08 * lift).clamp(0.0, 1.0),
            (0.44 * fixture_phi + 0.26 * goal_phi + 0.20 * grasped.float() + 0.10 * lift).clamp(0.0, 1.0),
            (0.50 * grasped.float() + 0.34 * goal_phi + 0.16 * reach_phi).clamp(0.0, 1.0),
            (0.70 * place_phi + 0.20 * grasped.float() + 0.10 * tight_goal_phi).clamp(0.0, 1.0),
            (0.72 * tight_goal_phi + 0.18 * place_phi + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            (tcp_bridge <= 0.060) | grasped,
            grasped,
            near_fixture,
            grasped,
            aligned,
            success,
        ),
        maintenance=(
            0.10 * reach_phi + 0.03 * close_near,
            0.12 * near_grasp_phi + 0.18 * close_near + 0.20 * grasped.float(),
            keep + 0.08 * fixture_phi + 0.05 * goal_phi,
            keep + 0.10 * goal_phi,
            keep + 0.18 * place_phi,
            0.24 * tight_goal_phi + 0.08 * success.float(),
        ),
        safety=zeros,
    )


def _fmb_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v7: contact/transport funnel for bridge placement.

    FMB success only checks the bridge center within 5 mm of the goal pose.
    Earlier variants waited for a strict grasp bit before giving meaningful
    transport reward; empirically that leaves the policy parked in stage 1.
    This version keeps a simple visualizable recipe: reach/contact bridge,
    start moving/lifting it, coarse place, fine place, settle at success.
    """
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp = _tcp(env)
    tcp_grasp = _distance(tcp, grasp_pose.p)
    tcp_center = _distance(tcp, env.bridge.pose.p)
    tcp_bridge = torch.minimum(tcp_grasp, tcp_center)
    grasped = _grasp(env, env.bridge)

    bridge_pos = env.bridge.pose.p
    goal_pos_vec = env.goal_bridge_pose.p
    goal_dist = _distance(bridge_pos, goal_pos_vec)
    goal_xy = _distance(bridge_pos, goal_pos_vec, xy=True)
    goal_z = torch.abs(bridge_pos[:, 2] - goal_pos_vec[:, 2])
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    speed = torch.linalg.norm(env.bridge.linear_velocity, dim=1)

    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    contact_like = tcp_bridge <= 0.075
    close_contact = contact_like & (closed >= 0.35)
    lift = torch.clamp((bridge_pos[:, 2] - 0.024) / 0.070, 0.0, 1.0)

    reach_phi = _phi(tcp_bridge, 7.0)
    contact_phi = _phi(tcp_bridge, 18.0)
    long_goal_phi = (1.0 - goal_dist / 0.42).clamp(0.0, 1.0)
    mid_goal_phi = (1.0 - goal_dist / 0.22).clamp(0.0, 1.0)
    goal_phi = _phi(goal_dist, 8.0)
    tight_goal_phi = _phi(goal_dist, 120.0)
    xy_phi = _phi(goal_xy, 10.0)
    z_phi = _phi(goal_z, 45.0)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)

    carrying = grasped | close_contact | ((tcp_bridge <= 0.10) & (lift >= 0.15))
    transport_started = carrying & ((goal_dist <= 0.34) | (lift >= 0.15))
    coarse = goal_dist <= 0.16
    near = goal_dist <= 0.055
    fine = goal_dist <= 0.025
    success = _success(env, info)
    static = env.bridge.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    place_phi = (0.48 * goal_phi + 0.24 * xy_phi + 0.20 * z_phi + 0.08 * angle_phi).clamp(0.0, 1.0)
    fine_place_phi = (0.60 * tight_goal_phi + 0.20 * xy_phi + 0.15 * z_phi + 0.05 * success.float()).clamp(0.0, 1.0)
    zeros = torch.zeros_like(tcp_bridge)

    return _pack(
        (
            reach_phi,
            (0.36 * contact_phi + 0.22 * closed + 0.22 * grasped.float() + 0.20 * lift).clamp(0.0, 1.0),
            (0.48 * long_goal_phi + 0.20 * mid_goal_phi + 0.18 * carrying.float() + 0.14 * lift).clamp(0.0, 1.0),
            (0.55 * place_phi + 0.25 * mid_goal_phi + 0.20 * carrying.float()).clamp(0.0, 1.0),
            (0.70 * fine_place_phi + 0.20 * fine.float() + 0.10 * carrying.float()).clamp(0.0, 1.0),
            (0.65 * success.float() + 0.20 * fine.float() + 0.15 * static.float()).clamp(0.0, 1.0),
        ),
        (
            contact_like | grasped,
            carrying,
            transport_started,
            coarse,
            near | fine,
            success,
        ),
        maintenance=(
            0.08 * reach_phi,
            0.12 * contact_phi + 0.10 * closed + 0.12 * grasped.float(),
            0.12 * long_goal_phi + 0.08 * mid_goal_phi + 0.08 * carrying.float(),
            0.16 * place_phi + 0.06 * carrying.float(),
            0.22 * fine_place_phi + 0.05 * fine.float(),
            0.12 * success.float(),
        ),
        safety=zeros,
    )


def _fmb_v8(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v8: open transport immediately after bridge contact."""
    base = _fmb_v7(env, info)
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp = _tcp(env)
    tcp_bridge = torch.minimum(_distance(tcp, grasp_pose.p), _distance(tcp, env.bridge.pose.p))
    grasped = _grasp(env, env.bridge)
    bridge_pos = env.bridge.pose.p
    goal_pos_vec = env.goal_bridge_pose.p
    goal_dist = _distance(bridge_pos, goal_pos_vec)
    goal_xy = _distance(bridge_pos, goal_pos_vec, xy=True)
    goal_z = torch.abs(bridge_pos[:, 2] - goal_pos_vec[:, 2])
    speed = torch.linalg.norm(env.bridge.linear_velocity, dim=1)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    contact_like = tcp_bridge <= 0.085
    close_contact = contact_like & (closed >= 0.25)
    lift = torch.clamp((bridge_pos[:, 2] - 0.024) / 0.070, 0.0, 1.0)
    carrying = grasped | close_contact | ((tcp_bridge <= 0.11) & (lift >= 0.10))
    long_goal_phi = (1.0 - goal_dist / 0.42).clamp(0.0, 1.0)
    mid_goal_phi = (1.0 - goal_dist / 0.24).clamp(0.0, 1.0)
    goal_phi = _phi(goal_dist, 8.0)
    tight_goal_phi = _phi(goal_dist, 120.0)
    xy_phi = _phi(goal_xy, 10.0)
    z_phi = _phi(goal_z, 45.0)
    contact_phi = _phi(tcp_bridge, 18.0)
    placed_loose = goal_dist <= 0.16
    placed_near = goal_dist <= 0.055
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 1] = (0.36 * contact_phi + 0.24 * closed + 0.20 * lift + 0.20 * grasped.float()).clamp(0.0, 1.0)
    potentials[:, 2] = (0.44 * long_goal_phi + 0.20 * mid_goal_phi + 0.18 * lift + 0.12 * contact_phi + 0.06 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.46 * goal_phi + 0.22 * xy_phi + 0.20 * z_phi + 0.12 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 4] = (0.62 * tight_goal_phi + 0.20 * z_phi + 0.10 * placed_near.float() + 0.08 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 5] = (0.70 * success.float() + 0.18 * placed_near.float() + 0.12 * _phi(speed, 8.0)).clamp(0.0, 1.0)
    gates[:, 1] = contact_like | grasped
    gates[:, 2] = (goal_dist <= 0.30) | (lift >= 0.10) | carrying
    gates[:, 3] = placed_loose
    gates[:, 4] = placed_near
    gates[:, 5] = success
    maintenance[:, 1] = 0.12 * contact_phi + 0.10 * closed + 0.10 * lift + 0.08 * grasped.float()
    maintenance[:, 2] = 0.18 * long_goal_phi + 0.10 * mid_goal_phi + 0.10 * lift + 0.06 * contact_like.float()
    maintenance[:, 3] = 0.18 * goal_phi + 0.10 * xy_phi + 0.10 * z_phi + 0.06 * carrying.float()
    maintenance[:, 4] = 0.24 * tight_goal_phi + 0.08 * z_phi + 0.06 * placed_near.float()
    maintenance[:, 5] = 0.12 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _fmb_v9(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v9: force one-step contact stage into transport shaping."""
    base = _fmb_v8(env, info)
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp = _tcp(env)
    tcp_bridge = torch.minimum(_distance(tcp, grasp_pose.p), _distance(tcp, env.bridge.pose.p))
    grasped = _grasp(env, env.bridge)
    bridge_pos = env.bridge.pose.p
    goal_dist = _distance(bridge_pos, env.goal_bridge_pose.p)
    lift = torch.clamp((bridge_pos[:, 2] - 0.024) / 0.070, 0.0, 1.0)
    contact_like = tcp_bridge <= 0.10
    long_goal_phi = (1.0 - goal_dist / 0.42).clamp(0.0, 1.0)
    mid_goal_phi = (1.0 - goal_dist / 0.24).clamp(0.0, 1.0)
    success = _success(env, info)
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials = base.potentials.clone()
    gates[:, 1] = torch.ones_like(gates[:, 1], dtype=torch.bool)
    gates[:, 2] = (goal_dist <= 0.30) | (lift >= 0.08) | grasped | contact_like
    gates[:, 3] = goal_dist <= 0.16
    gates[:, 4] = goal_dist <= 0.055
    gates[:, 5] = success
    potentials[:, 2] = (0.50 * long_goal_phi + 0.22 * mid_goal_phi + 0.18 * lift + 0.10 * contact_like.float()).clamp(0.0, 1.0)
    maintenance[:, 2] = 0.24 * long_goal_phi + 0.12 * mid_goal_phi + 0.12 * lift + 0.06 * contact_like.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _fmb_v10(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v10: transport shaping with directed bridge velocity."""
    base = _fmb_v9(env, info)
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp = _tcp(env)
    tcp_bridge = torch.minimum(_distance(tcp, grasp_pose.p), _distance(tcp, env.bridge.pose.p))
    bridge_pos = env.bridge.pose.p
    goal_vec = env.goal_bridge_pose.p - bridge_pos
    goal_dist = torch.linalg.norm(goal_vec, dim=1)
    goal_dir = goal_vec / goal_dist[:, None].clamp_min(1e-6)
    directed_speed = (env.bridge.linear_velocity * goal_dir).sum(dim=1)
    positive_motion = torch.clamp(torch.tanh(8.0 * directed_speed), 0.0, 1.0)
    lift = torch.clamp((bridge_pos[:, 2] - 0.024) / 0.070, 0.0, 1.0)
    contact_like = tcp_bridge <= 0.11
    long_goal_phi = (1.0 - goal_dist / 0.42).clamp(0.0, 1.0)
    mid_goal_phi = (1.0 - goal_dist / 0.24).clamp(0.0, 1.0)
    near_goal_phi = (1.0 - goal_dist / 0.12).clamp(0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 2] = (0.42 * long_goal_phi + 0.22 * mid_goal_phi + 0.16 * lift + 0.12 * positive_motion + 0.08 * contact_like.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.42 * mid_goal_phi + 0.24 * near_goal_phi + 0.16 * lift + 0.12 * positive_motion + 0.06 * contact_like.float()).clamp(0.0, 1.0)
    gates[:, 2] = (goal_dist <= 0.30) | (lift >= 0.08) | (positive_motion >= 0.55) | contact_like
    gates[:, 3] = (goal_dist <= 0.18) | ((goal_dist <= 0.22) & (lift >= 0.20))
    gates[:, 4] = goal_dist <= 0.060
    gates[:, 5] = success
    maintenance[:, 2] = 0.20 * long_goal_phi + 0.12 * mid_goal_phi + 0.10 * lift + 0.12 * positive_motion + 0.04 * contact_like.float()
    maintenance[:, 3] = 0.22 * mid_goal_phi + 0.14 * near_goal_phi + 0.10 * lift + 0.10 * positive_motion
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _fmb_v11(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v11: relax transport gates and add pose alignment."""
    base = _fmb_v10(env, info)
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp = _tcp(env)
    tcp_bridge = torch.minimum(_distance(tcp, grasp_pose.p), _distance(tcp, env.bridge.pose.p))
    grasped = _grasp(env, env.bridge)
    bridge_pos = env.bridge.pose.p
    goal_vec = env.goal_bridge_pose.p - bridge_pos
    goal_dist = torch.linalg.norm(goal_vec, dim=1)
    goal_xy = _distance(bridge_pos, env.goal_bridge_pose.p, xy=True)
    goal_z = torch.abs(bridge_pos[:, 2] - env.goal_bridge_pose.p[:, 2])
    goal_dir = goal_vec / goal_dist[:, None].clamp_min(1e-6)
    directed_speed = (env.bridge.linear_velocity * goal_dir).sum(dim=1)
    positive_motion = torch.clamp(torch.tanh(8.0 * directed_speed), 0.0, 1.0)
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(goal_angle / 0.75, 0.0, 1.0)).clamp(0.0, 1.0)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    contact_like = tcp_bridge <= 0.11
    lift = torch.clamp((bridge_pos[:, 2] - 0.024) / 0.070, 0.0, 1.0)
    carrying = grasped | (contact_like & (closed >= 0.20)) | ((tcp_bridge <= 0.13) & (lift >= 0.08))
    long_goal_phi = (1.0 - goal_dist / 0.45).clamp(0.0, 1.0)
    mid_goal_phi = (1.0 - goal_dist / 0.26).clamp(0.0, 1.0)
    near_goal_phi = (1.0 - goal_dist / 0.11).clamp(0.0, 1.0)
    tight_goal_phi = _phi(goal_dist, 90.0)
    xy_phi = _phi(goal_xy, 10.0)
    z_phi = _phi(goal_z, 45.0)
    pose_phi = (0.40 * mid_goal_phi + 0.20 * xy_phi + 0.18 * z_phi + 0.14 * angle_phi + 0.08 * carrying.float()).clamp(0.0, 1.0)
    fine_pose = (0.44 * tight_goal_phi + 0.22 * near_goal_phi + 0.18 * z_phi + 0.16 * tight_angle_phi).clamp(0.0, 1.0)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 1] = contact_like | grasped
    gates[:, 2] = carrying | (lift >= 0.08)
    gates[:, 3] = (goal_dist <= 0.28) | (positive_motion >= 0.45) | (pose_phi >= 0.38)
    gates[:, 4] = (goal_dist <= 0.12) | (fine_pose >= 0.45)
    gates[:, 5] = success
    potentials[:, 1] = (0.34 * _phi(tcp_bridge, 16.0) + 0.22 * closed + 0.22 * lift + 0.22 * grasped.float()).clamp(0.0, 1.0)
    potentials[:, 2] = (0.42 * long_goal_phi + 0.20 * mid_goal_phi + 0.16 * lift + 0.14 * positive_motion + 0.08 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 3] = pose_phi
    potentials[:, 4] = fine_pose
    potentials[:, 5] = (0.62 * fine_pose + 0.20 * success.float() + 0.18 * _phi(torch.linalg.norm(env.bridge.linear_velocity, dim=1), 8.0)).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.10 * _phi(tcp_bridge, 16.0) + 0.08 * closed + 0.08 * lift + 0.08 * grasped.float()
    maintenance[:, 2] = 0.18 * long_goal_phi + 0.10 * mid_goal_phi + 0.10 * lift + 0.08 * positive_motion
    maintenance[:, 3] = 0.18 * pose_phi + 0.08 * positive_motion + 0.06 * carrying.float()
    maintenance[:, 4] = 0.24 * fine_pose + 0.08 * near_goal_phi
    maintenance[:, 5] = 0.18 * fine_pose + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _fmb_v12(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v12: official-world direct bridge-to-goal funnel."""
    base = _fmb_v11(env, info)
    tcp = _tcp(env)
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp_grasp = _distance(tcp, grasp_pose.p)
    tcp_center = _distance(tcp, env.bridge.pose.p)
    tcp_bridge = torch.minimum(tcp_grasp, tcp_center)
    bridge_pos = env.bridge.pose.p
    goal_pos = env.goal_bridge_pose.p
    goal_dist = _distance(bridge_pos, goal_pos)
    goal_xy = _distance(bridge_pos, goal_pos, xy=True)
    goal_z = torch.abs(bridge_pos[:, 2] - goal_pos[:, 2])
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    grasped = _grasp(env, env.bridge)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    lift = torch.clamp((bridge_pos[:, 2] - 0.024) / 0.070, 0.0, 1.0)
    contact_like = tcp_bridge <= 0.12
    carrying = grasped | (contact_like & (closed >= 0.16)) | ((tcp_bridge <= 0.14) & (lift >= 0.06))
    goal_vec = goal_pos - bridge_pos
    goal_dir = goal_vec / torch.linalg.norm(goal_vec, dim=1, keepdim=True).clamp_min(1e-6)
    directed_speed = (env.bridge.linear_velocity * goal_dir).sum(dim=1)
    positive_motion = torch.clamp(torch.tanh(8.0 * directed_speed), 0.0, 1.0)
    speed = torch.linalg.norm(env.bridge.linear_velocity, dim=1)

    reach_phi = _phi(tcp_bridge, 12.0)
    long_goal_phi = (1.0 - goal_dist / 0.46).clamp(0.0, 1.0)
    mid_goal_phi = (1.0 - goal_dist / 0.22).clamp(0.0, 1.0)
    near_goal_phi = (1.0 - goal_dist / 0.075).clamp(0.0, 1.0)
    tight_goal_phi = _phi(goal_dist, 140.0)
    xy_phi = _phi(goal_xy, 14.0)
    z_phi = _phi(goal_z, 60.0)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    pose_phi = (0.40 * mid_goal_phi + 0.22 * xy_phi + 0.20 * z_phi + 0.10 * angle_phi + 0.08 * carrying.float()).clamp(0.0, 1.0)
    fine_pose = (0.48 * tight_goal_phi + 0.22 * near_goal_phi + 0.20 * z_phi + 0.10 * xy_phi).clamp(0.0, 1.0)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = contact_like | (reach_phi >= 0.35)
    gates[:, 1] = contact_like | carrying | (closed >= 0.20)
    gates[:, 2] = carrying | (lift >= 0.05) | (positive_motion >= 0.25)
    gates[:, 3] = (goal_dist <= 0.24) | (pose_phi >= 0.34) | (positive_motion >= 0.40)
    gates[:, 4] = (goal_dist <= 0.075) | (fine_pose >= 0.45)
    gates[:, 5] = success
    potentials[:, 0] = reach_phi
    potentials[:, 1] = (0.38 * reach_phi + 0.24 * closed + 0.20 * grasped.float() + 0.18 * lift).clamp(0.0, 1.0)
    potentials[:, 2] = (0.44 * long_goal_phi + 0.20 * mid_goal_phi + 0.16 * lift + 0.12 * positive_motion + 0.08 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 3] = pose_phi
    potentials[:, 4] = fine_pose
    potentials[:, 5] = (0.62 * fine_pose + 0.24 * success.float() + 0.14 * _phi(speed, 8.0)).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.10 * reach_phi
    maintenance[:, 1] = 0.12 * reach_phi + 0.10 * closed + 0.08 * grasped.float() + 0.06 * lift
    maintenance[:, 2] = 0.24 * long_goal_phi + 0.12 * mid_goal_phi + 0.10 * lift + 0.10 * positive_motion
    maintenance[:, 3] = 0.24 * pose_phi + 0.08 * positive_motion + 0.06 * carrying.float()
    maintenance[:, 4] = 0.30 * fine_pose + 0.10 * near_goal_phi
    maintenance[:, 5] = 0.20 * fine_pose + 0.12 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)

def _rotate_valve_level4(env: Any, info: dict[str, Any]) -> RuleInputs:
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)

    signed_rotation = (
        (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    )
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(5.0 * directed_velocity), 0.0, 1.0)
    success = _success(env, info)
    settled = success & (env.valve.qvel[:, 0].abs() <= 0.25)
    zeros = torch.zeros_like(contact_error)
    return _pack(
        (
            _phi(contact_error, 10.0),
            0.5 * _phi(contact_error, 12.0) + 0.5 * positive_motion,
            progress,
            0.5 * success.float() + 0.5 * _phi(env.valve.qvel[:, 0].abs(), 5.0),
        ),
        (
            contact_error <= 0.06,
            (contact_error <= 0.08) & ((directed_velocity > 0.03) | (progress > 0.05)),
            success,
            settled,
        ),
        maintenance=(
            zeros,
            zeros,
            _loss_penalty(contact_error <= 0.10),
            zeros,
        ),
    )


def _rotate_valve_level4_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v2: signed full-turn shaping for random direction."""
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_phi = _phi(contact_error, 10.0)
    contact = contact_error <= 0.10

    signed_rotation = (
        (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    )
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(5.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-5.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    mid_progress = torch.clamp(signed_rotation / (0.50 * env.success_threshold), 0.0, 1.0)
    turn_score = (
        0.50 * progress
        + 0.25 * mid_progress
        + 0.20 * positive_motion
        + 0.05 * contact.float()
    ).clamp(0.0, 1.0)
    success = _success(env, info)
    settled = success & (env.valve.qvel[:, 0].abs() <= 0.25)
    zeros = torch.zeros_like(contact_error)

    return _pack(
        (
            contact_phi,
            (0.55 * contact_phi + 0.45 * positive_motion).clamp(0.0, 1.0),
            turn_score,
            (0.55 * success.float() + 0.25 * progress + 0.20 * _phi(env.valve.qvel[:, 0].abs(), 5.0)).clamp(0.0, 1.0),
        ),
        (
            contact_error <= 0.08,
            contact & ((directed_velocity > 0.02) | (progress > 0.02)),
            success,
            settled,
        ),
        maintenance=(
            zeros,
            0.04 * contact.float() + 0.04 * positive_motion - 0.04 * wrong_motion,
            0.10 * contact.float() + 0.16 * turn_score - 0.06 * wrong_motion,
            zeros,
        ),
    )


def _rotate_valve_level4_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v3: reward sustained signed rotation more strongly."""
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_phi = _phi(contact_error, 10.0)
    contact = contact_error <= 0.11

    signed_rotation = (
        (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    )
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(5.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-5.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    half_progress = torch.clamp(signed_rotation / (0.50 * env.success_threshold), 0.0, 1.0)
    turn_score = (
        0.45 * progress
        + 0.30 * half_progress
        + 0.20 * positive_motion
        + 0.05 * contact.float()
    ).clamp(0.0, 1.0)
    success = _success(env, info)
    near_full_turn = progress >= 0.85
    settled = success & (env.valve.qvel[:, 0].abs() <= 0.25)
    zeros = torch.zeros_like(contact_error)

    return _pack(
        (
            contact_phi,
            (0.55 * contact_phi + 0.45 * positive_motion).clamp(0.0, 1.0),
            turn_score,
            (0.60 * success.float() + 0.25 * progress + 0.15 * _phi(env.valve.qvel[:, 0].abs(), 5.0)).clamp(0.0, 1.0),
        ),
        (
            contact_error <= 0.09,
            contact & ((directed_velocity > 0.015) | (progress > 0.02)),
            near_full_turn | success,
            settled,
        ),
        maintenance=(
            zeros,
            0.05 * contact.float() + 0.08 * positive_motion - 0.04 * wrong_motion,
            0.14 * contact.float() + 0.26 * progress + 0.28 * positive_motion - 0.08 * wrong_motion,
            0.24 * progress + 0.10 * success.float(),
        ),
    )


def _rotate_valve_level4_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v4: native-dense-like signed rotation, no early stage-2 exit."""
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_phi = _phi(contact_error, 10.0)
    contact = contact_error <= 0.11

    signed_rotation = (
        (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    )
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(5.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-5.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    half_progress = torch.clamp(signed_rotation / (0.50 * env.success_threshold), 0.0, 1.0)
    turn_score = (
        0.40 * progress
        + 0.25 * half_progress
        + 0.25 * positive_motion
        + 0.10 * contact.float()
    ).clamp(0.0, 1.0)
    success = _success(env, info)
    settled = success & (env.valve.qvel[:, 0].abs() <= 0.30)
    zeros = torch.zeros_like(contact_error)

    return _pack(
        (
            contact_phi,
            (0.50 * contact_phi + 0.50 * positive_motion).clamp(0.0, 1.0),
            turn_score,
            (0.70 * success.float() + 0.30 * _phi(env.valve.qvel[:, 0].abs(), 5.0)).clamp(0.0, 1.0),
        ),
        (
            contact_error <= 0.09,
            contact & ((directed_velocity > 0.012) | (progress > 0.02)),
            success,
            settled,
        ),
        maintenance=(
            zeros,
            0.05 * contact.float() + 0.10 * positive_motion - 0.05 * wrong_motion,
            0.16 * contact.float() + 0.34 * progress + 0.32 * positive_motion - 0.10 * wrong_motion,
            0.10 * success.float(),
        ),
    )


def _rotate_valve_level4_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v5: stronger full-turn progress/velocity credit."""
    base = _rotate_valve_level4_v4(env, info)
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact = contact_error <= 0.11
    signed_rotation = (
        (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    )
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(5.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-5.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    half_progress = torch.clamp(signed_rotation / (0.50 * env.success_threshold), 0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 2] = (
        0.62 * progress + 0.18 * half_progress + 0.16 * positive_motion + 0.04 * contact.float()
    ).clamp(0.0, 1.0)
    potentials[:, 3] = (0.75 * success.float() + 0.25 * progress).clamp(0.0, 1.0)
    maintenance[:, 1] = maintenance[:, 1] + 0.06 * positive_motion
    maintenance[:, 2] = 0.12 * contact.float() + 0.58 * progress + 0.48 * positive_motion - 0.18 * wrong_motion
    maintenance[:, 3] = 0.12 * success.float()
    return RuleInputs(potentials, base.gates, maintenance, base.safety_penalty)



def _rotate_valve_level4_v6(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v6: stronger native-like sustained signed rotation."""
    base = _rotate_valve_level4_v5(env, info)
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_phi = _phi(contact_error, 12.0)
    contact = contact_error <= 0.11
    signed_rotation = (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(5.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-5.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    half_progress = torch.clamp(signed_rotation / (0.50 * env.success_threshold), 0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 1] = (0.54 * contact_phi + 0.36 * positive_motion + 0.10 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 2] = (0.64 * progress + 0.16 * half_progress + 0.16 * positive_motion + 0.04 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.70 * success.float() + 0.30 * progress).clamp(0.0, 1.0)
    gates[:, 1] = contact & (directed_velocity > 0.01)
    gates[:, 2] = success
    gates[:, 3] = success
    maintenance[:, 1] = 0.18 * contact_phi + 0.16 * positive_motion - 0.08 * wrong_motion
    maintenance[:, 2] = 0.18 * contact.float() + 0.85 * progress + 0.70 * positive_motion - 0.25 * wrong_motion
    maintenance[:, 3] = 0.16 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _rotate_valve_level4_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v7: less brittle contact entry with stronger turn progress."""
    base = _rotate_valve_level4_v6(env, info)
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_phi = _phi(contact_error, 11.0)
    contact = contact_error <= 0.13
    signed_rotation = (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(4.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-4.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    early_progress = torch.clamp(signed_rotation / (0.25 * env.success_threshold), 0.0, 1.0)
    mid_progress = torch.clamp(signed_rotation / (0.55 * env.success_threshold), 0.0, 1.0)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = contact | (contact_phi >= 0.45)
    gates[:, 1] = contact & ((directed_velocity > -0.01) | (early_progress >= 0.12))
    gates[:, 2] = (mid_progress >= 0.75) | success
    gates[:, 3] = success
    potentials[:, 0] = contact_phi
    potentials[:, 1] = (0.46 * contact_phi + 0.34 * positive_motion + 0.20 * early_progress).clamp(0.0, 1.0)
    potentials[:, 2] = (0.56 * progress + 0.24 * mid_progress + 0.14 * positive_motion + 0.06 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.64 * progress + 0.24 * success.float() + 0.12 * positive_motion).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.08 * contact_phi
    maintenance[:, 1] = 0.18 * contact_phi + 0.20 * positive_motion + 0.08 * early_progress - 0.08 * wrong_motion
    maintenance[:, 2] = 0.95 * progress + 0.55 * positive_motion + 0.20 * mid_progress - 0.22 * wrong_motion
    maintenance[:, 3] = 0.28 * progress + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _rotate_valve_level4_v8(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v8: keep full-turn pressure until official success."""
    base = _rotate_valve_level4_v7(env, info)
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_phi = _phi(contact_error, 11.0)
    contact = contact_error <= 0.14
    signed_rotation = (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(4.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-4.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    early_progress = torch.clamp(signed_rotation / (0.25 * env.success_threshold), 0.0, 1.0)
    mid_progress = torch.clamp(signed_rotation / (0.60 * env.success_threshold), 0.0, 1.0)
    near_full = progress >= 0.92
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = contact | (contact_phi >= 0.40)
    gates[:, 1] = contact | (contact_phi >= 0.48) | (early_progress >= 0.08)
    gates[:, 2] = near_full | success
    gates[:, 3] = success
    potentials[:, 0] = contact_phi
    potentials[:, 1] = (0.48 * contact_phi + 0.32 * positive_motion + 0.20 * early_progress).clamp(0.0, 1.0)
    potentials[:, 2] = (0.64 * progress + 0.18 * mid_progress + 0.14 * positive_motion + 0.04 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.70 * progress + 0.18 * positive_motion + 0.12 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.08 * contact_phi
    maintenance[:, 1] = 0.18 * contact_phi + 0.22 * positive_motion + 0.10 * early_progress - 0.08 * wrong_motion
    maintenance[:, 2] = 1.15 * progress + 0.65 * positive_motion + 0.25 * mid_progress - 0.28 * wrong_motion
    maintenance[:, 3] = 0.34 * progress + 0.16 * positive_motion + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)

def _two_robot_stack_cube(env: Any, info: dict[str, Any]) -> RuleInputs:
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p

    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[1] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)
    reach = 0.5 * _phi(left_reach, 5.0) + 0.5 * _phi(right_reach, 5.0)

    cube_a_grasped = info.get(
        "is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)
    ).bool()
    cube_b_grasped = info.get(
        "is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)
    ).bool()
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get(
        "cubeB_placed", cube_b_goal < env.goal_radius
    ).bool()
    top_goal = cube_b.clone()
    top_goal[:, 2] += env.cube_half_size[2] * 2
    cube_a_top_error = _distance(cube_a, top_goal)
    cube_a_on_cube_b = info.get(
        "is_cubeA_on_cubeB", cube_a_top_error <= 0.01
    ).bool()
    success = _success(env, info)
    zeros = torch.zeros_like(left_reach)
    keep_top = _loss_penalty(cube_a_grasped | cube_a_on_cube_b)
    return _pack(
        (
            reach,
            cube_a_grasped.float(),
            0.5 * _phi(cube_b_goal, 5.0) + 0.5 * cube_a_grasped.float(),
            _phi(cube_a_top_error, 10.0),
            success.float(),
        ),
        (
            (left_reach <= 0.06) & (right_reach <= 0.08),
            cube_a_grasped,
            cube_b_placed & cube_a_grasped,
            cube_a_on_cube_b & cube_b_placed,
            success,
        ),
        maintenance=(zeros, zeros, keep_top, keep_top, zeros),
        safety=_safety(env.cubeA) + _safety(env.cubeB),
    )


def _two_robot_stack_cube_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v2: add dense grasp/push/release credit.

    The v1 funnel reached stage 1 but had no dense signal inside the grasp
    stage before the binary grasp predicate flipped. This variant keeps the
    official task order but gives each active stage a continuous target.
    """
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    goal_xy = env.goal_region.pose.p

    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[1] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)

    cube_a_grasped = info.get(
        "is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)
    ).bool()
    cube_b_grasped = info.get(
        "is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)
    ).bool()
    cube_b_goal = _distance(cube_b, goal_xy, xy=True)
    cube_b_placed = info.get(
        "cubeB_placed", cube_b_goal < env.goal_radius
    ).bool()
    top_goal = cube_b.clone()
    top_goal[:, 2] += env.cube_half_size[2] * 2
    cube_a_top_error = _distance(cube_a, top_goal)
    cube_a_on_cube_b = info.get(
        "is_cubeA_on_cubeB", cube_a_top_error <= 0.01
    ).bool()

    qlim_left = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    qlim_right = env.right_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (
        torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim_left
    ).clamp(0.0, 1.0)
    right_open = (
        torch.sum(env.right_agent.robot.get_qpos()[:, -2:], dim=1) / qlim_right
    ).clamp(0.0, 1.0)
    ungrasp_left = torch.where(cube_a_grasped, left_open, torch.ones_like(left_open))
    ungrasp_right = torch.where(cube_b_grasped, right_open, torch.ones_like(right_open))

    cube_a_lift = torch.clamp((cube_a[:, 2] - 0.02) / 0.06, 0.0, 1.0)
    right_clear = _phi((right_tcp[:, 1] - 0.20).abs(), 5.0)
    reach_score = (
        0.65 * _phi(left_reach, 8.0) + 0.35 * _phi(right_reach, 6.0)
    ).clamp(0.0, 1.0)
    grasp_score = (
        0.45 * _phi(left_reach, 12.0)
        + 0.45 * cube_a_grasped.float()
        + 0.10 * cube_a_lift
    ).clamp(0.0, 1.0)
    bottom_score = (
        0.55 * _phi(cube_b_goal, 8.0)
        + 0.25 * _phi(right_reach, 10.0)
        + 0.20 * cube_a_grasped.float()
    ).clamp(0.0, 1.0)
    stack_score = (
        0.72 * _phi(cube_a_top_error, 14.0)
        + 0.14 * cube_a_grasped.float()
        + 0.14 * right_clear
    ).clamp(0.0, 1.0)
    release_score = (
        0.40 * ungrasp_left
        + 0.40 * ungrasp_right
        + 0.20 * (cube_a_on_cube_b & cube_b_placed).float()
    ).clamp(0.0, 1.0)

    success = _success(env, info)
    zeros = torch.zeros_like(left_reach)
    keep_top = _loss_penalty(cube_a_grasped | cube_a_on_cube_b)
    keep_bottom = _loss_penalty(cube_b_placed | (cube_b_goal <= env.goal_radius * 1.5))
    return _pack(
        (
            reach_score,
            grasp_score,
            bottom_score,
            stack_score,
            release_score,
        ),
        (
            (left_reach <= 0.06) & (right_reach <= 0.10),
            cube_a_grasped,
            cube_b_placed & cube_a_grasped,
            cube_a_on_cube_b & cube_b_placed,
            success,
        ),
        maintenance=(
            zeros,
            _loss_penalty(left_reach <= 0.08),
            0.04 * cube_a_grasped.float() + keep_bottom,
            0.03 * cube_b_placed.float() + keep_top,
            0.05 * (cube_a_on_cube_b & cube_b_placed).float(),
        ),
        safety=_safety(env.cubeA) + _safety(env.cubeB),
    )


def _two_robot_stack_cube_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v3: v2 shaping with a less brittle reach-stage gate."""
    base = _two_robot_stack_cube_v2(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[1] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)
    gates = base.gates.clone()
    gates[:, 0] = (left_reach <= 0.07) | ((left_reach <= 0.09) & (right_reach <= 0.14))
    return RuleInputs(base.potentials, gates, base.maintenance, base.safety_penalty)


def _two_robot_stack_cube_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v4: v1 entry funnel plus v2 downstream shaping."""
    base = _two_robot_stack_cube_v2(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[1] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    potentials[:, 0] = (0.5 * _phi(left_reach, 5.0) + 0.5 * _phi(right_reach, 5.0)).clamp(0.0, 1.0)
    gates[:, 0] = (left_reach <= 0.06) & (right_reach <= 0.08)
    return RuleInputs(potentials, gates, base.maintenance, base.safety_penalty)


def _two_robot_stack_cube_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v5: warm-start entry stage, keep v2 task gates."""
    base = _two_robot_stack_cube_v2(env, info)
    gates = base.gates.clone()
    gates[:, 0] = torch.ones_like(gates[:, 0], dtype=torch.bool)
    return RuleInputs(base.potentials, gates, base.maintenance, base.safety_penalty)


def _two_robot_stack_cube_v6(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v6: v5 entry plus explicit close-gripper grasp shaping."""
    base = _two_robot_stack_cube_v5(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    left_reach = _distance(left_tcp, cube_a)
    cube_a_grasped = info.get(
        "is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)
    ).bool()
    qlim_left = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (
        torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim_left
    ).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    near_left = _phi(left_reach, 14.0)
    close_near = (near_left * left_closed).clamp(0.0, 1.0)
    lift = torch.clamp((cube_a[:, 2] - 0.02) / 0.06, 0.0, 1.0)
    potentials = base.potentials.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 1] = (
        0.36 * near_left
        + 0.28 * close_near
        + 0.28 * cube_a_grasped.float()
        + 0.08 * lift
    ).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.08 * near_left + 0.08 * close_near + 0.12 * cube_a_grasped.float()
    return RuleInputs(potentials, base.gates, maintenance, base.safety_penalty)


def _two_robot_stack_cube_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v7: let near-closed attempts enter bottom-push practice."""
    base = _two_robot_stack_cube_v6(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    left_reach = _distance(left_tcp, cube_a)
    cube_a_grasped = info.get(
        "is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)
    ).bool()
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get("cubeB_placed", cube_b_goal < env.goal_radius).bool()
    top_goal = cube_b.clone()
    top_goal[:, 2] += env.cube_half_size[2] * 2
    cube_a_top_error = _distance(cube_a, top_goal)
    cube_a_on_cube_b = info.get("is_cubeA_on_cubeB", cube_a_top_error <= 0.01).bool()
    qlim_left = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim_left).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    close_attempt = (left_reach <= 0.045) & (left_closed >= 0.55)
    gates = base.gates.clone()
    gates[:, 1] = cube_a_grasped | close_attempt
    gates[:, 2] = cube_b_placed & (cube_a_grasped | close_attempt)
    gates[:, 3] = cube_a_on_cube_b & cube_b_placed
    gates[:, 4] = _success(env, info)
    maintenance = base.maintenance.clone()
    maintenance[:, 2] = maintenance[:, 2] + 0.08 * close_attempt.float() + 0.08 * cube_a_grasped.float()
    return RuleInputs(base.potentials, gates, maintenance, base.safety_penalty)


def _two_robot_stack_cube_v8(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v8: strong left-grasp acquisition before stacking."""
    base = _two_robot_stack_cube_v7(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[1] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)
    cube_a_grasped = info.get(
        "is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)
    ).bool()
    qlim_left = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim_left).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    near_left = _phi(left_reach, 18.0)
    close_near = (near_left * left_closed).clamp(0.0, 1.0)
    lift = torch.clamp((cube_a[:, 2] - 0.02) / 0.06, 0.0, 1.0)
    potentials = base.potentials.clone()
    maintenance = base.maintenance.clone()
    gates = base.gates.clone()
    potentials[:, 0] = (0.55 * _phi(left_reach, 8.0) + 0.45 * _phi(right_reach, 5.0)).clamp(0.0, 1.0)
    potentials[:, 1] = (
        0.42 * near_left
        + 0.28 * close_near
        + 0.24 * cube_a_grasped.float()
        + 0.06 * lift
    ).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.18 * near_left + 0.22 * close_near + 0.24 * cube_a_grasped.float() + 0.04 * lift
    gates[:, 1] = cube_a_grasped | ((left_reach <= 0.05) & (left_closed >= 0.45))
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _two_robot_stack_cube_v9(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v9: independent gripper-close credit for grasp acquisition."""
    base = _two_robot_stack_cube_v8(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    left_reach = _distance(left_tcp, cube_a)
    cube_a_grasped = info.get(
        "is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)
    ).bool()
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get("cubeB_placed", cube_b_goal < env.goal_radius).bool()
    top_goal = cube_b.clone()
    top_goal[:, 2] += env.cube_half_size[2] * 2
    cube_a_top_error = _distance(cube_a, top_goal)
    cube_a_on_cube_b = info.get("is_cubeA_onCubeB", cube_a_top_error <= 0.01)
    if not isinstance(cube_a_on_cube_b, torch.Tensor):
        cube_a_on_cube_b = info.get("is_cubeA_on_cubeB", cube_a_top_error <= 0.01)
    cube_a_on_cube_b = cube_a_on_cube_b.bool()
    qlim_left = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim_left).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    near_left = _phi(left_reach, 18.0)
    close_near = (near_left * left_closed).clamp(0.0, 1.0)
    lift = torch.clamp((cube_a[:, 2] - 0.02) / 0.06, 0.0, 1.0)
    close_attempt = (left_reach <= 0.065) & (left_closed >= 0.30)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 1] = (
        0.32 * near_left
        + 0.24 * left_closed
        + 0.24 * close_near
        + 0.16 * cube_a_grasped.float()
        + 0.04 * lift
    ).clamp(0.0, 1.0)
    gates[:, 1] = cube_a_grasped | close_attempt
    gates[:, 2] = cube_b_placed & (cube_a_grasped | close_attempt)
    gates[:, 3] = cube_a_on_cube_b & cube_b_placed
    gates[:, 4] = _success(env, info)
    maintenance[:, 1] = 0.16 * near_left + 0.14 * left_closed + 0.20 * close_near + 0.20 * cube_a_grasped.float()
    maintenance[:, 2] = maintenance[:, 2] + 0.08 * close_attempt.float() + 0.08 * cube_a_grasped.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)



def _two_robot_stack_cube_v10(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v10: stage-wise version of the official dense reward."""
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[0] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)
    left_reach_phi = _phi(left_reach, 5.0)
    right_reach_phi = _phi(right_reach, 5.0)
    reach_reward = ((left_reach_phi + right_reach_phi) / 2).clamp(0.0, 1.0)

    cube_a_grasped = info.get("is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)).bool()
    cube_b_grasped = info.get("is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)).bool()
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get("cubeB_placed", cube_b_goal < env.goal_radius).bool()
    bottom_place = _phi(cube_b_goal, 5.0)

    top_goal = torch.hstack([cube_b[:, :2], (cube_b[:, 2] + env.cube_half_size[2] * 2)[:, None]])
    cube_a_top_error = _distance(cube_a, top_goal)
    top_place = _phi(cube_a_top_error, 5.0)
    right_clear = _phi((right_tcp[:, 1] - 0.2).abs(), 5.0)
    cube_a_on_cube_b = info.get("is_cubeA_on_cubeB", cube_a_top_error <= 0.01).bool()

    gripper_width = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / gripper_width).clamp(0.0, 1.0)
    right_open = (torch.sum(env.right_agent.robot.get_qpos()[:, -2:], dim=1) / gripper_width).clamp(0.0, 1.0)
    ungrasp_left = torch.where(cube_a_grasped, left_open, torch.ones_like(left_open))
    ungrasp_right = torch.where(cube_b_grasped, right_open, torch.ones_like(right_open))
    release_score = ((ungrasp_left + ungrasp_right) / 2).clamp(0.0, 1.0)
    success = _success(env, info)
    zeros = torch.zeros_like(left_reach)

    stage1_score = ((left_reach_phi + cube_a_grasped.float()) / 2).clamp(0.0, 1.0)
    stage2_score = ((bottom_place + cube_a_grasped.float()) / 2).clamp(0.0, 1.0)
    stage3_score = ((2.0 * top_place + right_clear) / 3.0).clamp(0.0, 1.0)
    return _pack(
        (
            reach_reward,
            stage1_score,
            stage2_score,
            stage3_score,
            (0.75 * release_score + 0.25 * success.float()).clamp(0.0, 1.0),
        ),
        (
            (left_reach <= 0.08) | cube_a_grasped,
            cube_a_grasped,
            cube_b_placed & cube_a_grasped,
            cube_a_on_cube_b & cube_b_placed,
            success,
        ),
        maintenance=(
            0.20 * reach_reward,
            0.30 * left_reach_phi + 0.35 * cube_a_grasped.float(),
            0.30 * bottom_place + 0.20 * cube_a_grasped.float(),
            0.42 * top_place + 0.18 * right_clear + 0.12 * cube_a_grasped.float(),
            0.35 * release_score + 0.12 * success.float(),
        ),
        safety=zeros,
    )


def _two_robot_stack_cube_v11(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v11: native order with soft top-carry gate."""
    base = _two_robot_stack_cube_v10(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    cube_a_grasped = info.get("is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)).bool()
    cube_b_grasped = info.get("is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)).bool()
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get("cubeB_placed", cube_b_goal < env.goal_radius).bool()
    top_goal = torch.hstack([cube_b[:, :2], (cube_b[:, 2] + env.cube_half_size[2] * 2)[:, None]])
    cube_a_top_error = _distance(cube_a, top_goal)
    cube_a_on_cube_b = info.get("is_cubeA_on_cubeB", cube_a_top_error <= 0.01).bool()
    left_reach = _distance(left_tcp, cube_a)
    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[0] + 0.005
    right_reach = _distance(right_tcp, right_push_pose)
    qlim = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    right_open = (torch.sum(env.right_agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    top_lift = torch.clamp((cube_a[:, 2] - 0.02) / 0.060, 0.0, 1.0)
    close_attempt = (left_reach <= 0.065) & (left_closed >= 0.30)
    top_carry = cube_a_grasped | close_attempt | ((left_reach <= 0.09) & (top_lift >= 0.12))
    bottom_phi = _phi(cube_b_goal, 7.0)
    top_phi = _phi(cube_a_top_error, 9.0)
    right_clear = _phi((right_tcp[:, 1] - 0.2).abs(), 5.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 1] = top_carry
    gates[:, 2] = cube_b_placed & top_carry
    gates[:, 3] = cube_a_on_cube_b & cube_b_placed
    gates[:, 4] = success
    potentials[:, 1] = (0.36 * _phi(left_reach, 12.0) + 0.24 * left_closed + 0.22 * top_carry.float() + 0.18 * top_lift).clamp(0.0, 1.0)
    potentials[:, 2] = (0.54 * bottom_phi + 0.22 * _phi(right_reach, 8.0) + 0.14 * top_carry.float() + 0.10 * top_lift).clamp(0.0, 1.0)
    potentials[:, 3] = (0.58 * top_phi + 0.18 * top_carry.float() + 0.14 * right_clear + 0.10 * cube_b_placed.float()).clamp(0.0, 1.0)
    potentials[:, 4] = (0.55 * ((left_open + right_open) / 2).clamp(0.0, 1.0) + 0.25 * success.float() + 0.20 * (cube_a_on_cube_b & cube_b_placed).float()).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.14 * _phi(left_reach, 12.0) + 0.12 * left_closed + 0.12 * top_carry.float() + 0.08 * top_lift
    maintenance[:, 2] = 0.18 * bottom_phi + 0.08 * top_carry.float() + 0.06 * _phi(right_reach, 8.0)
    maintenance[:, 3] = 0.20 * top_phi + 0.08 * right_clear + 0.08 * cube_b_placed.float()
    maintenance[:, 4] = 0.10 * success.float() + 0.06 * (cube_a_on_cube_b & cube_b_placed).float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _two_robot_stack_cube_v12(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v12: separate bottom placement from top-cube carry."""
    base = _two_robot_stack_cube_v11(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    left_reach = _distance(left_tcp, cube_a)
    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[0] + 0.005
    right_reach = _distance(right_tcp, right_push_pose)
    cube_a_grasped = info.get("is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)).bool()
    cube_b_grasped = info.get("is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)).bool()
    qlim = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    right_open = (torch.sum(env.right_agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    right_closed = (1.0 - right_open).clamp(0.0, 1.0)
    top_lift = torch.clamp((cube_a[:, 2] - 0.02) / 0.070, 0.0, 1.0)
    top_carry = cube_a_grasped | ((left_reach <= 0.09) & (left_closed >= 0.25)) | ((left_reach <= 0.11) & (top_lift >= 0.10))
    bottom_carry = cube_b_grasped | ((right_reach <= 0.10) & (right_closed >= 0.20))
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get("cubeB_placed", cube_b_goal < env.goal_radius).bool()
    bottom_phi = _phi(cube_b_goal, 8.0)
    top_goal = torch.hstack([cube_b[:, :2], (cube_b[:, 2] + env.cube_half_size[2] * 2)[:, None]])
    cube_a_top_error = _distance(cube_a, top_goal)
    top_phi = _phi(cube_a_top_error, 9.0)
    cube_a_on_cube_b = info.get("is_cubeA_on_cubeB", cube_a_top_error <= 0.01).bool()
    right_clear = _phi((right_tcp[:, 1] - 0.2).abs(), 5.0)
    release_score = ((left_open + right_open) / 2).clamp(0.0, 1.0)
    success = _success(env, info)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = (left_reach <= 0.12) | (right_reach <= 0.12) | top_carry | bottom_carry
    gates[:, 1] = top_carry | bottom_carry | cube_b_placed
    gates[:, 2] = (cube_b_goal <= 0.16) | cube_b_placed
    gates[:, 3] = cube_b_placed & ((top_phi >= 0.42) | (top_lift >= 0.25) | cube_a_on_cube_b)
    gates[:, 4] = success
    potentials[:, 0] = (0.38 * _phi(left_reach, 7.0) + 0.32 * _phi(right_reach, 7.0) + 0.15 * top_carry.float() + 0.15 * bottom_carry.float()).clamp(0.0, 1.0)
    potentials[:, 1] = (0.42 * _phi(left_reach, 10.0) + 0.26 * top_carry.float() + 0.18 * bottom_phi + 0.14 * bottom_carry.float()).clamp(0.0, 1.0)
    potentials[:, 2] = (0.58 * bottom_phi + 0.18 * bottom_carry.float() + 0.14 * top_carry.float() + 0.10 * cube_b_placed.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.54 * top_phi + 0.18 * top_carry.float() + 0.16 * cube_b_placed.float() + 0.12 * right_clear).clamp(0.0, 1.0)
    potentials[:, 4] = (0.48 * top_phi + 0.20 * cube_a_on_cube_b.float() + 0.14 * release_score + 0.10 * cube_b_placed.float() + 0.08 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.08 * _phi(left_reach, 7.0) + 0.08 * _phi(right_reach, 7.0)
    maintenance[:, 1] = 0.12 * top_carry.float() + 0.08 * bottom_carry.float() + 0.08 * bottom_phi
    maintenance[:, 2] = 0.24 * bottom_phi + 0.08 * cube_b_placed.float() + 0.06 * top_carry.float()
    maintenance[:, 3] = 0.22 * top_phi + 0.10 * right_clear + 0.08 * cube_b_placed.float()
    maintenance[:, 4] = 0.18 * top_phi + 0.10 * cube_a_on_cube_b.float() + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _two_robot_stack_cube_v13(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v13: strict official-reference staged geometry.

    This mirrors the native reward's observable progress variables, but keeps
    our FSM reward form. The main correction from v12 is removing loose
    top-lift/top-distance exits: release can only start after the official
    bottom-place and top-stack predicates are both true.
    """
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p

    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[0] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)
    left_reach_phi = _phi(left_reach, 7.0)
    right_reach_phi = _phi(right_reach, 7.0)

    cube_a_grasped = info.get(
        "is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)
    ).bool()
    cube_b_grasped = info.get(
        "is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)
    ).bool()
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get("cubeB_placed", cube_b_goal < env.goal_radius).bool()
    bottom_place = _phi(cube_b_goal, 5.0)

    top_goal = torch.hstack(
        [cube_b[:, :2], (cube_b[:, 2] + env.cube_half_size[2] * 2)[:, None]]
    )
    cube_a_top_error = _distance(cube_a, top_goal)
    top_place = _phi(cube_a_top_error, 5.0)
    cube_a_on_cube_b = info.get(
        "is_cubeA_on_cubeB", cube_a_top_error <= 0.01
    ).bool()
    right_clear = _phi((right_tcp[:, 1] - 0.2).abs(), 5.0)

    qlim = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (
        torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim
    ).clamp(0.0, 1.0)
    right_open = (
        torch.sum(env.right_agent.robot.get_qpos()[:, -2:], dim=1) / qlim
    ).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    right_closed = (1.0 - right_open).clamp(0.0, 1.0)
    release_score = torch.where(
        cube_a_grasped,
        left_open,
        torch.ones_like(left_open),
    ) + torch.where(
        cube_b_grasped,
        right_open,
        torch.ones_like(right_open),
    )
    release_score = (release_score / 2.0).clamp(0.0, 1.0)
    success = _success(env, info)
    stacked = cube_a_on_cube_b & cube_b_placed
    zeros = torch.zeros_like(left_reach)

    return _pack(
        (
            (0.50 * left_reach_phi + 0.50 * right_reach_phi).clamp(0.0, 1.0),
            (
                0.40 * left_reach_phi
                + 0.25 * left_closed
                + 0.25 * cube_a_grasped.float()
                + 0.10 * right_reach_phi
            ).clamp(0.0, 1.0),
            (
                0.62 * bottom_place
                + 0.18 * right_reach_phi
                + 0.14 * cube_a_grasped.float()
                + 0.06 * right_closed
            ).clamp(0.0, 1.0),
            (
                0.62 * top_place
                + 0.18 * right_clear
                + 0.14 * cube_a_grasped.float()
                + 0.06 * cube_b_placed.float()
            ).clamp(0.0, 1.0),
            (
                0.42 * top_place
                + 0.30 * release_score
                + 0.18 * stacked.float()
                + 0.10 * success.float()
            ).clamp(0.0, 1.0),
        ),
        (
            ((left_reach <= 0.10) & (right_reach <= 0.12)) | cube_a_grasped,
            cube_a_grasped,
            cube_b_placed & cube_a_grasped,
            stacked,
            success,
        ),
        maintenance=(
            0.08 * left_reach_phi + 0.08 * right_reach_phi,
            0.12 * left_reach_phi + 0.10 * left_closed + 0.12 * cube_a_grasped.float(),
            0.18 * bottom_place + 0.16 * cube_a_grasped.float(),
            0.20 * top_place + 0.10 * right_clear + 0.12 * cube_b_placed.float(),
            0.16 * release_score + 0.12 * stacked.float() + 0.08 * success.float(),
        ),
        safety=zeros,
    )


def _two_robot_stack_cube_v14(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v14: v12 entry with official-world terminal geometry."""
    base = _two_robot_stack_cube_v13(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    right_push_pose = cube_b.clone()
    right_push_pose[:, 1] += env.cube_half_size[0] + 0.005
    left_reach = _distance(left_tcp, cube_a)
    right_reach = _distance(right_tcp, right_push_pose)
    left_phi = _phi(left_reach, 8.0)
    right_phi = _phi(right_reach, 8.0)

    cube_a_grasped = info.get("is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)).bool()
    cube_b_grasped = info.get("is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)).bool()
    qlim = env.left_agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    left_open = (torch.sum(env.left_agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    right_open = (torch.sum(env.right_agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    left_closed = (1.0 - left_open).clamp(0.0, 1.0)
    right_closed = (1.0 - right_open).clamp(0.0, 1.0)
    top_lift = torch.clamp((cube_a[:, 2] - 0.02) / 0.070, 0.0, 1.0)
    top_carry = cube_a_grasped | ((left_reach <= 0.10) & (left_closed >= 0.20)) | ((left_reach <= 0.12) & (top_lift >= 0.08))
    bottom_carry = cube_b_grasped | ((right_reach <= 0.11) & (right_closed >= 0.18))

    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    cube_b_placed = info.get("cubeB_placed", cube_b_goal < env.goal_radius).bool()
    bottom_phi = _phi(cube_b_goal, 6.0)
    bottom_loose = cube_b_goal <= 0.11
    bottom_near = cube_b_goal <= 0.075
    top_goal = torch.hstack([cube_b[:, :2], (cube_b[:, 2] + env.cube_half_size[2] * 2)[:, None]])
    cube_a_top_error = _distance(cube_a, top_goal)
    top_phi = _phi(cube_a_top_error, 7.0)
    top_loose = cube_a_top_error <= 0.060
    top_near = cube_a_top_error <= 0.035
    cube_a_on_cube_b = info.get("is_cubeA_on_cubeB", cube_a_top_error <= 0.01).bool()
    stacked_loose = bottom_near & top_loose
    stacked_tight = cube_b_placed & (top_near | cube_a_on_cube_b)
    right_clear = _phi((right_tcp[:, 1] - 0.2).abs(), 5.0)
    release_score = (torch.where(cube_a_grasped, left_open, torch.ones_like(left_open)) + torch.where(cube_b_grasped, right_open, torch.ones_like(right_open))) / 2.0
    release_score = release_score.clamp(0.0, 1.0)
    success = _success(env, info)
    zeros = torch.zeros_like(left_reach)

    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = (left_reach <= 0.13) | (right_reach <= 0.14) | top_carry | bottom_carry
    gates[:, 1] = top_carry | ((left_reach <= 0.11) & (left_closed >= 0.20))
    gates[:, 2] = bottom_loose | cube_b_placed | (bottom_phi >= 0.50)
    gates[:, 3] = stacked_loose | stacked_tight | ((bottom_near | cube_b_placed) & (top_phi >= 0.55))
    gates[:, 4] = success
    potentials[:, 0] = (0.48 * left_phi + 0.42 * right_phi + 0.05 * top_carry.float() + 0.05 * bottom_carry.float()).clamp(0.0, 1.0)
    potentials[:, 1] = (0.42 * left_phi + 0.24 * left_closed + 0.20 * top_carry.float() + 0.14 * top_lift).clamp(0.0, 1.0)
    potentials[:, 2] = (0.60 * bottom_phi + 0.18 * right_phi + 0.12 * top_carry.float() + 0.10 * bottom_carry.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.52 * top_phi + 0.20 * bottom_phi + 0.12 * top_carry.float() + 0.10 * right_clear + 0.06 * stacked_loose.float()).clamp(0.0, 1.0)
    potentials[:, 4] = (0.44 * top_phi + 0.18 * bottom_phi + 0.18 * release_score + 0.12 * stacked_tight.float() + 0.08 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.08 * left_phi + 0.08 * right_phi
    maintenance[:, 1] = 0.12 * left_phi + 0.10 * left_closed + 0.10 * top_carry.float() + 0.06 * top_lift
    maintenance[:, 2] = 0.24 * bottom_phi + 0.08 * bottom_carry.float() + 0.06 * top_carry.float()
    maintenance[:, 3] = 0.22 * top_phi + 0.10 * bottom_phi + 0.08 * right_clear + 0.06 * stacked_loose.float()
    maintenance[:, 4] = 0.18 * top_phi + 0.10 * release_score + 0.10 * stacked_tight.float() + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, zeros)

def _draw_triangle(env: Any, info: dict[str, Any]) -> RuleInputs:
    tcp = _tcp(env)
    covered = env.ref_dist.float().mean(dim=1).clamp(0.0, 1.0)
    started = (env.dots_dist > -1).any(dim=1)
    nearest_path = torch.linalg.norm(
        tcp[:, None, :2] - env.triangles[..., :2], dim=-1
    ).min(dim=1).values
    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    reach = 0.5 * _phi(nearest_path, 12.0) + 0.5 * _phi(z_error, 40.0)
    success = _success(env, info)
    return _pack(
        (
            reach,
            torch.clamp(covered / 0.33, 0.0, 1.0),
            torch.clamp((covered - 0.20) / 0.55, 0.0, 1.0),
            covered,
        ),
        (
            started | ((nearest_path <= 0.06) & (z_error <= 0.025)),
            covered >= 0.30,
            covered >= 0.70,
            success,
        ),
    )


def _draw_triangle_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v2: side-wise coverage with off-path maintenance penalty."""
    tcp = _tcp(env)
    ref = env.ref_dist.float()
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    covered = ref.mean(dim=1).clamp(0.0, 1.0)

    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(
        drawn_mask,
        env.dots_dist.clamp(min=0).to(dtype=torch.float32),
        torch.zeros_like(env.dots_dist, dtype=torch.float32),
    ).sum(dim=1)
    valid_fraction = torch.where(
        drawn_count > 0,
        valid_count / drawn_count.clamp(min=1.0),
        torch.ones_like(drawn_count),
    ).clamp(0.0, 1.0)
    started = drawn_count > 0

    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    z_score = _phi(z_error, 45.0)
    nearest_path = torch.linalg.norm(
        tcp[:, None, :2] - env.triangles[..., :2], dim=-1
    ).min(dim=1).values
    path_score = _phi(nearest_path, 18.0)
    start_error = torch.linalg.norm(
        tcp[:, :2] - env.triangles[:, 0, :2], dim=-1
    )
    start_score = 0.65 * _phi(start_error, 18.0) + 0.35 * z_score
    side01 = torch.minimum(side0, side1)
    remaining = 0.50 * side1 + 0.50 * side2

    success = _success(env, info)
    off_path = started & (valid_fraction < 0.90)
    high_brush = z_error > 0.035
    return _pack(
        (
            start_score.clamp(0.0, 1.0),
            (
                0.55 * side0
                + 0.20 * valid_fraction
                + 0.15 * path_score
                + 0.10 * z_score
            ).clamp(0.0, 1.0),
            (
                0.45 * remaining
                + 0.20 * side01
                + 0.20 * valid_fraction
                + 0.15 * path_score
            ).clamp(0.0, 1.0),
            (
                0.65 * covered
                + 0.20 * valid_fraction
                + 0.15 * success.float()
            ).clamp(0.0, 1.0),
        ),
        (
            started & (valid_fraction >= 0.80),
            side0 >= 0.55,
            (covered >= 0.72) & (torch.minimum(side1, side2) >= 0.55),
            success,
        ),
        maintenance=(
            -0.04 * off_path.float() - 0.02 * high_brush.float(),
            -0.06 * off_path.float() - 0.02 * high_brush.float(),
            -0.06 * off_path.float() - 0.02 * high_brush.float(),
            -0.04 * off_path.float(),
        ),
    )


def _draw_triangle_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v3: v2 shaping with permissive drawing-stage gates."""
    base = _draw_triangle_v2(env, info)
    tcp = _tcp(env)
    ref = env.ref_dist.float()
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(
        drawn_mask,
        env.dots_dist.clamp(min=0).to(dtype=torch.float32),
        torch.zeros_like(env.dots_dist, dtype=torch.float32),
    ).sum(dim=1)
    valid_fraction = torch.where(
        drawn_count > 0,
        valid_count / drawn_count.clamp(min=1.0),
        torch.ones_like(drawn_count),
    ).clamp(0.0, 1.0)
    started = drawn_count > 0
    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    start_error = torch.linalg.norm(tcp[:, :2] - env.triangles[:, 0, :2], dim=-1)
    gates = base.gates.clone()
    gates[:, 0] = ((start_error <= 0.08) & (z_error <= 0.04)) | (started & (valid_fraction >= 0.60))
    gates[:, 1] = side0 >= 0.35
    gates[:, 2] = (covered >= 0.58) & (torch.minimum(side1, side2) >= 0.30)
    return RuleInputs(base.potentials, gates, base.maintenance, base.safety_penalty)


def _draw_triangle_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v4: side-wise reward with native nearest-path entry gate."""
    base = _draw_triangle_v2(env, info)
    tcp = _tcp(env)
    ref = env.ref_dist.float()
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    started = drawn_count > 0
    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    nearest_path = torch.linalg.norm(
        tcp[:, None, :2] - env.triangles[..., :2], dim=-1
    ).min(dim=1).values
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    potentials[:, 0] = (0.5 * _phi(nearest_path, 12.0) + 0.5 * _phi(z_error, 40.0)).clamp(0.0, 1.0)
    gates[:, 0] = started | ((nearest_path <= 0.08) & (z_error <= 0.04))
    gates[:, 1] = side0 >= 0.30
    gates[:, 2] = (covered >= 0.55) & (torch.minimum(side1, side2) >= 0.25)
    return RuleInputs(potentials, gates, base.maintenance, base.safety_penalty)


def _draw_triangle_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v5: warm-start entry stage, keep side-wise coverage gates."""
    base = _draw_triangle_v4(env, info)
    gates = base.gates.clone()
    gates[:, 0] = torch.ones_like(gates[:, 0], dtype=torch.bool)
    return RuleInputs(base.potentials, gates, base.maintenance, base.safety_penalty)


def _draw_triangle_v6(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v6: positive path/height/coverage shaping without early penalties."""
    tcp = _tcp(env)
    ref = env.ref_dist.float()
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(
        drawn_mask,
        env.dots_dist.clamp(min=0).to(dtype=torch.float32),
        torch.zeros_like(env.dots_dist, dtype=torch.float32),
    ).sum(dim=1)
    valid_fraction = torch.where(
        drawn_count > 0,
        valid_count / drawn_count.clamp(min=1.0),
        torch.ones_like(drawn_count),
    ).clamp(0.0, 1.0)
    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    z_score = _phi(z_error, 45.0)
    nearest_path = torch.linalg.norm(
        tcp[:, None, :2] - env.triangles[..., :2], dim=-1
    ).min(dim=1).values
    path_score = _phi(nearest_path, 18.0)
    side01 = torch.minimum(side0, side1)
    remaining = 0.50 * side1 + 0.50 * side2
    success = _success(env, info)
    one = torch.ones_like(success, dtype=torch.bool)
    potentials = torch.stack(
        (
            (0.5 * path_score + 0.5 * z_score).clamp(0.0, 1.0),
            (0.42 * side0 + 0.24 * path_score + 0.20 * z_score + 0.14 * valid_fraction).clamp(0.0, 1.0),
            (0.42 * remaining + 0.20 * side01 + 0.20 * path_score + 0.18 * valid_fraction).clamp(0.0, 1.0),
            (0.62 * covered + 0.20 * valid_fraction + 0.18 * success.float()).clamp(0.0, 1.0),
        ),
        dim=1,
    ).to(dtype=torch.float32)
    gates = torch.stack(
        (
            one,
            (side0 >= 0.15) | (covered >= 0.12),
            (covered >= 0.38) & (torch.minimum(side1, side2) >= 0.10),
            success,
        ),
        dim=1,
    )
    maintenance = torch.stack(
        (
            0.02 * path_score + 0.02 * z_score,
            0.05 * path_score + 0.05 * z_score + 0.08 * side0,
            0.04 * path_score + 0.04 * valid_fraction + 0.10 * covered,
            0.10 * covered + 0.06 * success.float(),
        ),
        dim=1,
    ).to(dtype=torch.float32)
    safety = torch.zeros_like(covered)
    return RuleInputs(potentials, gates, maintenance, safety)


def _draw_triangle_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v7: guide the brush to the nearest currently-uncovered path point."""
    tcp = _tcp(env)
    ref = env.ref_dist.float()
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(
        drawn_mask,
        env.dots_dist.clamp(min=0).to(dtype=torch.float32),
        torch.zeros_like(env.dots_dist, dtype=torch.float32),
    ).sum(dim=1)
    valid_fraction = torch.where(
        drawn_count > 0,
        valid_count / drawn_count.clamp(min=1.0),
        torch.ones_like(drawn_count),
    ).clamp(0.0, 1.0)
    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    z_score = _phi(z_error, 45.0)
    dist = torch.linalg.norm(tcp[:, None, :2] - env.triangles[..., :2], dim=-1)
    nearest_path = dist.min(dim=1).values
    path_score = _phi(nearest_path, 18.0)
    uncovered_dist = dist + ref * 10.0
    nearest_uncovered = uncovered_dist.min(dim=1).values
    uncovered_score = _phi(nearest_uncovered, 18.0)
    success = _success(env, info)
    one = torch.ones_like(success, dtype=torch.bool)
    potentials = torch.stack(
        (
            (0.45 * path_score + 0.45 * z_score + 0.10 * uncovered_score).clamp(0.0, 1.0),
            (0.36 * covered + 0.30 * uncovered_score + 0.20 * z_score + 0.14 * valid_fraction).clamp(0.0, 1.0),
            (0.54 * covered + 0.24 * uncovered_score + 0.12 * path_score + 0.10 * valid_fraction).clamp(0.0, 1.0),
            (0.70 * covered + 0.18 * valid_fraction + 0.12 * success.float()).clamp(0.0, 1.0),
        ),
        dim=1,
    ).to(dtype=torch.float32)
    gates = torch.stack((one, covered >= 0.22, covered >= 0.58, success), dim=1)
    maintenance = torch.stack(
        (
            0.02 * path_score + 0.03 * z_score,
            0.07 * uncovered_score + 0.05 * z_score + 0.08 * covered,
            0.08 * uncovered_score + 0.04 * valid_fraction + 0.12 * covered,
            0.12 * covered + 0.06 * success.float(),
        ),
        dim=1,
    ).to(dtype=torch.float32)
    safety = torch.zeros_like(covered)
    return RuleInputs(potentials, gates, maintenance, safety)


def _draw_triangle_v8(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v8: earlier completion stage with valid-stroke late shaping."""
    base = _draw_triangle_v7(env, info)
    ref = env.ref_dist.float()
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(
        drawn_mask,
        env.dots_dist.clamp(min=0).to(dtype=torch.float32),
        torch.zeros_like(env.dots_dist, dtype=torch.float32),
    ).sum(dim=1)
    valid_fraction = torch.where(
        drawn_count > 0,
        valid_count / drawn_count.clamp(min=1.0),
        torch.ones_like(drawn_count),
    ).clamp(0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    late_clean = ((covered >= 0.35) & (valid_fraction >= 0.90)).float()
    potentials[:, 3] = (0.58 * covered + 0.30 * valid_fraction + 0.12 * success.float()).clamp(0.0, 1.0)
    gates[:, 2] = covered >= 0.35
    gates[:, 3] = success
    maintenance[:, 2] = maintenance[:, 2] + 0.08 * covered + 0.04 * valid_fraction
    maintenance[:, 3] = 0.16 * covered + 0.10 * valid_fraction + 0.06 * late_clean + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)



def _draw_triangle_v9(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v9: reward uncovered-reference coverage, not average coverage only."""
    tcp = _tcp(env)
    ref_bool = env.ref_dist.bool()
    ref = ref_bool.float()
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    min_side = torch.minimum(torch.minimum(side0, side1), side2)
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    all_covered = torch.all(ref_bool, dim=1)

    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(drawn_mask, env.dots_dist.clamp(min=0).to(dtype=torch.float32), torch.zeros_like(env.dots_dist, dtype=torch.float32)).sum(dim=1)
    valid_fraction = torch.where(drawn_count > 0, valid_count / drawn_count.clamp(min=1.0), torch.ones_like(drawn_count)).clamp(0.0, 1.0)
    started = drawn_count > 0
    invalid = started & (valid_fraction < 0.995)

    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    z_score = _phi(z_error, 50.0)
    dist = torch.linalg.norm(tcp[:, None, :2] - env.triangles[..., :2], dim=-1)
    nearest_path = dist.min(dim=1).values
    path_score = _phi(nearest_path, 22.0)
    nearest_uncovered = (dist + ref * 10.0).min(dim=1).values
    uncovered_score = _phi(nearest_uncovered, 24.0)
    clean_complete = (all_covered & (valid_fraction >= 0.995)).float()
    success = _success(env, info)
    one = torch.ones_like(success, dtype=torch.bool)

    return _pack(
        (
            (0.42 * path_score + 0.42 * z_score + 0.16 * uncovered_score).clamp(0.0, 1.0),
            (0.34 * side0 + 0.28 * uncovered_score + 0.20 * z_score + 0.18 * valid_fraction).clamp(0.0, 1.0),
            (0.36 * covered + 0.24 * min_side + 0.24 * uncovered_score + 0.16 * valid_fraction).clamp(0.0, 1.0),
            (0.44 * covered + 0.28 * min_side + 0.14 * all_covered.float() + 0.14 * success.float()).clamp(0.0, 1.0),
        ),
        (
            one,
            covered >= 0.18,
            (covered >= 0.55) & (min_side >= 0.12),
            success,
        ),
        maintenance=(
            0.04 * path_score + 0.05 * z_score - 0.10 * invalid.float(),
            0.08 * uncovered_score + 0.05 * z_score + 0.08 * side0 - 0.12 * invalid.float(),
            0.12 * uncovered_score + 0.08 * covered + 0.06 * min_side - 0.14 * invalid.float(),
            0.16 * covered + 0.12 * min_side + 0.16 * clean_complete + 0.10 * success.float() - 0.16 * invalid.float(),
        ),
        safety=torch.zeros_like(covered),
    )


def _draw_triangle_v10(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v10: softer side coverage with a lighter invalid-stroke penalty."""
    tcp = _tcp(env)
    ref_bool = env.ref_dist.bool()
    ref = ref_bool.float()
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    min_side = torch.minimum(torch.minimum(side0, side1), side2)
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(drawn_mask, env.dots_dist.clamp(min=0).to(dtype=torch.float32), torch.zeros_like(env.dots_dist, dtype=torch.float32)).sum(dim=1)
    valid_fraction = torch.where(drawn_count > 0, valid_count / drawn_count.clamp(min=1.0), torch.ones_like(drawn_count)).clamp(0.0, 1.0)
    started = drawn_count > 0
    invalid = started & (valid_fraction < 0.985)

    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    z_score = _phi(z_error, 55.0)
    dist = torch.linalg.norm(tcp[:, None, :2] - env.triangles[..., :2], dim=-1)
    nearest_path = dist.min(dim=1).values
    path_score = _phi(nearest_path, 26.0)
    nearest_uncovered = (dist + ref * 10.0).min(dim=1).values
    uncovered_score = _phi(nearest_uncovered, 28.0)
    side_balance = (0.50 * min_side + 0.50 * covered).clamp(0.0, 1.0)
    success = _success(env, info)
    one = torch.ones_like(success, dtype=torch.bool)

    return _pack(
        (
            (0.44 * path_score + 0.36 * z_score + 0.20 * uncovered_score).clamp(0.0, 1.0),
            (0.34 * side0 + 0.26 * uncovered_score + 0.22 * z_score + 0.18 * valid_fraction).clamp(0.0, 1.0),
            (0.40 * covered + 0.28 * side_balance + 0.20 * uncovered_score + 0.12 * valid_fraction).clamp(0.0, 1.0),
            (0.46 * covered + 0.30 * min_side + 0.14 * valid_fraction + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            one,
            started | (covered >= 0.08),
            (covered >= 0.38) & (min_side >= 0.05),
            success,
        ),
        maintenance=(
            0.05 * path_score + 0.06 * z_score - 0.04 * invalid.float(),
            0.10 * uncovered_score + 0.06 * z_score + 0.08 * side0 - 0.05 * invalid.float(),
            0.14 * uncovered_score + 0.10 * covered + 0.08 * side_balance - 0.06 * invalid.float(),
            0.18 * covered + 0.14 * min_side + 0.08 * valid_fraction + 0.08 * success.float() - 0.06 * invalid.float(),
        ),
        safety=torch.zeros_like(covered),
    )


def _draw_triangle_v11(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v11: coverage-first shaping after v10 stalls in stage 1."""
    tcp = _tcp(env)
    ref_bool = env.ref_dist.bool()
    ref = ref_bool.float()
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    min_side = torch.minimum(torch.minimum(side0, side1), side2)
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    all_covered = torch.all(ref_bool, dim=1).float()

    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(
        drawn_mask,
        env.dots_dist.clamp(min=0).to(dtype=torch.float32),
        torch.zeros_like(env.dots_dist, dtype=torch.float32),
    ).sum(dim=1)
    valid_fraction = torch.where(
        drawn_count > 0,
        valid_count / drawn_count.clamp(min=1.0),
        torch.ones_like(drawn_count),
    ).clamp(0.0, 1.0)
    started = drawn_count > 0
    invalid = started & (valid_fraction < 0.95)

    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    z_score = _phi(z_error, 50.0)
    dist = torch.linalg.norm(tcp[:, None, :2] - env.triangles[..., :2], dim=-1)
    nearest_path = dist.min(dim=1).values
    path_score = _phi(nearest_path, 24.0)
    nearest_uncovered = (dist + ref * 10.0).min(dim=1).values
    uncovered_score = _phi(nearest_uncovered, 30.0)
    balance = (0.55 * covered + 0.45 * min_side).clamp(0.0, 1.0)
    success = _success(env, info)
    one = torch.ones_like(success, dtype=torch.bool)

    return _pack(
        (
            (0.42 * path_score + 0.38 * z_score + 0.20 * uncovered_score).clamp(0.0, 1.0),
            (0.46 * covered + 0.24 * uncovered_score + 0.18 * z_score + 0.12 * valid_fraction).clamp(0.0, 1.0),
            (0.44 * balance + 0.26 * covered + 0.18 * uncovered_score + 0.12 * valid_fraction).clamp(0.0, 1.0),
            (0.48 * covered + 0.28 * min_side + 0.14 * all_covered + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            one,
            started | (covered >= 0.04),
            (covered >= 0.22) & (min_side >= 0.02),
            success,
        ),
        maintenance=(
            0.06 * path_score + 0.06 * z_score,
            0.22 * covered + 0.16 * uncovered_score + 0.08 * z_score - 0.02 * invalid.float(),
            0.30 * covered + 0.22 * balance + 0.12 * uncovered_score + 0.08 * valid_fraction - 0.03 * invalid.float(),
            0.36 * covered + 0.24 * min_side + 0.12 * valid_fraction + 0.10 * all_covered + 0.10 * success.float() - 0.03 * invalid.float(),
        ),
        safety=torch.zeros_like(covered),
    )


def _draw_triangle_v12(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v12: official-canvas coverage with low-friction gates."""
    tcp = _tcp(env)
    ref_bool = env.ref_dist.bool()
    ref = ref_bool.float()
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    min_side = torch.minimum(torch.minimum(side0, side1), side2)
    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_error = (tcp[:, 2] - target_z).abs()
    z_score = _phi(z_error, 48.0)
    dist = torch.linalg.norm(tcp[:, None, :2] - env.triangles[..., :2], dim=-1)
    nearest_path = dist.min(dim=1).values
    path_score = _phi(nearest_path, 22.0)
    nearest_uncovered = (dist + ref * 10.0).min(dim=1).values
    uncovered_score = _phi(nearest_uncovered, 28.0)
    drawn_mask = env.dots_dist > -1
    drawn_count = drawn_mask.sum(dim=1).to(dtype=torch.float32)
    valid_count = torch.where(drawn_mask, env.dots_dist.clamp(min=0).to(dtype=torch.float32), torch.zeros_like(env.dots_dist, dtype=torch.float32)).sum(dim=1)
    valid_fraction = torch.where(drawn_count > 0, valid_count / drawn_count.clamp(min=1.0), torch.ones_like(drawn_count)).clamp(0.0, 1.0)
    started = drawn_count > 0
    success = _success(env, info)
    one = torch.ones_like(success, dtype=torch.bool)
    balance = (0.50 * covered + 0.50 * min_side).clamp(0.0, 1.0)
    return _pack(
        (
            (0.42 * path_score + 0.38 * z_score + 0.20 * uncovered_score).clamp(0.0, 1.0),
            (0.42 * covered + 0.26 * uncovered_score + 0.18 * z_score + 0.14 * valid_fraction).clamp(0.0, 1.0),
            (0.44 * balance + 0.24 * covered + 0.20 * uncovered_score + 0.12 * valid_fraction).clamp(0.0, 1.0),
            (0.50 * covered + 0.28 * min_side + 0.12 * valid_fraction + 0.10 * success.float()).clamp(0.0, 1.0),
        ),
        (
            one,
            started | (covered >= 0.02) | (path_score >= 0.35),
            (covered >= 0.16) | (min_side >= 0.02),
            success,
        ),
        maintenance=(
            0.06 * path_score + 0.06 * z_score,
            0.26 * covered + 0.16 * uncovered_score + 0.08 * z_score,
            0.34 * covered + 0.24 * balance + 0.12 * uncovered_score + 0.08 * valid_fraction,
            0.40 * covered + 0.24 * min_side + 0.12 * valid_fraction + 0.10 * success.float(),
        ),
        safety=torch.zeros_like(covered),
    )

def _trifinger_rotate_cube_level4(env: Any, info: dict[str, Any]) -> RuleInputs:
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    goal_pos = env.obj_goal.pose.p
    tip_obj = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1).mean(dim=1)
    pos_error = _distance(obj_pos, goal_pos)
    rot_error = _quat_error(env.obj.pose.q, env.obj_goal.pose.q)
    pos_progress = _phi(pos_error, 8.0)
    rot_progress = 1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)
    pose_progress = 0.5 * pos_progress + 0.5 * rot_progress
    lifted_or_close = (obj_pos[:, 2] > 0.05) | (pos_error <= 0.12)
    near_goal_pose = (pos_error <= env.goal_radius * 2.5) & (rot_error <= 0.50)
    success = _success(env, info)
    zeros = torch.zeros_like(tip_obj)
    return _pack(
        (
            _phi(tip_obj, 8.0),
            pos_progress,
            pose_progress,
            success.float(),
        ),
        (
            tip_obj <= 0.08,
            lifted_or_close,
            near_goal_pose,
            success,
        ),
        maintenance=(zeros, zeros, zeros, zeros),
        safety=_safety(env.obj),
    )


def _trifinger_rotate_cube_level4_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TriFingerRotateCubeLevel4 v2: split contact, lift/position, rotation."""
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    obj_q = env.obj.pose.q
    goal_pos = env.obj_goal.pose.p
    goal_q = env.obj_goal.pose.q

    tip_dists = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    mean_tip = tip_dists.mean(dim=1)
    max_tip = tip_dists.max(dim=1).values
    contact_score = (
        0.65 * _phi(mean_tip, 10.0) + 0.35 * _phi(max_tip, 8.0)
    ).clamp(0.0, 1.0)
    all_near = max_tip <= 0.14

    pos_error = _distance(obj_pos, goal_pos)
    xy_error = _distance(obj_pos, goal_pos, xy=True)
    z_error = (obj_pos[:, 2] - goal_pos[:, 2]).abs()
    xy_score = _phi(xy_error, 10.0)
    z_score = _phi(z_error, 18.0)
    pos_score = _phi(pos_error, 9.0)
    init_z = torch.full_like(obj_pos[:, 2], env.size / 2 + 0.005)
    lift_denom = (goal_pos[:, 2] - init_z).abs().clamp(min=0.02)
    lift_score = torch.clamp((obj_pos[:, 2] - init_z) / lift_denom, 0.0, 1.0)

    rot_error = _quat_error(obj_q, goal_q)
    rot_score = (1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    pose_score = (
        0.40 * pos_score + 0.30 * rot_score + 0.20 * contact_score + 0.10 * lift_score
    ).clamp(0.0, 1.0)

    near_position = pos_error <= 0.07
    near_pose = (pos_error <= env.goal_radius * 2.0) & (rot_error <= 0.35)
    success = _success(env, info)
    zeros = torch.zeros_like(mean_tip)
    lifted = obj_pos[:, 2] >= torch.minimum(goal_pos[:, 2] - 0.01, init_z + 0.035)
    return _pack(
        (
            contact_score,
            (
                0.35 * xy_score
                + 0.35 * z_score
                + 0.15 * lift_score
                + 0.15 * contact_score
            ).clamp(0.0, 1.0),
            pose_score,
            (0.50 * success.float() + 0.30 * pos_score + 0.20 * rot_score).clamp(0.0, 1.0),
        ),
        (
            all_near,
            near_position | (lifted & (xy_error <= 0.10)),
            near_pose,
            success,
        ),
        maintenance=(
            zeros,
            0.03 * contact_score - 0.04 * (obj_pos[:, 2] < 0.025).float(),
            0.04 * contact_score + 0.05 * pos_score - 0.04 * (obj_pos[:, 2] < 0.025).float(),
            0.05 * (near_pose | success).float(),
        ),
        safety=_safety(env.obj),
    )


def _trifinger_rotate_cube_level4_v3(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TriFingerRotateCubeLevel4 v3: stronger orientation stage after position reach."""
    base = _trifinger_rotate_cube_level4_v2(env, info)
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    goal_pos = env.obj_goal.pose.p
    tip_dists = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    mean_tip = tip_dists.mean(dim=1)
    max_tip = tip_dists.max(dim=1).values
    contact_score = (0.65 * _phi(mean_tip, 10.0) + 0.35 * _phi(max_tip, 8.0)).clamp(0.0, 1.0)
    pos_error = _distance(obj_pos, goal_pos)
    rot_error = _quat_error(env.obj.pose.q, env.obj_goal.pose.q)
    pos_score = _phi(pos_error, 10.0)
    rot_score = (1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 2] = (0.60 * rot_score + 0.25 * pos_score + 0.15 * contact_score).clamp(0.0, 1.0)
    potentials[:, 3] = (0.60 * success.float() + 0.25 * pos_score + 0.15 * rot_score).clamp(0.0, 1.0)
    gates[:, 1] = pos_error <= 0.08
    gates[:, 2] = (pos_error <= env.goal_radius * 2.0) & (rot_error <= 0.25)
    gates[:, 3] = success
    maintenance[:, 2] = 0.08 * contact_score + 0.10 * pos_score + 0.14 * rot_score - 0.05 * (obj_pos[:, 2] < 0.025).float()
    maintenance[:, 3] = 0.08 * ((pos_error <= env.goal_radius * 2.0) & (rot_error <= 0.25)).float() + 0.06 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)



def _trifinger_rotate_cube_level4_v4(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TriFingerRotateCubeLevel4 v4: tight final pose funnel, keep long rollout."""
    base = _trifinger_rotate_cube_level4_v3(env, info)
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    goal_pos = env.obj_goal.pose.p
    tip_dists = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    mean_tip = tip_dists.mean(dim=1)
    max_tip = tip_dists.max(dim=1).values
    contact_score = (0.65 * _phi(mean_tip, 10.0) + 0.35 * _phi(max_tip, 8.0)).clamp(0.0, 1.0)
    pos_error = _distance(obj_pos, goal_pos)
    rot_error = _quat_error(env.obj.pose.q, env.obj_goal.pose.q)
    pos_score = _phi(pos_error, 10.0)
    tight_pos_score = _phi(pos_error, 35.0)
    rot_score = (1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_rot_score = (1.0 - torch.clamp(rot_error / 0.10, 0.0, 1.0)).clamp(0.0, 1.0)
    success = _success(env, info)
    near_pose_loose = (pos_error <= env.goal_radius * 2.5) & (rot_error <= 0.35)
    near_pose_tight = (pos_error <= env.goal_radius * 1.4) & (rot_error <= 0.16)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    potentials[:, 2] = (0.44 * rot_score + 0.26 * pos_score + 0.18 * contact_score + 0.12 * tight_rot_score).clamp(0.0, 1.0)
    potentials[:, 3] = (0.38 * tight_pos_score + 0.38 * tight_rot_score + 0.14 * near_pose_tight.float() + 0.10 * success.float()).clamp(0.0, 1.0)
    gates[:, 1] = pos_error <= 0.09
    gates[:, 2] = near_pose_loose
    gates[:, 3] = success
    maintenance[:, 2] = 0.10 * contact_score + 0.12 * pos_score + 0.18 * rot_score - 0.04 * (obj_pos[:, 2] < 0.025).float()
    maintenance[:, 3] = 0.16 * tight_pos_score + 0.16 * tight_rot_score + 0.10 * near_pose_tight.float() + 0.08 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _trifinger_rotate_cube_level4_v5(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TriFingerRotateCubeLevel4 v5: native-dense-like relative pose shaping."""
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    obj_q = env.obj.pose.q
    goal_pos = env.obj_goal.pose.p
    goal_q = env.obj_goal.pose.q
    tip_dists = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    mean_tip = tip_dists.mean(dim=1)
    max_tip = tip_dists.max(dim=1).values
    finger_score = (0.65 * _phi(mean_tip, 8.0) + 0.35 * _phi(max_tip, 7.0)).clamp(0.0, 1.0)
    pos_error = _distance(obj_pos, goal_pos)
    init_xyz = torch.tensor([0.0, 0.0, 0.032], dtype=obj_pos.dtype, device=env.device).reshape(1, 3)
    init_dist = torch.linalg.norm(init_xyz - goal_pos, dim=1).clamp_min(0.03)
    pos_progress = ((init_dist - pos_error) / init_dist).clamp(0.0, 1.0)
    z_error = torch.abs(obj_pos[:, 2] - goal_pos[:, 2])
    init_z = torch.full_like(obj_pos[:, 2], 0.032)
    init_z_dist = torch.abs(init_z - goal_pos[:, 2]).clamp_min(0.02)
    z_progress = ((init_z_dist - z_error) / init_z_dist).clamp(0.0, 1.0)
    goal_dir = (goal_pos - obj_pos) / pos_error[:, None].clamp_min(1e-6)
    directed_speed = (env.obj.linear_velocity * goal_dir).sum(dim=1)
    positive_motion = torch.clamp(torch.tanh(6.0 * directed_speed), 0.0, 1.0)
    rot_error = _quat_error(obj_q, goal_q)
    rot_score = (1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_pos_score = _phi(pos_error, 35.0)
    tight_rot_score = (1.0 - torch.clamp(rot_error / 0.10, 0.0, 1.0)).clamp(0.0, 1.0)
    success = _success(env, info)
    near_position = (pos_error <= 0.12) | (pos_progress >= 0.45) | (z_progress >= 0.45)
    near_pose = (pos_error <= env.goal_radius * 3.0) & (rot_error <= 0.55)
    native_pose_score = (
        0.30 * pos_progress
        + 0.25 * z_progress
        + 0.25 * rot_score
        + 0.12 * finger_score
        + 0.08 * positive_motion
    ).clamp(0.0, 1.0)
    zeros = torch.zeros_like(mean_tip)
    low_object = (obj_pos[:, 2] < 0.025).float()
    return _pack(
        (
            finger_score,
            (0.34 * pos_progress + 0.30 * z_progress + 0.20 * finger_score + 0.16 * positive_motion).clamp(0.0, 1.0),
            native_pose_score,
            (0.34 * tight_pos_score + 0.34 * tight_rot_score + 0.16 * native_pose_score + 0.16 * success.float()).clamp(0.0, 1.0),
        ),
        (
            (mean_tip <= 0.11) | (finger_score >= 0.55),
            near_position,
            near_pose,
            success,
        ),
        maintenance=(
            0.04 * finger_score,
            0.12 * pos_progress + 0.12 * z_progress + 0.08 * positive_motion + 0.04 * finger_score - 0.04 * low_object,
            0.18 * native_pose_score + 0.10 * rot_score + 0.08 * positive_motion - 0.04 * low_object,
            0.16 * tight_pos_score + 0.16 * tight_rot_score + 0.10 * native_pose_score + 0.08 * success.float(),
        ),
        safety=_safety(env.obj),
    )

def _trifinger_rotate_cube_level4_v6(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TriFingerRotateCubeLevel4 v6: wider pose funnel with sustained contact."""
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    obj_q = env.obj.pose.q
    goal_pos = env.obj_goal.pose.p
    goal_q = env.obj_goal.pose.q
    tip_dists = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    mean_tip = tip_dists.mean(dim=1)
    max_tip = tip_dists.max(dim=1).values
    finger_score = (0.62 * _phi(mean_tip, 8.0) + 0.38 * _phi(max_tip, 7.0)).clamp(0.0, 1.0)
    pos_error = _distance(obj_pos, goal_pos)
    init_xyz = torch.tensor([0.0, 0.0, 0.032], dtype=obj_pos.dtype, device=env.device).reshape(1, 3)
    init_dist = torch.linalg.norm(init_xyz - goal_pos, dim=1).clamp_min(0.03)
    pos_progress = ((init_dist - pos_error) / init_dist).clamp(0.0, 1.0)
    z_error = torch.abs(obj_pos[:, 2] - goal_pos[:, 2])
    init_z = torch.full_like(obj_pos[:, 2], 0.032)
    init_z_dist = torch.abs(init_z - goal_pos[:, 2]).clamp_min(0.02)
    z_progress = ((init_z_dist - z_error) / init_z_dist).clamp(0.0, 1.0)
    goal_dir = (goal_pos - obj_pos) / pos_error[:, None].clamp_min(1e-6)
    directed_speed = (env.obj.linear_velocity * goal_dir).sum(dim=1)
    positive_motion = torch.clamp(torch.tanh(5.0 * directed_speed), 0.0, 1.0)
    rot_error = _quat_error(obj_q, goal_q)
    rot_score = (1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_pos_score = _phi(pos_error, 30.0)
    tight_rot_score = (1.0 - torch.clamp(rot_error / 0.18, 0.0, 1.0)).clamp(0.0, 1.0)
    pose_score = (0.30 * pos_progress + 0.22 * z_progress + 0.22 * rot_score + 0.16 * finger_score + 0.10 * positive_motion).clamp(0.0, 1.0)
    success = _success(env, info)
    near_position = (pos_error <= 0.16) | (pos_progress >= 0.30) | (z_progress >= 0.30)
    near_pose = ((pos_error <= env.goal_radius * 4.0) & (rot_error <= 0.90)) | (pose_score >= 0.55)
    low_object = (obj_pos[:, 2] < 0.022).float()
    return _pack(
        (
            finger_score,
            (0.32 * pos_progress + 0.28 * z_progress + 0.24 * finger_score + 0.16 * positive_motion).clamp(0.0, 1.0),
            pose_score,
            (0.34 * tight_pos_score + 0.30 * tight_rot_score + 0.20 * pose_score + 0.16 * success.float()).clamp(0.0, 1.0),
        ),
        (
            (mean_tip <= 0.13) | (finger_score >= 0.42),
            near_position,
            near_pose,
            success,
        ),
        maintenance=(
            0.06 * finger_score,
            0.14 * pos_progress + 0.14 * z_progress + 0.08 * positive_motion + 0.06 * finger_score - 0.03 * low_object,
            0.22 * pose_score + 0.10 * rot_score + 0.08 * positive_motion - 0.03 * low_object,
            0.16 * tight_pos_score + 0.14 * tight_rot_score + 0.12 * pose_score + 0.08 * success.float(),
        ),
        safety=0.5 * _safety(env.obj),
    )


def _plug_charger_v10(env: Any, info: dict[str, Any]) -> RuleInputs:
    """PlugCharger v10: official dense reach/close/pose funnel with proxy carrying."""
    base = _plug_charger_v9(env, info)
    tcp_obj = _distance(_tcp(env), env.charger_base_pose.p)
    grasped = _grasp(env, env.charger)
    obj_goal, angle = env._compute_distance()
    rel = (env.goal_pose.inv() * env.charger.pose).p
    lateral = torch.linalg.norm(rel[:, 1:], dim=1)
    axis = rel[:, 0].abs()
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    carrying = grasped | ((tcp_obj <= 0.085) & (closed >= 0.25))
    reach_phi = _phi(tcp_obj, 10.0)
    coarse_dist = (1.0 - obj_goal / 0.22).clamp(0.0, 1.0)
    mid_dist = (1.0 - obj_goal / 0.10).clamp(0.0, 1.0)
    tight_dist = _phi(obj_goal, 120.0)
    lateral_phi = _phi(lateral, 55.0)
    axis_phi = _phi(axis, 45.0)
    angle_phi = (1.0 - torch.clamp(angle / 1.40, 0.0, 1.0)).clamp(0.0, 1.0)
    tight_angle_phi = (1.0 - torch.clamp(angle / 0.25, 0.0, 1.0)).clamp(0.0, 1.0)
    coarse_pose = (0.30 * coarse_dist + 0.25 * lateral_phi + 0.17 * axis_phi + 0.18 * angle_phi + 0.10 * carrying.float()).clamp(0.0, 1.0)
    preinsert = (0.34 * mid_dist + 0.28 * lateral_phi + 0.22 * tight_angle_phi + 0.16 * axis_phi).clamp(0.0, 1.0)
    terminal = (0.42 * tight_dist + 0.24 * lateral_phi + 0.24 * tight_angle_phi + 0.10 * axis_phi).clamp(0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = (tcp_obj <= 0.085) | (reach_phi >= 0.45)
    gates[:, 1] = grasped | ((tcp_obj <= 0.070) & (closed >= 0.35))
    gates[:, 2] = carrying | (coarse_pose >= 0.25)
    gates[:, 3] = (obj_goal <= 0.090) | (preinsert >= 0.35)
    gates[:, 4] = success
    potentials[:, 1] = (0.30 * reach_phi + 0.48 * closed + 0.14 * grasped.float() + 0.08 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 2] = coarse_pose
    potentials[:, 3] = preinsert
    potentials[:, 4] = (0.62 * terminal + 0.26 * preinsert + 0.12 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.10 * reach_phi + 0.18 * closed + 0.10 * grasped.float()
    maintenance[:, 2] = 0.24 * coarse_pose + 0.12 * carrying.float()
    maintenance[:, 3] = 0.30 * preinsert + 0.10 * mid_dist
    maintenance[:, 4] = 0.34 * terminal + 0.12 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _stack_pyramid_v17(env: Any, info: dict[str, Any]) -> RuleInputs:
    """StackPyramid v17: official base/support geometry with an explicit blue approach stage."""
    base = _stack_pyramid_v16(env, info)
    grasp_a = _grasp(env, env.cubeA)
    grasp_c = _grasp(env, env.cubeC)
    tcp_a = _distance(_tcp(env), env.cubeA.pose.p)
    tcp_c = _distance(_tcp(env), env.cubeC.pose.p)
    reach_c = _phi(tcp_c, 7.0)
    beside_threshold = torch.linalg.norm(2 * env.cube_half_size[:2]) + 0.005
    red_green_xy = _distance(env.cubeA.pose.p, env.cubeB.pose.p, xy=True)
    base_loose = red_green_xy <= beside_threshold + 0.045
    base_score = (1.0 - red_green_xy / (beside_threshold + 0.060)).clamp(0.0, 1.0)
    blue_a_xy = _distance(env.cubeC.pose.p, env.cubeA.pose.p, xy=True)
    blue_b_xy = _distance(env.cubeC.pose.p, env.cubeB.pose.p, xy=True)
    support_xy_error = torch.maximum(
        blue_a_xy - beside_threshold, blue_b_xy - beside_threshold
    ).clamp_min(0.0)
    support_xy_phi = (1.0 - support_xy_error / 0.070).clamp(0.0, 1.0)
    min_top_gap = torch.minimum(
        env.cubeC.pose.p[:, 2] - env.cubeA.pose.p[:, 2],
        env.cubeC.pose.p[:, 2] - env.cubeB.pose.p[:, 2],
    )
    support_score = (0.65 * support_xy_phi + 0.20 * (min_top_gap > 0.01).float() + 0.15 * base_score).clamp(0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 3] = base_loose & (~grasp_a)
    gates[:, 4] = (support_score >= 0.30) | (reach_c >= 0.55) | grasp_c
    gates[:, 5] = (support_score >= 0.48) & grasp_c
    gates[:, 6] = success
    potentials[:, 3] = (0.62 * reach_c + 0.20 * base_score + 0.18 * grasp_c.float()).clamp(0.0, 1.0)
    potentials[:, 4] = (0.55 * support_score + 0.25 * reach_c + 0.20 * grasp_c.float()).clamp(0.0, 1.0)
    potentials[:, 5] = (0.62 * support_score + 0.22 * grasp_c.float() + 0.16 * success.float()).clamp(0.0, 1.0)
    potentials[:, 6] = (0.55 * support_score + 0.45 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 3] = 0.22 * reach_c + 0.16 * base_score + 0.10 * grasp_c.float()
    maintenance[:, 4] = 0.28 * support_score + 0.12 * reach_c + 0.10 * grasp_c.float()
    maintenance[:, 5] = 0.34 * support_score + 0.12 * grasp_c.float()
    maintenance[:, 6] = 0.18 * support_score + 0.12 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _fmb_v13(env: Any, info: dict[str, Any]) -> RuleInputs:
    """FMBAssembly1Easy v13: official bridge-to-goal progress without a hard regrasp bottleneck."""
    base = _fmb_v12(env, info)
    tcp = _tcp(env)
    grasp_pose = env.bridge.pose * env.bridge_grasp_offset
    tcp_grasp = _distance(tcp, grasp_pose.p)
    tcp_center = _distance(tcp, env.bridge.pose.p)
    tcp_bridge = torch.minimum(tcp_grasp, tcp_center)
    bridge_pos = env.bridge.pose.p
    goal_pos = env.goal_bridge_pose.p
    goal_dist = _distance(bridge_pos, goal_pos)
    goal_xy = _distance(bridge_pos, goal_pos, xy=True)
    goal_z = torch.abs(bridge_pos[:, 2] - goal_pos[:, 2])
    goal_angle = _quat_error(env.bridge.pose.q, env.goal_bridge_pose.q)
    grasped = _grasp(env, env.bridge)
    qlim = env.agent.robot.get_qlimits()[0, -1, 1].to(env.device) * 2
    open_width = (torch.sum(env.agent.robot.get_qpos()[:, -2:], dim=1) / qlim).clamp(0.0, 1.0)
    closed = (1.0 - open_width).clamp(0.0, 1.0)
    lift = torch.clamp((bridge_pos[:, 2] - 0.024) / 0.070, 0.0, 1.0)
    carrying = grasped | ((tcp_bridge <= 0.14) & (closed >= 0.16)) | ((tcp_bridge <= 0.16) & (lift >= 0.05))
    goal_vec = goal_pos - bridge_pos
    goal_dir = goal_vec / torch.linalg.norm(goal_vec, dim=1, keepdim=True).clamp_min(1e-6)
    directed_speed = (env.bridge.linear_velocity * goal_dir).sum(dim=1)
    positive_motion = torch.clamp(torch.tanh(8.0 * directed_speed), 0.0, 1.0)
    speed = torch.linalg.norm(env.bridge.linear_velocity, dim=1)
    reach_phi = _phi(tcp_bridge, 12.0)
    long_goal_phi = (1.0 - goal_dist / 0.46).clamp(0.0, 1.0)
    mid_goal_phi = (1.0 - goal_dist / 0.22).clamp(0.0, 1.0)
    near_goal_phi = (1.0 - goal_dist / 0.075).clamp(0.0, 1.0)
    tight_goal_phi = _phi(goal_dist, 140.0)
    xy_phi = _phi(goal_xy, 14.0)
    z_phi = _phi(goal_z, 60.0)
    angle_phi = (1.0 - torch.clamp(goal_angle / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    pose_phi = (0.38 * mid_goal_phi + 0.24 * xy_phi + 0.20 * z_phi + 0.10 * angle_phi + 0.08 * carrying.float()).clamp(0.0, 1.0)
    fine_pose = (0.48 * tight_goal_phi + 0.22 * near_goal_phi + 0.20 * z_phi + 0.10 * xy_phi).clamp(0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = (reach_phi >= 0.30) | (tcp_bridge <= 0.13)
    gates[:, 1] = carrying | (closed >= 0.18)
    gates[:, 2] = carrying | (lift >= 0.04) | (positive_motion >= 0.20)
    gates[:, 3] = (goal_dist <= 0.30) | (pose_phi >= 0.25) | (positive_motion >= 0.25)
    gates[:, 4] = (goal_dist <= 0.12) | (fine_pose >= 0.35)
    gates[:, 5] = success
    potentials[:, 0] = reach_phi
    potentials[:, 1] = (0.34 * reach_phi + 0.24 * closed + 0.22 * grasped.float() + 0.20 * lift).clamp(0.0, 1.0)
    potentials[:, 2] = (0.42 * long_goal_phi + 0.22 * mid_goal_phi + 0.18 * lift + 0.12 * positive_motion + 0.06 * carrying.float()).clamp(0.0, 1.0)
    potentials[:, 3] = pose_phi
    potentials[:, 4] = fine_pose
    potentials[:, 5] = (0.60 * fine_pose + 0.24 * success.float() + 0.16 * _phi(speed, 8.0)).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.10 * reach_phi
    maintenance[:, 1] = 0.12 * reach_phi + 0.12 * closed + 0.08 * grasped.float() + 0.06 * lift
    maintenance[:, 2] = 0.24 * long_goal_phi + 0.14 * mid_goal_phi + 0.12 * lift + 0.10 * positive_motion
    maintenance[:, 3] = 0.26 * pose_phi + 0.10 * positive_motion + 0.08 * carrying.float()
    maintenance[:, 4] = 0.34 * fine_pose + 0.12 * near_goal_phi
    maintenance[:, 5] = 0.22 * fine_pose + 0.14 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _rotate_valve_level4_v9(env: Any, info: dict[str, Any]) -> RuleInputs:
    """RotateValveLevel4 v9: official directed rotation progress with a permissive near-full handoff."""
    base = _rotate_valve_level4_v8(env, info)
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_phi = _phi(contact_error, 11.0)
    contact = contact_error <= 0.14
    signed_rotation = (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    positive_motion = torch.clamp(torch.tanh(4.0 * directed_velocity), 0.0, 1.0)
    wrong_motion = torch.clamp(torch.tanh(-4.0 * directed_velocity), 0.0, 1.0)
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    mid_progress = torch.clamp(signed_rotation / (0.60 * env.success_threshold), 0.0, 1.0)
    near_full = progress >= 0.92
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = contact | (contact_phi >= 0.35)
    gates[:, 1] = contact | (contact_phi >= 0.42) | (progress >= 0.04)
    gates[:, 2] = near_full | ((progress >= 0.68) & (positive_motion >= 0.08)) | success
    gates[:, 3] = success
    potentials[:, 0] = contact_phi
    potentials[:, 1] = (0.44 * contact_phi + 0.34 * positive_motion + 0.22 * progress).clamp(0.0, 1.0)
    potentials[:, 2] = (0.66 * progress + 0.20 * mid_progress + 0.10 * positive_motion + 0.04 * contact.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.76 * progress + 0.16 * positive_motion + 0.08 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.08 * contact_phi
    maintenance[:, 1] = 0.16 * contact_phi + 0.22 * positive_motion + 0.10 * progress - 0.06 * wrong_motion
    maintenance[:, 2] = 1.20 * progress + 0.70 * positive_motion + 0.28 * mid_progress - 0.24 * wrong_motion
    maintenance[:, 3] = 0.36 * progress + 0.18 * positive_motion + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _two_robot_stack_cube_v15(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TwoRobotStackCube v15: official place/top/release reward with earlier top-cube approach."""
    base = _two_robot_stack_cube_v14(env, info)
    left_tcp = env.left_agent.tcp.pose.p
    right_tcp = env.right_agent.tcp.pose.p
    cube_a = env.cubeA.pose.p
    cube_b = env.cubeB.pose.p
    cube_b_goal = _distance(cube_b, env.goal_region.pose.p, xy=True)
    bottom_phi = _phi(cube_b_goal, 6.0)
    bottom_near = cube_b_goal <= 0.075
    top_goal = torch.hstack([cube_b[:, :2], (cube_b[:, 2] + env.cube_half_size[2] * 2)[:, None]])
    top_error = _distance(cube_a, top_goal)
    top_phi = _phi(top_error, 7.0)
    top_loose = top_error <= 0.075
    top_near = top_error <= 0.045
    cube_a_grasped = info.get("is_cubeA_grasped", env.left_agent.is_grasping(env.cubeA)).bool()
    cube_b_grasped = info.get("is_cubeB_grasped", env.right_agent.is_grasping(env.cubeB)).bool()
    cube_a_on_cube_b = info.get("is_cubeA_on_cubeB", top_error <= 0.012).bool()
    stacked_loose = (bottom_near | (bottom_phi >= 0.55)) & (top_loose | (top_phi >= 0.55))
    release_score = (torch.where(cube_a_grasped, 1.0 - cube_a_grasped.float(), torch.ones_like(bottom_phi)) + torch.where(cube_b_grasped, 1.0 - cube_b_grasped.float(), torch.ones_like(bottom_phi))) / 2.0
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 2] = (bottom_phi >= 0.42) | bottom_near | cube_a_grasped
    gates[:, 3] = stacked_loose | (top_phi >= 0.42)
    gates[:, 4] = success
    potentials[:, 2] = (0.62 * bottom_phi + 0.20 * top_phi + 0.10 * cube_a_grasped.float() + 0.08 * cube_b_grasped.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.68 * top_phi + 0.20 * bottom_phi + 0.12 * cube_a_on_cube_b.float()).clamp(0.0, 1.0)
    potentials[:, 4] = (0.50 * top_phi + 0.20 * bottom_phi + 0.18 * release_score + 0.12 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 2] = 0.28 * bottom_phi + 0.10 * top_phi + 0.08 * cube_a_grasped.float()
    maintenance[:, 3] = 0.36 * top_phi + 0.14 * bottom_phi + 0.10 * cube_a_on_cube_b.float()
    maintenance[:, 4] = 0.20 * top_phi + 0.12 * release_score + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)


def _draw_triangle_v13(env: Any, info: dict[str, Any]) -> RuleInputs:
    """DrawTriangle v13: official reference-point coverage and nearest-uncovered guidance."""
    base = _draw_triangle_v12(env, info)
    tcp = _tcp(env)
    ref = env.ref_dist.float()
    covered = ref.mean(dim=1).clamp(0.0, 1.0)
    side0 = ref[:, :51].mean(dim=1).clamp(0.0, 1.0)
    side1 = ref[:, 51:102].mean(dim=1).clamp(0.0, 1.0)
    side2 = ref[:, 102:].mean(dim=1).clamp(0.0, 1.0)
    min_side = torch.minimum(torch.minimum(side0, side1), side2)
    dist = torch.linalg.norm(tcp[:, None, :2] - env.triangles[..., :2], dim=-1)
    nearest_uncovered = (dist + ref * 10.0).min(dim=1).values
    uncovered_score = _phi(nearest_uncovered, 32.0)
    target_z = env.CANVAS_THICKNESS + env.DOT_THICKNESS / 2
    z_score = _phi((tcp[:, 2] - target_z).abs(), 48.0)
    drawn_mask = env.dots_dist > -1
    valid_count = torch.where(drawn_mask, env.dots_dist.clamp(min=0).float(), torch.zeros_like(env.dots_dist)).sum(dim=1)
    drawn_count = drawn_mask.sum(dim=1).float()
    valid_fraction = torch.where(drawn_count > 0, valid_count / drawn_count.clamp(min=1.0), torch.ones_like(drawn_count)).clamp(0.0, 1.0)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 0] = torch.ones_like(success, dtype=torch.bool)
    gates[:, 1] = (covered >= 0.01) | (uncovered_score >= 0.35)
    gates[:, 2] = (covered >= 0.08) | (min_side >= 0.015) | (uncovered_score >= 0.45)
    gates[:, 3] = success
    potentials[:, 0] = (0.60 * uncovered_score + 0.30 * z_score + 0.10 * valid_fraction).clamp(0.0, 1.0)
    potentials[:, 1] = (0.46 * covered + 0.30 * uncovered_score + 0.16 * z_score + 0.08 * valid_fraction).clamp(0.0, 1.0)
    potentials[:, 2] = (0.48 * covered + 0.22 * min_side + 0.20 * uncovered_score + 0.10 * valid_fraction).clamp(0.0, 1.0)
    potentials[:, 3] = (0.60 * covered + 0.25 * min_side + 0.15 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 0] = 0.08 * uncovered_score + 0.05 * z_score
    maintenance[:, 1] = 0.28 * covered + 0.20 * uncovered_score + 0.10 * valid_fraction
    maintenance[:, 2] = 0.40 * covered + 0.28 * min_side + 0.18 * uncovered_score + 0.10 * valid_fraction
    maintenance[:, 3] = 0.46 * covered + 0.30 * min_side + 0.12 * valid_fraction + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, torch.zeros_like(covered))


def _trifinger_rotate_cube_level4_v7(env: Any, info: dict[str, Any]) -> RuleInputs:
    """TriFingerRotateCubeLevel4 v7: official rotation/pose progress with an earlier final funnel."""
    base = _trifinger_rotate_cube_level4_v6(env, info)
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    obj_q = env.obj.pose.q
    goal_pos = env.obj_goal.pose.p
    goal_q = env.obj_goal.pose.q
    tip_dists = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    mean_tip = tip_dists.mean(dim=1)
    max_tip = tip_dists.max(dim=1).values
    finger_score = (0.62 * _phi(mean_tip, 8.0) + 0.38 * _phi(max_tip, 7.0)).clamp(0.0, 1.0)
    pos_error = _distance(obj_pos, goal_pos)
    rot_error = _quat_error(obj_q, goal_q)
    pos_score = _phi(pos_error, 9.0)
    rot_score = (1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    near_pose = (pos_error <= env.goal_radius * 3.5) & (rot_error <= 0.75)
    success = _success(env, info)
    potentials = base.potentials.clone()
    gates = base.gates.clone()
    maintenance = base.maintenance.clone()
    gates[:, 1] = (pos_error <= 0.11) | (pos_score >= 0.46)
    gates[:, 2] = near_pose | (pos_score >= 0.62)
    gates[:, 3] = success
    potentials[:, 1] = (0.40 * pos_score + 0.28 * finger_score + 0.20 * rot_score + 0.12 * pos_score).clamp(0.0, 1.0)
    potentials[:, 2] = (0.44 * rot_score + 0.32 * pos_score + 0.16 * finger_score + 0.08 * near_pose.float()).clamp(0.0, 1.0)
    potentials[:, 3] = (0.48 * rot_score + 0.30 * pos_score + 0.12 * finger_score + 0.10 * success.float()).clamp(0.0, 1.0)
    maintenance[:, 1] = 0.12 * finger_score + 0.12 * pos_score + 0.08 * rot_score
    maintenance[:, 2] = 0.20 * rot_score + 0.16 * pos_score + 0.10 * finger_score
    maintenance[:, 3] = 0.24 * rot_score + 0.18 * pos_score + 0.10 * success.float()
    return RuleInputs(potentials, gates, maintenance, base.safety_penalty)



def _rotate_valve_half_turn(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Frame-VLM v1: contact the valve ring, then complete the half-turn."""
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_score = _phi(contact_error, 10.0)
    contact = contact_error <= 0.10

    signed_rotation = (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    motion_score = torch.clamp(torch.tanh(5.0 * directed_velocity), 0.0, 1.0)
    started = contact & ((directed_velocity > 0.02) | (progress > 0.02))
    success = _success(env, info)
    zeros = torch.zeros_like(progress)
    return _pack(
        (
            (0.80 * contact_score + 0.20 * motion_score).clamp(0.0, 1.0),
            (0.78 * progress + 0.14 * motion_score + 0.08 * contact_score).clamp(0.0, 1.0),
        ),
        (started, success),
        maintenance=(zeros, 0.08 * contact.float() + 0.04 * motion_score),
    )


def _trifinger_replacement(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Frame-VLM v1: secure/lift, position, then match orientation."""
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    goal_pos = env.obj_goal.pose.p
    tip_dist = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    mean_tip = tip_dist.mean(dim=1)
    max_tip = tip_dist.max(dim=1).values
    contact_score = (0.65 * _phi(mean_tip, 10.0) + 0.35 * _phi(max_tip, 8.0)).clamp(0.0, 1.0)
    contact = max_tip <= 0.14

    initial_z = torch.full_like(obj_pos[:, 2], env.size / 2 + 0.005)
    lift_score = torch.clamp((obj_pos[:, 2] - initial_z) / 0.04, 0.0, 1.0)
    lifted = obj_pos[:, 2] >= initial_z + 0.008
    pos_error = _distance(obj_pos, goal_pos)
    pos_score = _phi(pos_error, 9.0)
    rot_error = _quat_error(env.obj.pose.q, env.obj_goal.pose.q)
    rot_score = (1.0 - torch.clamp(rot_error / torch.pi, 0.0, 1.0)).clamp(0.0, 1.0)
    near_position = pos_error <= env.goal_radius * 2.0
    success = _success(env, info)
    below_table = obj_pos[:, 2] < env.size / 2 - 0.005
    return _pack(
        (
            (0.70 * contact_score + 0.30 * lift_score).clamp(0.0, 1.0),
            (0.72 * pos_score + 0.18 * contact_score + 0.10 * lift_score).clamp(0.0, 1.0),
            (0.68 * rot_score + 0.24 * pos_score + 0.08 * contact_score).clamp(0.0, 1.0),
        ),
        (contact & lifted, near_position, success),
        maintenance=(
            -0.04 * below_table.float(),
            0.05 * contact_score - 0.05 * below_table.float(),
            0.08 * pos_score + 0.04 * contact_score - 0.05 * below_table.float(),
        ),
        safety=_safety(env.obj),
    )



def _rotate_valve_half_turn_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Frame-VLM v2: maintain directed velocity throughout the half-turn."""
    tips = _tip_positions(env.agent)
    valve_xy = env.valve_link.pose.p[:, :2]
    tip_radius = torch.linalg.norm(tips[..., :2] - valve_xy[:, None, :], dim=-1)
    desired_radius = env.capsule_lens[:, None] - env.capsule_offset
    contact_error = torch.linalg.norm(tip_radius - desired_radius, dim=-1)
    contact_score = _phi(contact_error, 10.0)
    contact = contact_error <= 0.10

    angle_delta = (env.valve.qpos - env.rest_qpos)[:, 0]
    signed_rotation = angle_delta * env.rotate_direction
    directed_velocity = env.valve.qvel[:, 0] * env.rotate_direction
    progress = torch.clamp(signed_rotation / env.success_threshold, 0.0, 1.0)
    motion_score = torch.clamp(directed_velocity / 0.50, 0.0, 1.0)
    started = contact & (directed_velocity > 0.10) & (torch.abs(angle_delta) > 0.10)
    completed = (signed_rotation >= env.success_threshold) & (directed_velocity > 0.10)
    stationary = directed_velocity <= 0.02
    wrong_motion = directed_velocity < -0.02
    return _pack(
        (
            (0.85 * contact_score + 0.15 * motion_score).clamp(0.0, 1.0),
            (0.88 * progress + 0.12 * motion_score).clamp(0.0, 1.0),
        ),
        (started, completed),
        maintenance=(
            0.04 * contact_score,
            0.12 * motion_score
            - 0.03 * stationary.float()
            - 0.06 * wrong_motion.float(),
        ),
    )


def _trifinger_replacement_v2(env: Any, info: dict[str, Any]) -> RuleInputs:
    """Frame-VLM v2: split broad position approach from tight stabilization."""
    tips = _tip_positions(env.agent)
    obj_pos = env.obj.pose.p
    goal_pos = env.obj_goal.pose.p
    tip_dist = torch.linalg.norm(tips - obj_pos[:, None, :], dim=-1)
    min_tip = tip_dist.min(dim=1).values
    tip_score = _phi(min_tip, 20.0)
    supported = min_tip <= 0.05

    cube_half_size = env.size / 2
    cube_z = obj_pos[:, 2]
    lift_denominator = torch.as_tensor(
        cube_half_size, device=cube_z.device, dtype=cube_z.dtype
    ).clamp_min(1e-6)
    lift_score = torch.clamp(
        (cube_z - cube_half_size) / lift_denominator,
        0.0,
        1.0,
    )
    lifted = cube_z > cube_half_size * 2
    grasp_lift = (0.55 * lift_score + 0.45 * tip_score).clamp(0.0, 1.0)
    grasp_lift = torch.where(
        lifted & ~supported, torch.zeros_like(grasp_lift), grasp_lift
    )
    grasp_lift = torch.where(
        lifted & supported, torch.ones_like(grasp_lift), grasp_lift
    )

    pos_error = _distance(obj_pos, goal_pos)
    goal_radius = env.goal_radius
    position_approach = (
        1.0 - torch.clamp((pos_error / goal_radius) * 0.8, 0.0, 1.0)
    )
    position_stable = (
        1.0 - torch.clamp((pos_error / goal_radius) * 0.2, 0.0, 1.0)
    )
    rot_error = _quat_error(env.obj.pose.q, env.obj_goal.pose.q)
    orientation_stable = (
        1.0 - torch.clamp(rot_error / 0.5, 0.0, 1.0)
    )

    above_table = cube_z > cube_half_size * 2
    below_table = cube_z < cube_half_size - 0.005
    keep_support = 0.05 * tip_score - 0.05 * below_table.float()
    return _pack(
        (grasp_lift, position_approach, position_stable, orientation_stable),
        (
            above_table & supported & (pos_error < 0.20),
            above_table & (pos_error < goal_radius * 0.30),
            above_table & (pos_error < goal_radius * 0.10),
            above_table
            & (pos_error < goal_radius * 0.10)
            & (rot_error < 0.10),
        ),
        maintenance=(
            keep_support,
            keep_support,
            keep_support,
            keep_support + 0.04 * position_stable,
        ),
        safety=_safety(env.obj),
    )



def _inhand_signals(env: Any, info: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    tip_dist = info.get("obj_tip_dist")
    if tip_dist is None:
        tips = _tip_positions(env.agent)
        tip_dist = torch.linalg.norm(tips - env.obj.pose.p[:, None, :], dim=-1)
    mean_tip = tip_dist.mean(dim=1)
    max_tip = tip_dist.max(dim=1).values
    contact_score = (0.65 * _phi(mean_tip, 12.0) + 0.35 * _phi(max_tip, 9.0)).clamp(0.0, 1.0)
    stable_contact = max_tip <= 0.12
    fallen = info.get("obj_fall", env.obj.pose.p[:, 2] < env.hand_init_height - 0.05).bool()
    speed = info.get("obj_vel")
    if speed is None:
        speed = torch.linalg.norm(env.obj.get_linear_velocity(), dim=-1)
    stable_score = _phi(speed, 6.0)
    step_angle = info.get("rotation_angle", torch.zeros_like(speed)).clamp(min=0.0)
    motion_score = torch.clamp(step_angle / (torch.pi / 20), 0.0, 1.0)
    progress = torch.clamp(env.cum_rotation_angle / env.success_threshold, 0.0, 1.0)
    success = _success(env, info)
    safety = fallen.float() + torch.clamp((speed - 1.0) / 2.0, min=0.0, max=1.0)
    return contact_score, stable_contact, fallen, stable_score, motion_score, progress, success, safety


def _rotate_single_inhand_level1(env: Any, info: dict[str, Any]) -> RuleInputs:
    contact, stable, fallen, stable_score, motion, progress, success, safety = _inhand_signals(env, info)
    valid = ~fallen
    started = valid & stable & (env.cum_rotation_angle >= 0.10)
    return _pack(
        (
            (0.55 * contact + 0.25 * stable_score + 0.20 * motion).clamp(0.0, 1.0),
            (0.72 * progress + 0.18 * contact + 0.10 * motion).clamp(0.0, 1.0),
        ),
        (started, success),
        maintenance=(
            -0.10 * fallen.float(),
            0.06 * contact + 0.04 * stable_score - 0.10 * fallen.float(),
        ),
        safety=safety,
    )


def _rotate_single_inhand_level2(env: Any, info: dict[str, Any]) -> RuleInputs:
    contact, _, fallen, stable_score, motion, progress, success, safety = _inhand_signals(env, info)
    return _pack(
        ((0.68 * progress + 0.16 * contact + 0.10 * motion + 0.06 * stable_score).clamp(0.0, 1.0),),
        (success,),
        maintenance=(-0.10 * fallen.float(),),
        safety=safety,
    )


def _rotate_single_inhand_level3(env: Any, info: dict[str, Any]) -> RuleInputs:
    contact, stable, fallen, stable_score, motion, progress, success, safety = _inhand_signals(env, info)
    valid = ~fallen
    started = valid & stable & (env.cum_rotation_angle >= 0.10)
    rotating = valid & (env.cum_rotation_angle >= 0.50)
    return _pack(
        (
            (0.72 * contact + 0.28 * stable_score).clamp(0.0, 1.0),
            (0.52 * motion + 0.30 * progress + 0.18 * contact).clamp(0.0, 1.0),
            (0.76 * progress + 0.14 * contact + 0.10 * motion).clamp(0.0, 1.0),
        ),
        (started, rotating, success),
        maintenance=(
            -0.10 * fallen.float(),
            0.05 * contact - 0.10 * fallen.float(),
            0.08 * contact + 0.04 * stable_score - 0.10 * fallen.float(),
        ),
        safety=safety,
    )

_EXTRACTORS: dict[str, Extractor] = {
    "PickCube-v1": _pick_cube,
    "PushCube-v1": _push_cube,
    "PullCube-v1": _pull_cube,
    "LiftPegUpright-v1": _lift_peg,
    "PegInsertionSide-v1": _peg_insertion,
    "PlugCharger-v1": _plug_charger,
    "PokeCube-v1": _poke_cube,
    "PullCubeTool-v1": _pull_cube_tool,
    "PushT-v1": _push_t,
    "RollBall-v1": _roll_ball,
    "StackCube-v1": _stack_cube,
    "StackPyramid-v1": _stack_pyramid,
    "PickSingleYCB-v1": _pick_single_ycb,
    "PlaceSphere-v1": _place_sphere,
    "FMBAssembly1Easy-v1": _fmb,
    "RotateValveLevel4-v1": _rotate_valve_level4,
    "TwoRobotStackCube-v1": _two_robot_stack_cube,
    "DrawTriangle-v1": _draw_triangle,
    "TriFingerRotateCubeLevel4-v1": _trifinger_rotate_cube_level4,
    "RotateValveLevel1-v1": _rotate_valve_half_turn,
    "RotateValveLevel2-v1": _rotate_valve_half_turn,
    "TriFingerRotateCubeLevel1-v1": _trifinger_replacement,
    "TriFingerRotateCubeLevel3-v1": _trifinger_replacement,
    "RotateSingleObjectInHandLevel1-v1": _rotate_single_inhand_level1,
    "RotateSingleObjectInHandLevel2-v1": _rotate_single_inhand_level2,
    "RotateSingleObjectInHandLevel3-v1": _rotate_single_inhand_level3,
}


def _select_extractors() -> tuple[dict[str, Extractor], dict[str, str]]:
    variant = os.environ.get("OURS_RULE_VARIANT", "rule_v1").strip().lower()
    extractors = dict(_EXTRACTORS)
    versions = {env_id: "rule_v1" for env_id in extractors}
    if variant in {"", "rule_v1", "v1"}:
        return extractors, versions
    if variant in {"pusht_v2", "push_t_v2", "rule_v2_pusht"}:
        extractors["PushT-v1"] = _push_t_v2
        versions["PushT-v1"] = "rule_v2_pusht"
        return extractors, versions
    if variant in {
        "stackpyramid_v2",
        "stack_pyramid_v2",
        "rule_v2_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v2
        versions["StackPyramid-v1"] = "rule_v2_stackpyramid"
        return extractors, versions
    if variant in {
        "stackpyramid_v3",
        "stack_pyramid_v3",
        "rule_v3_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v3
        versions["StackPyramid-v1"] = "rule_v3_stackpyramid"
        return extractors, versions
    if variant in {
        "stackpyramid_v4",
        "stack_pyramid_v4",
        "rule_v4_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v4
        versions["StackPyramid-v1"] = "rule_v4_stackpyramid"
        return extractors, versions
    if variant in {
        "stackpyramid_v5",
        "stack_pyramid_v5",
        "rule_v5_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v5
        versions["StackPyramid-v1"] = "rule_v5_stackpyramid"
        return extractors, versions
    if variant in {
        "stackpyramid_v6",
        "stack_pyramid_v6",
        "rule_v6_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v6
        versions["StackPyramid-v1"] = "rule_v6_stackpyramid"
        return extractors, versions
    if variant in {
        "stackpyramid_v7",
        "stack_pyramid_v7",
        "rule_v7_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v7
        versions["StackPyramid-v1"] = "rule_v7_stackpyramid"
        return extractors, versions
    if variant in {
        "stackpyramid_v8",
        "stack_pyramid_v8",
        "rule_v8_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v8
        versions["StackPyramid-v1"] = "rule_v8_stackpyramid"
        return extractors, versions
    if variant in {
        "stackpyramid_v9",
        "stack_pyramid_v9",
        "rule_v9_stackpyramid",
    }:
        extractors["StackPyramid-v1"] = _stack_pyramid_v9
        versions["StackPyramid-v1"] = "rule_v9_stackpyramid"
        return extractors, versions
    if variant in {"stackpyramid_v10", "stack_pyramid_v10", "rule_v10_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v10
        versions["StackPyramid-v1"] = "rule_v10_stackpyramid"
        return extractors, versions
    if variant in {"stackpyramid_v11", "stack_pyramid_v11", "rule_v11_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v11
        versions["StackPyramid-v1"] = "rule_v11_stackpyramid"
        return extractors, versions
    if variant in {"stackpyramid_v12", "stack_pyramid_v12", "rule_v12_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v12
        versions["StackPyramid-v1"] = "rule_v12_stackpyramid"
        return extractors, versions
    if variant in {"stackpyramid_v13", "stack_pyramid_v13", "rule_v13_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v13
        versions["StackPyramid-v1"] = "rule_v13_stackpyramid"
        return extractors, versions
    if variant in {"stackpyramid_v14", "stack_pyramid_v14", "rule_v14_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v14
        versions["StackPyramid-v1"] = "rule_v14_stackpyramid"
        return extractors, versions
    if variant in {"stackpyramid_v15", "stack_pyramid_v15", "rule_v15_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v15
        versions["StackPyramid-v1"] = "rule_v15_stackpyramid"
        return extractors, versions
    if variant in {"stackpyramid_v16", "stack_pyramid_v16", "rule_v16_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v16
        versions["StackPyramid-v1"] = "rule_v16_stackpyramid"
        return extractors, versions
    if variant in {"peg_plug_v2", "rule_v2_peg_plug"}:
        replacements = {
            "PegInsertionSide-v1": (_peg_insertion_v2, "rule_v2_peginsertion"),
            "PlugCharger-v1": (_plug_charger_v2, "rule_v2_plugcharger"),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"peg_plug_v3", "rule_v3_peg_plug"}:
        replacements = {
            "PegInsertionSide-v1": (_peg_insertion_v3, "rule_v3_peginsertion"),
            "PlugCharger-v1": (_plug_charger_v3, "rule_v3_plugcharger"),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"plug_charger_v4", "rule_v4_plug_charger"}:
        extractors["PlugCharger-v1"] = _plug_charger_v4
        versions["PlugCharger-v1"] = "rule_v4_plugcharger"
        return extractors, versions
    if variant in {"plug_charger_v5", "rule_v5_plug_charger"}:
        extractors["PlugCharger-v1"] = _plug_charger_v5
        versions["PlugCharger-v1"] = "rule_v5_plugcharger"
        return extractors, versions
    if variant in {"plug_charger_v6", "rule_v6_plug_charger"}:
        extractors["PlugCharger-v1"] = _plug_charger_v6
        versions["PlugCharger-v1"] = "rule_v6_plugcharger"
        return extractors, versions
    if variant in {"plug_charger_v7", "rule_v7_plug_charger"}:
        extractors["PlugCharger-v1"] = _plug_charger_v7
        versions["PlugCharger-v1"] = "rule_v7_plugcharger"
        return extractors, versions
    if variant in {"plug_charger_v8", "rule_v8_plug_charger"}:
        extractors["PlugCharger-v1"] = _plug_charger_v8
        versions["PlugCharger-v1"] = "rule_v8_plugcharger"
        return extractors, versions
    if variant in {"plug_charger_v9", "rule_v9_plug_charger"}:
        extractors["PlugCharger-v1"] = _plug_charger_v9
        versions["PlugCharger-v1"] = "rule_v9_plugcharger"
        return extractors, versions
    if variant in {"trifinger_v3", "trifinger_rotate_v3", "rule_v3_trifinger"}:
        extractors["TriFingerRotateCubeLevel4-v1"] = _trifinger_rotate_cube_level4_v3
        versions["TriFingerRotateCubeLevel4-v1"] = "rule_v3_trifinger"
        return extractors, versions
    if variant in {"trifinger_v4", "trifinger_rotate_v4", "rule_v4_trifinger"}:
        extractors["TriFingerRotateCubeLevel4-v1"] = _trifinger_rotate_cube_level4_v4
        versions["TriFingerRotateCubeLevel4-v1"] = "rule_v4_trifinger"
        return extractors, versions
    if variant in {"trifinger_v5", "trifinger_rotate_v5", "rule_v5_trifinger"}:
        extractors["TriFingerRotateCubeLevel4-v1"] = _trifinger_rotate_cube_level4_v5
        versions["TriFingerRotateCubeLevel4-v1"] = "rule_v5_trifinger"
        return extractors, versions
    if variant in {"trifinger_v6", "trifinger_rotate_v6", "rule_v6_trifinger"}:
        extractors["TriFingerRotateCubeLevel4-v1"] = _trifinger_rotate_cube_level4_v6
        versions["TriFingerRotateCubeLevel4-v1"] = "rule_v6_trifinger"
        return extractors, versions
    if variant in {"fmb_v2", "rule_v2_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v2
        versions["FMBAssembly1Easy-v1"] = "rule_v2_fmb"
        return extractors, versions
    if variant in {"fmb_v3", "rule_v3_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v3
        versions["FMBAssembly1Easy-v1"] = "rule_v3_fmb"
        return extractors, versions
    if variant in {"fmb_v4", "rule_v4_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v4
        versions["FMBAssembly1Easy-v1"] = "rule_v4_fmb"
        return extractors, versions
    if variant in {"fmb_v5", "rule_v5_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v5
        versions["FMBAssembly1Easy-v1"] = "rule_v5_fmb"
        return extractors, versions
    if variant in {"fmb_v6", "rule_v6_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v6
        versions["FMBAssembly1Easy-v1"] = "rule_v6_fmb"
        return extractors, versions
    if variant in {"fmb_v7", "rule_v7_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v7
        versions["FMBAssembly1Easy-v1"] = "rule_v7_fmb"
        return extractors, versions
    if variant in {"fmb_v8", "rule_v8_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v8
        versions["FMBAssembly1Easy-v1"] = "rule_v8_fmb"
        return extractors, versions
    if variant in {"fmb_v9", "rule_v9_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v9
        versions["FMBAssembly1Easy-v1"] = "rule_v9_fmb"
        return extractors, versions
    if variant in {"fmb_v10", "rule_v10_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v10
        versions["FMBAssembly1Easy-v1"] = "rule_v10_fmb"
        return extractors, versions
    if variant in {"fmb_v11", "rule_v11_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v11
        versions["FMBAssembly1Easy-v1"] = "rule_v11_fmb"
        return extractors, versions
    if variant in {"fmb_v12", "rule_v12_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v12
        versions["FMBAssembly1Easy-v1"] = "rule_v12_fmb"
        return extractors, versions
    if variant in {"rotate_valve_v2", "rule_v2_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v2
        versions["RotateValveLevel4-v1"] = "rule_v2_rotatevalve"
        return extractors, versions
    if variant in {"rotate_valve_v3", "rule_v3_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v3
        versions["RotateValveLevel4-v1"] = "rule_v3_rotatevalve"
        return extractors, versions
    if variant in {"rotate_valve_v4", "rule_v4_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v4
        versions["RotateValveLevel4-v1"] = "rule_v4_rotatevalve"
        return extractors, versions
    if variant in {"rotate_valve_v5", "rule_v5_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v5
        versions["RotateValveLevel4-v1"] = "rule_v5_rotatevalve"
        return extractors, versions
    if variant in {"rotate_valve_v6", "rule_v6_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v6
        versions["RotateValveLevel4-v1"] = "rule_v6_rotatevalve"
        return extractors, versions
    if variant in {"rotate_valve_v7", "rule_v7_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v7
        versions["RotateValveLevel4-v1"] = "rule_v7_rotatevalve"
        return extractors, versions
    if variant in {"rotate_valve_v8", "rule_v8_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v8
        versions["RotateValveLevel4-v1"] = "rule_v8_rotatevalve"
        return extractors, versions
    if variant in {"tworobot_v2", "two_robot_v2", "rule_v2_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v2
        versions["TwoRobotStackCube-v1"] = "rule_v2_tworobot"
        return extractors, versions
    if variant in {"drawtriangle_v2", "draw_triangle_v2", "rule_v2_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v2
        versions["DrawTriangle-v1"] = "rule_v2_drawtriangle"
        return extractors, versions
    if variant in {"tworobot_v3", "two_robot_v3", "rule_v3_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v3
        versions["TwoRobotStackCube-v1"] = "rule_v3_tworobot"
        return extractors, versions
    if variant in {"drawtriangle_v3", "draw_triangle_v3", "rule_v3_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v3
        versions["DrawTriangle-v1"] = "rule_v3_drawtriangle"
        return extractors, versions
    if variant in {"tworobot_v4", "two_robot_v4", "rule_v4_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v4
        versions["TwoRobotStackCube-v1"] = "rule_v4_tworobot"
        return extractors, versions
    if variant in {"drawtriangle_v4", "draw_triangle_v4", "rule_v4_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v4
        versions["DrawTriangle-v1"] = "rule_v4_drawtriangle"
        return extractors, versions
    if variant in {"tworobot_v5", "two_robot_v5", "rule_v5_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v5
        versions["TwoRobotStackCube-v1"] = "rule_v5_tworobot"
        return extractors, versions
    if variant in {"drawtriangle_v5", "draw_triangle_v5", "rule_v5_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v5
        versions["DrawTriangle-v1"] = "rule_v5_drawtriangle"
        return extractors, versions
    if variant in {"tworobot_v6", "two_robot_v6", "rule_v6_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v6
        versions["TwoRobotStackCube-v1"] = "rule_v6_tworobot"
        return extractors, versions
    if variant in {"drawtriangle_v6", "draw_triangle_v6", "rule_v6_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v6
        versions["DrawTriangle-v1"] = "rule_v6_drawtriangle"
        return extractors, versions
    if variant in {"tworobot_v7", "two_robot_v7", "rule_v7_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v7
        versions["TwoRobotStackCube-v1"] = "rule_v7_tworobot"
        return extractors, versions
    if variant in {"tworobot_v8", "two_robot_v8", "rule_v8_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v8
        versions["TwoRobotStackCube-v1"] = "rule_v8_tworobot"
        return extractors, versions
    if variant in {"tworobot_v9", "two_robot_v9", "rule_v9_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v9
        versions["TwoRobotStackCube-v1"] = "rule_v9_tworobot"
        return extractors, versions
    if variant in {"tworobot_v10", "two_robot_v10", "rule_v10_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v10
        versions["TwoRobotStackCube-v1"] = "rule_v10_tworobot"
        return extractors, versions
    if variant in {"tworobot_v11", "two_robot_v11", "rule_v11_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v11
        versions["TwoRobotStackCube-v1"] = "rule_v11_tworobot"
        return extractors, versions
    if variant in {"tworobot_v12", "two_robot_v12", "rule_v12_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v12
        versions["TwoRobotStackCube-v1"] = "rule_v12_tworobot"
        return extractors, versions
    if variant in {"tworobot_v13", "two_robot_v13", "rule_v13_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v13
        versions["TwoRobotStackCube-v1"] = "rule_v13_tworobot"
        return extractors, versions
    if variant in {"tworobot_v14", "two_robot_v14", "rule_v14_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v14
        versions["TwoRobotStackCube-v1"] = "rule_v14_tworobot"
        return extractors, versions
    if variant in {"drawtriangle_v7", "draw_triangle_v7", "rule_v7_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v7
        versions["DrawTriangle-v1"] = "rule_v7_drawtriangle"
        return extractors, versions
    if variant in {"drawtriangle_v8", "draw_triangle_v8", "rule_v8_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v8
        versions["DrawTriangle-v1"] = "rule_v8_drawtriangle"
        return extractors, versions
    if variant in {"drawtriangle_v9", "draw_triangle_v9", "rule_v9_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v9
        versions["DrawTriangle-v1"] = "rule_v9_drawtriangle"
        return extractors, versions
    if variant in {"drawtriangle_v10", "draw_triangle_v10", "rule_v10_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v10
        versions["DrawTriangle-v1"] = "rule_v10_drawtriangle"
        return extractors, versions
    if variant in {"drawtriangle_v11", "draw_triangle_v11", "rule_v11_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v11
        versions["DrawTriangle-v1"] = "rule_v11_drawtriangle"
        return extractors, versions
    if variant in {"drawtriangle_v12", "draw_triangle_v12", "rule_v12_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v12
        versions["DrawTriangle-v1"] = "rule_v12_drawtriangle"
        return extractors, versions
    if variant in {"trifinger_v2", "trifinger_rotate_v2", "rule_v2_trifinger"}:
        extractors["TriFingerRotateCubeLevel4-v1"] = _trifinger_rotate_cube_level4_v2
        versions["TriFingerRotateCubeLevel4-v1"] = "rule_v2_trifinger"
        return extractors, versions
    if variant in {"plug_charger_v10", "rule_v10_plug_charger"}:
        extractors["PlugCharger-v1"] = _plug_charger_v10
        versions["PlugCharger-v1"] = "rule_v10_plugcharger"
        return extractors, versions
    if variant in {"stackpyramid_v17", "stack_pyramid_v17", "rule_v17_stackpyramid"}:
        extractors["StackPyramid-v1"] = _stack_pyramid_v17
        versions["StackPyramid-v1"] = "rule_v17_stackpyramid"
        return extractors, versions
    if variant in {"fmb_v13", "rule_v13_fmb"}:
        extractors["FMBAssembly1Easy-v1"] = _fmb_v13
        versions["FMBAssembly1Easy-v1"] = "rule_v13_fmb"
        return extractors, versions
    if variant in {"rotate_valve_v9", "rule_v9_rotate_valve"}:
        extractors["RotateValveLevel4-v1"] = _rotate_valve_level4_v9
        versions["RotateValveLevel4-v1"] = "rule_v9_rotatevalve"
        return extractors, versions
    if variant in {"tworobot_v15", "two_robot_v15", "rule_v15_tworobot"}:
        extractors["TwoRobotStackCube-v1"] = _two_robot_stack_cube_v15
        versions["TwoRobotStackCube-v1"] = "rule_v15_tworobot"
        return extractors, versions
    if variant in {"drawtriangle_v13", "draw_triangle_v13", "rule_v13_drawtriangle"}:
        extractors["DrawTriangle-v1"] = _draw_triangle_v13
        versions["DrawTriangle-v1"] = "rule_v13_drawtriangle"
        return extractors, versions
    if variant in {"trifinger_v7", "trifinger_rotate_v7", "rule_v7_trifinger"}:
        extractors["TriFingerRotateCubeLevel4-v1"] = _trifinger_rotate_cube_level4_v7
        versions["TriFingerRotateCubeLevel4-v1"] = "rule_v7_trifinger"
        return extractors, versions
    if variant in {"weak4_v2", "rule_v2_weak4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v2, "rule_v2_pokecube"),
            "PullCubeTool-v1": (_pull_cube_tool_v2, "rule_v2_pullcubetool"),
            "RollBall-v1": (_roll_ball_v2, "rule_v2_rollball"),
            "PickSingleYCB-v1": (_pick_single_ycb_v2, "rule_v2_picksingleycb"),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"hard4_v3", "rule_v3_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v3, "rule_v3_pokecube"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v3, "rule_v3_pusht"),
            "RollBall-v1": (_roll_ball_v3, "rule_v3_rollball"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v3,
                "rule_v3_picksingleycb",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"hard4_v4", "rule_v4_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v4, "rule_v4_pokecube"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v4, "rule_v4_pusht"),
            "RollBall-v1": (_roll_ball_v4, "rule_v4_rollball"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v4,
                "rule_v4_picksingleycb",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"hard4_v5", "rule_v5_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v4, "rule_v4_pokecube_frozen_candidate"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v5, "rule_v5_pusht"),
            "RollBall-v1": (_roll_ball_v5, "rule_v5_rollball"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v4,
                "rule_v4_picksingleycb_frozen_candidate",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"hard4_v7", "rule_v7_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v4, "rule_v4_pokecube_frozen_candidate"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v7, "rule_v7_pusht"),
            "RollBall-v1": (_roll_ball_v7, "rule_v7_rollball"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v4,
                "rule_v4_picksingleycb_frozen_candidate",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"hard4_v8", "rule_v8_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v4, "rule_v4_pokecube_frozen_candidate"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v8, "rule_v8_pusht"),
            "RollBall-v1": (_roll_ball_v5, "rule_v5_rollball_frozen_candidate"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v4,
                "rule_v4_picksingleycb_frozen_candidate",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"hard4_v9", "rule_v9_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v4, "rule_v4_pokecube_frozen_candidate"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v9, "rule_v9_pusht_native_like"),
            "RollBall-v1": (_roll_ball_v5, "rule_v5_rollball_frozen_candidate"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v4,
                "rule_v4_picksingleycb_frozen_candidate",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"hard4_v10", "rule_v10_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v4, "rule_v4_pokecube_frozen_candidate"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v10, "rule_v10_pusht_official_dense_strict"),
            "RollBall-v1": (_roll_ball_v5, "rule_v5_rollball_frozen_candidate"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v4,
                "rule_v4_picksingleycb_frozen_candidate",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"pusht_v11", "push_t_v11", "rule_v11_pusht"}:
        extractors["PushT-v1"] = _push_t_v11
        versions["PushT-v1"] = "rule_v11_pusht"
        return extractors, versions
    if variant in {"pusht_v12", "push_t_v12", "rule_v12_pusht"}:
        extractors["PushT-v1"] = _push_t_v12
        versions["PushT-v1"] = "rule_v12_pusht"
        return extractors, versions
    if variant in {"pusht_v13", "push_t_v13", "rule_v13_pusht"}:
        extractors["PushT-v1"] = _push_t_v13
        versions["PushT-v1"] = "rule_v13_pusht"
        return extractors, versions
    if variant in {"hard4_v11", "rule_v11_hard4"}:
        replacements = {
            "PokeCube-v1": (_poke_cube_v4, "rule_v4_pokecube_frozen_candidate"),
            "PullCubeTool-v1": (
                _pull_cube_tool_v2,
                "rule_v2_pullcubetool_frozen_candidate",
            ),
            "PushT-v1": (_push_t_v11, "rule_v11_pusht"),
            "RollBall-v1": (_roll_ball_v5, "rule_v5_rollball_frozen_candidate"),
            "PickSingleYCB-v1": (
                _pick_single_ycb_v4,
                "rule_v4_picksingleycb_frozen_candidate",
            ),
        }
        for env_id, (extractor, version) in replacements.items():
            extractors[env_id] = extractor
            versions[env_id] = version
        return extractors, versions
    if variant in {"replacement_low2_v2", "rule_v2_replacement_low2"}:
        extractors["RotateValveLevel1-v1"] = _rotate_valve_half_turn_v2
        versions["RotateValveLevel1-v1"] = "frame_vlm_v2_rotate_valve_level1"
        extractors["TriFingerRotateCubeLevel3-v1"] = _trifinger_replacement_v2
        versions["TriFingerRotateCubeLevel3-v1"] = "frame_vlm_v2_trifinger_level3"
        return extractors, versions
    raise ValueError(
        "Unsupported OURS_RULE_VARIANT="
        f"{variant!r}; expected rule_v1, pusht_v2, peg_plug_v2, peg_plug_v3, plug_charger_v4, plug_charger_v5, plug_charger_v6, rotate_valve_v2, rotate_valve_v3, rotate_valve_v4, rotate_valve_v5, rotate_valve_v6, stackpyramid_v2, stackpyramid_v3, stackpyramid_v4, stackpyramid_v5, stackpyramid_v6, stackpyramid_v7, stackpyramid_v8, stackpyramid_v9, stackpyramid_v10, stackpyramid_v11, tworobot_v2, drawtriangle_v2, tworobot_v3, drawtriangle_v3, tworobot_v4, drawtriangle_v4, tworobot_v5, drawtriangle_v5, tworobot_v6, drawtriangle_v6, tworobot_v7, tworobot_v8, tworobot_v9, tworobot_v10, tworobot_v13, drawtriangle_v7, drawtriangle_v8, drawtriangle_v9, trifinger_v2, trifinger_v3, trifinger_v4, fmb_v2, fmb_v3, "
        "weak4_v2, hard4_v3, hard4_v4, hard4_v5, hard4_v7, hard4_v8, hard4_v9, or hard4_v10"
    )


def load_rule_stage_specs() -> dict[str, RuleStageSpec]:
    extractors, versions = _select_extractors()
    tasks = load_task_specs()
    missing = {task.env_id for task in tasks} - set(extractors)
    if missing:
        raise ValueError(f"RuleStage registry is missing task IDs: {sorted(missing)}")
    return {
        task.env_id: RuleStageSpec(
            env_id=task.env_id,
            stage_names=task.stages,
            config=_config(len(task.stages)),
            extractor=extractors[task.env_id],
            version=versions[task.env_id],
        )
        for task in tasks
    }


def get_rule_stage_spec(env_id: str) -> RuleStageSpec:
    try:
        return load_rule_stage_specs()[env_id]
    except KeyError as error:
        raise KeyError(f"No RuleStage specification for {env_id}") from error

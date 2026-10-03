"""Task-local CORE instantiation for observable object-move/push tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from .reward_form_modes import (
    STAGED_RESET,
    dense_sum,
    remember_stage_quality,
    reward_form_mode,
    sparse_step_reward,
    transition_reward,
)

ACTIVE_STAGE_COUNT: Final[int] = 3
# The exported constants are legacy defaults for tasks without a multi-waypoint path.
STAGE_COUNT: Final[int] = 4
TERMINAL_STAGE: Final[int] = 3
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ObjectMoveConfig:
    gamma: float = 0.99
    reward_form_mode: str = STAGED_RESET
    dense_scales: tuple[float, ...] = (4.0, 6.0, 12.0)
    transition_bonuses: tuple[float, ...] = (5.0, 10.0, 25.0)
    required_holds: tuple[int, ...] = (3, 3, 3)
    prepush_offset_m: float = 0.06
    prepush_height_m: float = 0.08
    contact_height_m: float = 0.025
    prepush_enter_m: float = 0.050
    prepush_exit_m: float = 0.070
    contact_enter_m: float = 0.050
    contact_exit_m: float = 0.070
    target_enter_m: float = 0.050
    target_exit_m: float = 0.070
    action_rate_weight: float = 0.020
    action_curve_weight: float = 0.010
    boundary_weight: float = 0.020
    boundary_free_limit: float = 0.80
    boundary_hard_limit: float = 1.00
    wall_waypoint_xy: tuple[float, float] | None = None
    path_waypoints_xy: tuple[tuple[float, float], ...] = ()
    waypoint_enter_m: float = 0.060
    waypoint_exit_m: float = 0.080
    directional_contact_floor_weight: float = 0.0
    candidate_requires_directional_contact: bool = False
    directional_candidate_enter_m: float = 0.080
    directional_candidate_exit_m: float = 0.100
    use_directional_contact: bool = False
    conjunctive_directional_contact: bool = False
    # Optional explicit contact-mode switch between two path legs. Defaults
    # preserve all historical object-move revisions.
    reacquire_contact_between_waypoints: bool = False
    reacquire_contact_quality_scale_m: float = 0.18
    reacquire_progress_weight: float = 0.0
    reacquire_progress_quality_scale_m: float = 0.16
    reacquire_progress_enter_m: float | None = None
    reacquire_progress_exit_m: float | None = None
    # Optional official MetaWorld action interface. False preserves every
    # historical fixed-gripper object checkpoint.
    actor_controls_gripper: bool = False
    # Optional observable aperture factor for contact-dependent object motion.
    # Defaults are disabled so historical rewards remain bit-for-bit unchanged.
    contact_gripper_target: float | None = None
    contact_gripper_scale: float = 0.20
    contact_gripper_coupling_weight: float = 0.0
    smooth_actor_gripper: bool = False

    # A final push is only causal while the TCP remains on the target-facing
    # opposite side of the object. This optional observable state term cannot
    # authorize a transition; zero preserves every historical revision.
    final_contact_invariant_penalty_weight: float = 0.0
    final_contact_invariant_floor: float = 0.20
    # Optional direct observable object-to-target channel in the last active
    # stage. It changes only dense control credit; the existing candidate and
    # Mandatory-Qwen authorization remain unchanged.
    final_target_direct_weight: float = 0.0



    target_distance_mode: str = "xy"
    target_distance_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    target_quality_scale_m: float = 0.25


@dataclass(frozen=True)
class ObjectMoveFeatures:
    tcp: Array
    obj: Array
    target: Array
    direction_xy: Array
    prepush_point: Array
    contact_point: Array
    target_contact_point: Array
    prepush_distance: float
    contact_distance: float
    target_contact_distance: float
    tcp_obj_distance: float
    obj_target_xy_distance: float
    obj_target_distance: float
    gripper_opening: float = 1.0
    obj_waypoint_xy_distance: float | None = None
    obj_path_waypoint_xy_distances: tuple[float, ...] = ()
    tcp_path_contact_distances: tuple[float, ...] = ()


@dataclass
class ObjectMoveState:
    stage: int = 0
    stable_count: int = 0
    candidate_epoch: int = 0
    request_armed: bool = True
    previous_quality: float | None = None
    previous_stage_qualities: dict[int, float] = field(default_factory=dict)
    previous_action: Array = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    previous_previous_action: Array = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    qwen_confirmed_mask: int = 0

    def reset(self, quality: float) -> None:
        self.stage = 0
        self.stable_count = 0
        self.candidate_epoch = 0
        self.request_armed = True
        self.previous_quality = float(quality)
        self.previous_stage_qualities.clear()
        remember_stage_quality(self.previous_stage_qualities, 0, quality)
        self.previous_action.fill(0.0)
        self.previous_previous_action.fill(0.0)
        self.qwen_confirmed_mask = 0


@dataclass(frozen=True)
class ObjectMoveStep:
    reward: float
    request_due: bool
    strict_candidate: bool
    old_stage: int
    new_stage: int


def path_waypoints(config: ObjectMoveConfig) -> tuple[Array, ...]:
    """Return validated task waypoints while preserving legacy one-waypoint runs."""
    if config.path_waypoints_xy and config.wall_waypoint_xy is not None:
        raise ValueError("set path_waypoints_xy or wall_waypoint_xy, not both")
    raw = config.path_waypoints_xy
    if not raw and config.wall_waypoint_xy is not None:
        raw = (config.wall_waypoint_xy,)
    result: list[Array] = []
    for index, item in enumerate(raw):
        waypoint = np.asarray(item, dtype=np.float64)
        if waypoint.shape != (2,) or not np.all(np.isfinite(waypoint)):
            raise ValueError(f"path waypoint {index} must be a finite two-vector")
        result.append(waypoint)
    return tuple(result)


def active_stage_count(config: ObjectMoveConfig) -> int:
    """Approach, zero-or-more path stages, and a final target stage."""
    waypoint_count = len(path_waypoints(config))
    inserted = int(config.reacquire_contact_between_waypoints)
    # Preserve stage cardinality encoded by historical recipes even when the
    # path geometry is implicit (those recipes store five dense/hold entries
    # but no explicit waypoint list). This keeps checkpoint observation shapes
    # reconstructible while retaining the three-stage default.
    encoded_count = max(
        len(config.dense_scales), len(config.transition_bonuses), len(config.required_holds)
    )
    return max(ACTIVE_STAGE_COUNT, waypoint_count + 2 + inserted, encoded_count)



def stage_count(config: ObjectMoveConfig) -> int:
    return active_stage_count(config) + 1


def terminal_stage(config: ObjectMoveConfig) -> int:
    return active_stage_count(config)


def validate_stage_config(config: ObjectMoveConfig) -> None:
    reward_form_mode(config)
    expected = active_stage_count(config)
    for name in ("dense_scales", "transition_bonuses", "required_holds"):
        if len(getattr(config, name)) != expected:
            raise ValueError(f"{name} must have {expected} entries for this path")

    if not 0.0 <= config.directional_contact_floor_weight <= 1.0:
        raise ValueError("directional_contact_floor_weight must lie in [0, 1]")
    if config.reacquire_contact_between_waypoints:
        if len(path_waypoints(config)) != 2:
            raise ValueError(
                "contact reacquisition currently requires exactly two path waypoints"
            )
        if not config.use_directional_contact:
            raise ValueError("contact reacquisition requires directional contact")
    if config.reacquire_contact_quality_scale_m <= 0.0:
        raise ValueError("reacquire contact quality scale must be positive")
    if not 0.0 <= config.reacquire_progress_weight <= 1.0:
        raise ValueError("reacquire progress weight must lie in [0, 1]")
    if config.reacquire_progress_quality_scale_m <= 0.0:
        raise ValueError("reacquire progress quality scale must be positive")
    if (config.reacquire_progress_enter_m is None) != (
        config.reacquire_progress_exit_m is None
    ):
        raise ValueError("reacquire progress hysteresis must set both thresholds")
    if config.reacquire_progress_enter_m is not None:
        if not config.reacquire_contact_between_waypoints:
            raise ValueError("reacquire progress requires the contact-mode stage")
        if not 0.0 < config.reacquire_progress_enter_m < config.reacquire_progress_exit_m:
            raise ValueError("reacquire progress hysteresis is invalid")
    if config.directional_candidate_enter_m <= 0.0:
        raise ValueError("directional_candidate_enter_m must be positive")
    if config.directional_candidate_exit_m < config.directional_candidate_enter_m:
        raise ValueError(
            "directional_candidate_exit_m must be at least the enter threshold")
    if config.contact_gripper_target is not None:
        if not np.isfinite(config.contact_gripper_target):
            raise ValueError("contact_gripper_target must be finite")
        if not 0.0 <= config.contact_gripper_target <= 1.0:
            raise ValueError("contact_gripper_target must lie in [0, 1]")
    if config.contact_gripper_scale <= 0.0:
        raise ValueError("contact_gripper_scale must be positive")
    if not 0.0 <= config.contact_gripper_coupling_weight <= 1.0:
        raise ValueError("contact_gripper_coupling_weight must lie in [0, 1]")
    if config.smooth_actor_gripper and not config.actor_controls_gripper:
        raise ValueError("smooth_actor_gripper requires actor_controls_gripper")
    if config.final_contact_invariant_penalty_weight < 0.0:
        raise ValueError("final contact invariant weight must be non-negative")
    if not 0.0 < config.final_contact_invariant_floor <= 1.0:
        raise ValueError("final contact invariant floor must lie in (0, 1]")
    if not 0.0 <= config.final_target_direct_weight <= 1.0:
        raise ValueError("final target direct weight must lie in [0, 1]")


def _vec3(value: npt.ArrayLike, name: str) -> Array:

    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def features(
    tcp: npt.ArrayLike,
    obj: npt.ArrayLike,
    target: npt.ArrayLike,
    config: ObjectMoveConfig,
    *,
    gripper_opening: float = 1.0,
) -> ObjectMoveFeatures:
    tcp_v, obj_v, target_v = _vec3(tcp, 'tcp'), _vec3(obj, 'obj'), _vec3(target, 'target')
    waypoints = path_waypoints(config)
    first_goal_xy = target_v[:2]
    opening = float(gripper_opening)
    if not np.isfinite(opening):
        raise ValueError("gripper_opening must be finite")
    if waypoints:
        first_goal_xy = waypoints[0]
    waypoint_distances = tuple(float(np.linalg.norm(obj_v[:2] - waypoint)) for waypoint in waypoints)
    waypoint_distance = waypoint_distances[0] if waypoint_distances else None
    delta_xy = first_goal_xy - obj_v[:2]
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1.0e-6:
        direction_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction_xy = delta_xy / norm
    contact = obj_v.copy()
    contact[:2] = obj_v[:2] - config.prepush_offset_m * direction_xy
    contact[2] = obj_v[2] + config.contact_height_m
    prepush = contact.copy()
    prepush[2] = obj_v[2] + config.prepush_height_m
    target_delta_xy = target_v[:2] - obj_v[:2]
    target_norm = float(np.linalg.norm(target_delta_xy))
    target_direction_xy = (
        np.array([1.0, 0.0], dtype=np.float64)
        if target_norm < 1.0e-6
        else target_delta_xy / target_norm
    )
    target_contact = obj_v.copy()
    target_contact[:2] = obj_v[:2] - config.prepush_offset_m * target_direction_xy
    path_contact_distances: list[float] = []
    for waypoint in waypoints:
        path_delta = waypoint - obj_v[:2]
        path_norm = float(np.linalg.norm(path_delta))
        path_direction = (
            np.array([1.0, 0.0], dtype=np.float64)
            if path_norm < 1.0e-6
            else path_delta / path_norm
        )
        path_contact = obj_v.copy()
        path_contact[:2] = obj_v[:2] - config.prepush_offset_m * path_direction
        path_contact[2] = obj_v[2] + config.contact_height_m
        path_contact_distances.append(float(np.linalg.norm(tcp_v - path_contact)))

    target_contact[2] = obj_v[2] + config.contact_height_m
    obj_target_xy = float(np.linalg.norm(obj_v[:2] - target_v[:2]))
    if config.target_distance_mode == "xy":
        obj_target = obj_target_xy
    elif config.target_distance_mode == "scaled_xyz":
        scales = np.asarray(config.target_distance_scale_xyz, dtype=np.float64)
        if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("target_distance_scale_xyz must be a positive finite three-vector")
        obj_target = float(np.linalg.norm((obj_v - target_v) * scales))
    else:
        raise ValueError("target_distance_mode must be xy or scaled_xyz")
    if config.target_quality_scale_m <= 0.0:
        raise ValueError("target_quality_scale_m must be positive")

    return ObjectMoveFeatures(
        tcp=tcp_v,
        obj=obj_v,
        target=target_v,
        direction_xy=direction_xy,
        prepush_point=prepush,
        contact_point=contact,
        target_contact_point=target_contact,
        prepush_distance=float(np.linalg.norm(tcp_v - prepush)),
        contact_distance=float(np.linalg.norm(tcp_v - contact)),
        target_contact_distance=float(np.linalg.norm(tcp_v - target_contact)),
        tcp_obj_distance=float(np.linalg.norm(tcp_v - obj_v)),
        obj_target_xy_distance=obj_target_xy,
        obj_waypoint_xy_distance=waypoint_distance,
        obj_target_distance=obj_target,
        obj_path_waypoint_xy_distances=waypoint_distances,
        tcp_path_contact_distances=tuple(path_contact_distances),
        gripper_opening=opening,
    )


def _quality(distance: float, scale: float) -> float:
    return float(1.0 - np.tanh(max(float(distance), 0.0) / scale))


def _base_stage_quality(stage: int, value: ObjectMoveFeatures, config: ObjectMoveConfig | None = None) -> float:
    cfg = config or ObjectMoveConfig()
    waypoints = path_waypoints(cfg)
    waypoint_count = len(waypoints)
    if stage == 0:
        return _quality(value.prepush_distance, 0.20)
    if cfg.reacquire_contact_between_waypoints and waypoint_count == 2:
        if stage == 2:
            contact_quality = _quality(value.tcp_path_contact_distances[1], cfg.reacquire_contact_quality_scale_m)
            proximity = _quality(value.tcp_obj_distance, 0.16)
            contact_only = 0.85 * contact_quality + 0.15 * proximity
            if cfg.reacquire_progress_weight <= 0.0:
                return contact_only
            progress_quality = _quality(
                value.obj_path_waypoint_xy_distances[1],
                cfg.reacquire_progress_quality_scale_m,
            )
            coupled = float(np.sqrt(contact_quality * progress_quality))
            return float(
                (1.0 - cfg.reacquire_progress_weight) * contact_only
                + cfg.reacquire_progress_weight * coupled
            )
        if stage == 3:
            waypoint_distance = value.obj_path_waypoint_xy_distances[1]
            contact_distance = value.tcp_path_contact_distances[1]
            progress_quality = _quality(waypoint_distance, 0.22)
            contact_quality = _quality(contact_distance, 0.12)
            if cfg.conjunctive_directional_contact:
                coupled = float(np.sqrt(progress_quality * contact_quality))
                weight = cfg.directional_contact_floor_weight
                return float(weight * contact_quality + (1.0 - weight) * coupled)
            return 0.65 * progress_quality + 0.35 * contact_quality
        if stage == 4:
            progress_quality = _quality(
                value.obj_target_distance, cfg.target_quality_scale_m
            )
            contact_quality = _quality(value.target_contact_distance, 0.12)
            if cfg.conjunctive_directional_contact:
                coupled = float(np.sqrt(progress_quality * contact_quality))
                weight = cfg.directional_contact_floor_weight
                return float(weight * contact_quality + (1.0 - weight) * coupled)
            return 0.65 * progress_quality + 0.35 * contact_quality
    if waypoint_count and 1 <= stage <= waypoint_count:
        index = stage - 1
        waypoint_distance = value.obj_path_waypoint_xy_distances[index]
        if cfg.use_directional_contact:
            contact_distance = value.tcp_path_contact_distances[index]
            if cfg.conjunctive_directional_contact:
                progress_quality = _quality(waypoint_distance, 0.22)
                contact_quality = _quality(contact_distance, 0.12)
                coupled = float(np.sqrt(progress_quality * contact_quality))
                weight = cfg.directional_contact_floor_weight
                return float(weight * contact_quality + (1.0 - weight) * coupled)
            return 0.65 * _quality(waypoint_distance, 0.22) + 0.35 * _quality(contact_distance, 0.12)
        return 0.75 * _quality(waypoint_distance, 0.22) + 0.25 * _quality(value.tcp_obj_distance, 0.14)
    if stage == 1 and not waypoints:
        return 0.65 * _quality(value.contact_distance, 0.12) + 0.35 * _quality(value.tcp_obj_distance, 0.12)
    final_stage = max(2, waypoint_count + 1)
    if stage == final_stage:
        if cfg.use_directional_contact:
            if cfg.conjunctive_directional_contact:
                progress_quality = _quality(value.obj_target_distance, cfg.target_quality_scale_m)
                contact_quality = _quality(value.target_contact_distance, 0.12)
                coupled = float(np.sqrt(progress_quality * contact_quality))
                weight = cfg.directional_contact_floor_weight
                return float(weight * contact_quality + (1.0 - weight) * coupled)
            return 0.65 * _quality(value.obj_target_distance, cfg.target_quality_scale_m) + 0.35 * _quality(value.target_contact_distance, 0.12)
        return 0.75 * _quality(value.obj_target_distance, cfg.target_quality_scale_m) + 0.25 * _quality(value.tcp_obj_distance, 0.14)
    if stage == terminal_stage(cfg):
        return 1.0
    raise ValueError('invalid object-move stage')


def stage_quality(stage: int, value: ObjectMoveFeatures, config: ObjectMoveConfig | None = None) -> float:
    """Return stage quality with an optional non-substitutable aperture factor."""
    cfg = config or ObjectMoveConfig()
    base = _base_stage_quality(stage, value, cfg)
    weight = cfg.contact_gripper_coupling_weight
    if (
        stage == 0
        or stage == terminal_stage(cfg)
        or cfg.contact_gripper_target is None
        or weight <= 0.0
    ):
        result = base
    else:
        aperture = _quality(
            abs(value.gripper_opening - cfg.contact_gripper_target),
            cfg.contact_gripper_scale,
        )
        conjunctive = float(np.sqrt(max(base, 0.0) * aperture))
        result = float((1.0 - weight) * base + weight * conjunctive)
    direct_weight = cfg.final_target_direct_weight
    if stage == terminal_stage(cfg) - 1 and direct_weight > 0.0:
        direct = _quality(value.obj_target_distance, cfg.target_quality_scale_m)
        result = float((1.0 - direct_weight) * result + direct_weight * direct)
    return result


def _candidate(stage: int, value: ObjectMoveFeatures, config: ObjectMoveConfig, was_candidate: bool) -> bool:
    waypoint_count = len(path_waypoints(config))

    if stage == 0:
        return value.prepush_distance <= (config.prepush_exit_m if was_candidate else config.prepush_enter_m)
    if config.reacquire_contact_between_waypoints and waypoint_count == 2:
        if stage == 2:
            contact_threshold = (
                config.directional_candidate_exit_m
                if was_candidate
                else config.directional_candidate_enter_m
            )
            contact_ok = value.tcp_path_contact_distances[1] <= contact_threshold
            if config.reacquire_progress_enter_m is None:
                return contact_ok
            progress_threshold = (
                config.reacquire_progress_exit_m
                if was_candidate
                else config.reacquire_progress_enter_m
            )
            return contact_ok and value.obj_path_waypoint_xy_distances[1] <= progress_threshold
        if stage == 3:
            threshold = (
                config.waypoint_exit_m if was_candidate else config.waypoint_enter_m
            )
            waypoint_ok = value.obj_path_waypoint_xy_distances[1] <= threshold
            if not config.candidate_requires_directional_contact:
                return waypoint_ok
            contact_threshold = (
                config.directional_candidate_exit_m
                if was_candidate
                else config.directional_candidate_enter_m
            )
            return waypoint_ok and value.tcp_path_contact_distances[1] <= contact_threshold
        if stage == 4:
            target_ok = value.obj_target_distance <= (
                config.target_exit_m if was_candidate else config.target_enter_m
            )
            if not config.candidate_requires_directional_contact:
                return target_ok
            contact_threshold = (
                config.directional_candidate_exit_m
                if was_candidate
                else config.directional_candidate_enter_m
            )
            return target_ok and value.target_contact_distance <= contact_threshold
    if waypoint_count and 1 <= stage <= waypoint_count:
        if len(value.obj_path_waypoint_xy_distances) != waypoint_count:
            return False
        threshold = config.waypoint_exit_m if was_candidate else config.waypoint_enter_m
        waypoint_ok = value.obj_path_waypoint_xy_distances[stage - 1] <= threshold
        if not config.candidate_requires_directional_contact:
            return waypoint_ok
        contact_threshold = (
            config.directional_candidate_exit_m
            if was_candidate
            else config.directional_candidate_enter_m
        )
        return waypoint_ok and value.tcp_path_contact_distances[stage - 1] <= contact_threshold
    if stage == 1 and waypoint_count == 0:
        return value.contact_distance <= (config.contact_exit_m if was_candidate else config.contact_enter_m)
    final_stage = max(2, waypoint_count + 1)
    if stage == final_stage:
        target_ok = value.obj_target_distance <= (
            config.target_exit_m if was_candidate else config.target_enter_m)
        if not config.candidate_requires_directional_contact:
            return target_ok
        contact_threshold = (
            config.directional_candidate_exit_m
            if was_candidate
            else config.directional_candidate_enter_m
        )
        return target_ok and value.target_contact_distance <= contact_threshold
    return False


def _action_vector(value: npt.ArrayLike, name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape not in {(3,), (4,)} or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three- or four-vector")
    return vector


def _smooth(action: npt.ArrayLike, previous: npt.ArrayLike, before: npt.ArrayLike, config: ObjectMoveConfig) -> float:
    action_v = _action_vector(action, 'action')
    previous_v = _action_vector(previous, 'previous_action')
    before_v = _action_vector(before, 'previous_previous_action')
    if previous_v.shape != action_v.shape or before_v.shape != action_v.shape:
        previous_v = np.zeros_like(action_v)
        before_v = np.zeros_like(action_v)
    rate = float(np.mean(np.square(action_v - previous_v)))
    curve = float(np.mean(np.square(action_v - 2.0 * previous_v + before_v)))
    boundary = np.clip((np.abs(action_v) - config.boundary_free_limit) / (config.boundary_hard_limit - config.boundary_free_limit), 0.0, 1.0)
    return float(config.action_rate_weight * rate + config.action_curve_weight * curve + config.boundary_weight * np.mean(np.square(boundary)))


def step(state: ObjectMoveState, value: ObjectMoveFeatures, action: npt.ArrayLike, config: ObjectMoveConfig) -> ObjectMoveStep:
    old_stage = int(state.stage)
    if old_stage == terminal_stage(config):
        return ObjectMoveStep(0.0, False, False, old_stage, old_stage)
    mode = reward_form_mode(config)
    if mode == STAGED_RESET:
        quality = stage_quality(old_stage, value, config)
        previous = quality if state.previous_quality is None else state.previous_quality
        dense = config.dense_scales[old_stage] * (config.gamma * quality - previous)
    else:
        dense, quality = dense_sum(
            mode=mode,
            current_stage=old_stage,
            active_stage_count=active_stage_count(config),
            dense_scales=config.dense_scales,
            gamma=config.gamma,
            previous_stage_qualities=state.previous_stage_qualities,
            quality_fn=lambda stage: stage_quality(stage, value, config),
        )
    smooth = _smooth(action, state.previous_action, state.previous_previous_action, config)
    final_stage = terminal_stage(config) - 1
    if (
        old_stage == final_stage
        and config.final_contact_invariant_penalty_weight > 0.0
    ):
        contact = _quality(value.target_contact_distance, 0.12)
        normalized_shortfall = max(
            config.final_contact_invariant_floor - contact, 0.0
        ) / config.final_contact_invariant_floor
        smooth += (
            config.final_contact_invariant_penalty_weight
            * normalized_shortfall * normalized_shortfall
        )
    strict = _candidate(old_stage, value, config, state.stable_count > 0)
    if strict:
        state.stable_count += 1
    else:
        if state.stable_count > 0 or not state.request_armed:
            state.candidate_epoch += 1
        state.stable_count = 0
        state.request_armed = True
    due = bool(strict and state.request_armed and state.stable_count >= config.required_holds[old_stage])
    state.previous_quality = quality
    state.previous_previous_action = state.previous_action.copy()
    state.previous_action = _action_vector(action, 'action').copy()
    return ObjectMoveStep(sparse_step_reward(mode, dense - smooth), due, strict, old_stage, old_stage)


def apply_qwen(state: ObjectMoveState, value: ObjectMoveFeatures, decision: bool, config: ObjectMoveConfig, *, bypass_candidate: bool = False) -> float:
    old_stage = int(state.stage)
    if old_stage < 0 or old_stage >= active_stage_count(config) or (
        not bypass_candidate
        and (
            not state.request_armed
            or state.stable_count < config.required_holds[old_stage]
        )
    ):
        raise ValueError('no armed object-move candidate')
    state.request_armed = False
    state.stable_count = 0
    state.request_armed = True
    if not decision:
        state.candidate_epoch += 1
        return 0.0
    state.qwen_confirmed_mask |= 1 << old_stage
    state.stage = old_stage + 1
    state.previous_quality = stage_quality(state.stage, value, config)
    remember_stage_quality(
        state.previous_stage_qualities,
        state.stage,
        state.previous_quality,
    )
    return transition_reward(
        reward_form_mode(config),
        original_bonus=config.transition_bonuses[old_stage],
        new_stage=state.stage,
        terminal_stage=terminal_stage(config),
    )

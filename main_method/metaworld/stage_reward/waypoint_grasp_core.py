"""Four-active-stage grasp/transport/place revision of the shared CORE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .grasp_insert_core import GraspInsertConfig, smoothness_penalty
from .reward_form_modes import reward_form_mode, sparse_step_reward, transition_reward

ACTIVE_STAGE_COUNT = 4
STAGE_COUNT = 5
TERMINAL_STAGE = 4
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class WaypointGraspConfig(GraspInsertConfig):
    dense_scales: tuple[float, ...] = (6.0, 22.0, 72.0, 110.0)
    transition_bonuses: tuple[float, ...] = (8.0, 36.0, 110.0, 170.0)
    required_holds: tuple[int, ...] = (2, 4, 3, 2)
    waypoint_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.10)
    waypoint_enter_m: float = 0.080
    waypoint_exit_m: float = 0.105

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (len(self.dense_scales) == len(self.transition_bonuses) == len(self.required_holds) == ACTIVE_STAGE_COUNT):
            raise ValueError("waypoint revision requires four active-stage parameters")
        offset = np.asarray(self.waypoint_offset_xyz, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("waypoint offset must be a finite three-vector")
        if not 0.0 < self.waypoint_enter_m < self.waypoint_exit_m:
            raise ValueError("waypoint hysteresis is invalid")


@dataclass(frozen=True)
class WaypointFeatures:
    tcp: Array
    obj: Array
    target: Array
    initial_obj: Array
    waypoint: Array
    tcp_to_pregrasp: float
    pregrasp_xy_distance: float
    tcp_above_obj: float
    tcp_to_obj: float
    object_lift: float
    obj_to_waypoint: float
    obj_to_target: float


def _vec3(value: npt.ArrayLike, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def features(tcp: npt.ArrayLike, obj: npt.ArrayLike, target: npt.ArrayLike, initial_obj: npt.ArrayLike, config: WaypointGraspConfig) -> WaypointFeatures:
    tcp_v, obj_v = _vec3(tcp, "tcp"), _vec3(obj, "obj")
    target_v, initial_v = _vec3(target, "target"), _vec3(initial_obj, "initial_obj")
    waypoint = target_v + np.asarray(config.waypoint_offset_xyz, dtype=np.float64)
    pregrasp = obj_v + np.array((0.0, 0.0, 0.10), dtype=np.float64)
    return WaypointFeatures(
        tcp=tcp_v,
        obj=obj_v,
        target=target_v,
        initial_obj=initial_v,
        waypoint=waypoint,
        tcp_to_pregrasp=float(np.linalg.norm(tcp_v - pregrasp)),
        pregrasp_xy_distance=float(np.linalg.norm(tcp_v[:2] - obj_v[:2])),
        tcp_above_obj=float(tcp_v[2] - obj_v[2]),
        tcp_to_obj=float(np.linalg.norm(tcp_v - obj_v)),
        object_lift=float(obj_v[2] - initial_v[2]),
        obj_to_waypoint=float(np.linalg.norm(obj_v - waypoint)),
        obj_to_target=float(np.linalg.norm(obj_v - target_v)),
    )


def _quality(distance: float, scale: float) -> float:
    return float(1.0 - np.tanh(max(float(distance), 0.0) / scale))


def stage_quality(stage: int, value: WaypointFeatures, config: WaypointGraspConfig) -> float:
    if stage == 0:
        return _quality(value.tcp_to_pregrasp, 0.15)
    if stage == 1:
        grasp = _quality(value.tcp_to_obj, 0.08)
        lift = float(np.clip(value.object_lift / config.capture_lift_enter_m, 0.0, 1.0))
        return 0.40 * grasp + 0.60 * lift
    if stage == 2:
        return 0.72 * _quality(value.obj_to_waypoint, 0.24) + 0.28 * _quality(value.tcp_to_obj, 0.10)
    if stage == 3:
        return 0.78 * _quality(value.obj_to_target, 0.16) + 0.22 * _quality(value.tcp_to_obj, 0.10)
    if stage == TERMINAL_STAGE:
        return 1.0
    raise ValueError("invalid waypoint-grasp stage")


def candidate(stage: int, value: WaypointFeatures, config: WaypointGraspConfig, was_candidate: bool) -> bool:
    if stage == 0:
        xy = config.pregrasp_xy_exit_m if was_candidate else config.pregrasp_xy_enter_m
        return bool(value.pregrasp_xy_distance <= xy and config.pregrasp_height_low_m <= value.tcp_above_obj <= config.pregrasp_height_high_m)
    if stage == 1:
        lift = config.capture_move_exit_m if was_candidate else config.capture_lift_enter_m
        return bool(value.object_lift >= lift and value.tcp_to_obj <= 0.070)
    if stage == 2:
        waypoint = config.waypoint_exit_m if was_candidate else config.waypoint_enter_m
        return bool(value.obj_to_waypoint <= waypoint and value.tcp_to_obj <= 0.090)
    if stage == 3:
        target = config.insert_exit_m if was_candidate else config.insert_enter_m
        return bool(value.obj_to_target <= target)
    return False


@dataclass
class WaypointState:
    stage: int = 0
    stable_count: int = 0
    candidate_epoch: int = 0
    request_armed: bool = True
    previous_quality: float | None = None
    previous_action: Array = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    before_previous_action: Array = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    def reset(self, quality: float) -> None:
        self.stage = 0
        self.stable_count = 0
        self.candidate_epoch = 0
        self.request_armed = True
        self.previous_quality = float(quality)
        self.previous_action.fill(0.0)
        self.before_previous_action.fill(0.0)


@dataclass(frozen=True)
class StepResult:
    reward: float
    request_due: bool
    stage: int


def step(state: WaypointState, value: WaypointFeatures, action: npt.ArrayLike, config: WaypointGraspConfig) -> StepResult:
    if state.stage == TERMINAL_STAGE:
        return StepResult(0.0, False, state.stage)
    mode = reward_form_mode(config)
    quality = stage_quality(state.stage, value, config)
    previous = quality if state.previous_quality is None else state.previous_quality
    dense = config.dense_scales[state.stage] * (config.gamma * quality - previous)
    action_v = np.asarray(action, dtype=np.float64)
    penalty = smoothness_penalty(action_v, state.previous_action, state.before_previous_action, config)
    strict = candidate(state.stage, value, config, state.stable_count > 0)
    if strict:
        state.stable_count += 1
    else:
        if state.stable_count > 0 or not state.request_armed:
            state.candidate_epoch += 1
        state.stable_count = 0
        state.request_armed = True
    due = bool(state.request_armed and state.stable_count >= config.required_holds[state.stage])
    state.previous_quality = quality
    state.before_previous_action = state.previous_action.copy()
    state.previous_action = action_v.copy()
    return StepResult(sparse_step_reward(mode, dense - penalty), due, state.stage)


def apply_decision(
    state: WaypointState,
    decision: bool,
    config: WaypointGraspConfig,
    *,
    bypass_candidate: bool = False,
) -> float:
    if state.stage == TERMINAL_STAGE:
        raise ValueError("terminal stage has no candidate")
    if (
        not bypass_candidate
        and (
            not state.request_armed
            or state.stable_count < config.required_holds[state.stage]
        )
    ):
        raise ValueError("there is no armed candidate")
    state.request_armed = False
    if decision:
        old = state.stage
        state.stage += 1
        state.stable_count = 0
        state.request_armed = True
        state.previous_quality = None
        return transition_reward(
            reward_form_mode(config),
            original_bonus=config.transition_bonuses[old],
            new_stage=state.stage,
            terminal_stage=TERMINAL_STAGE,
        )
    state.candidate_epoch += 1
    state.stable_count = 0
    state.request_armed = True
    return 0.0


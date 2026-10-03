"""Pure NumPy CORE instantiation for grasp-and-insert MetaWorld tasks.

The stages are semantic task data, while the reward composition and
fail-closed Qwen transition semantics remain task-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


ACTIVE_STAGE_COUNT = 3
STAGE_COUNT = ACTIVE_STAGE_COUNT + 1
TERMINAL_STAGE = ACTIVE_STAGE_COUNT
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GraspInsertConfig:
    gamma: float = 0.99
    reward_form_mode: str = STAGED_RESET
    dense_scales: tuple[float, float, float] = (4.0, 7.0, 10.0)
    transition_bonuses: tuple[float, float, float] = (5.0, 10.0, 20.0)
    required_holds: tuple[int, int, int] = (3, 3, 3)
    pregrasp_xy_enter_m: float = 0.025
    pregrasp_xy_exit_m: float = 0.040
    pregrasp_height_low_m: float = 0.060
    pregrasp_height_high_m: float = 0.130
    capture_move_enter_m: float = 0.040
    capture_move_exit_m: float = 0.025
    capture_lift_enter_m: float = 0.0
    capture_progress_mode: str = "displacement"
    insert_enter_m: float = 0.045
    insert_exit_m: float = 0.060
    stage2_tcp_goal_weight: float = 0.0
    stage2_tcp_obj_weight: float = 0.0
    # Optional observable grasp-retention mixture in the transport stage.
    # Zero defaults reproduce every historical reward exactly.
    stage2_closed_weight: float = 0.0
    stage2_held_weight: float = 0.0
    stage2_obj_scale_m: float = 0.20
    stage2_tcp_goal_z_offset_m: float = 0.08
    stage2_fine_obj_weight: float = 0.0
    stage2_fine_obj_scale_m: float = 0.08
    stage2_precision_action_weight: float = 0.0
    stage2_precision_scale_m: float = 0.12
    target_z_override_m: float | None = None
    # Observable object-relative landmarks. Defaults preserve historical runs.
    grasp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_gripper_caging: bool = False
    open_gripper_scale: float = 0.70
    closed_gripper_scale: float = 0.45
    candidate_open_min: float = 0.35
    candidate_closed_max: float = 0.82
    action_rate_weight: float = 0.020
    action_curve_weight: float = 0.010
    boundary_weight: float = 0.020
    boundary_free_limit: float = 0.80
    boundary_hard_limit: float = 1.00

    def _valid_capture_progress_modes(self) -> frozenset[str]:
        return frozenset(("displacement", "target_progress"))

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        reward_form_mode(self)
        if any(v <= 0.0 for v in self.dense_scales):
            raise ValueError("dense scales must be positive")
        # A zero transition bonus is a valid controlled ablation. Production
        # recipes remain strictly positive; allowing zero here lets their
        # pure-flat checkpoints be reconstructed for validation and testing.
        if any(v < 0.0 for v in self.transition_bonuses):
            raise ValueError("transition bonuses must be non-negative")
        if any(v < 1 for v in self.required_holds):
            raise ValueError("required holds must be positive")
        if not 0.0 < self.pregrasp_xy_enter_m < self.pregrasp_xy_exit_m:
            raise ValueError("pregrasp hysteresis is invalid")
        if not 0.0 < self.capture_move_exit_m < self.capture_move_enter_m:
            raise ValueError("capture-move hysteresis is invalid")
        if self.capture_lift_enter_m < 0.0:
            raise ValueError("capture lift threshold must be non-negative")
        if self.capture_lift_enter_m > 0.0 and self.capture_move_exit_m >= self.capture_lift_enter_m:
            raise ValueError("capture lift hysteresis is invalid")
        valid_capture_modes = self._valid_capture_progress_modes()
        if self.capture_progress_mode not in valid_capture_modes:
            choices = " or ".join(sorted(valid_capture_modes))
            raise ValueError(f"capture_progress_mode must be {choices}")
        if not 0.0 < self.insert_enter_m < self.insert_exit_m:
            raise ValueError("insert hysteresis is invalid")
        if self.stage2_tcp_goal_weight < 0.0 or self.stage2_tcp_obj_weight < 0.0:
            raise ValueError("stage2 TCP weights must be non-negative")
        stage2_aux_weight = (
            self.stage2_tcp_goal_weight
            + self.stage2_tcp_obj_weight
            + self.stage2_closed_weight
            + self.stage2_held_weight
        )
        if min(self.stage2_closed_weight, self.stage2_held_weight) < 0.0 or stage2_aux_weight >= 1.0:
            raise ValueError("stage2 auxiliary weights must be non-negative and leave positive object-target weight")
        if self.stage2_obj_scale_m <= 0.0:
            raise ValueError("stage2_obj_scale_m must be positive")
        if not 0.0 <= self.stage2_fine_obj_weight <= 1.0:
            raise ValueError("stage2_fine_obj_weight must be in [0, 1]")
        if self.stage2_fine_obj_scale_m <= 0.0:
            raise ValueError("stage2_fine_obj_scale_m must be positive")
        if self.stage2_precision_action_weight < 0.0:
            raise ValueError("stage2_precision_action_weight must be non-negative")
        if self.stage2_precision_scale_m <= 0.0:
            raise ValueError("stage2_precision_scale_m must be positive")
        if self.boundary_hard_limit <= self.boundary_free_limit:
            raise ValueError("action boundary limits are invalid")
        for name, value in (
            ("grasp_offset_xyz", self.grasp_offset_xyz),
            ("target_offset_xyz", self.target_offset_xyz),
        ):
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite three-vector")
        if self.open_gripper_scale <= 0.0 or self.closed_gripper_scale <= 0.0:
            raise ValueError("gripper caging scales must be positive")
        if not 0.0 <= self.candidate_open_min <= self.candidate_closed_max <= 1.0:
            raise ValueError("gripper candidate thresholds are invalid")



@dataclass(frozen=True)
class GraspInsertFeatures:
    tcp: Array
    obj: Array
    target: Array
    initial_obj: Array
    tcp_to_pregrasp: float
    gripper_opening: float
    pregrasp_xy_distance: float
    tcp_above_obj: float
    tcp_to_obj: float
    object_displacement: float
    object_lift: float
    object_target_progress: float
    obj_to_target: float


def _vec3(value: npt.ArrayLike, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def features(
    tcp: npt.ArrayLike,
    obj: npt.ArrayLike,
    target: npt.ArrayLike,
    initial_obj: npt.ArrayLike,
    config: GraspInsertConfig | None = None,
    gripper_opening: float = 1.0,
) -> GraspInsertFeatures:
    cfg = config or GraspInsertConfig()
    tcp_v, obj_v = _vec3(tcp, "tcp"), _vec3(obj, "obj")
    target_v, initial_v = _vec3(target, "target"), _vec3(initial_obj, "initial_obj")
    grasp_point = obj_v + np.asarray(cfg.grasp_offset_xyz, dtype=np.float64)
    final_point = target_v + np.asarray(cfg.target_offset_xyz, dtype=np.float64)
    pregrasp = grasp_point + np.array((0.0, 0.0, 0.10), dtype=np.float64)
    return GraspInsertFeatures(
        tcp=tcp_v,
        obj=obj_v,
        target=target_v,
        initial_obj=initial_v,
        tcp_to_pregrasp=float(np.linalg.norm(tcp_v - pregrasp)),
        gripper_opening=float(gripper_opening),
        pregrasp_xy_distance=float(np.linalg.norm(tcp_v[:2] - grasp_point[:2])),
        tcp_above_obj=float(tcp_v[2] - grasp_point[2]),
        tcp_to_obj=float(np.linalg.norm(tcp_v - grasp_point)),
        object_displacement=float(np.linalg.norm(obj_v[:2] - initial_v[:2])),
        object_lift=float(obj_v[2] - initial_v[2]),
        object_target_progress=float(
            np.linalg.norm(initial_v - final_point) - np.linalg.norm(obj_v - final_point)
        ),
        obj_to_target=float(np.linalg.norm(obj_v - final_point)),
    )


def _quality(distance: float, scale: float) -> float:
    return float(1.0 - np.tanh(max(float(distance), 0.0) / scale))


def _capture_progress(value: GraspInsertFeatures, config: GraspInsertConfig) -> float:
    if config.capture_progress_mode == "target_progress":
        return value.object_target_progress
    return value.object_displacement


def stage_quality(stage: int, value: GraspInsertFeatures, config: GraspInsertConfig | None = None) -> float:
    cfg = config or GraspInsertConfig()
    if stage == 0:
        distance = _quality(value.tcp_to_pregrasp, 0.15)
        if not cfg.use_gripper_caging:
            return distance
        opening = float(np.clip(value.gripper_opening / cfg.open_gripper_scale, 0.0, 1.0))
        return 0.90 * distance + 0.10 * opening
    if stage == 1:
        grasp = _quality(value.tcp_to_obj, 0.08)
        if cfg.use_gripper_caging:
            closed = float(np.clip((1.0 - value.gripper_opening) / cfg.closed_gripper_scale, 0.0, 1.0))
            held = grasp * closed
        else:
            held = grasp
        if cfg.capture_lift_enter_m > 0.0:
            lift = float(np.clip(value.object_lift / cfg.capture_lift_enter_m, 0.0, 1.0))
            if cfg.use_gripper_caging:
                return 0.25 * grasp + 0.25 * held + 0.50 * lift
            return 0.45 * grasp + 0.55 * lift
        move = float(np.clip(_capture_progress(value, cfg) / 0.10, 0.0, 1.0))
        if cfg.use_gripper_caging:
            return 0.20 * grasp + 0.25 * held + 0.55 * move
        return 0.40 * grasp + 0.60 * move
    if stage == 2:
        obj_weight = 1.0 - (
            cfg.stage2_tcp_goal_weight
            + cfg.stage2_tcp_obj_weight
            + cfg.stage2_closed_weight
            + cfg.stage2_held_weight
        )
        coarse_obj_quality = _quality(value.obj_to_target, cfg.stage2_obj_scale_m)
        fine_obj_quality = _quality(value.obj_to_target, cfg.stage2_fine_obj_scale_m)
        obj_quality = (
            (1.0 - cfg.stage2_fine_obj_weight) * coarse_obj_quality
            + cfg.stage2_fine_obj_weight * fine_obj_quality
        )
        tcp_goal = (
            value.target
            + np.asarray(cfg.target_offset_xyz, dtype=np.float64)
            + np.array((0.0, 0.0, cfg.stage2_tcp_goal_z_offset_m), dtype=np.float64)
        )
        tcp_goal_quality = _quality(float(np.linalg.norm(value.tcp - tcp_goal)), 0.25)
        proximity = _quality(value.tcp_to_obj, 0.10)
        tcp_obj_quality = proximity
        closed = 1.0
        held = proximity
        if cfg.use_gripper_caging:
            closed = float(np.clip((1.0 - value.gripper_opening) / cfg.closed_gripper_scale, 0.0, 1.0))
            held = proximity * closed
            tcp_obj_quality = proximity * (0.50 + 0.50 * closed)
        return (
            obj_weight * obj_quality
            + cfg.stage2_tcp_goal_weight * tcp_goal_quality
            + cfg.stage2_tcp_obj_weight * tcp_obj_quality
            + cfg.stage2_closed_weight * closed
            + cfg.stage2_held_weight * held
        )
    if stage == TERMINAL_STAGE:
        return 1.0
    raise ValueError("invalid stage")


def candidate(
    stage: int,
    value: GraspInsertFeatures,
    config: GraspInsertConfig,
    *,
    was_candidate: bool,
) -> bool:
    if stage == 0:
        xy_limit = (
            config.pregrasp_xy_exit_m if was_candidate else config.pregrasp_xy_enter_m
        )
        return bool(
            value.pregrasp_xy_distance <= xy_limit
            and config.pregrasp_height_low_m
            <= value.tcp_above_obj
            <= config.pregrasp_height_high_m
            and (not config.use_gripper_caging or value.gripper_opening >= config.candidate_open_min)
        )
    if stage == 1:
        closed_ok = not config.use_gripper_caging or value.gripper_opening <= config.candidate_closed_max
        if config.capture_lift_enter_m > 0.0:
            threshold = config.capture_move_exit_m if was_candidate else config.capture_lift_enter_m
            return bool(value.object_lift >= threshold and value.tcp_to_obj <= 0.070 and closed_ok)
        threshold = config.capture_move_exit_m if was_candidate else config.capture_move_enter_m
        progress = _capture_progress(value, config)
        return bool(progress >= threshold and value.tcp_to_obj <= 0.060 and closed_ok)
    if stage == 2:
        threshold = config.insert_exit_m if was_candidate else config.insert_enter_m
        return bool(value.obj_to_target <= threshold)
    return False


def smoothness_penalty(
    action: npt.ArrayLike,
    previous: npt.ArrayLike,
    before_previous: npt.ArrayLike,
    config: GraspInsertConfig,
) -> float:
    action_v = np.asarray(action, dtype=np.float64)
    previous_v = np.asarray(previous, dtype=np.float64)
    before_v = np.asarray(before_previous, dtype=np.float64)
    if action_v.shape != (4,) or previous_v.shape != (4,) or before_v.shape != (4,):
        raise ValueError("grasp-and-insert actions must be finite four-vectors")
    if not all(np.all(np.isfinite(v)) for v in (action_v, previous_v, before_v)):
        raise ValueError("actions must be finite")
    rate = float(np.mean(np.square(action_v - previous_v)))
    curve = float(np.mean(np.square(action_v - 2.0 * previous_v + before_v)))
    boundary = np.clip(
        (np.abs(action_v) - config.boundary_free_limit)
        / (config.boundary_hard_limit - config.boundary_free_limit),
        0.0,
        1.0,
    )
    return float(
        config.action_rate_weight * rate
        + config.action_curve_weight * curve
        + config.boundary_weight * np.mean(np.square(boundary))
    )


@dataclass
class GraspInsertState:
    stage: int = 0
    stable_count: int = 0
    candidate_epoch: int = 0
    request_armed: bool = True
    previous_quality: float | None = None
    previous_stage_qualities: dict[int, float] = field(default_factory=dict)
    previous_action: Array = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    before_previous_action: Array = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    def reset(self, quality: float) -> None:
        self.stage = 0
        self.stable_count = 0
        self.candidate_epoch = 0
        self.request_armed = True
        self.previous_quality = float(quality)
        self.previous_stage_qualities.clear()
        remember_stage_quality(self.previous_stage_qualities, 0, quality)
        self.previous_action.fill(0.0)
        self.before_previous_action.fill(0.0)


@dataclass(frozen=True)
class StepResult:
    reward: float
    request_due: bool
    stage: int


def step(
    state: GraspInsertState,
    value: GraspInsertFeatures,
    action: npt.ArrayLike,
    config: GraspInsertConfig,
) -> StepResult:
    if state.stage == TERMINAL_STAGE:
        return StepResult(0.0, False, state.stage)
    mode = reward_form_mode(config)
    if mode == STAGED_RESET:
        quality = stage_quality(state.stage, value, config)
        previous_quality = quality if state.previous_quality is None else state.previous_quality
        dense = config.dense_scales[state.stage] * (
            config.gamma * quality - previous_quality
        )
    else:
        dense, quality = dense_sum(
            mode=mode,
            current_stage=state.stage,
            active_stage_count=ACTIVE_STAGE_COUNT,
            dense_scales=config.dense_scales,
            gamma=config.gamma,
            previous_stage_qualities=state.previous_stage_qualities,
            quality_fn=lambda stage: stage_quality(stage, value, config),
        )
    action_v = np.asarray(action, dtype=np.float64)
    penalty = smoothness_penalty(
        action_v, state.previous_action, state.before_previous_action, config
    )
    if state.stage == 2 and config.stage2_precision_action_weight > 0.0:
        near_goal = _quality(value.obj_to_target, config.stage2_precision_scale_m)
        penalty += (
            config.stage2_precision_action_weight
            * near_goal
            * float(np.mean(np.square(action_v[:3])))
        )
    was_candidate = state.stable_count > 0
    if candidate(state.stage, value, config, was_candidate=was_candidate):
        state.stable_count += 1
    else:
        if state.stable_count > 0 or not state.request_armed:
            state.candidate_epoch += 1
        state.stable_count = 0
        state.request_armed = True
    due = bool(
        state.request_armed
        and state.stable_count >= config.required_holds[state.stage]
    )
    state.previous_quality = quality
    state.before_previous_action = state.previous_action.copy()
    state.previous_action = action_v.copy()
    return StepResult(sparse_step_reward(mode, dense - penalty), due, state.stage)


def apply_qwen_decision(
    state: GraspInsertState,
    decision: bool,
    config: GraspInsertConfig,
    *,
    bypass_candidate: bool = False,
) -> float:
    if state.stage == TERMINAL_STAGE:
        raise ValueError("terminal stage has no pending decision")
    if (
        not bypass_candidate
        and (
            not state.request_armed
            or state.stable_count < config.required_holds[state.stage]
        )
    ):
        raise ValueError("there is no armed Qwen candidate")
    state.request_armed = False
    if bool(decision):
        old_stage = state.stage
        state.stage += 1
        state.stable_count = 0
        state.request_armed = True
        state.previous_quality = None
        return transition_reward(
            reward_form_mode(config),
            original_bonus=config.transition_bonuses[old_stage],
            new_stage=state.stage,
            terminal_stage=TERMINAL_STAGE,
        )
    state.candidate_epoch += 1
    state.stable_count = 0
    # Fail closed, but re-arm the still-valid physical candidate.  The sampled
    # Mandatory-Qwen adapter returns UNKNOWN/REJECT between real calls and
    # permits a transition only after an explicit Qwen SUCCESS; without
    # re-arming here its documented retry interval could never fire unless the
    # policy first destroyed and recreated the candidate.
    state.request_armed = True
    return 0.0


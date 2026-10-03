"""Four-stage observable reward machine for tool-mediated manipulation.

The actor never receives native reward, contact flags, or hidden joint state.
Stages use only TCP pose, gripper opening, tool pose, secondary-object pose,
goal pose, and action history; a Rule or Mandatory-Qwen gate authorizes each
candidate transition.
"""

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
class ToolEffectConfig(GraspInsertConfig):
    dense_scales: tuple[float, ...] = (6.0, 36.0, 150.0, 240.0)
    transition_bonuses: tuple[float, ...] = (8.0, 55.0, 210.0, 320.0)
    required_holds: tuple[int, ...] = (2, 3, 2, 1)
    use_gripper_caging: bool = True
    grasp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.02)
    effect_offset_xyz: tuple[float, float, float] = (0.05, 0.0, 0.0)
    rotate_tool_landmarks: bool = False
    tool_quaternion_order: str = "xyzw"
    effect_axis_enter_limits_xyz: tuple[float, float, float] | None = None
    effect_axis_exit_limits_xyz: tuple[float, float, float] | None = None
    capture_lift_enter_m: float = 0.025
    capture_move_exit_m: float = 0.014
    # Explicit stage-2 retention weights. Defaults reproduce the historical
    # 0.65-contact/0.20-held/0.15-grasp decomposition exactly.
    stage2_closed_weight: float = 0.0
    stage2_held_weight: float = 0.20
    stage2_grasp_weight: float = 0.15
    stage2_conjunctive_contact: bool = False
    effect_enter_m: float = 0.070
    effect_exit_m: float = 0.090
    secondary_enter_m: float = 0.100
    secondary_exit_m: float = 0.120
    require_effect_x_at_secondary: bool = False
    alignment_mode: str = "none"
    alignment_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alignment_scale_m: float = 0.18
    alignment_weight: float = 0.0
    # Optional decomposition for tasks where alignment and tool insertion are
    # physically distinct. Defaults preserve every historical revision.
    stage2_alignment_only_candidate: bool = False
    stage2_alignment_enter_m: float = 0.065
    stage2_alignment_exit_m: float = 0.085
    # Stage-3 defaults reproduce 0.70 outcome + 0.20 contact + 0.10 held.
    stage3_contact_weight: float = 0.20
    stage3_held_weight: float = 0.10
    # Persistent observable held-tool state cost after verified capture.
    post_capture_invariant_penalty_weight: float = 0.0
    post_capture_invariant_floor: float = 0.15
    post_capture_invariant_penalty_stage_start: int = 2
    # A positive MetaWorld gripper action maintains closure. This optional
    # post-capture command cost prevents the held tool from being released in
    # later stages. Zero preserves every historical reward revision.
    post_capture_gripper_action_penalty_weight: float = 0.0
    post_capture_gripper_action_min_command: float = 0.25
    post_capture_gripper_action_penalty_stage_start: int = 2
    # Optional persistent Stage-3 state costs. They use the same observable
    # distances as the potential and are disabled for historical revisions.
    stage3_contact_invariant_penalty_weight: float = 0.0
    stage3_contact_invariant_floor: float = 0.35
    stage3_contact_invariant_scale_m: float = 0.14
    stage3_outcome_invariant_penalty_weight: float = 0.0
    stage3_outcome_invariant_floor: float = 0.50
    stage3_outcome_invariant_scale_m: float = 0.25
    # Optional action-space guidance for the effect suffix. Before contact it
    # points the held tool endpoint toward the secondary object; after contact
    # it points the secondary object toward its observable terminal target.
    # Zero keeps every historical reward exactly unchanged.
    stage3_action_guidance_penalty_weight: float = 0.0
    stage3_action_guidance_min_projection: float = 0.25
    stage3_action_guidance_deadband_m: float = 0.03
    stage3_action_guidance_contact_switch_m: float = 0.09
    # Optional component-wise target prevents a correct Cartesian component
    # from masking an opposite command on another axis.
    stage3_action_target_penalty_weight: float = 0.0
    stage3_action_target_magnitude: float = 0.75
    # Optional vector targets for approach, capture/lift, and tool contact.
    # Defaults preserve all historical revisions.
    prefix_action_target_penalty_weights: tuple[float, float, float] = (
        0.0, 0.0, 0.0
    )
    prefix_action_target_magnitude: float = 0.75
    # A near-grasp Stage-1 positive command closes the gripper before lift.
    stage1_gripper_action_penalty_weight: float = 0.0
    stage1_gripper_action_min_command: float = 0.50
    stage1_gripper_action_deadband_m: float = 0.04

    # Optional long-range components keep potential gradients visible before
    # the policy reaches the short-range manipulation basin. Zero preserves
    # every v1--v4 checkpoint exactly.
    multiscale_grasp_weight: float = 0.0
    multiscale_effect_weight: float = 0.0
    multiscale_outcome_weight: float = 0.0
    multiscale_pregrasp_weight: float = 0.0
    grasp_coarse_scale_m: float = 0.24
    effect_coarse_scale_m: float = 0.55
    outcome_coarse_scale_m: float = 0.50
    pregrasp_coarse_scale_m: float = 0.40
    # Once the secondary object itself is in the terminal region, continued
    # tool contact is not part of MetaWorld's task outcome. Older revisions
    # retain their stricter association check by default.
    final_require_tool_association: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (
            len(self.dense_scales)
            == len(self.transition_bonuses)
            == len(self.required_holds)
            == ACTIVE_STAGE_COUNT
        ):
            raise ValueError("tool-effect config requires four active-stage parameters")
        for name, offset in (
            ("effect_offset_xyz", self.effect_offset_xyz),
            ("alignment_offset_xyz", self.alignment_offset_xyz),
        ):
            vector = np.asarray(offset, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite three-vector")
        if self.tool_quaternion_order not in {"wxyz", "xyzw"}:
            raise ValueError("tool_quaternion_order must be wxyz or xyzw")
        axis_limits = (
            self.effect_axis_enter_limits_xyz,
            self.effect_axis_exit_limits_xyz,
        )
        if (axis_limits[0] is None) != (axis_limits[1] is None):
            raise ValueError("effect-axis enter/exit limits must be set together")
        if axis_limits[0] is not None:
            enter = np.asarray(axis_limits[0], dtype=np.float64)
            exit_ = np.asarray(axis_limits[1], dtype=np.float64)
            if (
                enter.shape != (3,)
                or exit_.shape != (3,)
                or not np.all(np.isfinite(enter))
                or not np.all(np.isfinite(exit_))
                or np.any(enter <= 0.0)
                or np.any(exit_ < enter)
            ):
                raise ValueError("effect-axis limits require 0 < enter <= exit")
        retention_weights = (
            self.stage2_closed_weight,
            self.stage2_held_weight,
            self.stage2_grasp_weight,
        )
        if any(weight < 0.0 for weight in retention_weights):
            raise ValueError("stage2 retention weights must be non-negative")
        if self.alignment_weight + sum(retention_weights) >= 1.0:
            raise ValueError("stage2 weights must leave positive contact weight")
        if self.alignment_mode not in {"none", "tcp_xz_target", "tcp_yz_secondary"}:
            raise ValueError("unsupported alignment_mode")
        if self.alignment_scale_m <= 0.0:
            raise ValueError("alignment_scale_m must be positive")
        if not 0.0 <= self.alignment_weight <= 0.65:
            raise ValueError("alignment_weight must be in [0, 0.65]")
        if not 0.0 < self.stage2_alignment_enter_m < self.stage2_alignment_exit_m:
            raise ValueError("stage2 alignment hysteresis is invalid")
        if (
            self.stage3_contact_weight < 0.0
            or self.stage3_held_weight < 0.0
            or self.stage3_contact_weight + self.stage3_held_weight >= 1.0
        ):
            raise ValueError("stage3 contact/held weights must leave positive outcome weight")
        if self.post_capture_invariant_penalty_weight < 0.0:
            raise ValueError("post-capture invariant weight must be non-negative")
        if not 0.0 < self.post_capture_invariant_floor <= 1.0:
            raise ValueError("post-capture invariant floor must be in (0, 1]")
        if self.post_capture_invariant_penalty_stage_start not in {2, 3}:
            raise ValueError("post-capture invariant stage start must be 2 or 3")
        if self.post_capture_gripper_action_penalty_weight < 0.0:
            raise ValueError("post-capture gripper-action weight must be non-negative")
        if not 0.0 < self.post_capture_gripper_action_min_command <= 1.0:
            raise ValueError("post-capture gripper-action minimum must be in (0, 1]")
        if self.post_capture_gripper_action_penalty_stage_start not in {2, 3}:
            raise ValueError("post-capture gripper-action stage start must be 2 or 3")
        for name, weight in (
            ("stage3 contact invariant weight", self.stage3_contact_invariant_penalty_weight),
            ("stage3 outcome invariant weight", self.stage3_outcome_invariant_penalty_weight),
        ):
            if weight < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, floor in (
            ("stage3 contact invariant floor", self.stage3_contact_invariant_floor),
            ("stage3 outcome invariant floor", self.stage3_outcome_invariant_floor),
        ):
            if not 0.0 < floor <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name, scale in (
            ("stage3 contact invariant scale", self.stage3_contact_invariant_scale_m),
            ("stage3 outcome invariant scale", self.stage3_outcome_invariant_scale_m),
        ):
            if scale <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.stage3_action_guidance_penalty_weight < 0.0:
            raise ValueError("stage-3 action-guidance weight must be non-negative")
        if not 0.0 < self.stage3_action_guidance_min_projection <= 1.0:
            raise ValueError("stage-3 action-guidance minimum projection must be in (0, 1]")
        if self.stage3_action_guidance_deadband_m <= 0.0:
            raise ValueError("stage-3 action-guidance deadband must be positive")
        if self.stage3_action_guidance_contact_switch_m <= 0.0:
            raise ValueError("stage-3 action-guidance contact switch must be positive")
        if self.stage3_action_target_penalty_weight < 0.0:
            raise ValueError("stage-3 action-guidance target weight must be non-negative")
        if not 0.0 < self.stage3_action_target_magnitude <= 1.0:
            raise ValueError("stage-3 action-guidance target magnitude must be in (0, 1]")
        if (
            len(self.prefix_action_target_penalty_weights) != 3
            or any(
                not np.isfinite(weight) or weight < 0.0
                for weight in self.prefix_action_target_penalty_weights
            )
        ):
            raise ValueError(
                "prefix action-target weights require three non-negative values"
            )
        if not 0.0 < self.prefix_action_target_magnitude <= 1.0:
            raise ValueError("prefix action-target magnitude must be in (0, 1]")
        if self.stage1_gripper_action_penalty_weight < 0.0:
            raise ValueError("stage-1 gripper-action weight must be non-negative")
        if not 0.0 < self.stage1_gripper_action_min_command <= 1.0:
            raise ValueError("stage-1 gripper-action minimum must be in (0, 1]")
        if self.stage1_gripper_action_deadband_m <= 0.0:
            raise ValueError("stage-1 gripper-action deadband must be positive")
        for name, weight in (
            ("multiscale_pregrasp_weight", self.multiscale_pregrasp_weight),
            ("multiscale_grasp_weight", self.multiscale_grasp_weight),
            ("multiscale_effect_weight", self.multiscale_effect_weight),
            ("multiscale_outcome_weight", self.multiscale_outcome_weight),
        ):
            if not 0.0 <= weight <= 0.85:
                raise ValueError(f"{name} must be in [0, 0.85]")
        for name, scale in (
            ("pregrasp_coarse_scale_m", self.pregrasp_coarse_scale_m),
            ("grasp_coarse_scale_m", self.grasp_coarse_scale_m),
            ("effect_coarse_scale_m", self.effect_coarse_scale_m),
            ("outcome_coarse_scale_m", self.outcome_coarse_scale_m),
        ):
            if scale <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.effect_enter_m < self.effect_exit_m:
            raise ValueError("effect hysteresis is invalid")
        if not 0.0 < self.secondary_enter_m < self.secondary_exit_m:
            raise ValueError("secondary outcome hysteresis is invalid")


@dataclass(frozen=True)
class ToolEffectFeatures:
    tcp: Array
    tool: Array
    secondary: Array
    target: Array
    initial_tool: Array
    grasp_point: Array
    effect_point: Array
    effect_delta: Array
    secondary_target: Array
    gripper_opening: float
    tcp_to_pregrasp: float
    pregrasp_xy_distance: float
    tcp_above_grasp: float
    tcp_to_grasp: float
    tool_lift: float
    alignment_distance: float
    effect_to_secondary: float
    secondary_to_target: float


def _vec3(value: npt.ArrayLike, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def _rotate_quaternion(
    quaternion: npt.ArrayLike, vector: Array, order: str
) -> Array:
    """Rotate a body-frame landmark using an observed tool quaternion."""

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("tool quaternion must be a finite four-vector")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-9:
        raise ValueError("tool quaternion has zero norm")
    if order == "wxyz":
        w, x, y, z = quat / norm
    else:
        x, y, z, w = quat / norm
    rotation = np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return rotation @ vector


def features(
    tcp: npt.ArrayLike,
    tool: npt.ArrayLike,
    secondary: npt.ArrayLike,
    target: npt.ArrayLike,
    initial_tool: npt.ArrayLike,
    gripper_opening: float,
    config: ToolEffectConfig,
    tool_quaternion: npt.ArrayLike | None = None,
) -> ToolEffectFeatures:
    tcp_v = _vec3(tcp, "tcp")
    tool_v = _vec3(tool, "tool")
    secondary_v = _vec3(secondary, "secondary")
    target_v = _vec3(target, "target")
    initial_v = _vec3(initial_tool, "initial_tool")
    grasp_offset = np.asarray(config.grasp_offset_xyz, dtype=np.float64)
    effect_offset = np.asarray(config.effect_offset_xyz, dtype=np.float64)
    if config.rotate_tool_landmarks:
        if tool_quaternion is None:
            raise ValueError("a quaternion is required for body-frame tool landmarks")
        grasp_offset = _rotate_quaternion(
            tool_quaternion, grasp_offset, config.tool_quaternion_order
        )
        effect_offset = _rotate_quaternion(
            tool_quaternion, effect_offset, config.tool_quaternion_order
        )
    grasp = tool_v + grasp_offset
    effect = tool_v + effect_offset
    secondary_target = target_v + np.asarray(config.target_offset_xyz, dtype=np.float64)
    alignment_offset = np.asarray(config.alignment_offset_xyz, dtype=np.float64)
    if config.alignment_mode == "none":
        alignment_distance = 0.0
    elif config.alignment_mode == "tcp_xz_target":
        point = target_v + alignment_offset
        alignment_distance = float(np.linalg.norm(tcp_v[[0, 2]] - point[[0, 2]]))
    elif config.alignment_mode == "tcp_yz_secondary":
        point = secondary_v + alignment_offset
        alignment_distance = float(np.linalg.norm(tcp_v[[1, 2]] - point[[1, 2]]))
    else:
        raise ValueError("unsupported alignment_mode")
    pregrasp = grasp + np.array((0.0, 0.0, 0.10), dtype=np.float64)
    return ToolEffectFeatures(
        tcp=tcp_v,
        tool=tool_v,
        secondary=secondary_v,
        target=target_v,
        initial_tool=initial_v,
        grasp_point=grasp,
        effect_point=effect,
        effect_delta=effect - secondary_v,
        secondary_target=secondary_target,
        gripper_opening=float(gripper_opening),
        tcp_to_pregrasp=float(np.linalg.norm(tcp_v - pregrasp)),
        pregrasp_xy_distance=float(np.linalg.norm(tcp_v[:2] - grasp[:2])),
        tcp_above_grasp=float(tcp_v[2] - grasp[2]),
        tcp_to_grasp=float(np.linalg.norm(tcp_v - grasp)),
        tool_lift=float(tool_v[2] - initial_v[2]),
        alignment_distance=alignment_distance,
        effect_to_secondary=float(np.linalg.norm(effect - secondary_v)),
        secondary_to_target=float(np.linalg.norm(secondary_v - secondary_target)),
    )


def _quality(distance: float, scale: float) -> float:
    return float(1.0 - np.tanh(max(float(distance), 0.0) / scale))


def _open(value: ToolEffectFeatures, config: ToolEffectConfig) -> float:
    return float(np.clip(value.gripper_opening / config.open_gripper_scale, 0.0, 1.0))


def _closed(value: ToolEffectFeatures, config: ToolEffectConfig) -> float:
    return float(
        np.clip((1.0 - value.gripper_opening) / config.closed_gripper_scale, 0.0, 1.0)
    )


def stage_quality(stage: int, value: ToolEffectFeatures, config: ToolEffectConfig) -> float:
    grasp_fine = _quality(value.tcp_to_grasp, 0.08)
    grasp_coarse = _quality(value.tcp_to_grasp, config.grasp_coarse_scale_m)
    grasp = (
        (1.0 - config.multiscale_grasp_weight) * grasp_fine
        + config.multiscale_grasp_weight * grasp_coarse
    )
    held = grasp * _closed(value, config)
    if stage == 0:
        pregrasp_fine = _quality(value.tcp_to_pregrasp, 0.15)
        pregrasp_coarse = _quality(
            value.tcp_to_pregrasp, config.pregrasp_coarse_scale_m
        )
        pregrasp = (
            (1.0 - config.multiscale_pregrasp_weight) * pregrasp_fine
            + config.multiscale_pregrasp_weight * pregrasp_coarse
        )
        return 0.90 * pregrasp + 0.10 * _open(value, config)
    if stage == 1:
        lift = float(np.clip(value.tool_lift / config.capture_lift_enter_m, 0.0, 1.0))
        return 0.25 * grasp + 0.30 * held + 0.45 * lift
    if stage == 2:
        contact_fine = _quality(value.effect_to_secondary, 0.18)
        contact_coarse = _quality(
            value.effect_to_secondary, config.effect_coarse_scale_m
        )
        contact = (
            (1.0 - config.multiscale_effect_weight) * contact_fine
            + config.multiscale_effect_weight * contact_coarse
        )
        alignment = _quality(value.alignment_distance, config.alignment_scale_m)
        if config.stage2_conjunctive_contact:
            contact_target = (
                (1.0 - config.alignment_weight) * contact
                + config.alignment_weight * alignment
            )
            return float(np.sqrt(max(contact_target * held, 0.0)))
        contact_weight = 1.0 - (
            config.alignment_weight
            + config.stage2_closed_weight
            + config.stage2_held_weight
            + config.stage2_grasp_weight
        )
        return contact_weight * contact + config.alignment_weight * alignment + config.stage2_closed_weight * _closed(value, config) + config.stage2_held_weight * held + config.stage2_grasp_weight * grasp
    if stage == 3:
        outcome_fine = _quality(value.secondary_to_target, 0.25)
        outcome_coarse = _quality(
            value.secondary_to_target, config.outcome_coarse_scale_m
        )
        outcome = (
            (1.0 - config.multiscale_outcome_weight) * outcome_fine
            + config.multiscale_outcome_weight * outcome_coarse
        )
        contact = _quality(value.effect_to_secondary, 0.14)
        outcome_weight = 1.0 - (
            config.stage3_contact_weight + config.stage3_held_weight
        )
        return (
            outcome_weight * outcome
            + config.stage3_contact_weight * contact
            + config.stage3_held_weight * held
        )
    if stage == TERMINAL_STAGE:
        return 1.0
    raise ValueError("invalid tool-effect stage")


def candidate(
    stage: int,
    value: ToolEffectFeatures,
    config: ToolEffectConfig,
    *,
    was_candidate: bool,
) -> bool:
    axis_limits = (
        config.effect_axis_exit_limits_xyz
        if was_candidate
        else config.effect_axis_enter_limits_xyz
    )
    axis_ok = bool(
        axis_limits is None
        or np.all(np.abs(value.effect_delta) <= np.asarray(axis_limits))
    )
    directional = bool(
        not config.require_effect_x_at_secondary
        or value.effect_delta[0] >= -0.010
    )
    if stage == 0:
        xy = config.pregrasp_xy_exit_m if was_candidate else config.pregrasp_xy_enter_m
        return bool(
            value.pregrasp_xy_distance <= xy
            and config.pregrasp_height_low_m
            <= value.tcp_above_grasp
            <= config.pregrasp_height_high_m
            and value.gripper_opening >= config.candidate_open_min
        )
    if stage == 1:
        lift = config.capture_move_exit_m if was_candidate else config.capture_lift_enter_m
        return bool(
            value.tool_lift >= lift
            and value.tcp_to_grasp <= 0.075
            and value.gripper_opening <= config.candidate_closed_max
        )
    if stage == 2:
        if config.stage2_alignment_only_candidate:
            alignment_limit = (
                config.stage2_alignment_exit_m
                if was_candidate
                else config.stage2_alignment_enter_m
            )
            return bool(
                value.alignment_distance <= alignment_limit
                and value.tcp_to_grasp <= 0.100
                and value.gripper_opening <= config.candidate_closed_max
            )
        limit = config.effect_exit_m if was_candidate else config.effect_enter_m
        return bool(
            value.effect_to_secondary <= limit
            and value.tool_lift >= config.capture_move_exit_m
            and value.tcp_to_grasp <= 0.100
            and value.gripper_opening <= config.candidate_closed_max
            and directional
            and axis_ok
        )
    if stage == 3:
        limit = config.secondary_exit_m if was_candidate else config.secondary_enter_m
        if not config.final_require_tool_association:
            return bool(value.secondary_to_target <= limit)
        return bool(
            value.secondary_to_target <= limit
            and value.effect_to_secondary <= config.effect_exit_m
            and value.tool_lift >= config.capture_move_exit_m
            and value.tcp_to_grasp <= 0.110
            and value.gripper_opening <= config.candidate_closed_max
            and directional
            and axis_ok
        )
    return False


@dataclass
class ToolEffectState:
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


def step(
    state: ToolEffectState,
    value: ToolEffectFeatures,
    action: npt.ArrayLike,
    config: ToolEffectConfig,
) -> StepResult:
    if state.stage == TERMINAL_STAGE:
        return StepResult(0.0, False, state.stage)
    mode = reward_form_mode(config)
    quality = stage_quality(state.stage, value, config)
    previous = quality if state.previous_quality is None else state.previous_quality
    dense = config.dense_scales[state.stage] * (config.gamma * quality - previous)
    action_v = np.asarray(action, dtype=np.float64)
    penalty = smoothness_penalty(
        action_v, state.previous_action, state.before_previous_action, config
    )
    if state.stage <= 2:
        target_weight = config.prefix_action_target_penalty_weights[state.stage]
        desired: Array | None = None
        if target_weight > 0.0:
            if state.stage == 0:
                pregrasp = value.grasp_point + np.array(
                    (0.0, 0.0, 0.10), dtype=np.float64
                )
                desired = pregrasp - value.tcp
            elif state.stage == 1:
                if value.tcp_to_grasp > config.stage1_gripper_action_deadband_m:
                    desired = value.grasp_point - value.tcp
                elif value.gripper_opening <= config.candidate_closed_max:
                    desired = np.array((0.0, 0.0, 1.0), dtype=np.float64)
            else:
                desired = -value.effect_delta
        if desired is not None:
            desired_distance = float(np.linalg.norm(desired))
            if desired_distance > config.stage3_action_guidance_deadband_m:
                direction = desired / desired_distance
                target_action = config.prefix_action_target_magnitude * direction
                component_error = np.clip(action_v[:3], -1.0, 1.0) - target_action
                penalty += target_weight * float(np.mean(component_error**2))
    if (
        state.stage == 1
        and config.stage1_gripper_action_penalty_weight > 0.0
        and value.tcp_to_grasp <= config.stage1_gripper_action_deadband_m
    ):
        gripper_command = float(np.clip(action_v[3], -1.0, 1.0))
        normalized_shortfall = max(
            config.stage1_gripper_action_min_command - gripper_command,
            0.0,
        ) / (config.stage1_gripper_action_min_command + 1.0)
        penalty += (
            config.stage1_gripper_action_penalty_weight
            * normalized_shortfall**2
        )
    if (
        state.stage >= config.post_capture_invariant_penalty_stage_start
        and config.post_capture_invariant_penalty_weight > 0.0
    ):
        proximity = _quality(value.tcp_to_grasp, 0.08)
        held = proximity * _closed(value, config)
        normalized_shortfall = max(
            config.post_capture_invariant_floor - held,
            0.0,
        ) / config.post_capture_invariant_floor
        penalty += (
            config.post_capture_invariant_penalty_weight
            * normalized_shortfall * normalized_shortfall
        )
    if (
        state.stage >= config.post_capture_gripper_action_penalty_stage_start
        and config.post_capture_gripper_action_penalty_weight > 0.0
    ):
        gripper_command = float(np.clip(action_v[3], -1.0, 1.0))
        normalized_shortfall = max(
            config.post_capture_gripper_action_min_command - gripper_command,
            0.0,
        ) / (config.post_capture_gripper_action_min_command + 1.0)
        penalty += (
            config.post_capture_gripper_action_penalty_weight * normalized_shortfall**2
        )
    if state.stage == 3:
        for weight, floor, distance, scale in (
            (
                config.stage3_contact_invariant_penalty_weight,
                config.stage3_contact_invariant_floor,
                value.effect_to_secondary,
                config.stage3_contact_invariant_scale_m,
            ),
            (
                config.stage3_outcome_invariant_penalty_weight,
                config.stage3_outcome_invariant_floor,
                value.secondary_to_target,
                config.stage3_outcome_invariant_scale_m,
            ),
        ):
            if weight <= 0.0:
                continue
            state_quality = _quality(distance, scale)
            normalized_shortfall = max(
                floor - state_quality,
                0.0,
            ) / floor
            penalty += weight * normalized_shortfall * normalized_shortfall
        if (
            config.stage3_action_guidance_penalty_weight > 0.0
            or config.stage3_action_target_penalty_weight > 0.0
        ):
            if value.effect_to_secondary > config.stage3_action_guidance_contact_switch_m:
                desired = -value.effect_delta
            else:
                desired = value.secondary_target - value.secondary
            desired_distance = float(np.linalg.norm(desired))
            if desired_distance > config.stage3_action_guidance_deadband_m:
                direction = desired / desired_distance
                clipped_action = np.clip(action_v[:3], -1.0, 1.0)
                if config.stage3_action_guidance_penalty_weight > 0.0:
                    projection = float(np.dot(clipped_action, direction))
                    normalized_shortfall = max(
                        config.stage3_action_guidance_min_projection - projection,
                        0.0,
                    ) / (config.stage3_action_guidance_min_projection + 1.0)
                    penalty += (
                        config.stage3_action_guidance_penalty_weight
                        * normalized_shortfall * normalized_shortfall
                    )
                if config.stage3_action_target_penalty_weight > 0.0:
                    target_action = config.stage3_action_target_magnitude * direction
                    component_error = clipped_action - target_action
                    penalty += config.stage3_action_target_penalty_weight * float(
                        np.mean(component_error * component_error)
                    )
    if candidate(state.stage, value, config, was_candidate=state.stable_count > 0):
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


def apply_decision(
    state: ToolEffectState, decision: bool, config: ToolEffectConfig
) -> float:
    if (
        state.stage == TERMINAL_STAGE
        or not state.request_armed
        or state.stable_count < config.required_holds[state.stage]
    ):
        raise ValueError("there is no armed tool-effect candidate")
    state.request_armed = False
    if not bool(decision):
        state.candidate_epoch += 1
        state.stable_count = 0
        state.request_armed = True
        return 0.0
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

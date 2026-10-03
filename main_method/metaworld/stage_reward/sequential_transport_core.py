"""Five-stage causal grasp/clearance/transport/final reward CORE.

This keeps the shared stage-conditioned potential-shaping formulation while
splitting the physically distinct vertical-clearance and horizontal-transport
operations that the earlier diagonal waypoint merged into one bottleneck.
Only RGB-deployable proprioceptive quantities are used: TCP pose, object pose,
goal pose, gripper opening, and action history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .grasp_insert_core import GraspInsertConfig, smoothness_penalty
from .reward_form_modes import (
    STAGED_RESET,
    dense_sum,
    remember_stage_quality,
    reward_form_mode,
    sparse_step_reward,
    transition_reward,
)


ACTIVE_STAGE_COUNT = 5
STAGE_COUNT = 6
TERMINAL_STAGE = 5
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SequentialTransportConfig(GraspInsertConfig):
    dense_scales: tuple[float, ...] = (6.0, 26.0, 80.0, 110.0, 150.0)
    transition_bonuses: tuple[float, ...] = (8.0, 42.0, 120.0, 160.0, 220.0)
    required_holds: tuple[int, ...] = (2, 3, 3, 3, 1)
    # Dense approach landmark above the task-local grasp point. The default
    # preserves all historical revisions; hard spherical grasps may use a
    # nearer open-gripper approach basin before capture.
    pregrasp_offset_z_m: float = 0.10
    grasp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.02)
    success_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Known local offset for an off-center rigid-body observation landmark.
    observed_site_body_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # MetaWorld mixes MuJoCo wxyz and SciPy xyzw across tasks.
    observed_site_quaternion_order: str = "wxyz"
    # Axis scaling used by the task's observable terminal geometry.
    final_distance_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Optional obstacle-aware TCP waypoint expressed from the visible target.
    # None preserves the historical final-XY / safe-Z construction.
    transport_tcp_target_offset_xyz: tuple[float, float, float] | None = None
    # Normal placement aligns XY; side insertion aligns YZ before moving in X.
    transport_alignment_axes: tuple[int, int] = (0, 1)
    final_tcp_obj_weight: float = 0.15
    final_closed_weight: float = 0.10
    final_open_weight: float = 0.0
    # Optional factor-coupling controls. Defaults preserve historical revisions.
    stage0_open_weight: float = 0.10
    stage1_conjunctive_capture: bool = False
    # Observable association mixtures for clearance/transport. Zero preserves
    # all historical stage qualities exactly.
    stage2_retention_weight: float = 0.0
    stage3_retention_weight: float = 0.0
    # When enabled, retention is conjunctive with task progress rather than an
    # additive substitute for it. This prevents a policy from collecting a
    # useful transport quality after visibly losing the manipulated object.
    stage2_conjunctive_retention: bool = False
    stage3_conjunctive_retention: bool = False
    # Generic causal factor couplings. Zero preserves historical revisions:
    # task progress cannot be replaced by mere gripper proximity, and a
    # task-local TCP landmark cannot replace the object outcome.
    stage2_progress_coupling_weight: float = 0.0
    stage3_tcp_target_coupling_weight: float = 0.0
    # Optional terminal-aligned transport shaping.  Zero/False preserve every
    # historical revision.  This lets a transport stage learn the exact final
    # object outcome before a sparse verification-only stage, without changing
    # the outer stage-conditioned potential formulation.
    stage3_final_outcome_weight: float = 0.0
    # Optional direct convex blend of the same observable final-outcome
    # potential. Unlike ``stage3_final_outcome_weight`` (which factor-couples
    # outcome with the current transport basin), this preserves a standalone
    # terminal gradient when auxiliary TCP/retention quality is initially poor.
    # Zero preserves every historical revision.
    stage3_final_outcome_direct_weight: float = 0.0
    stage3_candidate_uses_final_distance: bool = False
    # Some tasks are complete once the visible object outcome is achieved;
    # continued TCP proximity/gripper retention is then not part of the task.
    # When enabled, the stage-3 candidate uses exactly the same terminal
    # outcome predicate as stage 4 and omits only the obsolete actuator-
    # association conjunct. False preserves every historical revision.
    stage3_terminal_outcome_only_candidate: bool = False

    # Ones preserve historical rewards while allowing suffix-only stabilization.
    action_penalty_stage_multipliers: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)

    # If stage 3 already satisfies the exact terminal geometry, stage 4 is a
    # verification-only micro-event: freeze simulation and query its gate on
    # the same evidence instead of executing an otherwise untrained action.
    # False preserves all historical revisions and ordinary inference.
    frozen_terminal_verification: bool = False

    # A verified capture creates an observable causal invariant: during later
    # transport stages the gripper should remain associated with the object.
    # This is a state term R_k(x_t), not a transition predicate or privileged
    # simulator signal. Zero disables it for historical/release tasks.
    post_capture_invariant_penalty_weight: float = 0.0
    post_capture_invariant_floor: float = 0.15
    # Earliest later stage where the persistent association cost is active.
    # Two preserves all historical behavior; task revisions may defer it.
    post_capture_invariant_penalty_stage_start: int = 2
    # A positive MetaWorld gripper action maintains closure. This optional
    # post-capture command cost prevents an otherwise solved transport prefix
    # from actively opening in later stages. Zero preserves historical runs.
    post_capture_gripper_action_penalty_weight: float = 0.0
    post_capture_gripper_action_min_command: float = 0.25
    post_capture_gripper_action_penalty_stage_start: int = 2
    # Incremental potential shaping can stop providing a learning signal after
    # a policy moves away from a Stage-3 waypoint and settles there. This
    # optional observable state cost stays active until the TCP reaches the
    # configured transport basin. Zero preserves every historical revision.
    stage3_tcp_target_invariant_penalty_weight: float = 0.0
    stage3_tcp_target_invariant_floor: float = 0.35
    stage3_tcp_target_invariant_scale_m: float = 0.18
    # Persistent observable object-outcome cost for Stage 3. Zero preserves
    # every historical reward revision.
    stage3_final_outcome_invariant_penalty_weight: float = 0.0
    stage3_final_outcome_invariant_floor: float = 0.50
    stage3_final_outcome_invariant_scale_m: float = 0.25
    # Optional action-space guidance for the transport suffix. It penalizes an
    # action whose Cartesian component fails to project toward the observable
    # TCP waypoint. Zero keeps every historical reward exactly unchanged.
    stage3_action_guidance_penalty_weight: float = 0.0
    stage3_action_guidance_min_projection: float = 0.25
    stage3_action_guidance_deadband_m: float = 0.03
    # Historical rewards guide the TCP toward its transport waypoint. Tasks
    # whose Stage 3 directly optimizes terminal object geometry may instead
    # guide the held outcome object toward its exact final point.
    stage3_action_direction_mode: str = "transport_tcp"
    # A dot-product constraint can hide a wrong-sign Cartesian component
    # behind progress on another axis. This optional vector target makes each
    # commanded component match the observable waypoint direction. Zero
    # preserves every historical reward revision.
    stage3_action_target_penalty_weight: float = 0.0
    stage3_action_target_magnitude: float = 0.75
    # Optional vector targets for stages 0/1/2 preserve the approach, capture,
    # and clearance prefix while a rare transport suffix is refined. The three
    # weights correspond to those stages and default to zero.
    prefix_action_target_penalty_weights: tuple[float, float, float] = (
        0.0, 0.0, 0.0
    )
    prefix_action_target_magnitude: float = 0.75
    # Optional phase-aware Stage-1 guidance: approach the observable grasp
    # point, close while near it, then move along the configured clearance axis.
    # All weights default to zero and preserve historical reward revisions.
    stage1_capture_action_guidance_penalty_weight: float = 0.0
    stage1_capture_action_guidance_min_projection: float = 0.25
    stage1_capture_action_guidance_grasp_deadband_m: float = 0.04
    stage1_capture_gripper_action_penalty_weight: float = 0.0
    stage1_capture_gripper_action_min_command: float = 0.50
    # Keep the gripper open while it is still outside the same grasp deadband.
    # Zero weight preserves every historical reward revision.
    stage1_capture_far_open_gripper_action_penalty_weight: float = 0.0
    stage1_capture_far_open_gripper_action_max_command: float = -0.50
    # After a verified pregrasp, stage 1 should recover toward the grasp point
    # rather than accepting a one-time potential drop and remaining far away.
    # This is observable state reward R_1(x_t), never a transition predicate.
    capture_proximity_invariant_penalty_weight: float = 0.0
    capture_proximity_invariant_floor: float = 0.20


    # Coarse-to-fine terminal potential. Zero blend preserves old revisions.
    final_fine_outcome_weight: float = 0.0
    final_fine_scale_m: float = 0.07
    # Couple terminal outcome to observable object retention. Zero preserves
    # historical revisions and tasks that legitimately require release.
    final_conjunctive_retention_weight: float = 0.0
    clearance_axis: int = 2
    clearance_direction: float = 1.0
    clearance_progress_mode: str = "axis"
    capture_progress_mode: str = "clearance"
    # Preserve revisions <= 11, whose waypoint stage used the capture threshold.
    # Revision 12 explicitly switches this to the task-axis clearance threshold.
    stage2_clearance_candidate: bool = False
    clearance_lift_enter_m: float = 0.10
    clearance_lift_exit_m: float = 0.075
    transport_min_lift_m: float = 0.075
    transport_xy_enter_m: float = 0.060
    transport_xy_exit_m: float = 0.085
    final_enter_m: float = 0.045
    final_exit_m: float = 0.060
    terminal_mode: str = "distance"
    terminal_axis_enter_m: float = 0.100
    terminal_axis_exit_m: float = 0.080
    preterminal_z_margin_m: float = 0.020
    # Optional task-calibrated stable grasp aperture. None preserves the
    # historical monotone "more closed is better" quality.
    held_gripper_target: float | None = None
    held_gripper_scale: float = 0.25
    # Lower aperture bound for capture/retention candidates. Zero preserves
    # every historical transition predicate.
    candidate_capture_open_min: float = 0.0
    terminal_z_offset_m: float = 0.0
    closed_gripper_scale: float = 0.45
    use_tcp_waypoints: bool = False
    require_tcp_waypoint_candidates: bool = True
    safe_tcp_height_above_initial_m: float = 0.0
    vertical_tcp_enter_m: float = 0.070
    vertical_tcp_exit_m: float = 0.095
    transport_tcp_enter_m: float = 0.080
    transport_tcp_exit_m: float = 0.110
    extraction_transport_at_initial_xy: bool = False

    def _valid_capture_progress_modes(self) -> frozenset[str]:
        return frozenset(("clearance", "displacement"))

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (
            len(self.dense_scales)
            == len(self.transition_bonuses)
            == len(self.required_holds)
            == ACTIVE_STAGE_COUNT
        ):
            raise ValueError("sequential transport requires five active-stage parameters")
        for name, value in (
            ("grasp_offset_xyz", self.grasp_offset_xyz),
            ("success_offset_xyz", self.success_offset_xyz),
            ("observed_site_body_offset_xyz", self.observed_site_body_offset_xyz),
            ("final_distance_scale_xyz", self.final_distance_scale_xyz),
        ):
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite three-vector")
        if self.observed_site_quaternion_order not in {"wxyz", "xyzw"}:
            raise ValueError("observed site quaternion order must be wxyz or xyzw")
        if any(scale <= 0.0 for scale in self.final_distance_scale_xyz):
            raise ValueError("final distance scales must be positive")
        if self.clearance_axis not in {0, 1, 2}:
            raise ValueError("clearance_axis must be an XYZ axis")
        if not np.isfinite(self.pregrasp_offset_z_m) or self.pregrasp_offset_z_m <= 0.0:
            raise ValueError("pregrasp_offset_z_m must be positive and finite")
        if self.clearance_direction not in {-1.0, 1.0}:
            raise ValueError("clearance_direction must be -1 or +1")
        if self.clearance_progress_mode not in {"axis", "goal"}:
            raise ValueError("clearance_progress_mode must be axis or goal")
        if self.capture_progress_mode not in {"clearance", "displacement"}:
            raise ValueError("capture_progress_mode must be clearance or displacement")
        if not 0.0 < self.clearance_lift_exit_m < self.clearance_lift_enter_m:
            raise ValueError("clearance lift hysteresis is invalid")
        if not 0.0 < self.transport_xy_enter_m < self.transport_xy_exit_m:
            raise ValueError("transport XY hysteresis is invalid")
        if not 0.0 < self.final_enter_m < self.final_exit_m:
            raise ValueError("final distance hysteresis is invalid")
        if self.terminal_mode not in {
            "distance", "z_above_target", "xy_below_target", "axis_progress"
        }:
            raise ValueError("unsupported terminal_mode")
        if not 0.0 < self.terminal_axis_exit_m < self.terminal_axis_enter_m:
            raise ValueError("terminal axis-progress hysteresis is invalid")
        if (
            len(self.action_penalty_stage_multipliers) != ACTIVE_STAGE_COUNT
            or any(value < 0.0 for value in self.action_penalty_stage_multipliers)
        ):
            raise ValueError(
                "action-penalty stage multipliers must contain five non-negative values"
            )
        if self.transport_tcp_target_offset_xyz is not None:
            offset = np.asarray(self.transport_tcp_target_offset_xyz, dtype=np.float64)
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise ValueError("transport TCP target offset must be a finite three-vector")
        axes = tuple(self.transport_alignment_axes)
        if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in {0, 1, 2} for axis in axes):
            raise ValueError("transport_alignment_axes must contain two distinct XYZ axes")
        if not 0.0 <= self.stage0_open_weight <= 1.0:
            raise ValueError("stage0_open_weight must be in [0, 1]")
        if (
            not 0.0 <= self.stage2_retention_weight <= 1.0
            or not 0.0 <= self.stage3_retention_weight <= 1.0
            or not 0.0 <= self.final_fine_outcome_weight <= 1.0
            or not 0.0 <= self.stage3_final_outcome_weight <= 1.0
            or not 0.0 <= self.final_conjunctive_retention_weight <= 1.0
            or not 0.0 <= self.stage2_progress_coupling_weight <= 1.0
            or not 0.0 <= self.stage3_tcp_target_coupling_weight <= 1.0
            or not 0.0 <= self.stage3_final_outcome_direct_weight <= 1.0
        ):
            raise ValueError("retention, coupling, and fine-outcome weights must be in [0, 1]")
        if self.post_capture_invariant_penalty_weight < 0.0:
            raise ValueError("post-capture invariant penalty weight must be non-negative")
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
        if self.stage3_tcp_target_invariant_penalty_weight < 0.0:
            raise ValueError("stage-3 TCP-target invariant weight must be non-negative")
        if not 0.0 < self.stage3_tcp_target_invariant_floor <= 1.0:
            raise ValueError("stage-3 TCP-target invariant floor must be in (0, 1]")
        if self.stage3_tcp_target_invariant_scale_m <= 0.0:
            raise ValueError("stage-3 TCP-target invariant scale must be positive")
        if self.stage3_final_outcome_invariant_penalty_weight < 0.0:
            raise ValueError("stage-3 final-outcome invariant weight must be non-negative")
        if not 0.0 < self.stage3_final_outcome_invariant_floor <= 1.0:
            raise ValueError("stage-3 final-outcome invariant floor must be in (0, 1]")
        if self.stage3_final_outcome_invariant_scale_m <= 0.0:
            raise ValueError("stage-3 final-outcome invariant scale must be positive")
        if self.stage3_action_guidance_penalty_weight < 0.0:
            raise ValueError("stage-3 action-guidance weight must be non-negative")
        if not 0.0 < self.stage3_action_guidance_min_projection <= 1.0:
            raise ValueError("stage-3 action-guidance minimum projection must be in (0, 1]")
        if self.stage3_action_guidance_deadband_m <= 0.0:
            raise ValueError("stage-3 action-guidance deadband must be positive")
        if self.stage3_action_direction_mode not in {"transport_tcp", "final_outcome"}:
            raise ValueError(
                "stage-3 action-guidance direction mode must be transport_tcp or final_outcome"
            )
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
        if self.stage1_capture_action_guidance_penalty_weight < 0.0:
            raise ValueError("stage-1 capture action-guidance weight must be non-negative")
        if not 0.0 < self.stage1_capture_action_guidance_min_projection <= 1.0:
            raise ValueError("stage-1 capture action-guidance projection must be in (0, 1]")
        if self.stage1_capture_action_guidance_grasp_deadband_m <= 0.0:
            raise ValueError("stage-1 capture action-guidance deadband must be positive")
        if self.stage1_capture_gripper_action_penalty_weight < 0.0:
            raise ValueError("stage-1 capture gripper-action weight must be non-negative")
        if not 0.0 < self.stage1_capture_gripper_action_min_command <= 1.0:
            raise ValueError("stage-1 capture gripper-action minimum must be in (0, 1]")
        if self.stage1_capture_far_open_gripper_action_penalty_weight < 0.0:
            raise ValueError("stage-1 capture far-open gripper-action weight must be non-negative")
        if not -1.0 <= self.stage1_capture_far_open_gripper_action_max_command < 0.0:
            raise ValueError("stage-1 capture far-open gripper-action maximum must be in [-1, 0)")
        if self.capture_proximity_invariant_penalty_weight < 0.0:
            raise ValueError("capture-proximity invariant weight must be non-negative")
        if not 0.0 < self.capture_proximity_invariant_floor <= 1.0:
            raise ValueError("capture-proximity invariant floor must be in (0, 1]")
        if self.final_fine_scale_m <= 0.0:
            raise ValueError("final_fine_scale_m must be positive")
        if (
            self.final_tcp_obj_weight < 0.0
            or self.final_closed_weight < 0.0
            or self.final_open_weight < 0.0
            or self.final_tcp_obj_weight
            + self.final_closed_weight
            + self.final_open_weight
            >= 1.0
        ):
            raise ValueError("final-stage weights must be non-negative and sum below one")
        if self.frozen_terminal_verification:
            if not self.stage3_candidate_uses_final_distance:
                raise ValueError(
                    "frozen terminal verification requires terminal-aligned stage 3"
                )
            if self.required_holds[4] != 1:
                raise ValueError(
                    "frozen terminal verification requires a one-frame terminal hold"
                )
        if self.closed_gripper_scale <= 0.0:
            raise ValueError("closed_gripper_scale must be positive")
        if self.held_gripper_target is not None:
            if not 0.0 <= self.held_gripper_target <= 1.0:
                raise ValueError("held_gripper_target must be in [0, 1]")
            if self.held_gripper_scale <= 0.0:
                raise ValueError("held_gripper_scale must be positive")
        if not 0.0 <= self.candidate_capture_open_min <= self.candidate_closed_max:
            raise ValueError(
                "capture aperture lower bound must not exceed candidate_closed_max"
            )
        if self.use_tcp_waypoints:
            if self.safe_tcp_height_above_initial_m <= 0.0:
                raise ValueError("safe TCP height must be positive when waypoints are enabled")
            if not 0.0 < self.vertical_tcp_enter_m < self.vertical_tcp_exit_m:
                raise ValueError("vertical TCP hysteresis is invalid")
            if not 0.0 < self.transport_tcp_enter_m < self.transport_tcp_exit_m:
                raise ValueError("transport TCP hysteresis is invalid")


@dataclass(frozen=True)
class SequentialTransportFeatures:
    tcp: Array
    obj: Array
    outcome_obj: Array
    target: Array
    initial_obj: Array
    grasp_point: Array
    final_point: Array
    vertical_tcp_point: Array
    transport_tcp_point: Array
    gripper_opening: float
    tcp_to_pregrasp: float
    pregrasp_xy_distance: float
    tcp_above_grasp: float
    tcp_to_grasp: float
    tcp_to_vertical: float
    tcp_to_transport: float
    object_lift: float
    object_displacement: float
    object_clearance: float
    object_goal_progress: float
    object_transport_to_final: float
    object_xy_to_final: float
    object_to_final: float
    object_z_minus_target: float


def _vec3(value: npt.ArrayLike, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def _rotate_quaternion(
    quaternion: npt.ArrayLike, vector: Array, order: str
) -> Array:
    """Rotate a local vector under the task-declared quaternion convention."""

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("object quaternion must be a finite four-vector")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        raise ValueError("object quaternion has zero norm")
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
    obj: npt.ArrayLike,
    target: npt.ArrayLike,
    initial_obj: npt.ArrayLike,
    gripper_opening: float,
    config: SequentialTransportConfig,
    object_quaternion: npt.ArrayLike | None = None,
) -> SequentialTransportFeatures:
    tcp_v, obj_v = _vec3(tcp, "tcp"), _vec3(obj, "obj")
    target_v, initial_v = _vec3(target, "target"), _vec3(initial_obj, "initial_obj")
    site_offset = np.asarray(config.observed_site_body_offset_xyz, dtype=np.float64)
    if np.linalg.norm(site_offset) > 0.0:
        if object_quaternion is None:
            raise ValueError("a quaternion is required for an off-center outcome landmark")
        outcome_obj = obj_v - _rotate_quaternion(
            object_quaternion,
            site_offset,
            config.observed_site_quaternion_order,
        )
    else:
        outcome_obj = obj_v
    grasp = obj_v + np.asarray(config.grasp_offset_xyz, dtype=np.float64)
    final = target_v + np.asarray(config.success_offset_xyz, dtype=np.float64)
    pregrasp = grasp + np.array((0.0, 0.0, config.pregrasp_offset_z_m), dtype=np.float64)
    safe_z = initial_v[2] + config.safe_tcp_height_above_initial_m
    vertical_tcp = np.array((grasp[0], grasp[1], safe_z), dtype=np.float64)
    grasp_offset = np.asarray(config.grasp_offset_xyz, dtype=np.float64)
    transport_xy = (
        initial_v[:2] + grasp_offset[:2]
        if config.extraction_transport_at_initial_xy
        else final[:2] + grasp_offset[:2]
    )
    if config.transport_tcp_target_offset_xyz is None:
        transport_tcp = np.array(
            (float(transport_xy[0]), float(transport_xy[1]), safe_z),
            dtype=np.float64,
        )
    else:
        transport_tcp = target_v + np.asarray(
            config.transport_tcp_target_offset_xyz, dtype=np.float64
        )
    axes = np.asarray(config.transport_alignment_axes, dtype=np.int64)
    return SequentialTransportFeatures(
        tcp=tcp_v,
        obj=obj_v,
        outcome_obj=outcome_obj,
        target=target_v,
        initial_obj=initial_v,
        grasp_point=grasp,
        final_point=final,
        vertical_tcp_point=vertical_tcp,
        transport_tcp_point=transport_tcp,
        gripper_opening=float(gripper_opening),
        tcp_to_pregrasp=float(np.linalg.norm(tcp_v - pregrasp)),
        pregrasp_xy_distance=float(np.linalg.norm(tcp_v[:2] - grasp[:2])),
        tcp_above_grasp=float(tcp_v[2] - grasp[2]),
        tcp_to_grasp=float(np.linalg.norm(tcp_v - grasp)),
        tcp_to_vertical=float(np.linalg.norm(tcp_v - vertical_tcp)),
        tcp_to_transport=float(np.linalg.norm(tcp_v - transport_tcp)),
        object_lift=float(obj_v[2] - initial_v[2]),
        object_displacement=float(np.linalg.norm(obj_v - initial_v)),
        object_clearance=float(
            config.clearance_direction
            * (obj_v[config.clearance_axis] - initial_v[config.clearance_axis])
        ),
        object_goal_progress=float(
            np.linalg.norm(initial_v - final) - np.linalg.norm(outcome_obj - final)
        ),
        object_transport_to_final=float(np.linalg.norm(outcome_obj[axes] - final[axes])),
        object_xy_to_final=float(np.linalg.norm(outcome_obj[:2] - final[:2])),
        object_to_final=float(
            np.linalg.norm(
                (outcome_obj - final)
                * np.asarray(config.final_distance_scale_xyz, dtype=np.float64)
            )
        ),
        object_z_minus_target=float(outcome_obj[2] - target_v[2]),
    )


def _quality(distance: float, scale: float) -> float:
    return float(1.0 - np.tanh(max(float(distance), 0.0) / scale))


def _closed(value: SequentialTransportFeatures, config: SequentialTransportConfig) -> float:
    if config.held_gripper_target is not None:
        return _quality(
            abs(value.gripper_opening - config.held_gripper_target),
            config.held_gripper_scale,
        )
    return float(np.clip((1.0 - value.gripper_opening) / config.closed_gripper_scale, 0.0, 1.0))


def _clearance_progress(
    value: SequentialTransportFeatures, config: SequentialTransportConfig
) -> float:
    if config.clearance_progress_mode == "goal":
        return value.object_goal_progress
    return value.object_clearance



def _capture_progress(
    value: SequentialTransportFeatures, config: SequentialTransportConfig
) -> float:
    if config.capture_progress_mode == "displacement":
        return value.object_displacement
    return _clearance_progress(value, config)


def _retained_quality(
    base: float,
    held: float,
    weight: float,
    *,
    conjunctive: bool,
) -> float:
    """Blend task progress with an observable gripper-object invariant."""

    retained = float(np.sqrt(max(base, 0.0) * max(held, 0.0))) if conjunctive else held
    return float((1.0 - weight) * base + weight * retained)


def stage_quality(stage: int, value: SequentialTransportFeatures, config: SequentialTransportConfig) -> float:
    proximity = _quality(value.tcp_to_grasp, 0.08)
    held = proximity * _closed(value, config)
    if stage == 0:
        approach = _quality(value.tcp_to_pregrasp, 0.15)
        if not config.use_gripper_caging:
            return approach
        opening = float(np.clip(value.gripper_opening / config.open_gripper_scale, 0.0, 1.0))
        return (1.0 - config.stage0_open_weight) * approach + config.stage0_open_weight * opening
    if stage == 1:
        capture = float(
            np.clip(_capture_progress(value, config) / config.capture_lift_enter_m, 0.0, 1.0)
        )
        if config.stage1_conjunctive_capture:
            coupled = float(np.sqrt(max(capture, 0.0) * max(held, 0.0)))
            # Preserve separate approach/closure gradients before object motion.
            return 0.20 * proximity + 0.20 * held + 0.60 * coupled
        return 0.30 * proximity + 0.25 * held + 0.45 * capture
    if stage == 2:
        clearance = float(
            np.clip(_clearance_progress(value, config) / config.clearance_lift_enter_m, 0.0, 1.0)
        )
        if config.use_tcp_waypoints:
            waypoint = _quality(value.tcp_to_vertical, 0.18)
            base = 0.50 * waypoint + 0.25 * clearance + 0.15 * proximity + 0.10 * held
        else:
            base = 0.60 * clearance + 0.25 * proximity + 0.15 * held
        progress_coupled = float(
            np.sqrt(max(clearance, 0.0) * max(held, 0.0))
        )
        coupling_weight = config.stage2_progress_coupling_weight
        base = (1.0 - coupling_weight) * base + coupling_weight * progress_coupled
        return _retained_quality(
            base,
            held,
            config.stage2_retention_weight,
            conjunctive=config.stage2_conjunctive_retention,
        )
    if stage == 3:
        if config.terminal_mode in {"distance", "xy_below_target", "axis_progress"}:
            transport = _quality(value.object_transport_to_final, 0.22)
            height = float(
                np.clip(_clearance_progress(value, config) / config.transport_min_lift_m, 0.0, 1.0)
            )
            if config.use_tcp_waypoints:
                tcp_transport = _quality(value.tcp_to_transport, 0.24)
                base = (
                    0.50 * tcp_transport
                    + 0.25 * transport
                    + 0.10 * height
                    + 0.10 * proximity
                    + 0.05 * held
                )
            else:
                base = 0.55 * transport + 0.20 * height + 0.15 * proximity + 0.10 * held
        else:
            remaining = max(
                value.target[2] - config.preterminal_z_margin_m - value.obj[2], 0.0
            )
            transport = _quality(remaining, 0.12)
            height = float(
                np.clip(_clearance_progress(value, config) / config.clearance_lift_enter_m, 0.0, 1.0)
            )
            if config.use_tcp_waypoints:
                tcp_transport = _quality(value.tcp_to_transport, 0.18)
                base = (
                    0.50 * tcp_transport
                    + 0.25 * transport
                    + 0.10 * height
                    + 0.10 * proximity
                    + 0.05 * held
                )
            else:
                base = 0.55 * transport + 0.20 * height + 0.15 * proximity + 0.10 * held
        tcp_weight = config.stage3_tcp_target_coupling_weight
        if tcp_weight > 0.0:
            tcp_target = _quality(value.tcp_to_transport, 0.18)
            tcp_coupled = float(
                np.sqrt(max(base, 0.0) * max(tcp_target, 0.0))
            )
            base = (1.0 - tcp_weight) * base + tcp_weight * tcp_coupled
        terminal_weight = config.stage3_final_outcome_weight
        if terminal_weight > 0.0:
            final_outcome = _quality(
                value.object_to_final, config.final_fine_scale_m
            )
            coupled = float(np.sqrt(max(base, 0.0) * max(final_outcome, 0.0)))
            base = (1.0 - terminal_weight) * base + terminal_weight * coupled
        direct_weight = config.stage3_final_outcome_direct_weight
        if direct_weight > 0.0:
            final_outcome = _quality(
                value.object_to_final, config.final_fine_scale_m
            )
            base = (1.0 - direct_weight) * base + direct_weight * final_outcome

        return _retained_quality(
            base,
            held,
            config.stage3_retention_weight,
            conjunctive=config.stage3_conjunctive_retention,
        )
    if stage == 4:
        if config.terminal_mode == "axis_progress":
            axis_progress = float(
                np.clip(_clearance_progress(value, config) / config.terminal_axis_enter_m, 0.0, 1.0)
            )
            target_progress = _quality(value.object_to_final, 0.16)
            outcome = 0.65 * axis_progress + 0.35 * target_progress
            fine_outcome = _quality(value.object_to_final, config.final_fine_scale_m)
        elif config.terminal_mode == "distance":
            outcome = _quality(value.object_to_final, 0.16)
            fine_outcome = _quality(value.object_to_final, config.final_fine_scale_m)
        elif config.terminal_mode == "xy_below_target":
            alignment = _quality(value.object_xy_to_final, 0.08)
            remaining = max(
                value.outcome_obj[2] - value.target[2] - config.terminal_z_offset_m,
                0.0,
            )
            outcome = 0.75 * alignment + 0.25 * _quality(remaining, 0.05)
            fine_outcome = (
                0.75 * _quality(value.object_xy_to_final, config.final_fine_scale_m)
                + 0.25 * _quality(remaining, config.final_fine_scale_m)
            )
        else:
            remaining = max(
                value.target[2] + config.terminal_z_offset_m - value.obj[2], 0.0
            )
            outcome = _quality(remaining, 0.08)
            fine_outcome = _quality(remaining, config.final_fine_scale_m)
        outcome = (
            (1.0 - config.final_fine_outcome_weight) * outcome
            + config.final_fine_outcome_weight * fine_outcome
        )
        # Prevent additive terminal progress from paying after the controlled
        # object is lost while preserving a smooth gradient in both factors.
        retention_weight = config.final_conjunctive_retention_weight
        coupled_outcome = float(
            np.sqrt(max(outcome, 0.0) * max(held, 0.0))
        )
        outcome = (1.0 - retention_weight) * outcome + retention_weight * coupled_outcome
        opening = float(
            np.clip(value.gripper_opening / config.open_gripper_scale, 0.0, 1.0)
        )
        object_weight = (
            1.0
            - config.final_tcp_obj_weight
            - config.final_closed_weight
            - config.final_open_weight
        )
        return (
            object_weight * outcome
            + config.final_tcp_obj_weight * proximity
            + config.final_closed_weight * held
            + config.final_open_weight * outcome * outcome * opening
        )
    if stage == TERMINAL_STAGE:
        return 1.0
    raise ValueError("invalid sequential-transport stage")


def candidate(
    stage: int,
    value: SequentialTransportFeatures,
    config: SequentialTransportConfig,
    *,
    was_candidate: bool,
) -> bool:
    closed_limit = (
        config.candidate_closed_max
        if config.use_gripper_caging
        else 0.82
    )
    aperture_ok = bool(
        config.candidate_capture_open_min
        <= value.gripper_opening
        <= closed_limit
    )
    if stage == 0:
        xy = config.pregrasp_xy_exit_m if was_candidate else config.pregrasp_xy_enter_m
        return bool(
            value.pregrasp_xy_distance <= xy
            and config.pregrasp_height_low_m
            <= value.tcp_above_grasp
            <= config.pregrasp_height_high_m
            and (not config.use_gripper_caging or value.gripper_opening >= config.candidate_open_min)
        )
    if stage == 1:
        progress = config.capture_move_exit_m if was_candidate else config.capture_lift_enter_m
        return bool(
            _capture_progress(value, config) >= progress
            and value.tcp_to_grasp <= 0.080
            and aperture_ok
        )
    if stage == 2:
        if config.use_tcp_waypoints:
            waypoint = (
                config.vertical_tcp_exit_m
                if was_candidate
                else config.vertical_tcp_enter_m
            )
            if config.stage2_clearance_candidate:
                clearance = (
                    config.clearance_lift_exit_m
                    if was_candidate
                    else config.clearance_lift_enter_m
                )
            else:
                clearance = (
                    config.capture_move_exit_m
                    if was_candidate
                    else config.capture_lift_enter_m
                )
            waypoint_ok = bool(
                not config.require_tcp_waypoint_candidates
                or value.tcp_to_vertical <= waypoint
            )
            return bool(
                waypoint_ok
                and _clearance_progress(value, config) >= clearance
                and value.tcp_to_grasp <= 0.100
                and aperture_ok
            )
        clearance = (
            config.clearance_lift_exit_m
            if was_candidate
            else config.clearance_lift_enter_m
        )
        return bool(
            _clearance_progress(value, config) >= clearance
            and value.tcp_to_grasp <= 0.090
            and aperture_ok
        )
    if stage == 3:
        if config.stage3_terminal_outcome_only_candidate:
            distance = config.final_exit_m if was_candidate else config.final_enter_m
            if config.terminal_mode == "axis_progress":
                axis_threshold = (
                    config.terminal_axis_exit_m
                    if was_candidate
                    else config.terminal_axis_enter_m
                )
                return bool(
                    _clearance_progress(value, config) >= axis_threshold
                    and value.object_to_final <= distance
                )
            if config.terminal_mode == "distance":
                return bool(value.object_to_final <= distance)
            if config.terminal_mode == "xy_below_target":
                z_slack = 0.008 if was_candidate else 0.0
                return bool(
                    value.object_xy_to_final <= distance
                    and value.outcome_obj[2]
                    <= value.target[2] + config.terminal_z_offset_m + z_slack
                )
            threshold = value.target[2] + config.terminal_z_offset_m
            if was_candidate:
                threshold -= 0.008
            return bool(value.obj[2] >= threshold)
        if config.use_tcp_waypoints:
            waypoint = (
                config.transport_tcp_exit_m
                if was_candidate
                else config.transport_tcp_enter_m
            )
            if config.terminal_mode in {"distance", "xy_below_target", "axis_progress"}:
                xy = config.transport_xy_exit_m if was_candidate else config.transport_xy_enter_m
                if config.stage3_candidate_uses_final_distance:
                    progress_limit = (
                        config.final_exit_m if was_candidate else config.final_enter_m
                    )
                    progress_ok = value.object_to_final <= progress_limit
                else:
                    progress_ok = value.object_transport_to_final <= xy
                waypoint_ok = bool(
                    not config.require_tcp_waypoint_candidates
                    or value.tcp_to_transport <= waypoint
                )
                return bool(
                    waypoint_ok
                    and progress_ok
                    and _clearance_progress(value, config) >= config.transport_min_lift_m
                    and value.tcp_to_grasp <= 0.110
                    and aperture_ok
                )
            threshold = value.target[2] - config.preterminal_z_margin_m
            if was_candidate:
                threshold -= 0.012
            waypoint_ok = bool(
                not config.require_tcp_waypoint_candidates
                or value.tcp_to_transport <= waypoint
            )
            return bool(
                waypoint_ok
                and value.obj[2] >= threshold
                and value.tcp_to_grasp <= 0.110
                and aperture_ok
            )
        if config.terminal_mode in {"distance", "xy_below_target", "axis_progress"}:
            xy = config.transport_xy_exit_m if was_candidate else config.transport_xy_enter_m
            if config.stage3_candidate_uses_final_distance:
                progress_limit = (
                    config.final_exit_m if was_candidate else config.final_enter_m
                )
                progress_ok = value.object_to_final <= progress_limit
            else:
                progress_ok = value.object_transport_to_final <= xy
            return bool(
                progress_ok
                and _clearance_progress(value, config) >= config.transport_min_lift_m
                and value.tcp_to_grasp <= 0.100
                and (config.held_gripper_target is None or aperture_ok)
            )
        threshold = value.target[2] - config.preterminal_z_margin_m
        if was_candidate:
            threshold -= 0.012
        return bool(
            value.obj[2] >= threshold
            and value.tcp_to_grasp <= 0.100
            and (config.held_gripper_target is None or aperture_ok)
        )
    if stage == 4:
        if config.terminal_mode == "axis_progress":
            axis_threshold = (
                config.terminal_axis_exit_m if was_candidate
                else config.terminal_axis_enter_m
            )
            distance_threshold = (
                config.final_exit_m if was_candidate else config.final_enter_m
            )
            return bool(
                _clearance_progress(value, config) >= axis_threshold
                and value.object_to_final <= distance_threshold
            )
        if config.terminal_mode == "distance":
            distance = config.final_exit_m if was_candidate else config.final_enter_m
            return bool(value.object_to_final <= distance)
        threshold = value.target[2] + config.terminal_z_offset_m
        if config.terminal_mode == "xy_below_target":
            distance = config.final_exit_m if was_candidate else config.final_enter_m
            z_slack = 0.008 if was_candidate else 0.0
            return bool(
                value.object_xy_to_final <= distance
                and value.outcome_obj[2]
                <= value.target[2] + config.terminal_z_offset_m + z_slack
            )
        if was_candidate:
            threshold -= 0.008
        return bool(value.obj[2] >= threshold)
    return False


@dataclass
class SequentialTransportState:
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
    state: SequentialTransportState,
    value: SequentialTransportFeatures,
    action: npt.ArrayLike,
    config: SequentialTransportConfig,
) -> StepResult:
    if state.stage == TERMINAL_STAGE:
        return StepResult(0.0, False, state.stage)
    mode = reward_form_mode(config)
    if mode == STAGED_RESET:
        quality = stage_quality(state.stage, value, config)
        previous = quality if state.previous_quality is None else state.previous_quality
        dense = config.dense_scales[state.stage] * (config.gamma * quality - previous)
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
    penalty *= config.action_penalty_stage_multipliers[state.stage]
    strict = candidate(
        state.stage, value, config, was_candidate=state.stable_count > 0
    )
    if state.stage <= 2:
        target_weight = config.prefix_action_target_penalty_weights[state.stage]
        desired: Array | None = None
        if target_weight > 0.0:
            if state.stage == 0:
                pregrasp = value.grasp_point + np.array(
                    (0.0, 0.0, config.pregrasp_offset_z_m), dtype=np.float64
                )
                desired = pregrasp - value.tcp
            elif state.stage == 1:
                if (
                    value.tcp_to_grasp
                    > config.stage1_capture_action_guidance_grasp_deadband_m
                ):
                    desired = value.grasp_point - value.tcp
                elif value.gripper_opening <= config.candidate_closed_max:
                    desired = np.zeros(3, dtype=np.float64)
                    desired[config.clearance_axis] = config.clearance_direction
            else:
                desired = value.vertical_tcp_point - value.tcp
        if desired is not None:
            desired_distance = float(np.linalg.norm(desired))
            if desired_distance > config.stage3_action_guidance_deadband_m:
                direction = desired / desired_distance
                target_action = config.prefix_action_target_magnitude * direction
                component_error = np.clip(action_v[:3], -1.0, 1.0) - target_action
                penalty += target_weight * float(np.mean(component_error**2))
    if state.stage == 1 and config.capture_proximity_invariant_penalty_weight > 0.0:
        proximity = _quality(value.tcp_to_grasp, 0.08)
        normalized_shortfall = max(
            config.capture_proximity_invariant_floor - proximity, 0.0
        ) / config.capture_proximity_invariant_floor
        penalty += (
            config.capture_proximity_invariant_penalty_weight
            * normalized_shortfall * normalized_shortfall
        )
    if state.stage == 1:
        if config.stage1_capture_action_guidance_penalty_weight > 0.0:
            desired: Array | None = None
            if value.tcp_to_grasp > config.stage1_capture_action_guidance_grasp_deadband_m:
                desired = value.grasp_point - value.tcp
            elif value.gripper_opening <= config.candidate_closed_max:
                desired = np.zeros(3, dtype=np.float64)
                desired[config.clearance_axis] = config.clearance_direction
            if desired is not None:
                direction = desired / np.linalg.norm(desired)
                projection = float(
                    np.dot(np.clip(action_v[:3], -1.0, 1.0), direction)
                )
                normalized_shortfall = max(
                    config.stage1_capture_action_guidance_min_projection - projection,
                    0.0,
                ) / (config.stage1_capture_action_guidance_min_projection + 1.0)
                penalty += (
                    config.stage1_capture_action_guidance_penalty_weight
                    * normalized_shortfall * normalized_shortfall
                )
        if (
            config.stage1_capture_far_open_gripper_action_penalty_weight > 0.0
            and value.tcp_to_grasp
            > config.stage1_capture_action_guidance_grasp_deadband_m
        ):
            gripper_command = float(np.clip(action_v[3], -1.0, 1.0))
            normalized_excess = max(
                gripper_command
                - config.stage1_capture_far_open_gripper_action_max_command,
                0.0,
            ) / (1.0 - config.stage1_capture_far_open_gripper_action_max_command)
            penalty += (
                config.stage1_capture_far_open_gripper_action_penalty_weight
                * normalized_excess**2
            )
        if (
            config.stage1_capture_gripper_action_penalty_weight > 0.0
            and value.tcp_to_grasp
            <= config.stage1_capture_action_guidance_grasp_deadband_m
        ):
            gripper_command = float(np.clip(action_v[3], -1.0, 1.0))
            normalized_shortfall = max(
                config.stage1_capture_gripper_action_min_command - gripper_command,
                0.0,
            ) / (config.stage1_capture_gripper_action_min_command + 1.0)
            penalty += (
                config.stage1_capture_gripper_action_penalty_weight
                * normalized_shortfall**2
            )
    if (
        state.stage >= config.post_capture_invariant_penalty_stage_start
        and config.post_capture_invariant_penalty_weight > 0.0
    ):
        proximity = _quality(value.tcp_to_grasp, 0.08)
        held = proximity * _closed(value, config)
        normalized_shortfall = max(
            config.post_capture_invariant_floor - held, 0.0
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
    if (
        state.stage == 3
        and config.stage3_tcp_target_invariant_penalty_weight > 0.0
    ):
        tcp_target = _quality(
            value.tcp_to_transport,
            config.stage3_tcp_target_invariant_scale_m,
        )
        normalized_shortfall = max(
            config.stage3_tcp_target_invariant_floor - tcp_target, 0.0
        ) / config.stage3_tcp_target_invariant_floor
        penalty += (
            config.stage3_tcp_target_invariant_penalty_weight
            * normalized_shortfall * normalized_shortfall
        )
    if (
        state.stage == 3
        and config.stage3_final_outcome_invariant_penalty_weight > 0.0
    ):
        final_outcome = _quality(
            value.object_to_final,
            config.stage3_final_outcome_invariant_scale_m,
        )
        normalized_shortfall = max(
            config.stage3_final_outcome_invariant_floor - final_outcome, 0.0
        ) / config.stage3_final_outcome_invariant_floor
        penalty += (
            config.stage3_final_outcome_invariant_penalty_weight
            * normalized_shortfall * normalized_shortfall
        )
    if (
        state.stage == 3
        and (
            config.stage3_action_guidance_penalty_weight > 0.0
            or config.stage3_action_target_penalty_weight > 0.0
        )
        and (
            float(np.linalg.norm(value.final_point - value.outcome_obj))
            if config.stage3_action_direction_mode == "final_outcome"
            else value.tcp_to_transport
        )
        > config.stage3_action_guidance_deadband_m
    ):
        desired = (
            value.final_point - value.outcome_obj
            if config.stage3_action_direction_mode == "final_outcome"
            else value.transport_tcp_point - value.tcp
        )
        direction = desired / np.linalg.norm(desired)
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
    if strict:
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
    state: SequentialTransportState,
    decision: bool,
    config: SequentialTransportConfig,
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

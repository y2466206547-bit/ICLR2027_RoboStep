"""Pure NumPy staged reward for MetaWorld ``drawer-open-v3``.

This module is the task-specific instantiation of the task-independent CORE in
``reward_model/stage_policy/stage_policy/reward_formulation.md``.  It does not
import MetaWorld, MuJoCo, Torch, or Qwen, which keeps reward and transition
semantics independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from .reward_form_modes import (
    STAGED_RESET,
    reward_form_mode,
    sparse_step_reward,
    transition_reward as mapped_transition_reward,
)


ACTIVE_STAGE_COUNT: Final[int] = 3
STAGE_COUNT: Final[int] = ACTIVE_STAGE_COUNT + 1
TERMINAL_STAGE: Final[int] = ACTIVE_STAGE_COUNT

STAGE_NAMES: Final[tuple[str, ...]] = (
    "align_above_handle",
    "descend_and_hook_handle",
    "pull_drawer_open",
)

STAGE_INSTRUCTIONS: Final[tuple[str, ...]] = (
    (
        "State has verified exact above-handle pregrasp geometry. Audit only the "
        "correct drawer scene and that the gripper is not obviously at a wrong part. "
        "Do not rejudge centimeter-scale alignment from pixels."
    ),
    (
        "State has verified exact handle-contact geometry. Audit only that there "
        "is no obvious empty relation, wrong-part contact, or lost drawer handle. "
        "Do not rejudge centimeter-scale contact from pixels."
    ),
    (
        "State has verified the drawer handle is within 3 cm of the task target. "
        "Audit only that the drawer visibly moved outward relative to the reference, "
        "with no closed drawer, robot-only motion, or re-closing."
    ),
)


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DrawerRewardConfig:
    """Task-specific terms plugged into the stable paper-level CORE."""

    gamma: float = 0.99
    reward_form_mode: str = STAGED_RESET
    dense_scales: tuple[float, ...] = (4.0, 6.0, 10.0)
    transition_bonuses: tuple[float, ...] = (5.0, 10.0, 20.0)
    required_holds: tuple[int, ...] = (3, 3, 3)

    # Stage-0: horizontally centered and still safely above the handle.
    align_xy_enter_m: float = 0.025
    align_xy_exit_m: float = 0.040
    align_height_min_m: float = 0.060
    align_height_max_m: float = 0.100

    # Stage-1: close enough for the open fingertips to hook the handle.
    hook_distance_enter_m: float = 0.025
    hook_distance_exit_m: float = 0.040
    hook_height_abs_max_m: float = 0.025

    # Stage-2/official success: handle within 3 cm of the open target.
    open_distance_enter_m: float = 0.030
    open_distance_exit_m: float = 0.045
    affordance_offset_xyz: tuple[float, float, float] = (0.0, -0.060, -0.020)

    # Generic normalized-action smoothness/saturation costs.
    action_rate_weight: float = 0.020
    action_curve_weight: float = 0.010
    boundary_weight: float = 0.020
    boundary_free_limit: float = 0.80
    boundary_hard_limit: float = 1.00

    def __post_init__(self) -> None:
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        for name, values in (
            ("dense_scales", self.dense_scales),
            ("transition_bonuses", self.transition_bonuses),
            ("required_holds", self.required_holds),
        ):
            if len(values) != ACTIVE_STAGE_COUNT:
                raise ValueError(f"{name} must have {ACTIVE_STAGE_COUNT} entries")
        if any(value <= 0 for value in self.dense_scales):
            raise ValueError("dense scales must be positive")
        # Zero is reserved for the pure-flat/no-transition-bonus ablation.
        if any(value < 0 for value in self.transition_bonuses):
            raise ValueError("transition bonuses must be non-negative")
        if any(value < 1 for value in self.required_holds):
            raise ValueError("required holds must be positive")
        if self.boundary_hard_limit <= self.boundary_free_limit:
            raise ValueError("boundary hard limit must exceed its free limit")
        offset = np.asarray(self.affordance_offset_xyz, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("affordance_offset_xyz must be a finite three-vector")



@dataclass
class DrawerStageState:
    """Per-environment reward-machine memory."""

    stage: int = 0
    stable_count: int = 0
    candidate_epoch: int = 0
    request_armed: bool = True
    previous_quality: float | None = None
    previous_action: Array = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    previous_previous_action: Array = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    qwen_confirmed_mask: int = 0

    def reset(self, initial_quality: float) -> None:
        self.stage = 0
        self.stable_count = 0
        self.candidate_epoch = 0
        self.request_armed = True
        self.previous_quality = float(initial_quality)
        self.previous_action = np.zeros(3, dtype=np.float64)
        self.previous_previous_action = np.zeros(3, dtype=np.float64)
        self.qwen_confirmed_mask = 0


@dataclass(frozen=True)
class DrawerFeatures:
    """Physical features used by reward and fast candidate proposals."""

    tcp: Array
    handle: Array
    target: Array
    align_xy_distance: float
    tcp_above_handle: float
    hook_distance: float
    open_distance: float
    pull_waypoint_distance: float
    open_progress: float


@dataclass(frozen=True)
class DrawerStepResult:
    """One CORE evaluation and optional verified transition."""

    reward: float
    dense_reward: float
    transition_reward: float
    smooth_penalty: float
    safety_penalty: float
    strict_candidate: bool
    request_due: bool
    transitioned: bool
    old_stage: int
    new_stage: int
    stage_quality: float


@dataclass(frozen=True)
class QwenTransitionResult:
    """Result of applying a decision to an already-frozen due candidate."""

    accepted: bool
    transitioned: bool
    transition_reward: float
    old_stage: int
    new_stage: int


def _as_vec3(value: npt.ArrayLike, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def drawer_features(
    tcp: npt.ArrayLike,
    handle: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_open_distance_m: float = 0.20,
    active_waypoint_offset_xyz: npt.ArrayLike = (0.0, -0.060, -0.020),
) -> DrawerFeatures:
    tcp_v = _as_vec3(tcp, "tcp")
    handle_v = _as_vec3(handle, "handle")
    target_v = _as_vec3(target, "target")
    open_distance = float(np.linalg.norm(handle_v - target_v))
    # A handle-relative infeasible waypoint creates sustained contact force;
    # its sign distinguishes opening from closing without hidden state.
    affordance_offset = _as_vec3(active_waypoint_offset_xyz, "active_waypoint_offset_xyz")
    pull_waypoint = handle_v + affordance_offset
    return DrawerFeatures(
        tcp=tcp_v,
        handle=handle_v,
        target=target_v,
        align_xy_distance=float(np.linalg.norm(tcp_v[:2] - handle_v[:2])),
        tcp_above_handle=float(tcp_v[2] - handle_v[2]),
        hook_distance=float(np.linalg.norm(tcp_v - handle_v)),
        open_distance=open_distance,
        pull_waypoint_distance=float(
            np.linalg.norm(tcp_v - pull_waypoint)
        ),
        open_progress=float(
            np.clip(1.0 - open_distance / max_open_distance_m, 0.0, 1.0)
        ),
    )


def _long_tail_quality(distance: float, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("quality scale must be positive")
    return float(1.0 - np.tanh(max(float(distance), 0.0) / scale))


def stage_quality(stage: int, features: DrawerFeatures) -> float:
    """Bounded Frame-VLM-authored potential for the active stage."""
    if stage == 0:
        # Align XY first; a broad height quality keeps the hand above the handle
        # without prescribing an expert action direction.
        xy = _long_tail_quality(features.align_xy_distance, 0.12)
        desired_height = 0.085
        height = _long_tail_quality(
            abs(features.tcp_above_handle - desired_height), 0.10
        )
        return 0.75 * xy + 0.25 * height
    if stage == 1:
        distance = _long_tail_quality(features.hook_distance, 0.10)
        height = _long_tail_quality(abs(features.tcp_above_handle), 0.08)
        return 0.70 * distance + 0.30 * height
    if stage == 2:
        # The moving affordance waypoint supplies gradient before static
        # friction is overcome and keeps demanding handle-relative contact.
        open_quality = _long_tail_quality(features.open_distance, 0.20)
        pull_affordance = _long_tail_quality(
            features.pull_waypoint_distance, 0.10
        )
        return 0.45 * open_quality + 0.55 * pull_affordance
    if stage == TERMINAL_STAGE:
        return 1.0
    raise ValueError(f"invalid stage {stage}")


def strict_candidate(
    stage: int,
    features: DrawerFeatures,
    config: DrawerRewardConfig,
    *,
    was_strict: bool,
) -> bool:
    """Hysteretic privileged rule that may propose, but never authorize, a stage."""
    if stage == 0:
        xy_limit = (
            config.align_xy_exit_m if was_strict else config.align_xy_enter_m
        )
        return bool(
            features.align_xy_distance <= xy_limit
            and config.align_height_min_m
            <= features.tcp_above_handle
            <= config.align_height_max_m
        )
    if stage == 1:
        distance_limit = (
            config.hook_distance_exit_m
            if was_strict
            else config.hook_distance_enter_m
        )
        return bool(
            features.hook_distance <= distance_limit
            and abs(features.tcp_above_handle) <= config.hook_height_abs_max_m
        )
    if stage == 2:
        distance_limit = (
            config.open_distance_exit_m
            if was_strict
            else config.open_distance_enter_m
        )
        return bool(features.open_distance <= distance_limit)
    return False


def smoothness_penalty(
    action: npt.ArrayLike,
    previous_action: npt.ArrayLike,
    previous_previous_action: npt.ArrayLike,
    config: DrawerRewardConfig,
) -> float:
    """Task-independent normalized delta-EEF action quality cost."""
    action_v = _as_vec3(action, "action")
    previous_v = _as_vec3(previous_action, "previous_action")
    previous_previous_v = _as_vec3(
        previous_previous_action, "previous_previous_action"
    )
    rate = float(np.mean(np.square(action_v - previous_v)))
    curve = float(
        np.mean(np.square(action_v - 2.0 * previous_v + previous_previous_v))
    )
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


def drawer_stage_step(
    state: DrawerStageState,
    features: DrawerFeatures,
    action: npt.ArrayLike,
    *,
    qwen_decision: bool | None,
    config: DrawerRewardConfig,
    safety_penalty: float = 0.0,
) -> DrawerStepResult:
    """Evaluate one transition under the stable CORE.

    ``qwen_decision`` must be ``None`` until a request is due.  On a due
    candidate, ``True`` authorizes exactly one transition and ``False`` blocks
    it for that candidate epoch.  The physical predicate is therefore a
    scheduler only; it can never advance the stage by itself.
    """
    if state.stage < 0 or state.stage > TERMINAL_STAGE:
        raise ValueError("state.stage is invalid")
    action_v = _as_vec3(action, "action")
    old_stage = int(state.stage)
    mode = reward_form_mode(config)
    if old_stage == TERMINAL_STAGE:
        return DrawerStepResult(
            reward=0.0,
            dense_reward=0.0,
            transition_reward=0.0,
            smooth_penalty=0.0,
            safety_penalty=0.0,
            strict_candidate=False,
            request_due=False,
            transitioned=False,
            old_stage=old_stage,
            new_stage=old_stage,
            stage_quality=1.0,
        )

    quality = stage_quality(old_stage, features)
    previous_quality = quality if state.previous_quality is None else state.previous_quality
    dense = config.dense_scales[old_stage] * (
        config.gamma * quality - previous_quality
    )
    smooth = smoothness_penalty(
        action_v,
        state.previous_action,
        state.previous_previous_action,
        config,
    )
    if safety_penalty < 0.0 or not np.isfinite(safety_penalty):
        raise ValueError("safety_penalty must be finite and non-negative")

    was_strict = state.stable_count > 0
    candidate = strict_candidate(
        old_stage, features, config, was_strict=was_strict
    )
    if candidate:
        state.stable_count += 1
    else:
        if state.stable_count > 0 or not state.request_armed:
            state.candidate_epoch += 1
        state.stable_count = 0
        state.request_armed = True

    request_due = bool(
        candidate
        and state.request_armed
        and state.stable_count >= config.required_holds[old_stage]
    )
    if not request_due and qwen_decision is not None:
        raise ValueError("a Qwen decision was supplied without a due request")

    transitioned = False
    transition_reward = 0.0
    if request_due:
        if qwen_decision is None:
            # The caller freezes the simulator and returns after obtaining the
            # mandatory decision; unresolved requests always fail closed.
            pass
        else:
            state.request_armed = False
            if bool(qwen_decision):
                transitioned = True
                transition_reward = mapped_transition_reward(
                    mode,
                    original_bonus=config.transition_bonuses[old_stage],
                    new_stage=old_stage + 1,
                    terminal_stage=TERMINAL_STAGE,
                )
                state.qwen_confirmed_mask |= 1 << old_stage
                state.stage = old_stage + 1
                state.stable_count = 0
                state.request_armed = True
                state.previous_quality = stage_quality(state.stage, features)
            else:
                state.candidate_epoch += 1
                state.stable_count = 0
                state.request_armed = True

    if not transitioned:
        state.previous_quality = quality
    state.previous_previous_action = state.previous_action.copy()
    state.previous_action = action_v.copy()

    return DrawerStepResult(
        reward=float(
            sparse_step_reward(mode, dense - smooth - safety_penalty)
            + transition_reward
        ),
        dense_reward=sparse_step_reward(mode, dense),
        transition_reward=float(transition_reward),
        smooth_penalty=float(smooth),
        safety_penalty=float(safety_penalty),
        strict_candidate=candidate,
        request_due=request_due,
        transitioned=transitioned,
        old_stage=old_stage,
        new_stage=int(state.stage),
        stage_quality=float(quality),
    )


def apply_due_qwen_decision(
    state: DrawerStageState,
    features: DrawerFeatures,
    decision: bool,
    config: DrawerRewardConfig,
) -> QwenTransitionResult:
    """Apply one exact Qwen response after ``drawer_stage_step`` requested it.

    The environment calls ``drawer_stage_step(..., qwen_decision=None)`` first,
    freezes before any further physics, obtains the response, and calls this
    function. Separating the two operations prevents reward/action history
    from being counted twice while the simulator is frozen.
    """
    old_stage = int(state.stage)
    if old_stage < 0 or old_stage >= ACTIVE_STAGE_COUNT:
        raise ValueError("no active stage can accept a Qwen decision")
    if (
        not state.request_armed
        or state.stable_count < config.required_holds[old_stage]
    ):
        raise ValueError("there is no armed due candidate")
    state.request_armed = False
    transitioned = bool(decision)
    transition_reward = 0.0
    if transitioned:
        transition_reward = mapped_transition_reward(
            reward_form_mode(config),
            original_bonus=config.transition_bonuses[old_stage],
            new_stage=old_stage + 1,
            terminal_stage=TERMINAL_STAGE,
        )
        state.qwen_confirmed_mask |= 1 << old_stage
        state.stage = old_stage + 1
        state.stable_count = 0
        state.request_armed = True
        state.previous_quality = stage_quality(state.stage, features)
    else:
        state.candidate_epoch += 1
        state.stable_count = 0
        state.request_armed = True
    return QwenTransitionResult(
        accepted=bool(decision),
        transitioned=transitioned,
        transition_reward=transition_reward,
        old_stage=old_stage,
        new_stage=int(state.stage),
    )

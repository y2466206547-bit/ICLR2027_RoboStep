"""One-stage CORE instantiation for observable TCP-to-target tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .reward_form_modes import (
    STAGED_RESET,
    reward_form_mode,
    sparse_step_reward,
    transition_reward,
)


STAGE_COUNT = 2
TERMINAL_STAGE = 1
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TargetReachConfig:
    gamma: float = 0.99
    reward_form_mode: str = STAGED_RESET
    dense_scale: float = 10.0
    transition_bonus: float = 20.0
    required_hold: int = 3
    enter_distance_m: float = 0.050
    exit_distance_m: float = 0.065
    quality_scale_m: float = 0.20
    # Semantic point offset from the task-provided goal marker to the visible
    # TCP target (for example, the mouth of a recessed aperture). The zero
    # default preserves the historical reach/reach-wall formulation.
    target_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_rate_weight: float = 0.020
    action_curve_weight: float = 0.010
    boundary_weight: float = 0.020
    boundary_free_limit: float = 0.80
    boundary_hard_limit: float = 1.00


@dataclass
class TargetReachState:
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

    def reset(self, quality: float) -> None:
        self.stage = 0
        self.stable_count = 0
        self.candidate_epoch = 0
        self.request_armed = True
        self.previous_quality = float(quality)
        self.previous_action.fill(0.0)
        self.previous_previous_action.fill(0.0)


def _vec3(value: npt.ArrayLike, name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def quality(distance_m: float, config: TargetReachConfig) -> float:
    return float(1.0 - np.tanh(max(float(distance_m), 0.0) / config.quality_scale_m))


def smoothness_penalty(
    action: npt.ArrayLike,
    previous_action: npt.ArrayLike,
    previous_previous_action: npt.ArrayLike,
    config: TargetReachConfig,
) -> float:
    action_v = _vec3(action, "action")
    previous_v = _vec3(previous_action, "previous_action")
    before_v = _vec3(previous_previous_action, "previous_previous_action")
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


@dataclass(frozen=True)
class TargetReachStep:
    reward: float
    request_due: bool
    transitioned: bool


def step(
    state: TargetReachState,
    *,
    distance_m: float,
    action: npt.ArrayLike,
    config: TargetReachConfig,
) -> TargetReachStep:
    if state.stage == TERMINAL_STAGE:
        return TargetReachStep(0.0, False, False)
    if state.stage != 0:
        raise ValueError("invalid target-reach stage")
    mode = reward_form_mode(config)
    q = quality(distance_m, config)
    previous = q if state.previous_quality is None else state.previous_quality
    dense = config.dense_scale * (config.gamma * q - previous)
    action_v = _vec3(action, "action")
    smooth = smoothness_penalty(
        action_v, state.previous_action, state.previous_previous_action, config
    )
    was_candidate = state.stable_count > 0
    threshold = config.exit_distance_m if was_candidate else config.enter_distance_m
    candidate = float(distance_m) <= threshold
    if candidate:
        state.stable_count += 1
    else:
        if state.stable_count > 0 or not state.request_armed:
            state.candidate_epoch += 1
        state.stable_count = 0
        state.request_armed = True
    due = bool(
        candidate and state.request_armed and state.stable_count >= config.required_hold
    )
    state.previous_quality = q
    state.previous_previous_action = state.previous_action.copy()
    state.previous_action = action_v.copy()
    return TargetReachStep(sparse_step_reward(mode, dense - smooth), due, False)


def apply_qwen_decision(
    state: TargetReachState, decision: bool, config: TargetReachConfig
) -> float:
    if state.stage != 0 or not state.request_armed or state.stable_count < config.required_hold:
        raise ValueError("there is no armed target-reach Qwen candidate")
    state.request_armed = False
    if bool(decision):
        state.stage = TERMINAL_STAGE
        state.stable_count = 0
        state.request_armed = True
        state.previous_quality = 1.0
        return transition_reward(
            reward_form_mode(config),
            original_bonus=config.transition_bonus,
            new_stage=state.stage,
            terminal_stage=TERMINAL_STAGE,
        )
    state.candidate_epoch += 1
    state.stable_count = 0
    state.request_armed = True
    return 0.0

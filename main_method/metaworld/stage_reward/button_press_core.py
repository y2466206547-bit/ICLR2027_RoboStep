"""Task-local three-stage CORE instantiation for top-down button press."""

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
STAGE_COUNT: Final[int] = 4
TERMINAL_STAGE: Final[int] = 3
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ButtonPressConfig:
    """Observable state geometry plugged into the unchanged shared CORE."""

    gamma: float = 0.99
    reward_form_mode: str = STAGED_RESET
    dense_scales: tuple[float, ...] = (4.0, 6.0, 48.0)
    transition_bonuses: tuple[float, ...] = (5.0, 10.0, 80.0)
    required_holds: tuple[int, ...] = (3, 3, 1)
    prepress_height_m: float = 0.10
    press_axis_xyz: tuple[float, float, float] = (0.0, 0.0, -1.0)
    object_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Some articulated fixtures require an obstacle-clearing Cartesian
    # waypoint instead of approaching opposite the actuation axis. None keeps
    # the original behavior used by every historical run.
    prepress_offset_xyz: tuple[float, float, float] | None = None
    prepress_enter_m: float = 0.030
    prepress_exit_m: float = 0.045
    contact_enter_m: float = 0.030
    contact_exit_m: float = 0.045
    press_enter_m: float = 0.024
    press_exit_m: float = 0.032
    press_quality_scale_m: float = 0.20
    tcp_press_axis_scale_m: float = 0.08
    use_object_target_distance: bool = False
    action_rate_weight: float = 0.020
    # Backward-compatible broad-to-fine stage-1 contact potential. Zero blend
    # preserves all historical revisions exactly.
    stage1_multiscale_weight: float = 0.0
    stage1_coarse_scale_m: float = 0.40
    stage1_distance_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    action_curve_weight: float = 0.010
    boundary_weight: float = 0.020
    # Preserve an object-causal progress gradient when an articulated contact
    # point moves on an arc and cannot remain pixel-perfectly registered.
    # Zero preserves every historical setting.
    stage2_contact_floor: float = 0.0
    # Optional broad-to-fine contact field for the active articulated stage.
    # Zero coarse weight preserves every historical checkpoint exactly.
    stage2_contact_multiscale_weight: float = 0.0
    stage2_contact_coarse_scale_m: float = 0.30
    boundary_free_limit: float = 0.80
    # Explicit terminal-basin mixture; defaults reproduce 0.70/0.20/0.10.
    stage2_near_target_weight: float = 0.20
    stage2_preload_weight: float = 0.10
    stage2_near_target_scale_m: float = 0.05
    boundary_hard_limit: float = 1.00

    def __post_init__(self) -> None:
        if not 0.0 <= self.stage2_contact_multiscale_weight <= 1.0:
            raise ValueError(
                "stage2_contact_multiscale_weight must be in [0, 1]"
            )
        if self.stage2_contact_coarse_scale_m <= 0.0:
            raise ValueError("stage2_contact_coarse_scale_m must be positive")
        if not 0.0 <= self.stage1_multiscale_weight <= 1.0:
            raise ValueError("stage1_multiscale_weight must be in [0, 1]")
        if self.stage1_coarse_scale_m <= 0.0:
            raise ValueError("stage1_coarse_scale_m must be positive")
        scales = np.asarray(self.stage1_distance_scale_xyz, dtype=np.float64)
        if (
            scales.shape != (3,) or not np.all(np.isfinite(scales))
            or np.any(scales <= 0.0)
        ):
            raise ValueError("stage1_distance_scale_xyz must be finite positive")
        if (
            self.stage2_near_target_weight < 0.0
            or self.stage2_preload_weight < 0.0
            or self.stage2_near_target_weight + self.stage2_preload_weight >= 1.0
        ):
            raise ValueError("stage2 terminal weights must be non-negative and leave positive linear-progress weight")
        if self.stage2_near_target_scale_m <= 0.0:
            raise ValueError("stage2_near_target_scale_m must be positive")
        reward_form_mode(self)


@dataclass(frozen=True)
class ButtonPressFeatures:
    tcp: Array
    button: Array
    target: Array
    prepress_distance: float
    contact_distance: float
    press_distance: float
    object_target_distance: float
    tcp_target_distance: float
    tcp_press_axis_distance: float


@dataclass
class ButtonPressState:
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
class ButtonPressStep:
    reward: float
    request_due: bool
    strict_candidate: bool
    old_stage: int
    new_stage: int


def _vec3(value: npt.ArrayLike, name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def features(tcp: npt.ArrayLike, button: npt.ArrayLike, target: npt.ArrayLike, config: ButtonPressConfig) -> ButtonPressFeatures:
    tcp_v, button_v, target_v = _vec3(tcp, "tcp"), _vec3(button, "button"), _vec3(target, "target")
    axis = np.asarray(config.press_axis_xyz, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-9:
        raise ValueError("press_axis_xyz must be a finite nonzero three-vector")
    axis = axis / np.linalg.norm(axis)
    object_offset = np.asarray(config.object_offset_xyz, dtype=np.float64)
    if object_offset.shape != (3,) or not np.all(np.isfinite(object_offset)):
        raise ValueError("object_offset_xyz must be a finite three-vector")
    contact_offset = np.asarray(config.contact_offset_xyz, dtype=np.float64)
    if contact_offset.shape != (3,) or not np.all(np.isfinite(contact_offset)):
        raise ValueError("contact_offset_xyz must be a finite three-vector")
    object_point = button_v + object_offset
    contact_point = object_point + contact_offset
    if config.prepress_offset_xyz is None:
        prepress = contact_point - axis * config.prepress_height_m
    else:
        prepress_offset = np.asarray(config.prepress_offset_xyz, dtype=np.float64)
        if prepress_offset.shape != (3,) or not np.all(np.isfinite(prepress_offset)):
            raise ValueError("prepress_offset_xyz must be a finite three-vector")
        prepress = object_point + prepress_offset
    return ButtonPressFeatures(
        tcp=tcp_v,
        button=object_point,
        target=target_v,
        prepress_distance=float(np.linalg.norm(tcp_v - prepress)),
        contact_distance=float(np.linalg.norm(tcp_v - contact_point)),
        press_distance=float(abs(np.dot(target_v - object_point, axis))),
        object_target_distance=float(np.linalg.norm(target_v - object_point)),
        tcp_target_distance=float(np.linalg.norm(tcp_v - target_v)),
        tcp_press_axis_distance=float(abs(np.dot(tcp_v - target_v, axis))),
    )


def _quality(distance: float, scale: float) -> float:
    return float(1.0 - np.tanh(max(float(distance), 0.0) / scale))


def stage_quality(stage: int, value: ButtonPressFeatures, config: ButtonPressConfig | None = None) -> float:
    cfg = config or ButtonPressConfig()
    if stage == 0:
        return _quality(value.prepress_distance, 0.15)
    if stage == 1:
        contact_point = value.button + np.asarray(
            cfg.contact_offset_xyz, dtype=np.float64
        )
        scales = np.asarray(cfg.stage1_distance_scale_xyz, dtype=np.float64)
        scaled_distance = float(np.linalg.norm((value.tcp - contact_point) * scales))
        fine = _quality(scaled_distance, 0.10)
        coarse = _quality(scaled_distance, cfg.stage1_coarse_scale_m)
        return (
            (1.0 - cfg.stage1_multiscale_weight) * fine
            + cfg.stage1_multiscale_weight * coarse
        )
    if stage == 2:
        # Revision 5: active pressing must not reward static contact. Gate
        # press progress by visible TCP-button contact, then use mostly
        # button-to-target progress plus a small TCP-depth preload term.
        contact_fine = _quality(value.contact_distance, 0.03)
        contact_coarse = _quality(
            value.contact_distance, cfg.stage2_contact_coarse_scale_m
        )
        contact_gate = (
            (1.0 - cfg.stage2_contact_multiscale_weight) * contact_fine
            + cfg.stage2_contact_multiscale_weight * contact_coarse
        )
        terminal_distance = value.object_target_distance if cfg.use_object_target_distance else value.press_distance
        linear_press = float(
            np.clip(
                (cfg.press_quality_scale_m - terminal_distance)
                / cfg.press_quality_scale_m,
                0.0,
                1.0,
            )
        )
        near_target = _quality(terminal_distance, cfg.stage2_near_target_scale_m)
        preload = _quality(value.tcp_press_axis_distance, cfg.tcp_press_axis_scale_m)
        effective_contact = cfg.stage2_contact_floor + (1.0 - cfg.stage2_contact_floor) * contact_gate
        return effective_contact * (
            (
                1.0
                - cfg.stage2_near_target_weight
                - cfg.stage2_preload_weight
            )
            * linear_press
            + cfg.stage2_near_target_weight * near_target
            + cfg.stage2_preload_weight * preload
        )
    if stage == TERMINAL_STAGE:
        return 1.0
    raise ValueError("invalid button-press stage")


def _candidate(stage: int, value: ButtonPressFeatures, config: ButtonPressConfig, was_candidate: bool) -> bool:
    if stage == 0:
        return value.prepress_distance <= (config.prepress_exit_m if was_candidate else config.prepress_enter_m)
    if stage == 1:
        return value.contact_distance <= (config.contact_exit_m if was_candidate else config.contact_enter_m)
    if stage == 2:
        terminal_distance = value.object_target_distance if config.use_object_target_distance else value.press_distance
        return terminal_distance <= (config.press_exit_m if was_candidate else config.press_enter_m)
    return False


def _smooth(action: npt.ArrayLike, previous: npt.ArrayLike, before: npt.ArrayLike, config: ButtonPressConfig) -> float:
    action_v, previous_v, before_v = _vec3(action, "action"), _vec3(previous, "previous_action"), _vec3(before, "previous_previous_action")
    rate = float(np.mean(np.square(action_v - previous_v)))
    curve = float(np.mean(np.square(action_v - 2.0 * previous_v + before_v)))
    boundary = np.clip((np.abs(action_v) - config.boundary_free_limit) / (config.boundary_hard_limit - config.boundary_free_limit), 0.0, 1.0)
    return float(config.action_rate_weight * rate + config.action_curve_weight * curve + config.boundary_weight * np.mean(np.square(boundary)))


def step(state: ButtonPressState, value: ButtonPressFeatures, action: npt.ArrayLike, config: ButtonPressConfig) -> ButtonPressStep:
    old_stage = int(state.stage)
    if old_stage == TERMINAL_STAGE:
        return ButtonPressStep(0.0, False, False, old_stage, old_stage)
    mode = reward_form_mode(config)
    if mode == STAGED_RESET:
        quality = stage_quality(old_stage, value, config)
        previous = quality if state.previous_quality is None else state.previous_quality
        dense = config.dense_scales[old_stage] * (config.gamma * quality - previous)
    else:
        dense, quality = dense_sum(
            mode=mode,
            current_stage=old_stage,
            active_stage_count=ACTIVE_STAGE_COUNT,
            dense_scales=config.dense_scales,
            gamma=config.gamma,
            previous_stage_qualities=state.previous_stage_qualities,
            quality_fn=lambda stage: stage_quality(stage, value, config),
        )
    smooth = _smooth(action, state.previous_action, state.previous_previous_action, config)
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
    state.previous_action = _vec3(action, "action").copy()
    return ButtonPressStep(sparse_step_reward(mode, dense - smooth), due, strict, old_stage, old_stage)


def apply_qwen(state: ButtonPressState, value: ButtonPressFeatures, decision: bool, config: ButtonPressConfig, *, bypass_candidate: bool = False) -> float:
    old_stage = int(state.stage)
    if old_stage < 0 or old_stage >= ACTIVE_STAGE_COUNT or (
        not bypass_candidate
        and (
            not state.request_armed
            or state.stable_count < config.required_holds[old_stage]
        )
    ):
        raise ValueError("no armed button-press candidate")
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
        terminal_stage=TERMINAL_STAGE,
    )

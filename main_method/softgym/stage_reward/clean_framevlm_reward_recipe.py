"""Clean Frame-VLM generated reward recipe for SoftGym.

This module implements the reward-function side of the clean SoftGym test.  It
uses only task-language-compatible simulator state interfaces: particle states,
cup/object poses, task target metadata, and geometric relations.  The benchmark
metric fields and environment reward functions are intentionally not used here.

The returned value is a bounded stage potential phi_i(x), as required by
reward_model/formulation.md.  It is not a final trajectory score and is not a
benchmark-answer normalization.  Dense reward is produced by the stage wrapper
as discounted potential difference on the currently active stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CLEAN_FRAMEVLM_V1_VERSION = "clean_framevlm_v1"
CLEAN_FRAMEVLM_V1_SOURCE = "frame_vlm_generated_clean_v1_raw_state_potential"
CLOTHDROP_TARGET_V4_VERSION = "clothdrop_target_v4"
CLOTHDROP_TARGET_V4_SOURCE = "frame_vlm_generated_clothdrop_official_target_geometry_v1"
CLOTHDROP_TARGET_V5_VERSION = "clothdrop_target_v5"
CLOTHDROP_TARGET_V5_SOURCE = "frame_vlm_generated_clothdrop_official_target_alignment_v1"
CLOTHDROP_TARGET_V6_VERSION = "clothdrop_target_v6"
CLOTHDROP_TARGET_V6_SOURCE = "frame_vlm_generated_clothdrop_official_target_progress_alignment_v1"
CLEAN_FRAMEVLM_V1_ALLOWED_SIGNALS: dict[str, tuple[str, ...]] = {
    "ClothDrop": (
        "pyflex particle positions",
        "current cloth size metadata",
        "cloth particle spacing/radius",
    ),
    "ClothFlatten": (
        "pyflex particle positions",
        "current cloth size metadata",
        "cloth particle spacing/radius",
    ),
    "ClothFold": (
        "pyflex particle positions",
        "fold_group_a/fold_group_b particle indices",
        "initial cloth particle positions",
    ),
    "RopeFlatten": (
        "pyflex ordered rope particle positions",
        "rope_length metadata when available",
    ),
    "PassWater": (
        "cup x position and target x metadata",
        "water particle positions from env.get_state()",
        "env.in_glass geometric containment test",
        "glass geometry states/parameters",
    ),
    "PourWater": (
        "source cup pose and target cup geometry metadata",
        "water particle positions from env.get_state()",
        "env.in_glass geometric containment test",
        "source and target glass geometry states/parameters",
    ),
}


CLOTHDROP_TARGET_V4_ALLOWED_SIGNALS: dict[str, tuple[str, ...]] = {
    "ClothDrop": CLEAN_FRAMEVLM_V1_ALLOWED_SIGNALS["ClothDrop"]
    + ("task target_pos particle layout metadata (simulator target geometry)",),
}


def _clip01(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    den = float(den)
    if abs(den) < 1e-8:
        return float(default)
    return float(num) / den


def _positions() -> np.ndarray:
    import pyflex

    return np.asarray(pyflex.get_positions(), dtype=np.float32).reshape(-1, 4)[:, :3]


def _xz_bbox_area(pos: np.ndarray) -> float:
    if len(pos) == 0:
        return 0.0
    span = np.ptp(pos[:, [0, 2]], axis=0)
    return float(max(span[0] * span[1], 0.0))


def _cloth_target_bbox_area(env: Any, fallback_area: float) -> float:
    try:
        dim_x, dim_z = env.get_current_config()["ClothSize"]
        spacing = float(getattr(env, "cloth_particle_radius", 0.00625))
        target = float(max(dim_x - 1, 1) * max(dim_z - 1, 1)) * spacing * spacing
    except Exception:
        target = float(fallback_area)
    return max(float(target), float(fallback_area) + 1e-6)


def _water_positions(env: Any) -> np.ndarray:
    state = env.get_state()
    return np.asarray(state["particle_pos"], dtype=np.float32).reshape(-1, env.dim_position)


def _inside_fraction(env: Any, water: np.ndarray, glass_states: Any, border: float, height: float) -> float:
    inside = env.in_glass(water, glass_states, border, height)
    arr = np.asarray(inside)
    if arr.shape == ():
        count = float(arr)
    else:
        count = float(np.sum(arr))
    return _clip01(_safe_div(count, max(len(water), 1), default=0.0))


def _pour_inside_fractions(env: Any) -> tuple[float, float, float]:
    water = _water_positions(env)
    target_inside = np.asarray(
        env.in_glass(water, env.poured_glass_states, env.poured_border, env.poured_height),
        dtype=np.float32,
    )
    source_inside = np.asarray(
        env.in_glass(water, env.glass_states, env.border, env.height),
        dtype=np.float32,
    )
    target_only = target_inside * (1.0 - source_inside)
    target_frac = _clip01(_safe_div(float(np.sum(target_only)), max(len(water), 1), default=0.0))
    source_frac = _clip01(_safe_div(float(np.sum(source_inside)), max(len(water), 1), default=0.0))
    retained_frac = _clip01(target_frac + source_frac)
    return target_frac, source_frac, retained_frac


@dataclass
class CleanFrameVlmRewardRecipe:
    """Frame-VLM-generated clean v1 stage potential recipe."""

    task: str
    initial: dict[str, float] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return CLEAN_FRAMEVLM_V1_VERSION

    @property
    def source(self) -> str:
        return CLEAN_FRAMEVLM_V1_SOURCE

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return CLEAN_FRAMEVLM_V1_ALLOWED_SIGNALS.get(self.task, ())

    def reset(self, env: Any) -> None:
        self.initial = {}
        task = self.task
        if task in {"ClothDrop", "ClothFlatten", "ClothFold", "RopeFlatten"}:
            pos = _positions()
        if task == "ClothDrop":
            self.initial["mean_height"] = float(np.mean(pos[:, 1])) if len(pos) else 0.0
            self.initial["xz_area"] = _xz_bbox_area(pos)
            self.initial["target_area"] = _cloth_target_bbox_area(env, self.initial["xz_area"])
        elif task == "ClothFlatten":
            area = _xz_bbox_area(pos)
            self.initial["xz_area"] = area
            self.initial["target_area"] = _cloth_target_bbox_area(env, area)
        elif task == "ClothFold":
            group_dist, anchor_shift = self._cloth_fold_features(env, pos)
            self.initial["group_dist"] = max(group_dist, 1e-6)
            self.initial["anchor_shift"] = max(anchor_shift, 1e-6)
        elif task == "RopeFlatten":
            endpoint_dist, path_len = self._rope_features(pos)
            self.initial["endpoint_dist"] = endpoint_dist
            self.initial["path_len"] = max(path_len, endpoint_dist + 1e-6)
            self.initial["rope_length"] = max(
                float(getattr(env, "rope_length", path_len)), endpoint_dist + 1e-6
            )
        elif task == "PassWater":
            self.initial["glass_x"] = float(env.glass_x)
            self.initial["target_x"] = float(env.terminal_x)
            self.initial["retained"] = self._pass_water_retained(env)
        elif task == "PourWater":
            self.initial["source_x"] = float(getattr(env, "glass_x", 0.0))
            target_x = float(getattr(env, "x_center", 0.0) + getattr(env, "glass_distance", 0.0))
            try:
                target_x = float(env.glass_params.get("poured_glass_x_center", target_x))
            except Exception:
                pass
            self.initial["target_x"] = target_x
            target_frac, source_frac, retained_frac = _pour_inside_fractions(env)
            self.initial["target_frac"] = target_frac
            self.initial["source_frac"] = source_frac
            self.initial["retained_frac"] = retained_frac

    def stage_potential(self, env: Any, stage_id: int) -> float:
        if stage_id >= 2:
            return 1.0
        task = self.task
        if task == "ClothDrop":
            return self._cloth_drop_potential(env, stage_id)
        if task == "ClothFlatten":
            return self._cloth_flatten_potential(env, stage_id)
        if task == "ClothFold":
            return self._cloth_fold_potential(env, stage_id)
        if task == "RopeFlatten":
            return self._rope_flatten_potential(env, stage_id)
        if task == "PassWater":
            return self._pass_water_potential(env, stage_id)
        if task == "PourWater":
            return self._pour_water_potential(env, stage_id)
        return 0.0

    def accepted(self, env: Any, stage_id: int, threshold: float) -> bool:
        return self.stage_potential(env, stage_id) >= float(threshold)

    def _cloth_area_growth(self, pos: np.ndarray) -> tuple[float, float, float]:
        area = _xz_bbox_area(pos)
        init = float(self.initial.get("xz_area", area))
        target = max(float(self.initial.get("target_area", area)), init + 1e-6)
        growth = _clip01(_safe_div(area - init, target - init, default=0.0))
        height_span = float(np.ptp(pos[:, 1])) if len(pos) else 0.0
        flatness = _clip01(1.0 - height_span / 0.18)
        return area, growth, flatness

    def _cloth_drop_potential(self, env: Any, stage_id: int) -> float:
        pos = _positions()
        if len(pos) == 0:
            return 0.0
        init_h = max(float(self.initial.get("mean_height", np.mean(pos[:, 1]))), 1e-6)
        mean_h = float(np.mean(pos[:, 1]))
        max_h = float(np.max(pos[:, 1]))
        height_span = float(np.ptp(pos[:, 1]))
        _, spread, _ = self._cloth_area_growth(pos)
        drop = _clip01(_safe_div(init_h - mean_h, max(init_h - 0.03, 1e-6), default=0.0))
        low = _clip01(1.0 - mean_h / 0.18)
        no_hanging = _clip01(1.0 - max_h / 0.35)
        flatness = _clip01(1.0 - height_span / 0.18)
        if stage_id == 0:
            return _clip01(0.80 * drop + 0.20 * no_hanging)
        return _clip01(0.35 * low + 0.25 * no_hanging + 0.25 * flatness + 0.15 * spread)

    def _cloth_flatten_potential(self, env: Any, stage_id: int) -> float:
        pos = _positions()
        if len(pos) == 0:
            return 0.0
        _, spread, flatness = self._cloth_area_growth(pos)
        if stage_id == 0:
            return _clip01(0.85 * spread + 0.15 * flatness)
        return _clip01(0.65 * spread + 0.35 * flatness)

    def _cloth_fold_features(self, env: Any, pos: np.ndarray) -> tuple[float, float]:
        group_a = np.asarray(env.fold_group_a, dtype=np.int64)
        group_b = np.asarray(env.fold_group_b, dtype=np.int64)
        if len(group_a) == 0 or len(group_b) == 0 or len(pos) == 0:
            return 0.0, 0.0
        paired_dist = float(np.mean(np.linalg.norm(pos[group_a] - pos[group_b], axis=1)))
        init_pos = np.asarray(env.init_pos, dtype=np.float32)
        anchor_shift = float(np.mean(np.linalg.norm(pos[group_b] - init_pos[group_b], axis=1)))
        return paired_dist, anchor_shift

    def _cloth_fold_potential(self, env: Any, stage_id: int) -> float:
        pos = _positions()
        paired_dist, anchor_shift = self._cloth_fold_features(env, pos)
        init_dist = max(float(self.initial.get("group_dist", paired_dist)), 1e-6)
        overlap = _clip01(1.0 - paired_dist / init_dist)
        anchor_ok = _clip01(1.0 - anchor_shift / max(0.35 * init_dist, 1e-6))
        if stage_id == 0:
            return overlap
        return _clip01(0.80 * overlap + 0.20 * anchor_ok)

    def _rope_features(self, pos: np.ndarray) -> tuple[float, float]:
        if len(pos) < 2:
            return 0.0, 0.0
        endpoint = float(np.linalg.norm(pos[0] - pos[-1]))
        path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        return endpoint, max(path_len, endpoint + 1e-6)

    def _rope_flatten_potential(self, env: Any, stage_id: int) -> float:
        pos = _positions()
        endpoint, path_len = self._rope_features(pos)
        init_endpoint = float(self.initial.get("endpoint_dist", endpoint))
        target_len = max(float(self.initial.get("rope_length", path_len)), path_len, init_endpoint + 1e-6)
        extension = _clip01(_safe_div(endpoint - init_endpoint, target_len - init_endpoint, default=0.0))
        straightness = _clip01(_safe_div(endpoint, path_len, default=0.0))
        height_span = float(np.ptp(pos[:, 1])) if len(pos) else 0.0
        on_plane = _clip01(1.0 - height_span / 0.12)
        if stage_id == 0:
            return _clip01(0.80 * extension + 0.20 * straightness)
        return _clip01(0.65 * straightness + 0.25 * extension + 0.10 * on_plane)

    def _pass_water_retained(self, env: Any) -> float:
        water = _water_positions(env)
        return _inside_fraction(env, water, env.glass_states, env.border, env.height)

    def _pass_water_potential(self, env: Any, stage_id: int) -> float:
        start = float(self.initial.get("glass_x", env.glass_x))
        target = float(self.initial.get("target_x", env.terminal_x))
        path = max(abs(target - start), 1e-6)
        direction = 1.0 if target >= start else -1.0
        moved = _clip01(_safe_div((float(env.glass_x) - start) * direction, path, default=0.0))
        close = _clip01(1.0 - _safe_div(abs(target - float(env.glass_x)), path, default=1.0))
        retained = self._pass_water_retained(env)
        retain_gate = _clip01(_safe_div(retained - 0.78, 0.20, default=0.0))
        if stage_id == 0:
            return _clip01(0.80 * moved * retain_gate + 0.20 * retain_gate)
        return _clip01(0.70 * close * retain_gate + 0.30 * retained)

    def _pour_water_potential(self, env: Any, stage_id: int) -> float:
        target_frac, source_frac, retained_frac = _pour_inside_fractions(env)
        source_x = float(getattr(env, "glass_x", 0.0))
        target_x = float(self.initial.get("target_x", source_x + getattr(env, "glass_distance", 0.0)))
        init_source = float(self.initial.get("source_x", source_x))
        path = max(abs(target_x - init_source), 1e-6)
        align = _clip01(1.0 - _safe_div(abs(target_x - source_x), path, default=1.0))
        tilt = _clip01(abs(float(getattr(env, "glass_rotation", 0.0))) / 0.70)
        early_transfer = _clip01(target_frac / 0.15)
        retain_gate = _clip01(_safe_div(retained_frac - 0.70, 0.25, default=0.0))
        if stage_id == 0:
            return _clip01((0.40 * align + 0.40 * tilt + 0.20 * early_transfer) * retain_gate)
        target_fill = _clip01(target_frac / 0.65)
        source_emptying = _clip01(1.0 - source_frac / 0.45)
        return _clip01(0.70 * target_fill + 0.20 * retained_frac + 0.10 * source_emptying)

@dataclass
class ClothDropTargetAwareRewardRecipe(CleanFrameVlmRewardRecipe):
    """Target-aware ClothDrop recipe generated from the simulator state API.

    This is intentionally a separate ablation from ``clean_framevlm_v1``.  It
    uses the environment's task target particle layout (``target_pos``) as
    geometric task metadata, but never reads SoftGym's ``performance`` or
    ``normalized_performance`` fields.  The potential remains bounded and is
    consumed by the same formulation.md potential-difference reward.
    """

    @property
    def version(self) -> str:
        return CLOTHDROP_TARGET_V4_VERSION

    @property
    def source(self) -> str:
        return CLOTHDROP_TARGET_V4_SOURCE

    def reset(self, env: Any) -> None:
        super().reset(env)
        target = self._target_positions(env)
        pos = _positions()
        if len(target) == len(pos) and len(target) > 0:
            self.initial["target_distance"] = max(
                float(np.mean(np.linalg.norm(pos - target, axis=1))), 1e-6
            )
        else:
            self.initial["target_distance"] = 0.0
        try:
            dim_x, dim_z = env.get_current_config()["ClothSize"]
            spacing = float(getattr(env, "cloth_particle_radius", 0.00625))
            extent = float(
                np.hypot(max(dim_x - 1, 1) * spacing, max(dim_z - 1, 1) * spacing)
            )
        except Exception:
            extent = 0.30
        self.initial["target_scale"] = max(0.5 * extent, 0.12)

    @staticmethod
    def _target_positions(env: Any) -> np.ndarray:
        try:
            target = env.get_current_config().get("target_pos")
        except Exception:
            target = None
        if target is None:
            return np.empty((0, 3), dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if target.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return target.reshape(-1, 3)

    def _target_alignment(self, env: Any) -> float:
        pos = _positions()
        target = self._target_positions(env)
        if len(pos) == 0 or len(target) != len(pos):
            return 0.0
        curr_dist = float(np.mean(np.linalg.norm(pos - target, axis=1)))
        scale = max(float(self.initial.get("target_scale", 0.20)), 1e-6)
        return _clip01(np.exp(-curr_dist / scale))

    def _target_progress(self, env: Any) -> float:
        pos = _positions()
        target = self._target_positions(env)
        init_dist = float(self.initial.get("target_distance", 0.0))
        if len(pos) == 0 or len(target) != len(pos) or init_dist <= 1e-8:
            return 0.0
        curr_dist = float(np.mean(np.linalg.norm(pos - target, axis=1)))
        # Target-geometry progress, not a read of SoftGym's official metric.
        return _clip01((init_dist - curr_dist) / init_dist)

    def _cloth_drop_potential(self, env: Any, stage_id: int) -> float:
        pos = _positions()
        if len(pos) == 0:
            return 0.0
        target_progress = self._target_progress(env)
        mean_h = float(np.mean(pos[:, 1]))
        max_h = float(np.max(pos[:, 1]))
        height_span = float(np.ptp(pos[:, 1]))
        low = _clip01(1.0 - mean_h / 0.18)
        no_hanging = _clip01(1.0 - max_h / 0.35)
        flatness = _clip01(1.0 - height_span / 0.18)
        _, spread, _ = self._cloth_area_growth(pos)
        if stage_id == 0:
            return _clip01(0.55 * target_progress + 0.30 * no_hanging + 0.15 * low)
        return _clip01(0.70 * target_progress + 0.15 * flatness + 0.10 * spread + 0.05 * low)

@dataclass
class ClothDropTargetAlignmentRewardRecipe(ClothDropTargetAwareRewardRecipe):
    """v5: continuous absolute alignment plus simple release/flatness cues."""

    @property
    def version(self) -> str:
        return CLOTHDROP_TARGET_V5_VERSION

    @property
    def source(self) -> str:
        return CLOTHDROP_TARGET_V5_SOURCE

    def _cloth_drop_potential(self, env: Any, stage_id: int) -> float:
        pos = _positions()
        if len(pos) == 0:
            return 0.0
        alignment = self._target_alignment(env)
        mean_h = float(np.mean(pos[:, 1]))
        max_h = float(np.max(pos[:, 1]))
        low = _clip01(1.0 - mean_h / 0.18)
        no_hanging = _clip01(1.0 - max_h / 0.35)
        flatness = _clip01(1.0 - float(np.ptp(pos[:, 1])) / 0.18)
        _, spread, _ = self._cloth_area_growth(pos)
        if stage_id == 0:
            return _clip01(0.25 * alignment + 0.45 * no_hanging + 0.30 * low)
        return _clip01(0.50 * alignment + 0.20 * flatness + 0.20 * spread + 0.10 * low)


@dataclass
class ClothDropTargetProgressAlignmentRewardRecipe(ClothDropTargetAwareRewardRecipe):
    """v6: combine relative target progress with continuous target alignment."""

    @property
    def version(self) -> str:
        return CLOTHDROP_TARGET_V6_VERSION

    @property
    def source(self) -> str:
        return CLOTHDROP_TARGET_V6_SOURCE

    def _cloth_drop_potential(self, env: Any, stage_id: int) -> float:
        pos = _positions()
        if len(pos) == 0:
            return 0.0
        progress = self._target_progress(env)
        alignment = self._target_alignment(env)
        mean_h = float(np.mean(pos[:, 1]))
        low = _clip01(1.0 - mean_h / 0.18)
        flatness = _clip01(1.0 - float(np.ptp(pos[:, 1])) / 0.18)
        _, spread, _ = self._cloth_area_growth(pos)
        if stage_id == 0:
            return _clip01(0.35 * progress + 0.30 * alignment + 0.20 * low + 0.15 * spread)
        return _clip01(0.45 * progress + 0.35 * alignment + 0.10 * flatness + 0.10 * spread)

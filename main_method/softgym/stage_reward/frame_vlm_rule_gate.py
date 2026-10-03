"""Frame-VLM-style geometric rule gates for SoftGym stages.

These predicates intentionally do not read SoftGym's official
``normalized_performance`` or ``performance`` fields. They compile the stage
text into simple state/geometry evidence that can be audited separately from the
benchmark metric used only for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(float(den)) < 1e-8:
        return default
    return float(num) / float(den)


def _positions() -> np.ndarray:
    import pyflex

    return np.asarray(pyflex.get_positions(), dtype=np.float32).reshape(-1, 4)[:, :3]


@dataclass
class FrameVlmRuleGate:
    """Deterministic stage-completion predicate from Frame-VLM task semantics."""

    task: str
    initial: dict[str, float] = field(default_factory=dict)

    def reset(self, env: Any) -> None:
        self.initial = {}
        task = self.task
        pos = _positions()
        if task == "PassWater":
            self.initial["glass_x"] = float(env.glass_x)
            self.initial["target_x"] = float(env.terminal_x)
            self.initial["water_retained"] = self._pass_water_retained(env)
        elif task == "PourWater":
            self.initial["source_x"] = float(env.glass_x)
            self.initial["target_x"] = float(env.glass_x + env.glass_distance)
            target_frac, source_frac = self._pour_water_fractions(env)
            self.initial["target_frac"] = target_frac
            self.initial["source_frac"] = source_frac
        elif task == "RopeFlatten":
            self.initial["endpoint_dist"] = self._rope_endpoint_distance(pos)
            self.initial["rope_length"] = float(env.rope_length)
        elif task == "ClothFlatten":
            area = self._xz_bbox_area(pos)
            self.initial["covered_area"] = area
            self.initial["max_area"] = self._cloth_flat_target_area(env, area)
        elif task == "ClothFold":
            group_dist, fixation_dist = self._cloth_fold_distances(env, pos)
            self.initial["group_dist"] = group_dist
            self.initial["fixation_dist"] = max(fixation_dist, 1e-6)
        elif task == "ClothDrop":
            self.initial["mean_height"] = float(np.mean(pos[:, 1]))
            self.initial["height_span"] = float(np.ptp(pos[:, 1]))
            self.initial["xz_area"] = self._xz_bbox_area(pos)

    def score(self, env: Any, stage_id: int) -> float:
        task = self.task
        if task == "PassWater":
            return self._score_pass_water(env, stage_id)
        if task == "PourWater":
            return self._score_pour_water(env, stage_id)
        pos = _positions()
        if task == "RopeFlatten":
            return self._score_rope_flatten(env, pos, stage_id)
        if task == "ClothFlatten":
            return self._score_cloth_flatten(env, pos, stage_id)
        if task == "ClothFold":
            return self._score_cloth_fold(env, pos, stage_id)
        if task == "ClothDrop":
            return self._score_cloth_drop(pos, stage_id)
        return 0.0

    def accepted(self, env: Any, stage_id: int, threshold: float) -> bool:
        return self.score(env, stage_id) >= float(threshold)

    def _pass_water_retained(self, env: Any) -> float:
        state = env.get_state()
        water = np.asarray(state["particle_pos"], dtype=np.float32).reshape(-1, env.dim_position)
        inside = float(env.in_glass(water, env.glass_states, env.border, env.height))
        return _clip01(_safe_div(inside, len(water)))

    def _score_pass_water(self, env: Any, stage_id: int) -> float:
        start = self.initial.get("glass_x", float(env.glass_x))
        target = self.initial.get("target_x", float(env.terminal_x))
        path = abs(target - start)
        direction = 1.0 if target >= start else -1.0
        progress = _clip01(_safe_div((float(env.glass_x) - start) * direction, path, default=0.0))
        closeness = _clip01(1.0 - _safe_div(abs(float(env.terminal_x) - float(env.glass_x)), path, default=1.0))
        retained = self._pass_water_retained(env)
        spill_gate = _clip01((retained - 0.55) / 0.35)
        if stage_id == 0:
            return _clip01(progress * spill_gate)
        return _clip01(min(closeness, retained))

    def _pour_water_fractions(self, env: Any) -> tuple[float, float]:
        state = env.get_state()
        water = np.asarray(state["particle_pos"], dtype=np.float32).reshape(-1, env.dim_position)
        target = np.asarray(env.in_glass(water, env.poured_glass_states, env.poured_border, env.poured_height), dtype=np.float32)
        source = np.asarray(env.in_glass(water, env.glass_states, env.border, env.height), dtype=np.float32)
        target_frac = float(np.sum(target * (1.0 - source))) / float(len(water))
        source_frac = float(np.sum(source)) / float(len(water))
        return _clip01(target_frac), _clip01(source_frac)

    def _score_pour_water(self, env: Any, stage_id: int) -> float:
        target_frac, source_frac = self._pour_water_fractions(env)
        rotation = abs(float(getattr(env, "glass_rotation", 0.0)))
        tilt_score = _clip01(rotation / 0.75)
        source_x = float(getattr(env, "glass_x", 0.0))
        target_x = float(getattr(env, "glass_x", 0.0) + getattr(env, "glass_distance", 0.0))
        init_source = self.initial.get("source_x", source_x)
        init_target = self.initial.get("target_x", target_x)
        approach = _clip01(_safe_div(abs(source_x - init_source), max(abs(init_target - init_source), 1e-6)))
        if stage_id == 0:
            begun_transfer = _clip01(target_frac / 0.12)
            has_water = _clip01((source_frac + target_frac - 0.25) / 0.60)
            return _clip01(max(tilt_score, begun_transfer, approach * 0.6) * has_water)
        return _clip01(target_frac / 0.75)

    def _rope_endpoint_distance(self, pos: np.ndarray) -> float:
        if len(pos) < 2:
            return 0.0
        return float(np.linalg.norm(pos[0] - pos[-1]))

    def _score_rope_flatten(self, env: Any, pos: np.ndarray, stage_id: int) -> float:
        init_dist = self.initial.get("endpoint_dist", self._rope_endpoint_distance(pos))
        rope_length = self.initial.get("rope_length", float(getattr(env, "rope_length", init_dist + 1e-6)))
        curr = self._rope_endpoint_distance(pos)
        stretch_progress = _clip01(_safe_div(curr - init_dist, rope_length - init_dist, default=0.0))
        straightness = _clip01(_safe_div(curr, rope_length, default=0.0))
        return stretch_progress if stage_id == 0 else straightness

    def _cloth_flat_target_area(self, env: Any, fallback_area: float) -> float:
        try:
            dim_x, dim_z = env.get_current_config()["ClothSize"]
            spacing = float(getattr(env, "cloth_particle_radius", 0.00625))
            target = max(float(dim_x - 1), 1.0) * max(float(dim_z - 1), 1.0) * spacing * spacing
        except Exception:
            target = float(fallback_area)
        return max(float(target), float(fallback_area) + 1e-6)

    def _score_cloth_flatten(self, env: Any, pos: np.ndarray, stage_id: int) -> float:
        area = self._xz_bbox_area(pos)
        init_area = self.initial.get("covered_area", area)
        max_area = max(self.initial.get("max_area", area), init_area + 1e-6)
        progress = _clip01(_safe_div(area - init_area, max_area - init_area, default=0.0))
        height_span = float(np.ptp(pos[:, 1])) if len(pos) else 0.0
        flatness = _clip01(1.0 - height_span / 0.18)
        if stage_id == 0:
            return progress
        return _clip01(0.75 * progress + 0.25 * flatness)

    def _cloth_fold_distances(self, env: Any, pos: np.ndarray) -> tuple[float, float]:
        group_a = np.asarray(env.fold_group_a, dtype=np.int64)
        group_b = np.asarray(env.fold_group_b, dtype=np.int64)
        if len(group_a) == 0 or len(group_b) == 0:
            return 0.0, 0.0
        pos_a = pos[group_a]
        pos_b = pos[group_b]
        group_dist = float(np.mean(np.linalg.norm(pos_a - pos_b, axis=1)))
        init_pos = np.asarray(env.init_pos, dtype=np.float32)
        fixation_dist = float(np.mean(np.linalg.norm(pos_b - init_pos[group_b], axis=1)))
        return group_dist, fixation_dist

    def _score_cloth_fold(self, env: Any, pos: np.ndarray, stage_id: int) -> float:
        group_dist, fixation_dist = self._cloth_fold_distances(env, pos)
        init_group = max(self.initial.get("group_dist", group_dist), 1e-6)
        overlap_progress = _clip01(1.0 - group_dist / init_group)
        anchor_ok = _clip01(1.0 - fixation_dist / max(0.35 * init_group, 1e-6))
        if stage_id == 0:
            return overlap_progress
        return _clip01(0.85 * overlap_progress + 0.15 * anchor_ok)

    def _xz_bbox_area(self, pos: np.ndarray) -> float:
        if len(pos) == 0:
            return 0.0
        span = np.ptp(pos[:, [0, 2]], axis=0)
        return float(max(span[0] * span[1], 0.0))

    def _score_cloth_drop(self, pos: np.ndarray, stage_id: int) -> float:
        if len(pos) == 0:
            return 0.0
        init_h = max(self.initial.get("mean_height", float(np.mean(pos[:, 1]))), 1e-6)
        mean_h = float(np.mean(pos[:, 1]))
        max_h = float(np.max(pos[:, 1]))
        height_span = float(np.ptp(pos[:, 1]))
        drop_progress = _clip01((init_h - mean_h) / max(init_h - 0.03, 1e-6))
        low = _clip01(1.0 - mean_h / 0.18)
        no_hanging = _clip01(1.0 - max_h / 0.35)
        flatness = _clip01(1.0 - height_span / 0.18)
        if stage_id == 0:
            return drop_progress
        return _clip01(0.45 * low + 0.35 * no_hanging + 0.20 * flatness)

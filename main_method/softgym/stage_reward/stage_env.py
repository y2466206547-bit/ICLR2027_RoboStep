"""Active-stage SoftGym wrapper shared by Rule and Qwen training/evaluation."""

from __future__ import annotations

from collections import deque
import importlib
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from .clean_framevlm_reward_recipe import CleanFrameVlmRewardRecipe
from .clean_framevlm_reward_recipe import ClothDropTargetAwareRewardRecipe
from .clean_framevlm_reward_recipe import ClothDropTargetAlignmentRewardRecipe
from .clean_framevlm_reward_recipe import ClothDropTargetProgressAlignmentRewardRecipe
from .compat import install_legacy_compat
from .frame_vlm_rule_gate import FrameVlmRuleGate
from .lrm_reward import LRMProgressClient
from .task_config import TaskSpec, build_env_kwargs


class SoftGymStageEnv:
    """One legacy SoftGym environment with an explicit active-stage machine."""

    def __init__(
        self,
        *,
        spec: TaskSpec,
        cache_path: Path,
        gate: str,
        seed: int,
        output_dir: Path,
        qwen_gpu: int = 0,
        qwen_model_path: str | Path | None = None,
        num_variations: int = 1000,
        reward_version: str = "v1",
        reward_form_mode: str = "staged_reset",
        append_stage_obs: bool = True,
        terminal_sparse_threshold: float | None = None,
        terminal_sparse_metric: str | None = None,
        benchmark_native_dense: bool = False,
        per_subtask_sparse: bool = False,
        query_mode: str = "candidate_retry",
        query_interval_steps: int = 5,
    ) -> None:
        install_legacy_compat()
        module = importlib.import_module(spec.env_module)
        env_class = getattr(module, spec.env_class)
        self.spec = spec
        self.gate_name = gate
        self.seed = int(seed)
        self.reward_version = str(reward_version)
        self.reward_form_mode = str(reward_form_mode)
        if self.reward_form_mode not in {"staged_reset", "all_dense_with_stage_actor"}:
            raise ValueError(f"unknown reward_form_mode: {self.reward_form_mode}")
        self.append_stage_obs = bool(append_stage_obs)
        self.terminal_sparse_threshold = terminal_sparse_threshold
        self.terminal_sparse_metric = terminal_sparse_metric
        self.benchmark_native_dense = bool(benchmark_native_dense)
        self.per_subtask_sparse = bool(per_subtask_sparse)
        if sum((self.benchmark_native_dense, self.per_subtask_sparse, terminal_sparse_threshold is not None)) > 1:
            raise ValueError("native dense, terminal sparse, and per-subtask sparse are mutually exclusive")
        if self.per_subtask_sparse and gate != "rule":
            raise ValueError("per-subtask sparse requires the frozen Rule transition gate")
        if terminal_sparse_threshold is not None:
            if not np.isfinite(terminal_sparse_threshold):
                raise ValueError("terminal sparse threshold must be finite")
            if terminal_sparse_metric not in {"normalized_performance", "clothfold_semantic"}:
                raise ValueError(f"invalid terminal sparse metric: {terminal_sparse_metric}")
            if terminal_sparse_metric == "clothfold_semantic" and spec.name != "ClothFold":
                raise ValueError("clothfold_semantic is only valid for ClothFold")
        self.query_mode = str(query_mode)
        if self.query_mode not in {"candidate_retry", "fixed_frequency"}:
            raise ValueError(f"unknown query_mode: {self.query_mode}")
        self.query_interval_steps = max(1, int(query_interval_steps))
        self.gate_records: list[dict[str, Any]] = []
        self.code_expression = None
        if self.reward_version == "code_as_reward":
            expression_path = os.environ.get("SOFTGYM_CODE_REWARD_EXPRESSION_FILE")
            if not expression_path:
                raise ValueError("code_as_reward requires SOFTGYM_CODE_REWARD_EXPRESSION_FILE")
            import json
            payload = json.loads(Path(expression_path).read_text(encoding="utf-8"))
            self.code_expression = str(payload.get("expression", "")).strip()
            if not self.code_expression:
                raise ValueError(f"empty SoftGym code-as-reward expression: {expression_path}")
        self.lrm_client = (
            LRMProgressClient(
                server_url=os.environ.get("LRM_SERVER_URL", "http://127.0.0.1:5002"),
                timeout_s=float(os.environ.get("LRM_TIMEOUT_S", "600")),
                query_interval=int(os.environ.get("LRM_QUERY_INTERVAL", "10")),
                reward_scale=float(os.environ.get("LRM_REWARD_SCALE", "1.0")),
                mode=os.environ.get("LRM_REWARD_MODE", "progress"),
                max_frames=int(os.environ.get("LRM_MAX_FRAMES", "15")),
            )
            if self.reward_version.startswith("lrm_progress_")
            else None
        )
        self.rng = np.random.default_rng(seed)
        self.episode_id = 0
        cache_path = cache_path.expanduser().resolve()
        self.env = env_class(**build_env_kwargs(spec, cache_path, num_variations=num_variations))
        self.action_space = self.env.action_space
        self.base_obs_dim = int(np.prod(self.env.observation_space.shape))
        self.stage_count = len(spec.stages) + 1
        self.clean_recipe = (
            CleanFrameVlmRewardRecipe(spec.name)
            if self.reward_version == "clean_framevlm_v1"
            else None
        )
        target_recipe_cls = {
            "clothdrop_target_v4": ClothDropTargetAwareRewardRecipe,
            "clothdrop_target_v5": ClothDropTargetAlignmentRewardRecipe,
            "clothdrop_target_v6": ClothDropTargetProgressAlignmentRewardRecipe,
        }.get(self.reward_version)
        self.target_recipe = (
            target_recipe_cls(spec.name)
            if target_recipe_cls is not None and spec.name == "ClothDrop"
            else None
        )
        self.gate = None
        self.rule_gate = None
        if gate == "qwen":
            from .qwen_gate import SoftGymQwenGate

            self.gate = SoftGymQwenGate(
                output_dir=output_dir / "qwen_gate",
                stage_names=spec.stages,
                instructions=spec.instructions,
                qwen_gpu=qwen_gpu,
                qwen_model_path=qwen_model_path,
            )
        elif gate == "rule":
            self.rule_gate = FrameVlmRuleGate(spec.name)
        else:
            raise ValueError(f"unknown gate: {gate}")
        frame_ring_size = int(os.environ.get("SOFTGYM_QWEN_RING_FRAMES", "8"))
        self._frames: deque[Any] = deque(maxlen=max(2, frame_ring_size))
        self._reference = None
        self._step = 0
        self._stage = 0
        self._lrm_progress = 0.0
        # Official Robometer buffers one frame per VLM trigger, not every simulator step.
        self._lrm_frame_buffer: list[np.ndarray] = []
        self._prev_phi = 0.0
        self._prev_phis = [0.0 for _ in self.spec.bounds]
        self._hold = 0
        self._candidate_start_step = None
        self._candidate_epoch = 0
        self._last_gate_step = -10_000
        self._last_prediction = None
        self._config_id = -1

    @property
    def obs_dim(self) -> int:
        return self.base_obs_dim + self.stage_count if self.append_stage_obs else self.base_obs_dim

    def _capture(self):
        from stage_policy.live_vlm_gate import FrozenRgbFrame

        image = np.asarray(self.env.get_image(320, 320), dtype=np.uint8)
        frame = FrozenRgbFrame(
            rgb=image.tobytes(),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            global_step=self._step,
            sim_time_s=float(self._step) * 0.1,
            source_frame_id=self._step,
        )
        self._frames.append(frame)
        return frame

    def _actor_obs(self, base_obs: np.ndarray) -> np.ndarray:
        base = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        if not self.append_stage_obs:
            return base
        stage = np.zeros(self.stage_count, dtype=np.float32)
        stage[self._stage] = 1.0
        return np.concatenate([base, stage])

    def _uses_lrm_progress(self) -> bool:
        return self.lrm_client is not None

    def _frame_array(self, frame):
        return np.frombuffer(frame.rgb, dtype=np.uint8).reshape(frame.height, frame.width, 3)

    def _task_description(self) -> str:
        return f"{self.spec.name}: " + "; ".join(self.spec.instructions)

    def _uses_clean_framevlm_progress(self) -> bool:
        return self.clean_recipe is not None

    def _uses_clothfold_semantic_progress(self) -> bool:
        return (
            self.spec.name == "ClothFold"
            and self.reward_version.startswith("clothfold_semantic_v")
        )

    def _uses_clothfold_coverage_progress(self) -> bool:
        return (
            self.spec.name == "ClothFold"
            and self.reward_version.startswith("clothfold_coverage_v")
        )

    def _uses_passwater_spillaware_progress(self) -> bool:
        return (
            self.spec.name == "PassWater"
            and self.reward_version.startswith("passwater_spillaware_v")
        )

    def _uses_clothdrop_target_progress(self) -> bool:
        return (
            self.spec.name == "ClothDrop"
            and self.reward_version.startswith("clothdrop_target_v")
        )

    def _clip01(self, value: float) -> float:
        if not np.isfinite(float(value)):
            return 0.0
        return float(np.clip(float(value), 0.0, 1.0))

    def _cloth_fold_overlap_progress(self) -> float:
        import pyflex

        group_a = np.asarray(self.env.fold_group_a, dtype=np.int64)
        group_b = np.asarray(self.env.fold_group_b, dtype=np.int64)
        if len(group_a) == 0 or len(group_b) == 0:
            return 0.0
        pos = np.asarray(pyflex.get_positions(), dtype=np.float32).reshape(-1, 4)[:, :3]
        final_group_dist = float(np.mean(np.linalg.norm(pos[group_a] - pos[group_b], axis=1)))
        initial_group_dist = float(getattr(self.env, "prev_dist", final_group_dist))
        if not np.isfinite(initial_group_dist) or initial_group_dist <= 1e-8:
            return 0.0
        return float(np.clip(1.0 - final_group_dist / initial_group_dist, 0.0, 1.0))

    def _nearest_coverage_fraction(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return 0.0
        tolerance = max(2.0 * float(getattr(self.env, "cloth_particle_radius", 0.00625)), 1e-4)
        tol_sq = tolerance * tolerance
        covered = 0
        target = target.astype(np.float32, copy=False)
        for begin in range(0, len(source), 512):
            block = source[begin : begin + 512].astype(np.float32, copy=False)
            diff = block[:, None, :] - target[None, :, :]
            dist_sq = (diff * diff).sum(axis=2)
            covered += int((dist_sq.min(axis=1) <= tol_sq).sum())
        return self._clip01(covered / max(float(len(source)), 1.0))

    def _cloth_fold_coverage_progress(self) -> float:
        import pyflex

        group_a = np.asarray(self.env.fold_group_a, dtype=np.int64)
        group_b = np.asarray(self.env.fold_group_b, dtype=np.int64)
        if len(group_a) == 0 or len(group_b) == 0:
            return 0.0
        pos = np.asarray(pyflex.get_positions(), dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(pos) == 0 or max(int(group_a.max()), int(group_b.max())) >= len(pos):
            return 0.0
        paired_dist = float(np.mean(np.linalg.norm(pos[group_a] - pos[group_b], axis=1)))
        initial_group_dist = float(getattr(self.env, "prev_dist", paired_dist))
        if not np.isfinite(initial_group_dist) or initial_group_dist <= 1e-8:
            approach = 0.0
        else:
            approach = self._clip01(1.0 - paired_dist / initial_group_dist)

        pos_a = pos[group_a][:, [0, 2]]
        pos_b = pos[group_b][:, [0, 2]]
        coverage = self._clip01(
            0.5
            * (
                self._nearest_coverage_fraction(pos_a, pos_b)
                + self._nearest_coverage_fraction(pos_b, pos_a)
            )
        )
        version = self.reward_version
        if version == "clothfold_coverage_v1":
            return coverage
        if version == "clothfold_coverage_v2":
            if self._stage == 0:
                return self._clip01(0.70 * approach + 0.30 * coverage)
            return coverage
        if version == "clothfold_coverage_v3":
            if self._stage == 0:
                return self._clip01(0.50 * approach + 0.50 * coverage)
            return self._clip01(0.85 * coverage + 0.15 * approach)
        return coverage

    def _pass_water_retained(self) -> float:
        state = self.env.get_state()
        water = np.asarray(state["particle_pos"], dtype=np.float32).reshape(-1, self.env.dim_position)
        inside = float(self.env.in_glass(water, self.env.glass_states, self.env.border, self.env.height))
        return self._clip01(inside / max(float(len(water)), 1.0))

    def _pass_water_spillaware_progress(self) -> float:
        start = float(getattr(self.env, "x_center", self.env.glass_x))
        target = float(self.env.terminal_x)
        path = max(abs(target - start), 1e-6)
        direction = 1.0 if target >= start else -1.0
        moved = self._clip01((float(self.env.glass_x) - start) * direction / path)
        close = self._clip01(1.0 - abs(target - float(self.env.glass_x)) / path)
        retained = self._pass_water_retained()
        version = self.reward_version
        if version in {
            "passwater_spillaware_v4",
            "passwater_spillaware_v5",
            "passwater_spillaware_v6",
            "passwater_spillaware_v7",
            "passwater_spillaware_v8",
            "passwater_spillaware_v9",
            "passwater_spillaware_v10",
        }:
            spill = self._clip01(1.0 - retained)
            if version == "passwater_spillaware_v4":
                retain_gate = self._clip01((retained - 0.86) / 0.10)
                if self._stage == 0:
                    return self._clip01(moved * retain_gate)
                return self._clip01(close - spill / 0.14)
            if version == "passwater_spillaware_v5":
                retain_gate = self._clip01((retained - 0.90) / 0.08)
                if self._stage == 0:
                    return self._clip01(moved * retain_gate)
                return self._clip01(close - spill / 0.10)
            if version == "passwater_spillaware_v6":
                retain_gate = self._clip01((retained - 0.82) / 0.16)
                if self._stage == 0:
                    return self._clip01(moved * retain_gate)
                return self._clip01(0.75 * close + 0.25 * retained - 1.25 * spill)
            if version == "passwater_spillaware_v7":
                if self._stage == 0:
                    return self._clip01(moved * retained * retained)
                return self._clip01(close * retained**4)
            if version == "passwater_spillaware_v8":
                retain_gate = self._clip01((retained - 0.88) / 0.10)
                if self._stage == 0:
                    return self._clip01(moved * retain_gate)
                return self._clip01((close**1.5) * retained**6)
            if version == "passwater_spillaware_v9":
                retain_gate = self._clip01((retained - 0.92) / 0.06)
                if self._stage == 0:
                    return self._clip01(0.80 * moved * retain_gate + 0.20 * retained)
                return self._clip01(min(close, retain_gate))
            retain_gate = self._clip01((retained - 0.94) / 0.05)
            if self._stage == 0:
                return self._clip01(moved * retain_gate)
            return self._clip01((0.70 * close + 0.30 * moved) * retain_gate)
        strict_retained = self._clip01((retained - 0.90) / 0.08)
        if self._stage == 0:
            return self._clip01(moved * strict_retained)
        return self._clip01(min(close, strict_retained))

    def _clothdrop_target_progress(self) -> float:
        import pyflex

        pos = np.asarray(pyflex.get_positions(), dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(pos) == 0:
            return 0.0
        config = self.env.get_current_config()
        dim_x, dim_z = config["ClothSize"]
        spacing = float(getattr(self.env, "cloth_particle_radius", 0.00625))
        target_center = np.array([0.5 * max(dim_x - 1, 1) * spacing, 0.0], dtype=np.float32)
        center = np.mean(pos[:, [0, 2]], axis=0)
        center_scale = max(0.5 * max(dim_x, dim_z) * spacing, 0.15)
        center_score = self._clip01(1.0 - float(np.linalg.norm(center - target_center)) / center_scale)
        span = np.ptp(pos[:, [0, 2]], axis=0)
        area = float(max(span[0] * span[1], 0.0))
        target_area = max(float(max(dim_x - 1, 1) * max(dim_z - 1, 1)) * spacing * spacing, 1e-6)
        area_score = self._clip01(area / target_area)
        init_h = max(float(getattr(self.env, "_stage_initial_mean_height", np.mean(pos[:, 1]))), 1e-6)
        mean_h = float(np.mean(pos[:, 1]))
        max_h = float(np.max(pos[:, 1]))
        height_span = float(np.ptp(pos[:, 1]))
        drop = self._clip01((init_h - mean_h) / max(init_h - 0.03, 1e-6))
        low = self._clip01(1.0 - mean_h / 0.18)
        no_hanging = self._clip01(1.0 - max_h / 0.35)
        flatness = self._clip01(1.0 - height_span / 0.18)
        if self._stage == 0:
            return self._clip01(0.80 * drop + 0.20 * center_score)
        return self._clip01(0.30 * low + 0.20 * no_hanging + 0.20 * flatness + 0.20 * area_score + 0.10 * center_score)

    def _stage_progress(self, official_performance: float) -> float:
        if self._uses_lrm_progress():
            return self._lrm_progress
        if self._uses_clean_framevlm_progress():
            return self.clean_recipe.stage_potential(self.env, self._stage)
        if self.target_recipe is not None:
            return self.target_recipe.stage_potential(self.env, self._stage)
        if self._uses_clothfold_coverage_progress():
            return self._cloth_fold_coverage_progress()
        if self._uses_clothfold_semantic_progress():
            return self._cloth_fold_overlap_progress()
        if self._uses_passwater_spillaware_progress():
            return self._pass_water_spillaware_progress()
        if self._uses_clothdrop_target_progress():
            return self._clothdrop_target_progress()
        if self.gate_name == "rule":
            if self.rule_gate is None:
                raise RuntimeError("rule gate was not initialized")
            if self._stage >= len(self.spec.bounds):
                return 1.0
            return float(self.rule_gate.score(self.env, self._stage))
        return float(official_performance)

    def _progress_source(self) -> str:
        if self._uses_lrm_progress():
            return "lrm_progress_absolute_hold"
        if self._uses_clean_framevlm_progress():
            return self.clean_recipe.source
        if self.target_recipe is not None:
            return self.target_recipe.source
        if self._uses_clothfold_coverage_progress():
            return "cloth_fold_projected_half_coverage_no_translation_penalty_v1"
        if self._uses_clothfold_semantic_progress():
            return "cloth_fold_overlap_no_anchor_penalty_v1"
        if self._uses_passwater_spillaware_progress():
            return "pass_water_distance_strict_retention_v1"
        if self._uses_clothdrop_target_progress():
            return "cloth_drop_flat_center_height_geometry_v1"
        if self.gate_name == "rule":
            return "frame_vlm_rule_geometry"
        return "official_normalized_performance"

    def _reward_settings(self) -> dict[str, float]:
        if self._uses_clean_framevlm_progress():
            if self.spec.name in {"PassWater", "PourWater"}:
                return {"scale": 4.0, "low": -0.30, "high": 0.30, "transition_bonus": 1.5, "semantic_keep": 0.0}
            if self.spec.name in {"ClothFold", "RopeFlatten"}:
                return {"scale": 3.0, "low": -0.30, "high": 0.30, "transition_bonus": 1.25, "semantic_keep": 0.0}
            return {"scale": 2.5, "low": -0.30, "high": 0.30, "transition_bonus": 1.0, "semantic_keep": 0.0}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v2":
            return {"scale": 3.0, "low": -0.10, "high": 0.45, "transition_bonus": 1.5, "semantic_keep": 0.05}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v3":
            return {"scale": 4.0, "low": 0.0, "high": 0.45, "transition_bonus": 2.0, "semantic_keep": 0.10}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v4":
            return {"scale": 4.5, "low": -0.05, "high": 0.50, "transition_bonus": 2.5, "semantic_keep": 0.20}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v5":
            return {"scale": 5.0, "low": 0.0, "high": 0.60, "transition_bonus": 3.0, "semantic_keep": 0.30}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v6":
            return {"scale": 4.0, "low": -0.15, "high": 0.50, "transition_bonus": 2.0, "semantic_keep": 0.15}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v7":
            return {"scale": 5.0, "low": -0.20, "high": 0.60, "transition_bonus": 2.5, "semantic_keep": 0.20}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v8":
            return {"scale": 5.0, "low": -0.10, "high": 0.60, "transition_bonus": 3.0, "semantic_keep": 0.25}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v9":
            return {"scale": 6.0, "low": 0.0, "high": 0.70, "transition_bonus": 3.0, "semantic_keep": 0.30}
        if self.spec.name == "PassWater" and self.reward_version == "passwater_spillaware_v10":
            return {"scale": 6.0, "low": -0.05, "high": 0.70, "transition_bonus": 3.5, "semantic_keep": 0.35}
        if self.spec.name == "ClothDrop" and self.reward_version == "clothdrop_target_v2":
            return {"scale": 3.0, "low": -0.10, "high": 0.45, "transition_bonus": 1.5, "semantic_keep": 0.05}
        if self.spec.name == "ClothDrop" and self.reward_version == "clothdrop_target_v3":
            return {"scale": 4.0, "low": 0.0, "high": 0.45, "transition_bonus": 2.0, "semantic_keep": 0.10}
        if self.spec.name == "ClothDrop" and self.reward_version == "clothdrop_target_v4":
            return {"scale": 4.5, "low": -0.05, "high": 0.60, "transition_bonus": 2.5, "semantic_keep": 0.10}
        if self.spec.name == "ClothDrop" and self.reward_version == "clothdrop_target_v5":
            return {"scale": 4.0, "low": -0.05, "high": 0.55, "transition_bonus": 2.0, "semantic_keep": 0.08}
        if self.spec.name == "ClothDrop" and self.reward_version == "clothdrop_target_v6":
            return {"scale": 4.5, "low": -0.05, "high": 0.60, "transition_bonus": 2.5, "semantic_keep": 0.10}
        if self.spec.name != "ClothFold":
            return {"scale": 2.0, "low": -0.30, "high": 0.30, "transition_bonus": 1.0, "semantic_keep": 0.0}
        if self.reward_version == "clothfold_coverage_v1":
            return {"scale": 3.0, "low": -0.30, "high": 0.40, "transition_bonus": 1.5, "semantic_keep": 0.0}
        if self.reward_version == "clothfold_coverage_v2":
            return {"scale": 4.0, "low": -0.10, "high": 0.45, "transition_bonus": 2.0, "semantic_keep": 0.0}
        if self.reward_version == "clothfold_coverage_v3":
            return {"scale": 4.5, "low": -0.05, "high": 0.50, "transition_bonus": 2.25, "semantic_keep": 0.0}
        if self.reward_version == "clothfold_semantic_v3":
            return {"scale": 3.0, "low": -0.30, "high": 0.40, "transition_bonus": 1.5, "semantic_keep": 0.0}
        if self.reward_version == "clothfold_semantic_v4":
            return {"scale": 3.0, "low": 0.0, "high": 0.40, "transition_bonus": 1.5, "semantic_keep": 0.0}
        if self.reward_version == "clothfold_semantic_v5":
            return {"scale": 4.0, "low": 0.0, "high": 0.45, "transition_bonus": 2.0, "semantic_keep": 0.20}
        return {"scale": 2.0, "low": -0.30, "high": 0.30, "transition_bonus": 1.0, "semantic_keep": 0.0}

    def _flat_dense_reward(self, official_performance: float) -> tuple[float, float, float]:
        """Sum all stage potentials while retaining the active FSM and bonus."""
        old_stage = self._stage
        old_prev = self._prev_phi
        total = 0.0
        active_progress = 0.0
        active_phi = 0.0
        for stage in range(len(self.spec.bounds)):
            self._stage = stage
            progress = self._stage_progress(official_performance)
            phi = self._phi(progress)
            self._prev_phi = self._prev_phis[stage]
            total += self._dense_reward(phi, progress)
            self._prev_phis[stage] = phi
            if stage == old_stage:
                active_progress, active_phi = progress, phi
        self._stage = old_stage
        self._prev_phi = old_prev
        return float(total), float(active_progress), float(active_phi)

    def _dense_reward(self, phi: float, progress: float) -> float:
        settings = self._reward_settings()
        delta = 0.995 * phi - self._prev_phi
        reward = settings["scale"] * float(np.clip(delta, settings["low"], settings["high"]))
        if settings["semantic_keep"] > 0.0 and (
            self._uses_clothfold_semantic_progress()
            or self._uses_clothfold_coverage_progress()
            or self._uses_passwater_spillaware_progress()
            or self._uses_clothdrop_target_progress()
        ):
            reward += settings["semantic_keep"] * float(progress)
        return reward

    def _phi(self, progress: float) -> float:
        if self._stage >= len(self.spec.bounds):
            return 1.0
        low, high = self.spec.bounds[self._stage]
        return float(np.clip((progress - low) / max(high - low, 1e-6), 0.0, 1.0))

    def _candidate_due(self, progress: float) -> bool:
        if self._stage >= len(self.spec.thresholds):
            return False
        if progress >= self.spec.thresholds[self._stage]:
            self._hold += 1
        else:
            self._hold = 0
        return self._hold >= 2 and self._step - self._last_gate_step >= self.query_interval_steps

    def _advance_if_authorized(self, *, progress: float, info: dict[str, Any]) -> bool:
        rule_candidate = bool(self._candidate_due(progress))
        periodic_due = (
            self.query_mode == "fixed_frequency"
            and self._stage < len(self.spec.thresholds)
            and self._step % self.query_interval_steps == 0
        )
        if not (periodic_due if self.query_mode == "fixed_frequency" else rule_candidate):
            return False
        if self._candidate_start_step is None:
            self._candidate_start_step = self._step
        active_stage = int(self._stage)
        # The training-matched audit label is the persistent RuleGate
        # candidate that actually triggers a Qwen request.
        ground_truth_complete = bool(rule_candidate)
        accepted = False
        prediction = "pending"
        raw_output = ""
        self._last_gate_step = self._step
        if self.gate_name == "rule":
            if (
                self._uses_lrm_progress()
                or self._uses_clean_framevlm_progress()
                or self._uses_clothfold_coverage_progress()
                or self._uses_clothfold_semantic_progress()
                or self._uses_passwater_spillaware_progress()
                or self._uses_clothdrop_target_progress()
            ):
                accepted = progress >= self.spec.thresholds[self._stage]
                prediction = f"{self._progress_source()}_{'success' if accepted else 'reject'}"
            else:
                if self.rule_gate is None:
                    raise RuntimeError("rule gate was not initialized")
                accepted = self.rule_gate.accepted(self.env, self._stage, self.spec.thresholds[self._stage])
                prediction = "frame_vlm_rule_success" if accepted else "frame_vlm_rule_reject"
        elif self.gate_name == "qwen":
            decision = self.gate.decide(
                env_id=0,
                episode_id=self.episode_id,
                stage_id=self._stage,
                candidate_epoch=self._candidate_epoch,
                step=self._step,
                candidate_start_step=int(self._candidate_start_step),
                reference=self._reference,
                ring_frames=tuple(self._frames),
                current=self._frames[-1],
            )
            accepted, prediction, raw_output = decision.accepted, decision.prediction, decision.raw_output
            self._candidate_epoch += 1
        self.gate_records.append(
            {
                "task": self.spec.name,
                "episode_id": int(self.episode_id),
                "config_id": int(self._config_id),
                "request_step": int(self._step),
                "stage_id": active_stage,
                "rule_candidate": rule_candidate,
                "periodic_due": periodic_due,
                "ground_truth_complete": ground_truth_complete,
                "prediction": str(prediction),
                "accepted": bool(accepted),
            }
        )
        info["gate_query"] = True
        info["gate_rule_candidate"] = rule_candidate
        info["gate_ground_truth_complete"] = ground_truth_complete
        if accepted:
            old_stage = self._stage
            self._stage += 1
            self._hold = 0
            self._candidate_start_step = None
            self._last_prediction = prediction
            info["stage_transition"] = [old_stage, self._stage]
            info["gate_prediction"] = prediction
            info["gate_raw_output"] = raw_output
            return True
        info["gate_prediction"] = prediction
        info["gate_raw_output"] = raw_output
        return False

    def gate_statistics(self) -> dict[str, Any]:
        rows = list(self.gate_records)
        positives = [r for r in rows if r.get("ground_truth_complete") is True]
        negatives = [r for r in rows if r.get("ground_truth_complete") is False]
        accepted = [r for r in rows if r.get("prediction") == "success"]
        true_accept = sum(1 for r in positives if r.get("prediction") == "success")
        false_reject = sum(1 for r in positives if r.get("prediction") != "success")
        false_accept = sum(1 for r in negatives if r.get("prediction") == "success")
        true_reject = sum(1 for r in negatives if r.get("prediction") != "success")
        return {
            "requests": len(rows),
            "accepted": len(accepted),
            "unknown": sum(1 for r in rows if r.get("prediction") == "unknown"),
            "labeled_requests": len(positives) + len(negatives),
            "gt_positive": len(positives),
            "gt_negative": len(negatives),
            "true_accept": true_accept,
            "true_reject": true_reject,
            "false_reject": false_reject,
            "false_accept": false_accept,
            "frr": (false_reject / len(positives)) if positives else None,
            "far": (false_accept / len(negatives)) if negatives else None,
        }

    def reset(self, config_id: int | None = None) -> np.ndarray:
        if config_id is None:
            config_id = int(self.rng.integers(0, 800))
        self._step = 0
        self._stage = 0
        self._lrm_progress = 0.0
        self._lrm_frame_buffer = []
        self._prev_phi = 0.0
        self._hold = 0
        self._candidate_start_step = None
        self._candidate_epoch = 0
        self._last_gate_step = -10_000
        self._last_prediction = None
        self._config_id = -1
        base_obs = self.env.reset(config_id=int(config_id))
        self._config_id = int(config_id)
        if self.clean_recipe is not None:
            self.clean_recipe.reset(self.env)
        if self.target_recipe is not None:
            self.target_recipe.reset(self.env)
        if self.rule_gate is not None:
            self.rule_gate.reset(self.env)
        try:
            import pyflex

            pos = np.asarray(pyflex.get_positions(), dtype=np.float32).reshape(-1, 4)[:, :3]
            self.env._stage_initial_mean_height = float(np.mean(pos[:, 1]))
        except Exception:
            pass
        self.episode_id += 1
        self._frames.clear()
        self._reference = self._capture()
        self._lrm_initial_image = self._frame_array(self._reference)
        # Formulation.md requires previous potential to be initialized on stage
        # entry; the first reward step must not receive free progress from zero.
        self._prev_phi = self._phi(self._stage_progress(0.0))
        self._prev_phis = []
        for stage in range(len(self.spec.bounds)):
            self._stage = stage
            self._prev_phis.append(self._phi(self._stage_progress(0.0)))
        self._stage = 0
        self._prev_phi = self._prev_phis[0] if self._prev_phis else 0.0
        return self._actor_obs(base_obs)

    def step(self, action: np.ndarray):
        base_obs, native_reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        info = dict(info)
        self._step += 1
        frame = self._capture()
        if self._uses_lrm_progress() and self._step % self.lrm_client.query_interval == 0:
            current_image = self._frame_array(frame).copy()
            if self.lrm_client.mode in {"robometer", "robometer_video", "roboreward", "roboreward_video"}:
                self._lrm_frame_buffer.append(current_image)
                if len(self._lrm_frame_buffer) > self.lrm_client.max_frames:
                    keep = np.linspace(
                        0, len(self._lrm_frame_buffer) - 1, self.lrm_client.max_frames, dtype=int
                    )
                    self._lrm_frame_buffer = [
                        self._lrm_frame_buffer[int(index)] for index in keep
                    ]
                lrm_frames = self._lrm_frame_buffer
            else:
                lrm_frames = [self._frame_array(item) for item in self._frames]
            self._lrm_progress = self.lrm_client.compute_progress(
                current_image,
                task_description=self._task_description(),
                initial_image=self._lrm_initial_image,
                frames=lrm_frames,
            )
        official_performance = float(info.get("normalized_performance", 0.0))
        if self.reward_form_mode == "all_dense_with_stage_actor":
            dense, progress, phi = self._flat_dense_reward(official_performance)
        else:
            progress = self._stage_progress(official_performance)
            phi = self._phi(progress)
            dense = self._dense_reward(phi, progress)
        transitioned = self._advance_if_authorized(progress=progress, info=info)
        reward = (self._lrm_progress if self._uses_lrm_progress() else dense + (self._reward_settings()["transition_bonus"] if transitioned else 0.0))
        if self.code_expression is not None:
            reward = float(eval(self.code_expression, {"__builtins__": {}, "np": np}, {
                "reward": reward, "native_reward": native_reward, "dense": dense,
                "progress": progress, "phi": phi, "stage": self._stage,
                "transitioned": transitioned,
            }))
        if self.benchmark_native_dense:
            reward = float(native_reward)
        if self.per_subtask_sparse:
            reward = float(transitioned)
        terminal_sparse_score = None
        if self.terminal_sparse_threshold is not None:
            if done:
                if self.terminal_sparse_metric == "clothfold_semantic":
                    from .evaluation import _semantic_final_metrics
                    final_metrics = _semantic_final_metrics(env=self.env, task=self.spec.name, final_info=info)
                    if "final_semantic_performance" not in final_metrics:
                        raise RuntimeError("ClothFold terminal semantic metric unavailable")
                    terminal_sparse_score = float(final_metrics["final_semantic_performance"])
                else:
                    if "normalized_performance" not in info:
                        raise RuntimeError("SoftGym terminal normalized performance unavailable")
                    terminal_sparse_score = official_performance
                if not np.isfinite(terminal_sparse_score):
                    raise RuntimeError("non-finite SoftGym terminal sparse score")
            reward = float(done and terminal_sparse_score is not None and terminal_sparse_score >= self.terminal_sparse_threshold)
        if transitioned:
            # Re-initialize the newly active potential only for staged-reset.
            # The all-dense control tracks every stage continuously.
            if self.reward_form_mode == "staged_reset":
                self._prev_phi = self._phi(self._stage_progress(official_performance))
            elif self._stage < len(self._prev_phis):
                self._prev_phi = self._prev_phis[self._stage]
        elif self.reward_form_mode == "staged_reset":
            self._prev_phi = phi
        info.update(
            {
                "stage_id": self._stage,
                "active_stage_reward": reward,
                "native_reward_excluded": not self.benchmark_native_dense,
                "benchmark_native_dense": self.benchmark_native_dense,
                "terminal_sparse_metric": self.terminal_sparse_metric,
                "terminal_sparse_threshold": self.terminal_sparse_threshold,
                "terminal_sparse_score": terminal_sparse_score,
                "official_normalized_performance": official_performance,
                "stage_progress": progress,
                "stage_progress_source": self._progress_source(),
                "reward_version": self.reward_version,
                "reward_form_mode": self.reward_form_mode,
                "actor_receives_stage_id": self.append_stage_obs,
                "critic_receives_stage_id": self.append_stage_obs,
                "rule_gate_version": (
                    "clean_framevlm_v1_progress_threshold"
                    if self.gate_name == "rule" and self._uses_clean_framevlm_progress()
                    else (
                        "versioned_geometry_progress_threshold"
                        if self.gate_name == "rule"
                        and (
                            self._uses_clothfold_coverage_progress()
                            or self._uses_clothfold_semantic_progress()
                            or self._uses_passwater_spillaware_progress()
                            or self._uses_clothdrop_target_progress()
                        )
                        else ("frame_vlm_rule_v1" if self.gate_name == "rule" else None)
                    )
                ),
                "config_id": int(getattr(self.env, "current_config_id", self._config_id)),
            }
        )
        return self._actor_obs(base_obs), float(reward), bool(done), info

    def close(self) -> None:
        if self.lrm_client is not None:
            self.lrm_client.close()
            self.lrm_client = None
        if self.gate is not None:
            self.gate.close()
        self.env.close()

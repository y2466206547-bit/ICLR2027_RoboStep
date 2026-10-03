"""Task and stage specifications for the SoftGym deformable pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_module: str
    env_class: str
    horizon: int
    action_repeat: int
    env_kwargs: dict[str, Any]
    stages: tuple[str, ...]
    instructions: tuple[str, ...]
    thresholds: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]


# All tasks use the same bounded active-stage construction. Rule runs use
# Frame-VLM-style geometric stage predicates rather than SoftGym's official
# normalized_performance for reward or transition decisions. The actor receives
# state plus a one-hot stage code, never task-language embeddings.
TASKS: dict[str, TaskSpec] = {
    "ClothDrop": TaskSpec(
        name="ClothDrop",
        env_module="softgym.envs.cloth_drop",
        env_class="ClothDropEnv",
        horizon=30,
        action_repeat=16,
        env_kwargs={
            "observation_mode": "key_point",
            "action_mode": "picker",
            "num_picker": 2,
            "render_mode": "cloth",
        },
        stages=("approach_release", "flat_completion"),
        instructions=(
            "Audit whether the cloth has visibly transitioned from the held vertical setup toward the floor without an obvious wrong object or empty scene.",
            "Audit whether the cloth is visibly laid broadly flat on the floor, with no clear large fold, hanging portion, or later contradiction.",
        ),
        thresholds=(0.35, 0.75),
        bounds=((0.0, 0.35), (0.35, 1.0)),
    ),
    "ClothFlatten": TaskSpec(
        name="ClothFlatten",
        env_module="softgym.envs.cloth_flatten",
        env_class="ClothFlattenEnv",
        horizon=100,
        action_repeat=8,
        env_kwargs={
            "observation_mode": "key_point",
            "action_mode": "picker",
            "num_picker": 2,
            "render_mode": "cloth",
        },
        stages=("spread_progress", "flat_completion"),
        instructions=(
            "Audit whether the crumpled cloth has visibly begun a coherent spreading motion across the floor rather than merely moving a picker.",
            "Audit whether the cloth is visibly substantially flattened and spread across the floor, without a clear large unspread region or later contradiction.",
        ),
        thresholds=(0.35, 0.75),
        bounds=((0.0, 0.35), (0.35, 1.0)),
    ),
    "ClothFold": TaskSpec(
        name="ClothFold",
        env_module="softgym.envs.cloth_fold",
        env_class="ClothFoldEnv",
        horizon=100,
        action_repeat=8,
        env_kwargs={
            "observation_mode": "key_point",
            "action_mode": "picker",
            "num_picker": 2,
            "render_mode": "cloth",
        },
        stages=("fold_progress", "fold_completion"),
        instructions=(
            "Audit whether the cloth has visibly begun a coherent folding motion, with one side moving toward its counterpart rather than only translating the whole cloth.",
            "Audit whether the cloth is visibly folded so the intended halves substantially overlap, without a clear later contradiction or empty scene.",
        ),
        thresholds=(0.35, 0.75),
        bounds=((0.0, 0.35), (0.35, 1.0)),
    ),
    "RopeFlatten": TaskSpec(
        name="RopeFlatten",
        env_module="softgym.envs.rope_flatten",
        env_class="RopeFlattenEnv",
        horizon=75,
        action_repeat=8,
        env_kwargs={
            "observation_mode": "key_point",
            "action_mode": "picker",
            "num_picker": 2,
            "render_mode": "cloth",
        },
        stages=("stretch_progress", "straight_completion"),
        instructions=(
            "Audit whether the rope has visibly begun to straighten or stretch from its initial tangled or curved shape rather than only moving a picker.",
            "Audit whether the rope is visibly mostly straight and extended, without a clear large bend, knot, or later contradiction.",
        ),
        thresholds=(0.35, 0.75),
        bounds=((0.0, 0.35), (0.35, 1.0)),
    ),
    "PassWater": TaskSpec(
        name="PassWater",
        env_module="softgym.envs.pass_water",
        env_class="PassWater1DEnv",
        horizon=75,
        action_repeat=8,
        env_kwargs={
            "observation_mode": "key_point",
            "action_mode": "direct",
            "render_mode": "fluid",
        },
        stages=("transport_progress", "water_delivery"),
        instructions=(
            "Audit transport_progress: success only if the cup has moved clearly beyond early motion toward the target side while still visibly carrying most water. Reject if motion is only slight/early, water retention is doubtful, the cup is empty or nearly empty, water is left behind, or there is a major spill.",
            "Audit water_delivery: success only if the cup/water is visibly on or near the target side/end and water has not been lost to a major spill. Reject if the cup is not near the target side/end, appears empty or nearly empty, leaves water behind, or the evidence is ambiguous.",
        ),
        thresholds=(0.35, 0.75),
        bounds=((0.0, 0.35), (0.35, 1.0)),
    ),
    "PourWater": TaskSpec(
        name="PourWater",
        env_module="softgym.envs.pour_water",
        env_class="PourWaterPosControlEnv",
        horizon=100,
        action_repeat=8,
        env_kwargs={
            "observation_mode": "key_point",
            "action_mode": "rotation_bottom",
            "render_mode": "fluid",
            "camera_name": "default_camera",
        },
        stages=("pour_progress", "target_fill_completion"),
        instructions=(
            "Audit whether the source cup has visibly begun a controlled pour toward the receiving cup without an obvious wrong scene or major spill.",
            "Audit whether the receiving cup is visibly filled by the poured water, with no obvious major spill or later contradiction.",
        ),
        thresholds=(0.35, 0.75),
        bounds=((0.0, 0.35), (0.35, 1.0)),
    ),
}


def build_env_kwargs(spec: TaskSpec, cache_path: Path, *, num_variations: int) -> dict[str, Any]:
    return {
        **spec.env_kwargs,
        "render": True,
        "headless": True,
        "horizon": spec.horizon,
        "action_repeat": spec.action_repeat,
        "num_variations": int(num_variations),
        "use_cached_states": True,
        "save_cached_states": False,
        "deterministic": False,
        "cached_states_path": str(cache_path),
    }

"""Frozen task and initial-state split for fair SoftGym comparisons."""

from __future__ import annotations

import os
from pathlib import Path


CANONICAL_TASKS = (
    "PassWater",
    "PourWater",
    "RopeFlatten",
    "ClothFlatten",
    "ClothFold",
    "ClothDrop",
)

DEFAULT_QWEN_MODEL_PATH = Path(
    os.environ.get("ROBOSTEP_QWEN_MODEL", "models/Qwen3-VL-8B-Instruct")
)


CACHE_FILENAMES = {
    "PassWater": "pass_water_init_states.pkl",
    "PourWater": "pour_water_init_states.pkl",
    "RopeFlatten": "rope_flatten_init_states.pkl",
    "ClothFlatten": "cloth_flatten_init_states.pkl",
    "ClothFold": "cloth_fold_init_states.pkl",
    "ClothDrop": "cloth_drop_init_states.pkl",
}

TOTAL_VARIATIONS = 1000
TRAINING_SEEDS = (42, 43, 44)
SPLIT_INDICES = {
    "train": tuple(range(0, 800)),
    "validation": tuple(range(800, 900)),
    "test": tuple(range(900, 1000)),
}


def protocol_manifest(cache_root: Path) -> dict:
    """Return a serializable manifest without opening any cache file."""

    cache_root = cache_root.expanduser().resolve()
    return {
        "tasks": list(CANONICAL_TASKS),
        "training_seeds": list(TRAINING_SEEDS),
        "total_variations_per_task": TOTAL_VARIATIONS,
        "splits": {
            name: {
                "count": len(indices),
                "first_config_id": indices[0],
                "last_config_id": indices[-1],
            }
            for name, indices in SPLIT_INDICES.items()
        },
        "cache_paths": {
            task: str(cache_root / CACHE_FILENAMES[task])
            for task in CANONICAL_TASKS
        },
        "checkpoint_selection": (
            "highest mean official normalized performance on validation config ids"
        ),
        "final_evaluation": (
            "one deterministic episode per untouched test config id for the fixed "
            "selected checkpoint"
        ),
        "primary_metric": "official normalized performance",
    }


def _validate_protocol() -> None:
    if set(CACHE_FILENAMES) != set(CANONICAL_TASKS):
        raise RuntimeError("cache filenames do not cover the canonical tasks")
    split_sets = {name: set(indices) for name, indices in SPLIT_INDICES.items()}
    if any(len(indices) != len(split_sets[name]) for name, indices in SPLIT_INDICES.items()):
        raise RuntimeError("a split contains duplicate config ids")
    if split_sets["train"] & split_sets["validation"]:
        raise RuntimeError("train and validation config ids overlap")
    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError("train and test config ids overlap")
    if split_sets["validation"] & split_sets["test"]:
        raise RuntimeError("validation and test config ids overlap")
    union = set().union(*split_sets.values())
    if union != set(range(TOTAL_VARIATIONS)):
        raise RuntimeError("frozen splits do not cover exactly 1000 config ids")


_validate_protocol()

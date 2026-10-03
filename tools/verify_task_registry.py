#!/usr/bin/env python3
"""Validate the package-local 75-task reward-machine registry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from task_registry import load_program, load_reward_machine


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "task_registry"


def main() -> None:
    data = json.loads((REGISTRY / "manifest.json").read_text(encoding="utf-8"))
    tasks = data["tasks"]
    assert data["total"] == 75, data["total"]
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["benchmark"]] = counts.get(task["benchmark"], 0) + 1
        assert task["K"] >= 1
        assert task["transitions"] == task["K"] - 1
        assert len(task["stages"]) == task["K"], task["task_id"]
        machine_path = REGISTRY / task["reward_machine"]
        assert machine_path.is_file(), machine_path
        program = load_program(task["task_id"], task["benchmark"])
        assert program.num_stages == task["K"]
        errors = program.validate()
        assert not errors, (task["task_id"], errors)
        assert all(stage.potential_terms for stage in program.stages), task["task_id"]
        machine = load_reward_machine(task["task_id"], task["benchmark"])
        feature_names = {term.name for stage in program.stages for term in stage.potential_terms}
        zeros = {name: 0.0 for name in feature_names}
        machine.reset(zeros)
        smoke = machine.step(zeros, np.zeros(task["K"], dtype=bool))
        assert smoke.reward.shape == (1,), task["task_id"]
    assert counts == {"Meta-World": 50, "ManiSkill": 19, "SoftGym": 6}, counts
    print(f"registry OK: {len(tasks)} task reward machines ({counts})")


if __name__ == "__main__":
    main()

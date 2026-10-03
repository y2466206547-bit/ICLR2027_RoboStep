"""Dependency-free loader for the self-contained 75-task reward registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def _records() -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"{MANIFEST} is missing; run tools/build_task_registry.py first"
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]


def list_tasks(benchmark: str | None = None) -> list[dict[str, Any]]:
    """Return registry records, optionally restricted to one benchmark."""
    records = _records()
    if benchmark is None:
        return records
    aliases = {"metaworld": "Meta-World", "maniskill": "ManiSkill", "softgym": "SoftGym"}
    wanted = aliases.get(benchmark.lower(), benchmark)
    return [record for record in records if record["benchmark"] == wanted]


def find_task(task_id: str, benchmark: str | None = None) -> dict[str, Any]:
    matches = [record for record in list_tasks(benchmark) if record["task_id"] == task_id]
    if not matches:
        raise KeyError(f"task not found: {benchmark}:{task_id}")
    if len(matches) > 1:
        raise KeyError(f"task id is ambiguous across benchmarks: {task_id}")
    return matches[0]


def load_task(task_id: str, benchmark: str | None = None) -> dict[str, Any]:
    """Load one complete task reward-machine record."""
    record = find_task(task_id, benchmark)
    recipe = ROOT / record["reward_machine"]
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    payload["_reward_machine_file"] = recipe.as_posix()
    return payload


def load_program(task_id: str, benchmark: str | None = None):
    """Load the validated :class:`robostep.schema.RewardProgram` for a task."""

    from robostep.schema import RewardProgram

    payload = load_task(task_id, benchmark)
    return RewardProgram.from_dict(payload["reward_program"])

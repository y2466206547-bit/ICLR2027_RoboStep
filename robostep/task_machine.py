"""Lazy access to the package-local task reward machines."""

from __future__ import annotations

from typing import Any


def load_reward_machine(task_id: str, benchmark: str | None = None, **kwargs: Any):
    """Load a frozen 75-task reward machine without importing the registry eagerly."""

    from task_registry.reward_machine import load_reward_machine as _load

    return _load(task_id, benchmark, **kwargs)


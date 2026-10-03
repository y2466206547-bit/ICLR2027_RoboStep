"""Task definitions for ManiSkill benchmark suites."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_PATH = Path(__file__).with_name("tasks_17.json")


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    horizon: int
    family: str
    official_demo: bool
    native_dense: bool
    groups: tuple[str, ...]
    stages: tuple[str, ...]


def resolve_manifest_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    configured = os.environ.get("OURS_TASK_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    return MANIFEST_PATH


def load_task_specs(path: Path | None = None) -> tuple[TaskSpec, ...]:
    manifest_path = resolve_manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload["tasks"] if isinstance(payload, dict) else payload
    specs = tuple(
        TaskSpec(
            env_id=row["env_id"],
            horizon=int(row["horizon"]),
            family=row["family"],
            official_demo=bool(row.get("official_demo", False)),
            native_dense=bool(row.get("native_dense", False)),
            groups=tuple(row.get("groups", ("all19",))),
            stages=tuple(row.get("stages", ())),
        )
        for row in rows
    )
    validate_task_specs(specs)
    return specs


def validate_task_specs(specs: Iterable[TaskSpec]) -> None:
    specs = tuple(specs)
    ids = [spec.env_id for spec in specs]
    if len(specs) != 19:
        raise ValueError(f"Expected 19 tasks, found {len(specs)}")
    if len(set(ids)) != len(ids):
        raise ValueError("Task manifest contains duplicate environment IDs")
    groups = {group for spec in specs for group in spec.groups}
    legacy_counts = {
        "all19": 19,
        "demo12": 12,
        "extended6": 6,
        "fmb1": 1,
        "anchor3": 3,
        "hard12": 12,
    }
    if set(legacy_counts).issubset(groups):
        for group, expected in legacy_counts.items():
            actual = sum(group in spec.groups for spec in specs)
            if actual != expected:
                raise ValueError(f"Expected {expected} tasks in {group}, found {actual}")
    elif sum("all19" in spec.groups for spec in specs) != 19:
        raise ValueError("Every replacement-suite task must belong to all19")
    for spec in specs:
        if not spec.stages:
            raise ValueError(f"{spec.env_id} has no stage blueprint")
        if spec.horizon <= 0:
            raise ValueError(f"{spec.env_id} has invalid horizon {spec.horizon}")


def select_tasks(group: str = "all19", path: Path | None = None) -> tuple[TaskSpec, ...]:
    return tuple(spec for spec in load_task_specs(path) if group in spec.groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="all19")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--ids-only", action="store_true")
    args = parser.parse_args()
    specs = select_tasks(args.group, args.manifest)
    if args.ids_only:
        print(" ".join(spec.env_id for spec in specs))
        return
    for spec in specs:
        flags = []
        if spec.official_demo:
            flags.append("demo")
        if spec.native_dense:
            flags.append("dense")
        print(
            f"{spec.env_id:40s} horizon={spec.horizon:3d} "
            f"family={spec.family:24s} flags={','.join(flags) or '-'} "
            f"stages={' -> '.join(spec.stages)}"
        )


if __name__ == "__main__":
    main()

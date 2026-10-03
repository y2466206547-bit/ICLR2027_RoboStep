#!/usr/bin/env python3
"""Rebuild the package-local 75-task manifest from reward-machine JSON files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "task_registry"
MACHINES = REGISTRY / "reward_machines"


def main() -> None:
    records = []
    for benchmark_dir, benchmark in (("metaworld", "Meta-World"), ("maniskill", "ManiSkill"), ("softgym", "SoftGym")):
        for path in sorted((MACHINES / benchmark_dir).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            program = payload.get("reward_program", {})
            record = {
                key: payload[key]
                for key in ("benchmark", "task_id", "task_type", "long_horizon", "K", "transitions", "horizon", "stages", "dense_terms_by_stage", "simulator_state_api", "reward_config", "gate_config")
            }
            record["reward_machine"] = path.relative_to(REGISTRY).as_posix()
            record["program_source"] = program.get("source", "")
            records.append(record)
    expected = {"Meta-World": 50, "ManiSkill": 19, "SoftGym": 6}
    counts = {name: sum(item["benchmark"] == name for item in records) for name in expected}
    if len(records) != 75 or counts != expected:
        raise SystemExit(f"unexpected task inventory: {counts}")
    (REGISTRY / "manifest.json").write_text(
        json.dumps({"schema": "robostep-task-registry-v2", "suite": expected, "total": len(records), "tasks": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = ["benchmark", "task_id", "task_type", "long_horizon", "K", "transitions", "horizon", "reward_machine", "program_source"]
    with (REGISTRY / "task_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fields} for item in records)
    print(json.dumps({"total": len(records), "suite": counts}, indent=2))


if __name__ == "__main__":
    main()

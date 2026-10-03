"""Parsers shared by visual rollback workers and lightweight tests."""

from __future__ import annotations

import re


def parse_reasoned_final_stage(text: str) -> str:
    value = text.strip().lower()
    match = re.search(
        r"final_stage\s*[:=]\s*[\"']?(stage[_\s-]?\d+|unchanged|unknown)",
        value,
    )
    if match is None:
        return "unknown"
    label = match.group(1).replace("-", "_").replace(" ", "_")
    if label.startswith("stage_"):
        return f"stage_{int(label.split('_', 1)[1])}"
    return label


def parse_consistent_predicate_stage(text: str) -> str:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip().upper()] = value.strip().lower()
    changed = fields.get("CHANGED", "unclear")
    failed = fields.get("FAILED_PREDICATES", "unknown")
    final = parse_reasoned_final_stage(text)
    if changed == "no":
        return "unchanged" if failed == "none" and final == "unchanged" else "unknown"
    if changed != "yes":
        return "unknown"
    indices: list[int] = []
    for token in failed.split(","):
        token = token.strip()
        if not token.startswith("stage_") or not token[6:].isdigit():
            return "unknown"
        indices.append(int(token[6:]))
    if not indices:
        return "unknown"
    expected = f"stage_{min(indices)}"
    return final if final == expected else "unknown"

"""Prompt selection for stage-transition scheduling ablations.

The production/candidate gate keeps its original conservative audit prompt.
Fixed-frequency experiments may opt into a neutral checker that does not tell
the VLM that a simulator predicate has already fired.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Sequence


ENV_PROMPT_MODE = "STAGE_TRANSITION_PROMPT_MODE"
ENV_QUERY_MODE = "STAGE_TRANSITION_QUERY_MODE"
PROMPT_MODE_CANDIDATE_AUDIT = "candidate_audit"
PROMPT_MODE_PERIODIC_NEUTRAL = "periodic_neutral"
PROMPT_MODE_HYBRID_NEUTRAL = "hybrid_neutral"
DEFAULT_PROMPT_MODE = PROMPT_MODE_CANDIDATE_AUDIT
VALID_PROMPT_MODES = {
    PROMPT_MODE_CANDIDATE_AUDIT,
    PROMPT_MODE_PERIODIC_NEUTRAL,
    PROMPT_MODE_HYBRID_NEUTRAL,
}
HYBRID_NEUTRAL_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "qwen_hybrid_stage_gate.md"
)
PERIODIC_NEUTRAL_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "qwen_periodic_neutral_stage_gate.md"
)


def resolve_transition_prompt_mode() -> str:
    mode = os.environ.get(ENV_PROMPT_MODE, DEFAULT_PROMPT_MODE).strip().lower()
    if not mode:
        mode = DEFAULT_PROMPT_MODE
    if mode not in VALID_PROMPT_MODES:
        raise ValueError(f"unsupported {ENV_PROMPT_MODE}: {mode}")
    query_mode = os.environ.get(ENV_QUERY_MODE, "candidate_retry").strip().lower()
    required_query_mode = {
        PROMPT_MODE_PERIODIC_NEUTRAL: "fixed_frequency",
        PROMPT_MODE_HYBRID_NEUTRAL: "hybrid_k64",
    }.get(mode)
    if required_query_mode is not None and query_mode != required_query_mode:
        raise ValueError(
            f"{mode} requires {ENV_QUERY_MODE}={required_query_mode}, "
            f"got {query_mode!r}"
        )
    return mode


def _task_goal(instruction: str) -> str:
    match = re.match(r"\s*(MetaWorld task .*?\.)\s*", instruction)
    if match:
        return match.group(1)
    return "MetaWorld manipulation task."


def _visual_condition(instruction: str, stage_name: str) -> str:
    patterns = (
        r"Simulator (?:geometry|state) verified(?: that)? (.*?)(?:\.\s|\.$|$)",
        r"State(?: geometry)? has verified(?: that)? (.*?)(?:\.\s|\.$|$)",
    )
    condition = ""
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            condition = match.group(1).strip()
            break
    if "terminal geometry" in condition.lower():
        return "the task outcome stated above is visibly achieved"
    if condition:
        return condition

    label = stage_name.replace("_", " ").strip()
    return f"the visual state matches the current stage named {label}"


def neutralize_stage_instruction(stage_name: str, instruction: str) -> str:
    """Retain the task/stage criterion without candidate-audit presuppositions."""

    goal = _task_goal(str(instruction))
    condition = _visual_condition(str(instruction), str(stage_name))
    label = str(stage_name).replace("_", " ").strip()
    return (
        f"{goal} Current stage: {label}. "
        f"Visual condition to evaluate: {condition}. "
        "Check the available images and decide whether this condition is satisfied now."
    )


def configure_transition_prompt(
    prompt_template: str | Path,
    stage_names: Sequence[str],
    stage_instructions: Sequence[str],
) -> tuple[str, Path, tuple[str, ...]]:
    """Return prompt mode, effective template, and effective stage instructions."""

    mode = resolve_transition_prompt_mode()
    original = tuple(str(instruction) for instruction in stage_instructions)
    if mode == PROMPT_MODE_CANDIDATE_AUDIT:
        return mode, Path(prompt_template).expanduser().resolve(), original
    neutral = tuple(
        neutralize_stage_instruction(name, instruction)
        for name, instruction in zip(stage_names, original, strict=True)
    )
    prompt = (
        HYBRID_NEUTRAL_PROMPT
        if mode == PROMPT_MODE_HYBRID_NEUTRAL
        else PERIODIC_NEUTRAL_PROMPT
    )
    return mode, prompt.resolve(), neutral

"""Shared reward-form ablation helpers for staged MetaWorld rewards."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from typing import Any, Final


STAGED_RESET: Final[str] = "staged_reset"
COMPLETED_CUMULATIVE: Final[str] = "completed_cumulative"
ALL_DENSE_WITH_STAGE_ACTOR: Final[str] = "all_dense_with_stage_actor"
CODE_AS_REWARD: Final[str] = "code_as_reward"
BENCHMARK_NATIVE_DENSE: Final[str] = "benchmark_native_dense"
TERMINAL_ONLY_SPARSE: Final[str] = "terminal_only_sparse"
PER_SUBTASK_SPARSE: Final[str] = "per_subtask_sparse"

REWARD_FORM_MODES: Final[tuple[str, ...]] = (
    STAGED_RESET,
    COMPLETED_CUMULATIVE,
    ALL_DENSE_WITH_STAGE_ACTOR,
    CODE_AS_REWARD,
    BENCHMARK_NATIVE_DENSE,
    TERMINAL_ONLY_SPARSE,
    PER_SUBTASK_SPARSE,
)

_ALIASES: Final[dict[str, str]] = {
    "active_only": STAGED_RESET,
    "main": STAGED_RESET,
    "reset": STAGED_RESET,
    "cumulative": COMPLETED_CUMULATIVE,
    "completed": COMPLETED_CUMULATIVE,
    "completed_cumsum": COMPLETED_CUMULATIVE,
    "dense_no_stage": ALL_DENSE_WITH_STAGE_ACTOR,
    "all_dense": ALL_DENSE_WITH_STAGE_ACTOR,
    "all_dense_actor_stage": ALL_DENSE_WITH_STAGE_ACTOR,
    "dense_no_stage_actor_stage": ALL_DENSE_WITH_STAGE_ACTOR,
    "code": CODE_AS_REWARD,
    "code_reward": CODE_AS_REWARD,
    "native_dense": BENCHMARK_NATIVE_DENSE,
    "benchmark_native": BENCHMARK_NATIVE_DENSE,
    "terminal_sparse": TERMINAL_ONLY_SPARSE,
    "terminal_only": TERMINAL_ONLY_SPARSE,
    "subtask_sparse": PER_SUBTASK_SPARSE,
    "stage_sparse": PER_SUBTASK_SPARSE,
}


def normalize_reward_form_mode(mode: str | None) -> str:
    """Return a canonical reward-form mode name."""

    value = (mode or STAGED_RESET).strip()
    canonical = _ALIASES.get(value, value)
    if canonical not in REWARD_FORM_MODES:
        choices = ", ".join(REWARD_FORM_MODES)
        raise ValueError(f"unknown reward_form mode {mode!r}; expected one of {choices}")
    return canonical


def reward_form_mode(config: Any) -> str:
    return normalize_reward_form_mode(getattr(config, "reward_form_mode", None))


def uses_benchmark_native_dense(config: Any) -> bool:
    """Whether a VecEnv must return the original Meta-World reward."""

    return reward_form_mode(config) == BENCHMARK_NATIVE_DENSE


def is_sparse_reward_mode(mode: str) -> bool:
    """Whether step shaping and action penalties are disabled."""

    return normalize_reward_form_mode(mode) in {
        TERMINAL_ONLY_SPARSE,
        PER_SUBTASK_SPARSE,
    }


def sparse_step_reward(mode: str, shaped_reward: float) -> float:
    """Return zero between accepted transitions for sparse controls."""

    if is_sparse_reward_mode(mode):
        return 0.0
    return float(shaped_reward)


def transition_reward(
    mode: str,
    *,
    original_bonus: float,
    new_stage: int,
    terminal_stage: int,
) -> float:
    """Map an accepted transition to the selected control reward.

    The stage machine and its candidate predicates are unchanged. Per-subtask
    sparse emits one unit at every accepted transition; terminal-only sparse
    emits one unit only upon entry to the terminal stage.
    """

    canonical = normalize_reward_form_mode(mode)
    if canonical == PER_SUBTASK_SPARSE:
        return 1.0
    if canonical == TERMINAL_ONLY_SPARSE:
        return 1.0 if int(new_stage) == int(terminal_stage) else 0.0
    return float(original_bonus)


def reward_stage_indices(
    mode: str,
    *,
    current_stage: int,
    active_stage_count: int,
) -> tuple[int, ...]:
    """Select dense-reward stages while leaving the stage FSM unchanged."""

    if current_stage < 0 or current_stage >= active_stage_count:
        return ()
    canonical = normalize_reward_form_mode(mode)
    if canonical in {STAGED_RESET, BENCHMARK_NATIVE_DENSE} or is_sparse_reward_mode(canonical):
        return (current_stage,)
    if canonical == COMPLETED_CUMULATIVE:
        return tuple(range(current_stage + 1))
    return tuple(range(active_stage_count))


def dense_sum(
    *,
    mode: str,
    current_stage: int,
    active_stage_count: int,
    dense_scales: Sequence[float],
    gamma: float,
    previous_stage_qualities: MutableMapping[int, float],
    quality_fn: Callable[[int], float],
) -> tuple[float, float]:
    """Compute potential-difference reward for one or more stage functions.

    Missing per-stage baselines are initialized to the current potential.  That
    avoids a positive reward spike when a stage first becomes included in a
    non-reset ablation condition.
    """

    total = 0.0
    active_quality = 0.0
    for stage in reward_stage_indices(
        mode,
        current_stage=current_stage,
        active_stage_count=active_stage_count,
    ):
        quality = float(quality_fn(stage))
        previous = float(previous_stage_qualities.get(stage, quality))
        total += float(dense_scales[stage]) * (float(gamma) * quality - previous)
        previous_stage_qualities[stage] = quality
        if stage == current_stage:
            active_quality = quality
    if is_sparse_reward_mode(mode):
        total = 0.0
    return float(total), float(active_quality)


def remember_stage_quality(
    previous_stage_qualities: MutableMapping[int, float],
    stage: int,
    quality: float,
) -> None:
    previous_stage_qualities[int(stage)] = float(quality)

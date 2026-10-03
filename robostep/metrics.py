"""Small reporting helpers that keep train/eval metrics separate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Iterable


@dataclass(frozen=True)
class SeedResult:
    benchmark: str
    task: str
    method: str
    seed: int
    success: float
    return_mean: float | None = None
    transitions: int | None = None
    checkpoint: str = ""
    evaluation_status: str = "held_out"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_seed_results(results: Iterable[SeedResult]) -> dict[str, float | int | str]:
    values = list(results)
    if not values:
        raise ValueError("cannot summarize an empty result list")
    scores = [float(item.success) for item in values]
    return {
        "benchmark": values[0].benchmark,
        "task": values[0].task,
        "method": values[0].method,
        "mean": mean(scores),
        "sd": stdev(scores) if len(scores) > 1 else 0.0,
        "n_seeds": len(scores),
        "seeds": ";".join(str(item.seed) for item in values),
    }

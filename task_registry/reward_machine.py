"""Callable reward machines for the frozen 75-task Frame-VLM programs.

The JSON files next to this module are the task-level programs used by the
main-table runs.  A benchmark adapter supplies normalized feature values and
completion predicates from its native simulator state; this module turns them
into the active-stage potential-difference reward and the shared stage FSM.
No benchmark checkout is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from robostep.reward import ActiveStageReward, RewardStep
from robostep.schema import RewardProgram

from .loader import load_task


def _as_feature_rows(features: Mapping[str, float] | Sequence[Mapping[str, float]], batch: int) -> list[Mapping[str, float]]:
    if isinstance(features, Mapping):
        return [features] * batch
    rows = list(features)
    if len(rows) != batch:
        raise ValueError(f"features must contain {batch} rows, got {len(rows)}")
    return rows


@dataclass
class TaskRewardMachine:
    """Stateful task reward machine backed by one frozen Frame-VLM program.

    Feature adapters return values in ``[0, 1]``.  A term with direction
    ``increase`` is interpreted as progress; a term with direction ``decrease``
    is interpreted as a normalized cost and is converted to ``1 - value``.
    The adapter must provide one candidate boolean per stage.  The official
    success signal is passed separately through ``terminal_success`` and is
    never used to construct dense potentials.
    """

    program: RewardProgram
    benchmark: str
    task_id: str
    batch_size: int = 1
    shaping_gamma: float = 0.99
    delta_max: float = 0.10

    def __post_init__(self) -> None:
        self.engine = ActiveStageReward(
            self.program,
            batch_size=self.batch_size,
            shaping_gamma=self.shaping_gamma,
            delta_max=self.delta_max,
        )

    @property
    def stage_count(self) -> int:
        return self.program.num_stages

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.program.stages)

    @property
    def feature_names(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(term.name for term in stage.potential_terms)
            for stage in self.program.stages
        )

    @classmethod
    def from_task(
        cls,
        task_id: str,
        benchmark: str | None = None,
        *,
        batch_size: int = 1,
        shaping_gamma: float | None = None,
        delta_max: float | None = None,
    ) -> "TaskRewardMachine":
        payload = load_task(task_id, benchmark)
        program_payload = payload.get("reward_program")
        if not isinstance(program_payload, Mapping):
            raise ValueError(f"{task_id} does not contain a reward_program")
        program = RewardProgram.from_dict(program_payload)
        config = payload.get("reward_config", {})
        return cls(
            program=program,
            benchmark=str(payload["benchmark"]),
            task_id=str(payload["task_id"]),
            batch_size=batch_size,
            shaping_gamma=float(config.get("gamma", 0.99) if shaping_gamma is None else shaping_gamma),
            delta_max=float(config.get("delta_max", 0.10) if delta_max is None else delta_max),
        )

    def reset(self, feature_rows: Mapping[str, float] | Sequence[Mapping[str, float]]) -> None:
        self.engine.reset(self.potentials(feature_rows))

    def potentials(self, feature_rows: Mapping[str, float] | Sequence[Mapping[str, float]]) -> np.ndarray:
        rows = _as_feature_rows(feature_rows, self.batch_size)
        values = np.zeros((self.batch_size, self.stage_count), dtype=np.float64)
        for batch_index, features in enumerate(rows):
            for stage_index, stage in enumerate(self.program.stages):
                terms = stage.potential_terms
                if not terms:
                    values[batch_index, stage_index] = 0.0
                    continue
                weighted = 0.0
                normalizer = 0.0
                for term in terms:
                    raw = float(features.get(term.name, 0.0))
                    raw = float(np.clip(raw, 0.0, 1.0))
                    progress = raw if term.direction == "increase" else 1.0 - raw
                    if term.clip_min is not None:
                        progress = max(progress, float(term.clip_min))
                    if term.clip_max is not None:
                        progress = min(progress, float(term.clip_max))
                    weighted += float(term.weight) * progress
                    normalizer += abs(float(term.weight))
                values[batch_index, stage_index] = float(np.clip(weighted / max(normalizer, 1e-8), 0.0, 1.0))
        return values

    def step(
        self,
        feature_rows: Mapping[str, float] | Sequence[Mapping[str, float]],
        candidates: Sequence[bool] | np.ndarray,
        *,
        maintenance: Sequence[float] | np.ndarray | None = None,
        safety_penalty: Sequence[float] | np.ndarray | None = None,
        transition_authorization: Sequence[bool] | np.ndarray | None = None,
        terminal_success: Sequence[bool] | np.ndarray | None = None,
        reward_mode: str = "staged_reset",
        transition_authority: str = "auto",
    ) -> RewardStep:
        potentials = self.potentials(feature_rows)
        candidates_array = np.asarray(candidates, dtype=bool)
        if candidates_array.shape == (self.stage_count,) and self.batch_size == 1:
            candidates_array = candidates_array[None, :]
        return self.engine.step(
            potentials,
            candidates_array,
            maintenance=None if maintenance is None else np.asarray(maintenance, dtype=np.float64),
            safety_penalty=None if safety_penalty is None else np.asarray(safety_penalty, dtype=np.float64),
            transition_authorization=None if transition_authorization is None else np.asarray(transition_authorization, dtype=bool),
            terminal_success=None if terminal_success is None else np.asarray(terminal_success, dtype=bool),
            reward_mode=reward_mode,
            transition_authority=transition_authority,
        )


def load_reward_machine(task_id: str, benchmark: str | None = None, **kwargs: object) -> TaskRewardMachine:
    """Load one of the 75 frozen main-table reward machines."""

    return TaskRewardMachine.from_task(task_id, benchmark, **kwargs)


"""Serializable reward-program schema emitted by a Frame-VLM compiler.

The schema deliberately stores feature names and coefficients rather than
Python callbacks.  Environment adapters map feature names to numeric
potentials, which makes a frozen recipe auditable and replayable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PotentialTerm:
    name: str
    weight: float = 1.0
    direction: str = "increase"
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'")
        if not self.name:
            raise ValueError("potential terms need a stable name")
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min > self.clip_max:
                raise ValueError("clip_min must not exceed clip_max")


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    semantic_objective: str
    potential_terms: tuple[PotentialTerm, ...] = ()
    dwell_steps: int = 1
    transition_bonus: float = 0.0
    dense_scale: float = 1.0
    progress_aggregation: str = "potential"
    completion_predicate: tuple[str, ...] = ()
    normalization: str = ""
    hysteresis: Mapping[str, float] = field(default_factory=dict)
    adversarial_counterexamples: tuple[str, ...] = ()
    maintenance_terms: tuple[str, ...] = ()
    safety_terms: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    recovery_policy: str = "gate"
    vlm_verification: bool = False
    visible_evidence: tuple[str, ...] = ()
    abstain_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dwell_steps < 1:
            raise ValueError("dwell_steps must be positive")
        if self.dense_scale < 0:
            raise ValueError("dense_scale must be non-negative")
        if self.progress_aggregation not in {"potential", "high_water", "event"}:
            raise ValueError("unsupported progress aggregation")
        if self.recovery_policy not in {"gate", "rollback", "recovery_stage", "terminate"}:
            raise ValueError("unsupported recovery policy")


@dataclass(frozen=True)
class RewardProgram:
    task_id: str
    task_description: str
    stages: tuple[StageSpec, ...]
    source: str = "frame_vlm"
    compiler_round: int = 0
    model_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a reward program needs at least one stage")
        ids = [s.stage_id for s in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("stage_id values must be unique and ordered")
        if self.compiler_round < 0:
            raise ValueError("compiler_round cannot be negative")

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for stage in self.stages:
            if not stage.semantic_objective.strip():
                errors.append(f"{stage.stage_id}: empty semantic objective")
            for term in stage.potential_terms:
                if term.name in stage.maintenance_terms:
                    errors.append(f"{stage.stage_id}: potential used as maintenance term")
            if stage.vlm_verification and not stage.visible_evidence:
                errors.append(f"{stage.stage_id}: VLM stage lacks visible evidence")
        return errors

    def to_dict(self) -> dict[str, Any]:
        # Build this explicitly instead of calling ``asdict(self)`` first:
        # dataclasses.asdict recursively converts PotentialTerm objects, which
        # would make a second asdict call fail for nested terms.
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "source": self.source,
            "compiler_round": self.compiler_round,
            "model_id": self.model_id,
            "metadata": dict(self.metadata),
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "semantic_objective": stage.semantic_objective,
                    "potential_terms": [asdict(term) for term in stage.potential_terms],
                    "dwell_steps": stage.dwell_steps,
                    "transition_bonus": stage.transition_bonus,
                    "dense_scale": stage.dense_scale,
                    "progress_aggregation": stage.progress_aggregation,
                    "completion_predicate": list(stage.completion_predicate),
                    "normalization": stage.normalization,
                    "hysteresis": dict(stage.hysteresis),
                    "adversarial_counterexamples": list(stage.adversarial_counterexamples),
                    "maintenance_terms": list(stage.maintenance_terms),
                    "safety_terms": list(stage.safety_terms),
                    "prerequisites": list(stage.prerequisites),
                    "recovery_policy": stage.recovery_policy,
                    "vlm_verification": stage.vlm_verification,
                    "visible_evidence": list(stage.visible_evidence),
                    "abstain_conditions": list(stage.abstain_conditions),
                }
                for stage in self.stages
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewardProgram":
        stages = []
        for raw_stage in data.get("stages", []):
            terms = tuple(PotentialTerm(**t) for t in raw_stage.get("potential_terms", []))
            clean = dict(raw_stage)
            clean["potential_terms"] = terms
            for key in ("completion_predicate", "adversarial_counterexamples", "maintenance_terms", "safety_terms", "prerequisites", "visible_evidence", "abstain_conditions"):
                clean[key] = tuple(clean.get(key, ()))
            stages.append(StageSpec(**clean))
        result = dict(data)
        result["stages"] = tuple(stages)
        return cls(**result)

    @classmethod
    def from_json(cls, path: str) -> "RewardProgram":
        import json
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_json(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

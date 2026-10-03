"""Stage transition observers used by the reward engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from .vlm import FrozenAsyncVLM, GateRequest


class GateDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class GateRule:
    """Privileged deterministic semantic gate.

    `predicate` receives an environment state mapping and returns a boolean.
    The class only supplies a candidate/authorization decision; persistence is
    owned by `ActiveStageReward` so Rule and VLM comparisons share counters.
    """

    predicate: Callable[[Mapping[str, Any], int], bool]

    def authorize(self, state: Mapping[str, Any], stage_index: int) -> bool:
        return bool(self.predicate(state, int(stage_index)))


class GateVLM:
    """Visual semantic verifier layered on top of a fast candidate gate."""

    def __init__(self, client: FrozenAsyncVLM, *, abstain_is_accept: bool = False) -> None:
        self.client = client
        self.abstain_is_accept = bool(abstain_is_accept)
        self.submitted = 0
        self.accepted = 0
        self.rejected = 0
        self.abstained = 0

    def request_if_candidate(self, request: GateRequest, candidate: bool) -> None:
        if candidate:
            self.submitted += 1
            self.client.submit(request)

    def drain(self, *, episode_id: int, stage_index: int, request_step: int | None = None) -> dict[str, bool]:
        decisions: dict[str, bool] = {}
        for request, raw in self.client.poll(current_episode=episode_id, current_stage=stage_index, current_request_step=request_step):
            if raw == GateDecision.ACCEPT.value:
                self.accepted += 1
                decisions[request.request_id] = True
            elif raw == GateDecision.REJECT.value:
                self.rejected += 1
                decisions[request.request_id] = False
            else:
                self.abstained += 1
                decisions[request.request_id] = self.abstain_is_accept
        return decisions

    def statistics(self) -> dict[str, int]:
        return {"submitted": self.submitted, "accepted": self.accepted, "rejected": self.rejected, "abstained": self.abstained}

"""Pure-RGB mandatory verifier for persistent state-rule candidates.

The predictor receives frozen RGB evidence and semantic text only. Simulator
state is intentionally absent from VisualStageRequest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class VisualStageRequest:
    env_index: int
    episode_id: int
    request_step: int
    stage_id: int
    task_id: str
    stage_name: str
    task_instruction: str
    reference_rgb: np.ndarray
    current_rgb: np.ndarray
    # Optional privileged label used only by offline/audit evaluators. It is
    # not sent to the VLM worker; the worker receives RGB and text only.
    ground_truth_complete: bool | None = None


class VisualStagePredictor(Protocol):
    """Closed-vocabulary synchronous predictor used by the wrapper."""

    def predict_stage_completion(
        self, requests: Sequence[VisualStageRequest]
    ) -> Sequence[str]:
        """Return one of success/failure/unknown for every request."""

    def close(self) -> None:
        """Release a persistent worker, if any."""


class SynchronousVisualStageObserver:
    """Event-triggered pure-RGB verifier with fail-closed transitions.

    This adapter is intentionally model-agnostic. The existing Qwen worker can
    implement VisualStagePredictor without receiving simulator state. A
    transition is authorized only when a persistent RuleGate candidate and an
    exact visual success response coincide. The first candidate is always
    queried; rejected candidates are retried at a fixed step interval.
    """

    allowed_predictions = frozenset({"success", "failure", "unknown"})

    def __init__(
        self,
        *,
        env_id: str,
        stage_names: Sequence[str],
        task_instruction: str,
        num_envs: int,
        device: torch.device | str,
        predictor: VisualStagePredictor,
        query_interval_steps: int = 10,
        stage_instructions: Sequence[str] | None = None,
        close_predictor: bool = True,
        query_mode: str = "candidate_retry",
    ) -> None:
        if query_interval_steps < 1:
            raise ValueError("query_interval_steps must be positive")
        if query_mode not in {"candidate_retry", "fixed_frequency"}:
            raise ValueError(f"unsupported query_mode: {query_mode}")
        if not stage_names:
            raise ValueError("stage_names must be nonempty")
        self.env_id = str(env_id)
        self.stage_names = tuple(str(name) for name in stage_names)
        self.task_instruction = str(task_instruction)
        if stage_instructions is None:
            self.stage_instructions = self.stage_names
        else:
            self.stage_instructions = tuple(str(item) for item in stage_instructions)
            if len(self.stage_instructions) != len(self.stage_names):
                raise ValueError("stage_instructions must align with stage_names")
        self.close_predictor = bool(close_predictor)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.predictor = predictor
        self.query_interval_steps = int(query_interval_steps)
        self.query_mode = str(query_mode)
        self.episode_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_query_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_stage = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.reference_rgb: list[np.ndarray | None] = [None] * self.num_envs
        self.requests = 0
        self.accepted = 0
        self.unknown = 0
        self.protocol_errors = 0
        self.records: list[dict[str, Any]] = []

    def reset(
        self, env: Any, env_idx: torch.Tensor | Sequence[int] | None = None
    ) -> None:
        if env_idx is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            )
        frames = self._freeze_rgb_batch(env.render())
        self.episode_id[indices] += 1
        self.episode_step[indices] = 0
        self.last_query_step[indices] = -1
        self.last_stage[indices] = -1
        for index in indices.tolist():
            self.reference_rgb[index] = frames[index].copy()

    def authorize(
        self,
        env: Any,
        current_stage: torch.Tensor,
        rule_candidate: torch.Tensor,
        ground_truth_complete: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_stage.shape != (self.num_envs,):
            raise ValueError(
                f"current_stage must have shape {(self.num_envs,)}"
            )
        if rule_candidate.shape != (self.num_envs,):
            raise ValueError(
                f"rule_candidate must have shape {(self.num_envs,)}"
            )
        if ground_truth_complete is not None and ground_truth_complete.shape != (
            self.num_envs,
        ):
            raise ValueError(
                "ground_truth_complete must have shape "
                f"{(self.num_envs,)}, found {tuple(ground_truth_complete.shape)}"
            )
        self.episode_step += 1
        active = current_stage < len(self.stage_names)
        stage_changed = current_stage != self.last_stage
        self.last_query_step = torch.where(
            stage_changed,
            torch.full_like(self.last_query_step, -1),
            self.last_query_step,
        )
        self.last_stage = current_stage.clone()
        if self.query_mode == "candidate_retry":
            # Original behavior: query only on persistent RuleGate candidates.
            # The first candidate is queried immediately; rejected candidates
            # are retried after query_interval_steps.
            due = active & rule_candidate.bool() & (
                (self.last_query_step < 0)
                | (
                    self.episode_step - self.last_query_step
                    >= self.query_interval_steps
                )
            )
        else:
            # Fixed-frequency ablation: rule_candidate is diagnostic only.
            # Query every K steps while the rollout is in a non-terminal stage.
            just_entered_stage = stage_changed | (self.last_query_step < 0)
            self.last_query_step = torch.where(
                just_entered_stage,
                self.episode_step,
                self.last_query_step,
            )
            due = active & (
                self.episode_step - self.last_query_step
                >= self.query_interval_steps
            )
        authorization = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        indices = torch.nonzero(due).flatten()
        if not len(indices):
            return authorization
        self.last_query_step[indices] = self.episode_step[indices]

        frames = self._freeze_rgb_batch(env.render())
        requests = []
        for index in indices.tolist():
            reference = self.reference_rgb[index]
            if reference is None:
                raise RuntimeError(
                    f"visual observer env {index} used before reset"
                )
            stage_id = int(current_stage[index].item())
            requests.append(
                VisualStageRequest(
                    env_index=index,
                    episode_id=int(self.episode_id[index].item()),
                    request_step=int(self.episode_step[index].item()),
                    stage_id=stage_id,
                    task_id=self.env_id,
                    stage_name=self.stage_names[stage_id],
                    task_instruction=(
                        f"{self.task_instruction} "
                        "Current visible completion contract: "
                        f"{self.stage_instructions[stage_id]}"
                    ),
                    reference_rgb=reference.copy(),
                    current_rgb=frames[index].copy(),
                    ground_truth_complete=(
                        bool(ground_truth_complete[index].item())
                        if ground_truth_complete is not None
                        else None
                    ),
                )
            )

        try:
            predictions = tuple(
                str(item).strip().lower()
                for item in self.predictor.predict_stage_completion(requests)
            )
            if len(predictions) != len(requests):
                raise ValueError(
                    "visual predictor returned a different batch length"
                )
            if any(item not in self.allowed_predictions for item in predictions):
                raise ValueError(
                    "visual predictor violated the closed vocabulary"
                )
        except Exception:
            self.protocol_errors += len(requests)
            predictions = ("unknown",) * len(requests)

        for index, prediction in zip(indices.tolist(), predictions, strict=True):
            accepted = prediction == "success"
            authorization[index] = accepted
            self.requests += 1
            self.accepted += int(accepted)
            self.unknown += int(prediction == "unknown")
            gt = (
                bool(ground_truth_complete[index].item())
                if ground_truth_complete is not None
                else None
            )
            self.records.append(
                {
                    "env_index": int(index),
                    "episode_id": int(self.episode_id[index].item()),
                    "request_step": int(self.episode_step[index].item()),
                    "stage_id": int(current_stage[index].item()),
                    "rule_candidate": bool(rule_candidate[index].item()),
                    "ground_truth_complete": gt,
                    "prediction": prediction,
                    "accepted": bool(accepted),
                }
            )
        return authorization

    def statistics(self) -> dict[str, Any]:
        """Return query counts and confusion statistics against optional GT labels."""
        labeled = [
            item for item in self.records if item["ground_truth_complete"] is not None
        ]
        positives = [item for item in labeled if item["ground_truth_complete"]]
        negatives = [item for item in labeled if not item["ground_truth_complete"]]
        false_rejects = [item for item in positives if not item["accepted"]]
        false_accepts = [item for item in negatives if item["accepted"]]
        # Counters are intentionally derived from the retained records so a
        # caller can clear ``records`` between independent rollouts without
        # accidentally reporting cumulative counts from an earlier seed.
        return {
            "requests": len(self.records),
            "accepted": sum(1 for item in self.records if item["accepted"]),
            "unknown": sum(1 for item in self.records if item["prediction"] == "unknown"),
            "protocol_errors": sum(1 for item in self.records if item.get("protocol_error", False)),
            "labeled_requests": len(labeled),
            "gt_positive": len(positives),
            "gt_negative": len(negatives),
            "true_accept": sum(1 for item in positives if item["accepted"]),
            "true_reject": sum(1 for item in negatives if not item["accepted"]),
            "false_reject": len(false_rejects),
            "false_accept": len(false_accepts),
            "frr": float(len(false_rejects) / len(positives)) if positives else None,
            "far": float(len(false_accepts) / len(negatives)) if negatives else None,
        }

    @staticmethod
    def _freeze_rgb_batch(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 3:
            value = value[None]
        if value.ndim != 4 or value.shape[-1] not in (3, 4):
            raise ValueError(
                "render() must return [N,H,W,3|4] RGB(A), found "
                f"{value.shape}"
            )
        return np.ascontiguousarray(value[..., :3]).copy()

    def close(self) -> None:
        if self.close_predictor:
            close = getattr(self.predictor, "close", None)
            if close is not None:
                close()

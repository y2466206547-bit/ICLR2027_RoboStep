"""Pure-RGB client for Qwen task-stage regression judgments."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from stage_reward.qwen3_client import GATE_SCHEMA, Qwen3VL8BClient


@dataclass(frozen=True)
class RollbackStageRequest:
    env_index: int
    episode_id: int
    request_step: int
    task_id: str
    task_instruction: str
    stage_id: int
    stage_descriptions: tuple[str, ...]
    reference_rgb: np.ndarray
    current_rgb: np.ndarray


class QwenRollbackClient(Qwen3VL8BClient):
    """Reuse the persistent Qwen transport with a stage-index vocabulary."""

    def predict_rollback_stage(
        self, requests: Sequence[RollbackStageRequest]
    ) -> Sequence[str]:
        if not requests:
            return ()
        items: list[dict[str, Any]] = []
        ids: list[int] = []
        for request in requests:
            request_id = self.request_id
            self.request_id += 1
            ids.append(request_id)
            audit_limit = int(os.environ.get("ROBOSTEP_ROLLBACK_RGB_AUDIT_LIMIT", "0"))
            if request_id < audit_limit:
                audit_dir = self.output_dir / "rgb_audit"
                audit_dir.mkdir(parents=True, exist_ok=True)
                stem = (
                    f"request_{request_id:04d}_env_{request.env_index:03d}"
                    f"_step_{request.request_step:04d}"
                )
                Image.fromarray(request.reference_rgb).convert("RGB").save(
                    audit_dir / f"{stem}_reference.png"
                )
                Image.fromarray(request.current_rgb).convert("RGB").save(
                    audit_dir / f"{stem}_current.png"
                )
            items.append(
                {
                    "request_id": request_id,
                    "env_index": request.env_index,
                    "episode_id": request.episode_id,
                    "request_step": request.request_step,
                    "task_id": request.task_id,
                    "task_instruction": request.task_instruction,
                    "stage_id": request.stage_id,
                    "stage_name": f"stage_{request.stage_id}",
                    "stage_descriptions": list(request.stage_descriptions),
                    "reference_jpeg_b64": self._jpeg_b64(request.reference_rgb),
                    "current_jpeg_b64": self._jpeg_b64(request.current_rgb),
                }
            )
        started = time.time()
        try:
            result = self._command({"schema": GATE_SCHEMA, "items": items})
            rows = list(result["results"])
            returned_ids = [int(row["request_id"]) for row in rows]
            if returned_ids != ids:
                raise RuntimeError(
                    f"Qwen rollback request order mismatch: {returned_ids} != {ids}"
                )
            predictions = [str(row["prediction"]).strip().lower() for row in rows]
            raw_outputs = [str(row.get("raw_output", "")) for row in rows]
        except Exception as error:
            self.protocol_errors += len(items)
            predictions = ["unknown"] * len(items)
            raw_outputs = [""] * len(items)
            self._write(
                {
                    "event": "rollback_protocol_error",
                    "request_ids": ids,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        self.gate_requests += len(items)
        self.gate_unknown += sum(item == "unknown" for item in predictions)
        for item, prediction, raw_output in zip(
            items, predictions, raw_outputs, strict=True
        ):
            self._write(
                {
                    "event": "rollback_stage_result",
                    "request_id": item["request_id"],
                    "task_id": item["task_id"],
                    "env_index": item["env_index"],
                    "request_step": item["request_step"],
                    "stage_before": item["stage_id"],
                    "prediction": prediction,
                    "raw_output": raw_output,
                }
            )
        self._write(
            {
                "event": "rollback_stage_batch",
                "batch_size": len(items),
                "predictions": predictions,
                "walltime_s": time.time() - started,
            }
        )
        return predictions

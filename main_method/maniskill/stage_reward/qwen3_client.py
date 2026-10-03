"""Persistent client and VisualStagePredictor for Qwen3-VL-8B."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from typing import Any, Sequence

from PIL import Image

from stage_reward.visual_observer import VisualStageRequest


READY_SCHEMA = "maniskill-qwen3-vl8b-ready-v1"
GATE_SCHEMA = "maniskill-qwen3-vl8b-gate-batch-v1"
DESIGN_SCHEMA = "maniskill-frame-vlm-design-batch-v1"
RESULT_SCHEMA = "maniskill-qwen3-vl8b-result-v1"


class Qwen3VL8BClient:
    """One persistent worker shared by Frame-VLM design and gate requests."""

    def __init__(
        self,
        *,
        python_executable: str | Path,
        worker_script: str | Path,
        model_path: str | Path,
        output_dir: str | Path,
        device_index: int = 1,
        timeout_s: float = 900.0,
        image_size: int = 320,
        gate_max_new_tokens: int = 4,
        design_max_new_tokens: int = 1800,
    ) -> None:
        self.python_executable = Path(python_executable).resolve()
        self.worker_script = Path(worker_script).resolve()
        self.model_path = Path(model_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self.image_size = int(image_size)
        self.request_id = 0
        self.gate_requests = 0
        self.gate_accepts = 0
        self.gate_unknown = 0
        self.protocol_errors = 0
        self.log_path = self.output_dir / "qwen3_vl8b_client.jsonl"
        self.stderr_path = self.output_dir / "qwen3_vl8b_worker.stderr.log"
        self._log = self.log_path.open("a", encoding="utf-8")
        self._stderr = self.stderr_path.open("a", encoding="utf-8")
        command = [
            str(self.python_executable),
            str(self.worker_script),
            "--model-path",
            str(self.model_path),
            "--device-index",
            str(int(device_index)),
            "--gate-max-new-tokens",
            str(int(gate_max_new_tokens)),
            "--design-max-new-tokens",
            str(int(design_max_new_tokens)),
        ]
        adapter_path = os.environ.get("ADAPTED_GATEVLM_ADAPTER_PATH", "").strip()
        if adapter_path:
            command.extend(["--adapter-path", adapter_path])
        # The evaluator may need a Python 3.10 PYTHONPATH for ManiSkill/SAPIEN.
        # The Qwen worker runs in a separate Python (usually 3.11); inheriting
        # that path can load incompatible binary wheels (for example PIL).
        worker_env = os.environ.copy()
        worker_env.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        ready = self._read_json(self.timeout_s)
        if ready.get("schema") != READY_SCHEMA:
            self.close()
            raise RuntimeError(f"invalid Qwen worker ready record: {ready}")
        self._write({"event": "worker_ready", "ready": ready, "command": command})

    def _write(self, value: dict[str, Any]) -> None:
        self._log.write(json.dumps(value, sort_keys=True) + "\n")
        self._log.flush()

    def _read_json(self, timeout_s: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Qwen worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(timeout_s)
        finally:
            selector.close()
        if not events:
            raise TimeoutError(f"Qwen worker timed out after {timeout_s}s")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(
                f"Qwen worker exited with code {code}; see {self.stderr_path}"
            )
        value = json.loads(line)
        if value.get("schema") not in {READY_SCHEMA, RESULT_SCHEMA}:
            raise RuntimeError(f"invalid Qwen worker schema: {value}")
        return value

    def _command(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None:
            raise RuntimeError("Qwen worker stdin is unavailable")
        self.process.stdin.write(json.dumps(command, sort_keys=True) + "\n")
        self.process.stdin.flush()
        result = self._read_json(self.timeout_s)
        if "error" in result:
            raise RuntimeError(
                f"{result.get('error_type', 'WorkerError')}: {result['error']}"
            )
        return result

    def design_batch(
        self, items: Sequence[dict[str, str]]
    ) -> list[dict[str, Any]]:
        started = time.time()
        result = self._command({"schema": DESIGN_SCHEMA, "items": list(items)})
        self._write(
            {
                "event": "frame_vlm_design_batch",
                "task_count": len(items),
                "worker_walltime_s": result.get("walltime_s"),
                "client_walltime_s": time.time() - started,
            }
        )
        return list(result["results"])

    def _jpeg_b64(self, value: Any) -> str:
        image = Image.fromarray(value).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=92, optimize=False)
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def predict_stage_completion(
        self, requests: Sequence[VisualStageRequest]
    ) -> Sequence[str]:
        if not requests:
            return ()
        items = []
        ids = []
        for request in requests:
            request_id = self.request_id
            self.request_id += 1
            ids.append(request_id)
            items.append(
                {
                    "request_id": request_id,
                    "env_index": request.env_index,
                    "episode_id": request.episode_id,
                    "request_step": request.request_step,
                    "stage_id": request.stage_id,
                    "task_id": request.task_id,
                    "stage_name": request.stage_name,
                    "task_instruction": request.task_instruction,
                    "reference_jpeg_b64": self._jpeg_b64(
                        request.reference_rgb
                    ),
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
                    f"Qwen worker request order mismatch: {returned_ids} != {ids}"
                )
            predictions = [str(row["prediction"]).lower() for row in rows]
            if any(
                item not in {"success", "failure", "unknown"}
                for item in predictions
            ):
                raise RuntimeError(
                    f"Qwen worker returned invalid predictions: {predictions}"
                )
            raw_outputs = [str(row.get("raw_output", "")) for row in rows]
        except Exception as error:
            self.protocol_errors += len(items)
            predictions = ["unknown"] * len(items)
            raw_outputs = [""] * len(items)
            self._write(
                {
                    "event": "gate_protocol_error",
                    "request_ids": ids,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        self.gate_requests += len(items)
        self.gate_accepts += sum(item == "success" for item in predictions)
        self.gate_unknown += sum(item == "unknown" for item in predictions)
        for item, prediction, raw_output in zip(
            items, predictions, raw_outputs, strict=True
        ):
            self._write(
                {
                    "event": "gate_result",
                    "request_id": item["request_id"],
                    "task_id": item["task_id"],
                    "env_index": item["env_index"],
                    "episode_id": item["episode_id"],
                    "stage_id": item["stage_id"],
                    "request_step": item["request_step"],
                    "prediction": prediction,
                    "raw_output": raw_output,
                }
            )
        self._write(
            {
                "event": "gate_batch",
                "batch_size": len(items),
                "predictions": predictions,
                "walltime_s": time.time() - started,
            }
        )
        return predictions

    def statistics(self) -> dict[str, int]:
        return {
            "qwen_requests": self.gate_requests,
            "qwen_accepts": self.gate_accepts,
            "qwen_unknown": self.gate_unknown,
            "qwen_protocol_errors": self.protocol_errors,
        }

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)
            self.process = None
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            self._write({"event": "client_close", **self.statistics()})
            log.close()
        stderr = getattr(self, "_stderr", None)
        if stderr is not None and not stderr.closed:
            stderr.close()

"""Task-configurable Mandatory-Qwen gate used by the MT50 sweep.

This intentionally keeps the frozen-packet, exact-token, fail-closed protocol
used for drawer opening, but makes stage text explicit task data rather than an
implicit import from one task's reward module.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence

from stage_policy.live_vlm_gate import (
    JsonlWorkerClient,
    LIVE_WORKER_MANIFEST_DIGEST,
    WorkerProvenanceExpectations,
    freeze_request_packet,
    validate_worker_result,
)

from .qwen_gate import GateCandidate, GateDecision
from .transition_prompt_runtime import configure_transition_prompt


class MandatoryQwenStageGate:
    """Persistent Qwen worker with task-owned stage names and instructions."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        python_executable: str | Path,
        worker_script: str | Path,
        model_path: str | Path,
        prompt_template: str | Path,
        stage_names: Sequence[str],
        stage_instructions: Sequence[str],
        qwen_gpu: int = 0,
        timeout_s: float = 600.0,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        packet_override = os.environ.get("STAGE_REWARD_QWEN_PACKET_ROOT")
        if packet_override:
            override_root = Path(packet_override).expanduser().resolve()
            digest = hashlib.blake2b(str(self.output_dir).encode("utf-8")).hexdigest()[:16]
            self.packet_root = override_root / f"{self.output_dir.name}_{digest}_{os.getpid()}" / "packets"
            self._packet_cleanup_guard = override_root
        else:
            self.packet_root = self.output_dir / "packets"
            self._packet_cleanup_guard = None
        self._cleanup_packets = os.environ.get(
            "STAGE_REWARD_QWEN_PACKET_CLEANUP", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.packet_root.mkdir(parents=True, exist_ok=False)
        self.log_path = self.output_dir / "qwen_gate.jsonl"
        self._log = self.log_path.open("x", encoding="utf-8")
        self.stage_names = tuple(str(name) for name in stage_names)
        self.original_stage_instructions = tuple(
            str(instruction) for instruction in stage_instructions
        )
        if not self.stage_names or len(self.stage_names) != len(
            self.original_stage_instructions
        ):
            raise ValueError("stage names and instructions must be nonempty and aligned")
        (
            self.prompt_mode,
            prompt,
            self.stage_instructions,
        ) = configure_transition_prompt(
            prompt_template,
            self.stage_names,
            self.original_stage_instructions,
        )
        self.timeout_s = float(timeout_s)
        self.request_id = 0
        self.batch_id = 0
        self.requests = 0
        self.accepted = 0
        self.protocol_errors = 0
        self._log_disabled = False

        worker = Path(worker_script).expanduser().resolve()
        model = Path(model_path).expanduser().resolve()
        expectations = WorkerProvenanceExpectations(
            model_path=model,
            prompt_template_digest=hashlib.blake2b(prompt.read_bytes()).hexdigest(),
            worker_script_digest=hashlib.blake2b(worker.read_bytes()).hexdigest(),
            worker_manifest_digest=LIVE_WORKER_MANIFEST_DIGEST,
        )
        qwen_python = os.environ.get("ADAPTED_GATEVLM_QWEN_PYTHON", "").strip()
        command = (
            str(Path(qwen_python or python_executable).expanduser().resolve()),
            str(worker),
            "--model-path",
            str(model),
            "--single-gpu-device",
            str(int(qwen_gpu)),
            "--max-new-tokens",
            "6",
            "--prompt-template-file",
            str(prompt),
        )
        adapter_path = os.environ.get("ADAPTED_GATEVLM_ADAPTER_PATH", "").strip()
        if adapter_path:
            command = command + ("--adapter-path", adapter_path)
        self.client = JsonlWorkerClient(
            command,
            startup_timeout_s=self.timeout_s,
            expected_provenance=expectations,
        )
        self._write(
            {
                "event": "worker_ready",
                "ready": self.client.ready_record,
                "command": list(command),
                "prompt_mode": self.prompt_mode,
                "prompt_template": str(prompt),
                "stage_names": list(self.stage_names),
                "stage_instructions": list(self.stage_instructions),
                "packet_root": str(self.packet_root),
                "packet_cleanup_enabled": self._cleanup_packets,
            }
        )

    def _write(self, record: dict[str, object]) -> None:
        if self._log_disabled or self._log.closed:
            return
        try:
            self._log.write(json.dumps(record, sort_keys=True) + "\n")
            self._log.flush()
        except OSError as error:
            if error.errno != errno.ENOSPC:
                raise
            self._log_disabled = True
            print(
                f"warning: disabling qwen gate JSONL logging after ENOSPC: {self.log_path}",
                file=sys.stderr,
            )

    def _cleanup_packet_root(self) -> None:
        if not self._cleanup_packets:
            return
        guard = self._packet_cleanup_guard
        if guard is None:
            raise RuntimeError("packet cleanup requires STAGE_REWARD_QWEN_PACKET_ROOT")
        try:
            self.packet_root.relative_to(guard)
        except ValueError as error:
            raise RuntimeError(
                f"refusing to clean packet_root outside override guard: {self.packet_root}"
            ) from error
        if self.packet_root.exists():
            shutil.rmtree(self.packet_root)
        self.packet_root.mkdir(parents=True, exist_ok=True)

    def decide_batch(
        self, candidates: Sequence[GateCandidate]
    ) -> list[GateDecision]:
        if not candidates:
            return []
        packets: list[dict[str, object]] = []
        tokens: list[dict[str, int]] = []
        for candidate in candidates:
            if candidate.stage_id < 0 or candidate.stage_id >= len(
                self.stage_instructions
            ):
                raise ValueError("candidate stage is outside the configured range")
            token = {
                "env_id": int(candidate.env_id),
                "episode_id": int(candidate.episode_id),
                "stage_id": int(candidate.stage_id),
                "candidate_epoch": int(candidate.candidate_epoch),
                "request_id": int(self.request_id),
                "request_step": int(candidate.request_step),
                "candidate_start_step": int(candidate.candidate_start_step),
            }
            self.request_id += 1
            packet = freeze_request_packet(
                self.packet_root,
                token=token,
                stage_name=self.stage_names[candidate.stage_id],
                camera_name=candidate.camera_name,
                instruction=self.stage_instructions[candidate.stage_id],
                reference=candidate.reference,
                ring_frames=candidate.ring_frames,
                current=candidate.current,
                ring_nominal_fps=candidate.ring_nominal_fps,
                source_update_period_s=candidate.source_update_period_s,
                target_samples_due_since_reset=(
                    candidate.target_samples_due_since_reset
                ),
                pixel_identical_duplicates_skipped_since_reset=(
                    candidate.duplicates_skipped_since_reset
                ),
            )
            tokens.append(token)
            packets.append(packet)
        try:
            response = self.client.request_batch(
                packets, batch_id=self.batch_id, timeout_s=self.timeout_s
            )
            self.batch_id += 1
            decisions: list[GateDecision] = []
            for packet, token, raw in zip(
                packets, tokens, response["results"]
            ):
                _, prediction = validate_worker_result(raw, token)
                decision = GateDecision(
                    prediction=prediction,
                    accepted=prediction == "success",
                    token=token,
                    packet_path=str(packet["packet_path"]),
                    raw_output=str(raw.get("raw_output", "")),
                )
                decisions.append(decision)
                self.requests += 1
                self.accepted += int(decision.accepted)
                self._write(
                    {
                        "event": "qwen_gate_result",
                        "token": token,
                        "packet_path": decision.packet_path,
                        "prediction": prediction,
                        "accepted": decision.accepted,
                        "raw_output": decision.raw_output,
                        "worker_batch_size": raw.get("worker_batch_size"),
                        "generation_walltime_s": raw.get("generation_walltime_s"),
                    }
                )
            return decisions
        except Exception as error:
            self.protocol_errors += len(candidates)
            self._write(
                {
                    "event": "qwen_gate_protocol_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "tokens": tokens,
                }
            )
            return [
                GateDecision(
                    prediction="unknown",
                    accepted=False,
                    token=token,
                    packet_path=str(packet["packet_path"]),
                    raw_output="",
                )
                for token, packet in zip(tokens, packets)
            ]
        finally:
            self._cleanup_packet_root()

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            client.close()
            self.client = None
        if not self._log.closed:
            self._write(
                {
                    "event": "gate_summary",
                    "requests": self.requests,
                    "accepted": self.accepted,
                    "protocol_errors": self.protocol_errors,
                }
            )
            try:
                self._log.close()
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
                print(
                    f"warning: qwen gate JSONL close hit ENOSPC: {self.log_path}",
                    file=sys.stderr,
                )

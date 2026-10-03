"""Frozen Frame-VLM compiler and asynchronous gate protocol.

No model weights are bundled.  A model-specific worker can be plugged in via
`call_backend`; after `freeze()` no cache miss is allowed to invoke a model.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .schema import RewardProgram


def canonical_key(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{zlib.crc32(payload):08x}"


@dataclass(frozen=True)
class FramePacket:
    frame_id: str
    timestamp: float
    image_path: str | None = None
    image_b64: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompileRequest:
    task_id: str
    task_description: str
    simulator_state_api: Mapping[str, Any]
    frames: tuple[FramePacket, ...] = ()
    previous_program: Mapping[str, Any] | None = None
    repair_round: int = 0
    training_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    gate_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def key_payload(self) -> dict[str, Any]:
        return {
            "kind": "compile",
            "task_id": self.task_id,
            "task_description": self.task_description,
            "simulator_state_api": self.simulator_state_api,
            "frames": [asdict(frame) for frame in self.frames],
            "previous_program": self.previous_program,
            "repair_round": self.repair_round,
            "training_diagnostics": self.training_diagnostics,
            "gate_diagnostics": self.gate_diagnostics,
        }


@dataclass(frozen=True)
class GateRequest:
    request_id: str
    task_id: str
    episode_id: int
    stage_index: int
    request_step: int
    task_description: str
    stage_contract: Mapping[str, Any]
    reference_frames: tuple[FramePacket, ...] = ()
    candidate_frames: tuple[FramePacket, ...] = ()

    def key_payload(self) -> dict[str, Any]:
        return {
            "kind": "gate",
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "stage_index": self.stage_index,
            "request_step": self.request_step,
            "task_description": self.task_description,
            "stage_contract": self.stage_contract,
            "reference_frames": [asdict(frame) for frame in self.reference_frames],
            "candidate_frames": [asdict(frame) for frame in self.candidate_frames],
        }


class FrozenAsyncVLM:
    """Asynchronous, cache-backed VLM client.

    Before freezing, a backend can answer cache misses.  After freezing,
    `submit()` only looks up the frozen cache and returns `abstain` on misses.
    The worker checks request identity at the consumer boundary, so delayed
    answers cannot advance a different episode or stage.
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        call_backend: Callable[[dict[str, Any]], str] | None = None,
        worker_count: int = 1,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        self._load_cache()
        self._backend = call_backend
        self._frozen = False
        self._queue: queue.Queue[GateRequest | None] = queue.Queue()
        self._results: queue.Queue[tuple[GateRequest, str]] = queue.Queue()
        self._threads: list[threading.Thread] = []
        for index in range(max(1, int(worker_count))):
            thread = threading.Thread(target=self._worker, name=f"robostep-vlm-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def add(self, key_payload: Mapping[str, Any], decision: str) -> None:
        if self._frozen:
            raise RuntimeError("cannot add entries after freeze()")
        decision = self._normalize(decision)
        key = canonical_key(key_payload)
        self._cache[key] = decision
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "decision": decision, "payload": key_payload}, sort_keys=True) + "\n")

    def freeze(self) -> None:
        self._frozen = True
        with self.cache_path.with_suffix(self.cache_path.suffix + ".frozen").open("w", encoding="utf-8") as handle:
            json.dump({"schema": "robostep-frozen-v1", "keys": sorted(self._cache), "frozen_at": time.time()}, handle, indent=2)
            handle.write("\n")

    def submit(self, request: GateRequest) -> None:
        self._queue.put(request)

    def poll(self, *, current_episode: int | None = None, current_stage: int | None = None, current_request_step: int | None = None) -> list[tuple[GateRequest, str]]:
        accepted: list[tuple[GateRequest, str]] = []
        while True:
            try:
                request, decision = self._results.get_nowait()
            except queue.Empty:
                break
            if current_episode is not None and request.episode_id != current_episode:
                continue
            if current_stage is not None and request.stage_index != current_stage:
                continue
            if current_request_step is not None and request.request_step != current_request_step:
                continue
            accepted.append((request, decision))
        return accepted

    def flush(self, timeout_s: float = 60.0) -> None:
        done = threading.Event()
        def wait_for_queue() -> None:
            self._queue.join()
            done.set()
        waiter = threading.Thread(target=wait_for_queue, daemon=True)
        waiter.start()
        done.wait(timeout=float(timeout_s))

    def close(self) -> None:
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=2.0)

    def _worker(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                self._queue.task_done()
                return
            payload = request.key_payload()
            key = canonical_key(payload)
            decision = self._cache.get(key)
            if decision is None and not self._frozen and self._backend is not None:
                decision = self._backend(payload)
                self.add(payload, decision)
            if decision is None:
                decision = "abstain"
            self._results.put((request, self._normalize(decision)))
            self._queue.task_done()

    def _load_cache(self) -> None:
        if not self.cache_path.is_file():
            return
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "key" in item and "decision" in item:
                self._cache[str(item["key"])] = self._normalize(item["decision"])

    @staticmethod
    def _normalize(value: str) -> str:
        value = str(value).strip().lower()
        aliases = {"success": "accept", "failure": "reject", "unknown": "abstain", "true": "accept", "false": "reject"}
        value = aliases.get(value, value)
        if value not in {"accept", "reject", "abstain"}:
            raise ValueError(f"invalid VLM gate decision: {value}")
        return value


class FrameVLMCompiler:
    """Offline compiler facade with strict JSON validation."""

    def __init__(self, backend: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None) -> None:
        self.backend = backend

    def compile(self, request: CompileRequest) -> RewardProgram:
        if self.backend is None:
            raise RuntimeError("no compiler backend configured; use a frozen recipe or provide one")
        raw = self.backend(request_payload(request))
        program = raw if isinstance(raw, RewardProgram) else RewardProgram.from_dict(raw)
        errors = program.validate()
        if errors:
            raise ValueError("invalid Frame-VLM reward program: " + "; ".join(errors))
        return program


class SubprocessJSONLBackend:
    """Adapter for a downloaded local VLM worker using one JSON per line.

    The worker is intentionally model-agnostic.  It receives the dictionary
    built by `request_payload` and must print either a reward-program object or
    a gate decision for each line.  Keeping this process outside PPO makes
    model download/versioning explicit and allows the cache to be frozen before
    training starts.
    """

    def __init__(self, command: Sequence[str], *, timeout_s: float = 900.0) -> None:
        self.command = list(command)
        self.timeout_s = float(timeout_s)
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()

    def __call__(self, payload: dict[str, Any]) -> Any:
        with self.lock:
            if self.process.stdin is None or self.process.stdout is None:
                raise RuntimeError("VLM worker pipes are unavailable")
            self.process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"VLM worker exited with code {self.process.poll()}")
            return json.loads(line)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()


def request_payload(request: CompileRequest) -> dict[str, Any]:
    payload = request.key_payload()
    payload["prompt"] = build_frame_vlm_prompt(request)
    return payload


def build_frame_vlm_prompt(request: CompileRequest) -> str:
    """Build a reproducible prompt; the model response is never executed."""
    api = json.dumps(request.simulator_state_api, sort_keys=True, indent=2)
    previous = json.dumps(request.previous_program, sort_keys=True, indent=2) if request.previous_program else "<none>"
    diagnostics = json.dumps(request.training_diagnostics, sort_keys=True, indent=2)
    gate_diagnostics = json.dumps(request.gate_diagnostics, sort_keys=True, indent=2)
    return f"""You are a reward-program compiler for a robot manipulation task.\nTask: {request.task_description}\n\nSimulator/API schema (privileged state may be used to write a reward, but not to fake visual evidence):\n{api}\n\nRepair round: {request.repair_round}. Previous program:\n{previous}\n\nTraining diagnostics:\n{diagnostics}\n\nGate diagnostics:\n{gate_diagnostics}\n\nReturn JSON only. Produce ordered stages. Each stage must specify bounded potential terms, a completion contract, dwell steps, a one-time transition bonus, maintenance terms, safety envelope, recovery policy, and explicit negative/abstain cases for any visual verifier. Use independent success evaluation; never use an official success flag as dense reward.\n"""

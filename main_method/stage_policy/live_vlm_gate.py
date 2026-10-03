"""Dependency-light helpers for a synchronous live VLM-gate proof run.

This module deliberately knows nothing about IsaacLab or Qwen.  It owns the
boundary between them: a time-sampled RGB ring, an immutable on-disk request
packet, a strict JSONL subprocess client, and post-result transition evidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import select
import string
import subprocess
from typing import Mapping, Sequence

from PIL import Image


REQUEST_SCHEMA = "stage_policy_live_vlm_request_v1"
MULTIVIEW_REQUEST_SCHEMA = "stage_policy_live_vlm_multiview_request_v1"
REQUEST_SCHEMAS = (REQUEST_SCHEMA, MULTIVIEW_REQUEST_SCHEMA)
COMMAND_SCHEMA = "stage_policy_live_vlm_command_v1"
RESULT_SCHEMA = "stage_policy_live_vlm_result_v1"
BATCH_RESULT_SCHEMA = "stage_policy_live_vlm_batch_result_v1"
READY_SCHEMA = "stage_policy_live_vlm_ready_v1"
LOG_SCHEMA = "stage_policy_live_vlm_proof_v1"
WORKER_MANIFEST_SCHEMA = "stage_policy_live_vlm_worker_manifest_v1"
LIVE_RING_TARGET_PERIOD_S = 0.05
LIVE_REQUIRED_HOLDS = (3, 6, 10, 10)

LIVE_STAGE_INSTRUCTIONS = (
    "State has verified exact pre-grasp geometry. Audit only that this is the intended Franka scene and the correct small colored cube is present; do not judge visible alignment or require contact or lift. Reject only a wrong/absent object or a clear scene mismatch.",
    "State has verified capture and lift initiation; audit only that there is no obvious empty grasp and the small colored cube is not visibly left behind on the table.",
    "State has verified exact lift height and control; audit only that the small colored cube moves with the gripper with no obvious drop, throw, or separation.",
    "State has verified exact goal and control thresholds; audit only that the controlled small colored cube is at the intended XYZ target marker with no obvious wrong-target transport or drop.",
)

TOKEN_FIELDS = (
    "env_id",
    "episode_id",
    "stage_id",
    "candidate_epoch",
    "request_id",
    "request_step",
    "candidate_start_step",
)


class LiveVlmProtocolError(RuntimeError):
    """Raised when a worker violates the one-line JSON protocol."""


def canonical_json_digest(value: object) -> str:
    """Hash one JSON value using the protocol's canonical serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(payload.encode("utf-8")).hexdigest()


LIVE_WORKER_MANIFEST: dict[str, object] = {
    "schema": WORKER_MANIFEST_SCHEMA,
    "protocol_command_schema": COMMAND_SCHEMA,
    "request_schemas": list(REQUEST_SCHEMAS),
    "single_result_schema": RESULT_SCHEMA,
    "batch_result_schema": BATCH_RESULT_SCHEMA,
    "supported_commands": ["infer", "infer_batch", "shutdown"],
    "batch_execution": "one_processor_apply_chat_template_plus_one_model_generate",
    "batch_result_order": "request_order",
}
LIVE_WORKER_MANIFEST_DIGEST = canonical_json_digest(LIVE_WORKER_MANIFEST)


def _normalized_digest(value: str, *, field_name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in string.hexdigits.lower() for character in digest):
        raise ValueError(f"{field_name} must be a 64-character fixed-length digest hex digest")
    return digest


@dataclass(frozen=True)
class WorkerProvenanceExpectations:
    """Independent values that a mandatory worker must prove before serving."""

    model_path: str | Path
    prompt_template_digest: str
    worker_script_digest: str
    worker_manifest_digest: str

    def __post_init__(self) -> None:
        if not str(self.model_path):
            raise ValueError("model_path must be non-empty")
        object.__setattr__(self, "model_path", str(Path(self.model_path).expanduser().resolve()))
        for field_name in (
            "prompt_template_digest",
            "worker_script_digest",
            "worker_manifest_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalized_digest(getattr(self, field_name), field_name=field_name),
            )


def validate_worker_ready_record(
    ready: Mapping[str, object],
    expected: WorkerProvenanceExpectations,
) -> dict[str, object]:
    """Fail closed unless the loaded worker exactly matches independent provenance."""
    if ready.get("schema") != READY_SCHEMA or ready.get("status") != "ready":
        raise LiveVlmProtocolError(f"worker did not send a valid ready record: {ready}")
    if ready.get("model_load_count") != 1:
        raise LiveVlmProtocolError("worker must report exactly one model load")
    actual_model_path = ready.get("model_path")
    if not isinstance(actual_model_path, str):
        raise LiveVlmProtocolError("worker ready record is missing model_path")
    if str(Path(actual_model_path).expanduser().resolve()) != expected.model_path:
        raise LiveVlmProtocolError("worker model_path does not match the expected model")
    for ready_name, expected_value in (
        ("prompt_template_digest", expected.prompt_template_digest),
        ("worker_script_digest", expected.worker_script_digest),
    ):
        actual = ready.get(ready_name)
        if not isinstance(actual, str) or actual.lower() != expected_value:
            raise LiveVlmProtocolError(f"worker {ready_name} does not match the expected digest")
    manifest = ready.get("worker_manifest")
    if not isinstance(manifest, Mapping):
        raise LiveVlmProtocolError("worker ready record is missing worker_manifest")
    manifest = dict(manifest)
    computed_manifest_sha = canonical_json_digest(manifest)
    reported_manifest_sha = ready.get("worker_manifest_digest")
    if not isinstance(reported_manifest_sha, str) or reported_manifest_sha.lower() != computed_manifest_sha:
        raise LiveVlmProtocolError("worker manifest digest does not match its manifest")
    if computed_manifest_sha != expected.worker_manifest_digest:
        raise LiveVlmProtocolError("worker manifest does not match the expected manifest")
    if manifest.get("schema") != WORKER_MANIFEST_SCHEMA:
        raise LiveVlmProtocolError("worker manifest has an unknown schema")
    commands = manifest.get("supported_commands")
    if not isinstance(commands, list) or not {"infer", "infer_batch", "shutdown"}.issubset(commands):
        raise LiveVlmProtocolError("worker manifest does not advertise required commands")
    request_schemas = manifest.get("request_schemas")
    if (
        not isinstance(request_schemas, list)
        or not set(REQUEST_SCHEMAS).issubset(request_schemas)
    ):
        raise LiveVlmProtocolError("worker manifest omits a required packet schema")
    if manifest.get("batch_execution") != "one_processor_apply_chat_template_plus_one_model_generate":
        raise LiveVlmProtocolError("worker manifest does not advertise true model batching")
    return dict(ready)


def _validate_worker_script_in_command(
    command: Sequence[str], expected_digest: str
) -> str:
    """Hash the actual Python script passed to the subprocess command."""
    candidates: list[Path] = []
    for value in command:
        path = Path(value).expanduser()
        if path.suffix == ".py" and path.is_file():
            candidates.append(path.resolve())
    if not candidates:
        raise LiveVlmProtocolError("worker command has no readable Python script to attest")
    matches = [
        path
        for path in candidates
        if hashlib.blake2b(path.read_bytes()).hexdigest() == expected_digest
    ]
    if len(matches) != 1:
        raise LiveVlmProtocolError("worker command script fixed-length digest does not match expectation")
    return str(matches[0])


@dataclass
class EpisodeServiceSelector:
    """Select either one exact episode or the first bounded set of episodes."""

    max_episodes: int = 1
    exact_episode_id: int | None = None
    selected_episode_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        if self.exact_episode_id is not None and self.exact_episode_id < 0:
            raise ValueError("exact_episode_id must be non-negative")

    @property
    def selection_policy(self) -> str:
        return "exact_episode_id" if self.exact_episode_id is not None else "first_distinct_episodes"

    def observe(self, episode_id: int) -> bool:
        episode_id = int(episode_id)
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        if self.exact_episode_id is not None:
            selected = episode_id == self.exact_episode_id
        else:
            selected = episode_id in self.selected_episode_ids or len(self.selected_episode_ids) < self.max_episodes
        if selected and episode_id not in self.selected_episode_ids:
            self.selected_episode_ids.append(episode_id)
        return selected

    def is_selected(self, episode_id: int) -> bool:
        return int(episode_id) in self.selected_episode_ids


def load_prompt_template(
    prompt_template_file: str | Path | None, *, default_template: str
) -> tuple[str, str | None, str]:
    """Load and validate a worker prompt, returning text, absolute path, and fixed-length digest."""
    if prompt_template_file is None:
        text = str(default_template)
        absolute_path = None
    else:
        path = Path(prompt_template_file).expanduser().resolve()
        text = path.read_text(encoding="utf-8")
        absolute_path = str(path)
    fields = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(text)
        if field_name is not None
    }
    if "instruction" not in fields:
        raise ValueError("prompt template must contain {instruction}")
    unknown = fields - {"instruction", "image_context"}
    if unknown:
        raise ValueError(f"unsupported prompt template fields: {sorted(unknown)}")
    # Exercise formatting at startup, before the model is loaded or a simulator freezes.
    text.format(instruction="test instruction", image_context="test image context")
    digest = hashlib.blake2b(text.encode("utf-8")).hexdigest()
    return text, absolute_path, digest


def normalize_token(token: object) -> dict[str, int]:
    """Return the exact integer identity fields accepted by the gate protocol."""
    result: dict[str, int] = {}
    for name in TOKEN_FIELDS:
        if isinstance(token, Mapping):
            if name not in token:
                raise LiveVlmProtocolError(f"token is missing {name}")
            value = token[name]
        else:
            if not hasattr(token, name):
                raise LiveVlmProtocolError(f"token is missing {name}")
            value = getattr(token, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise LiveVlmProtocolError(f"token field {name} must be an integer")
        result[name] = int(value)
    extra = set(token) - set(TOKEN_FIELDS) if isinstance(token, Mapping) else set()
    if extra:
        raise LiveVlmProtocolError(f"token has unexpected fields: {sorted(extra)}")
    return result


@dataclass(frozen=True)
class FrozenRgbFrame:
    """One immutable RGB24 camera frame copied out of the simulator."""

    rgb: bytes
    width: int
    height: int
    global_step: int
    sim_time_s: float
    source_frame_id: int = -1

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        expected = self.width * self.height * 3
        if len(self.rgb) != expected:
            raise ValueError(f"RGB24 frame has {len(self.rgb)} bytes; expected {expected}")


@dataclass(frozen=True)
class FrozenViewEvidence:
    """Immutable named evidence for one fixed, always-served camera view."""

    view_name: str
    camera_name: str
    reference: FrozenRgbFrame
    ring_frames: Sequence[FrozenRgbFrame]
    current: FrozenRgbFrame
    ring_nominal_fps: float
    source_update_period_s: float
    target_samples_due_since_reset: int
    pixel_identical_duplicates_skipped_since_reset: int

    def __post_init__(self) -> None:
        if not self.view_name or any(
            not (character.isalnum() or character in "_-")
            for character in self.view_name
        ):
            raise ValueError("view_name must contain only letters, digits, '_' or '-'")
        if not self.camera_name:
            raise ValueError("camera_name must be non-empty")
        ring_frames = tuple(self.ring_frames)
        if not ring_frames:
            raise ValueError("view evidence requires a non-empty frame ring")
        object.__setattr__(self, "ring_frames", ring_frames)
        if self.ring_nominal_fps <= 0.0 or self.source_update_period_s <= 0.0:
            raise ValueError("view frame rates must be positive")
        if (
            self.target_samples_due_since_reset < 0
            or self.pixel_identical_duplicates_skipped_since_reset < 0
        ):
            raise ValueError("view sample counters must be non-negative")
        dimensions = {
            (frame.width, frame.height)
            for frame in (*ring_frames, self.reference, self.current)
        }
        if len(dimensions) != 1:
            raise ValueError("all frames within one view must have identical dimensions")


class RgbFrameRing:
    """Select a target-rate ring from a faster source and drop pixel duplicates."""

    def __init__(self, *, frame_count: int = 21, capture_period_s: float = 0.05):
        if frame_count < 2:
            raise ValueError("frame_count must be at least two")
        if capture_period_s <= 0.0:
            raise ValueError("capture_period_s must be positive")
        self.frame_count = int(frame_count)
        self.capture_period_s = float(capture_period_s)
        self._frames: deque[FrozenRgbFrame] = deque(maxlen=self.frame_count)
        self._next_capture_time_s: float | None = None
        self.target_samples_due_since_reset = 0
        self.pixel_identical_duplicates_skipped_since_reset = 0
        self.distinct_frames_appended_since_reset = 0

    @property
    def nominal_fps(self) -> float:
        return 1.0 / self.capture_period_s

    @property
    def nominal_history_s(self) -> float:
        return (self.frame_count - 1) * self.capture_period_s

    @property
    def actual_history_s(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return float(self._frames[-1].sim_time_s - self._frames[0].sim_time_s)

    @property
    def frames(self) -> tuple[FrozenRgbFrame, ...]:
        return tuple(self._frames)

    def reset(self, seed: FrozenRgbFrame) -> None:
        """Start a new episode and make its first rendered frame the reference seed."""
        self._frames.clear()
        self._frames.append(seed)
        self._next_capture_time_s = seed.sim_time_s + self.capture_period_s
        self.target_samples_due_since_reset = 0
        self.pixel_identical_duplicates_skipped_since_reset = 0
        self.distinct_frames_appended_since_reset = 1

    def maybe_append(self, frame: FrozenRgbFrame) -> bool:
        """Append one due target sample only when its RGB differs from the last append."""
        if not self._frames or self._next_capture_time_s is None:
            self.reset(frame)
            return True
        if frame.width != self._frames[-1].width or frame.height != self._frames[-1].height:
            raise ValueError("camera dimensions changed within one frame ring")
        epsilon = max(1.0e-9, self.capture_period_s * 1.0e-6)
        if frame.sim_time_s + epsilon < self._next_capture_time_s:
            return False
        self.target_samples_due_since_reset += 1
        while self._next_capture_time_s <= frame.sim_time_s + epsilon:
            self._next_capture_time_s += self.capture_period_s
        if frame.rgb == self._frames[-1].rgb:
            self.pixel_identical_duplicates_skipped_since_reset += 1
            return False
        self._frames.append(frame)
        self.distinct_frames_appended_since_reset += 1
        return True


def _write_rgb_png(path: Path, frame: FrozenRgbFrame) -> None:
    """Write the exact RGB24 pixels using lossless PNG compression.

    The Qwen worker loads packet images through Pillow, so this changes only
    the on-disk representation: decoding the PNG yields the same RGB bytes as
    the former uncompressed PPM. Keeping every training request auditable is
    otherwise prohibitively large for long mandatory-gate sweeps.
    """
    image = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)
    image.save(path, format="PNG", compress_level=6, optimize=False)


def _store_rgb_png(root: Path, frame: FrozenRgbFrame) -> Path:
    """Store a lossless frame once and return its content-addressed path."""
    identity = f"{frame.width}x{frame.height}:".encode("ascii") + frame.rgb
    digest = hashlib.blake2b(identity).hexdigest()
    frame_root = root / "_frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    path = frame_root / f"{digest}.png"
    if not path.exists():
        _write_rgb_png(path, frame)
    return path


def freeze_request_packet(
    root: str | Path,
    *,
    token: object,
    stage_name: str,
    camera_name: str,
    instruction: str,
    reference: FrozenRgbFrame,
    ring_frames: Sequence[FrozenRgbFrame],
    current: FrozenRgbFrame,
    ring_nominal_fps: float,
    source_update_period_s: float,
    target_samples_due_since_reset: int,
    pixel_identical_duplicates_skipped_since_reset: int,
) -> dict[str, object]:
    """Write reference + ordered ring + current frame and return the JSON packet."""
    token_payload = normalize_token(token)
    if not ring_frames:
        raise ValueError("cannot freeze an empty frame ring")
    if ring_nominal_fps <= 0.0:
        raise ValueError("ring_nominal_fps must be positive")
    if source_update_period_s <= 0.0:
        raise ValueError("source_update_period_s must be positive")
    if target_samples_due_since_reset < 0 or pixel_identical_duplicates_skipped_since_reset < 0:
        raise ValueError("ring sample counters must be non-negative")
    dimensions = {(frame.width, frame.height) for frame in (*ring_frames, reference, current)}
    if len(dimensions) != 1:
        raise ValueError("all packet frames must have identical dimensions")

    request_dir = Path(root).expanduser().resolve() / (
        f"request_{token_payload['request_id']:06d}_env{token_payload['env_id']}_"
        f"ep{token_payload['episode_id']}_stage{token_payload['stage_id']}_"
        f"epoch{token_payload['candidate_epoch']}"
    )
    request_dir.mkdir(parents=True, exist_ok=False)
    packet_root = Path(root).expanduser().resolve()
    reference_path = _store_rgb_png(packet_root, reference)
    current_path = _store_rgb_png(packet_root, current)
    ring_paths = [
        str(_store_rgb_png(packet_root, frame)) for frame in ring_frames
    ]

    packet: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "image_storage": "lossless_png_digest_content_addressed",
        "token": token_payload,
        "stage_name": str(stage_name),
        "instruction": str(instruction),
        "camera_name": str(camera_name),
        "reference_image": str(reference_path),
        "ring_images": ring_paths,
        "current_image": str(current_path),
        "ring_capture_global_steps": [int(frame.global_step) for frame in ring_frames],
        "ring_capture_sim_time_s": [float(frame.sim_time_s) for frame in ring_frames],
        "ring_source_frame_ids": [int(frame.source_frame_id) for frame in ring_frames],
        "camera_source_update_period_s": float(source_update_period_s),
        "camera_source_nominal_fps": 1.0 / float(source_update_period_s),
        "ring_target_capture_period_s": 1.0 / float(ring_nominal_fps),
        "ring_target_nominal_fps": float(ring_nominal_fps),
        # Compatibility aliases: fps is the target; count/history are actual.
        "ring_nominal_fps": float(ring_nominal_fps),
        "ring_frame_count": len(ring_frames),
        "ring_history_s": float(ring_frames[-1].sim_time_s - ring_frames[0].sim_time_s),
        "ring_actual_distinct_frame_count": len(ring_frames),
        "ring_actual_history_s": float(ring_frames[-1].sim_time_s - ring_frames[0].sim_time_s),
        "ring_target_samples_due_since_episode_reset": int(target_samples_due_since_reset),
        "ring_pixel_identical_duplicates_skipped_since_episode_reset": int(
            pixel_identical_duplicates_skipped_since_reset
        ),
        "reference_global_step": int(reference.global_step),
        "reference_sim_time_s": float(reference.sim_time_s),
        "reference_source_frame_id": int(reference.source_frame_id),
        "current_global_step": int(current.global_step),
        "current_sim_time_s": float(current.sim_time_s),
        "current_source_frame_id": int(current.source_frame_id),
        "width": int(current.width),
        "height": int(current.height),
    }
    packet_path = request_dir / "request.json"
    packet["packet_path"] = str(packet_path)
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet


def freeze_multiview_request_packet(
    root: str | Path,
    *,
    token: object,
    stage_name: str,
    instruction: str,
    views: Sequence[FrozenViewEvidence],
) -> dict[str, object]:
    """Freeze every declared camera view into one immutable candidate packet."""
    token_payload = normalize_token(token)
    frozen_views = tuple(views)
    if not frozen_views:
        raise ValueError("multiview packet requires at least one view")
    view_names = [view.view_name for view in frozen_views]
    if len(set(view_names)) != len(view_names):
        raise ValueError("multiview packet contains duplicate view names")

    request_dir = Path(root).expanduser().resolve() / (
        f"request_{token_payload['request_id']:06d}_env{token_payload['env_id']}_"
        f"ep{token_payload['episode_id']}_stage{token_payload['stage_id']}_"
        f"epoch{token_payload['candidate_epoch']}"
    )
    request_dir.mkdir(parents=True, exist_ok=False)
    view_payloads: list[dict[str, object]] = []
    for view in frozen_views:
        packet_root = Path(root).expanduser().resolve()
        reference_path = _store_rgb_png(packet_root, view.reference)
        current_path = _store_rgb_png(packet_root, view.current)
        ring_paths = [
            str(_store_rgb_png(packet_root, frame))
            for frame in view.ring_frames
        ]
        history_s = float(
            view.ring_frames[-1].sim_time_s - view.ring_frames[0].sim_time_s
        )
        view_payloads.append(
            {
                "view_name": view.view_name,
                "camera_name": view.camera_name,
                "reference_image": str(reference_path),
                "ring_images": ring_paths,
                "current_image": str(current_path),
                "ring_capture_global_steps": [
                    int(frame.global_step) for frame in view.ring_frames
                ],
                "ring_capture_sim_time_s": [
                    float(frame.sim_time_s) for frame in view.ring_frames
                ],
                "ring_source_frame_ids": [
                    int(frame.source_frame_id) for frame in view.ring_frames
                ],
                "camera_source_update_period_s": float(
                    view.source_update_period_s
                ),
                "camera_source_nominal_fps": (
                    1.0 / float(view.source_update_period_s)
                ),
                "ring_target_capture_period_s": (
                    1.0 / float(view.ring_nominal_fps)
                ),
                "ring_target_nominal_fps": float(view.ring_nominal_fps),
                "ring_frame_count": len(view.ring_frames),
                "ring_actual_distinct_frame_count": len(view.ring_frames),
                "ring_actual_history_s": history_s,
                "ring_target_samples_due_since_episode_reset": int(
                    view.target_samples_due_since_reset
                ),
                "ring_pixel_identical_duplicates_skipped_since_episode_reset": int(
                    view.pixel_identical_duplicates_skipped_since_reset
                ),
                "reference_global_step": int(view.reference.global_step),
                "reference_sim_time_s": float(view.reference.sim_time_s),
                "reference_source_frame_id": int(view.reference.source_frame_id),
                "current_global_step": int(view.current.global_step),
                "current_sim_time_s": float(view.current.sim_time_s),
                "current_source_frame_id": int(view.current.source_frame_id),
                "width": int(view.current.width),
                "height": int(view.current.height),
            }
        )

    packet: dict[str, object] = {
        "schema": MULTIVIEW_REQUEST_SCHEMA,
        "image_storage": "lossless_png_digest_content_addressed",
        "token": token_payload,
        "stage_name": str(stage_name),
        "instruction": str(instruction),
        "view_policy": "fixed_all_views_in_declared_order_no_state_selection_no_crop",
        "view_order": view_names,
        "view_count": len(view_payloads),
        "views": view_payloads,
    }
    packet_path = request_dir / "request.json"
    packet["packet_path"] = str(packet_path)
    packet_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return packet


def validate_worker_result(
    result: Mapping[str, object], expected_token: object
) -> tuple[dict[str, int], str]:
    """Validate schema, exact token round-trip, and the closed prediction vocabulary."""
    if result.get("schema") != RESULT_SCHEMA:
        raise LiveVlmProtocolError(f"unexpected result schema: {result.get('schema')!r}")
    returned_token = normalize_token(result.get("token", {}))
    expected = normalize_token(expected_token)
    if returned_token != expected:
        mismatches = [name for name in TOKEN_FIELDS if returned_token[name] != expected[name]]
        raise LiveVlmProtocolError(f"worker token mismatch: {mismatches}")
    prediction = result.get("prediction")
    if prediction not in ("success", "failure", "unknown"):
        raise LiveVlmProtocolError(f"invalid worker prediction: {prediction!r}")
    return returned_token, str(prediction)


def should_issue_live_request(
    *,
    request_armed: bool,
    strict_candidate: bool,
    stage_id: int,
    stable_count: int,
    episode_eligible: bool,
    requests_issued: int,
    max_requests: int,
    required_holds: Sequence[int] = LIVE_REQUIRED_HOLDS,
) -> bool:
    """Issue only at the penultimate strict-dwell step for the active stage.

    The core may arm on the first strict sample, but a short strict spike must
    not consume the one expensive request for its candidate epoch. Freezing at
    ``required_hold - 1`` means one further strict control step is still needed
    after a positive VLM result; physical transition logic remains authoritative.
    """
    if stage_id == len(required_holds):
        return False
    if stage_id < 0 or stage_id > len(required_holds):
        raise ValueError("stage_id is outside required_holds")
    if requests_issued < 0 or max_requests < 0 or stable_count < 0:
        raise ValueError("request counts must be non-negative")
    required_hold = int(required_holds[stage_id])
    if required_hold < 1:
        raise ValueError("required holds must be positive")
    return bool(
        request_armed
        and strict_candidate
        and stable_count >= required_hold - 1
        and episode_eligible
        and requests_issued < max_requests
    )


class JsonlWorkerClient:
    """One persistent subprocess with one request and one response per JSON line."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        startup_timeout_s: float = 600.0,
        expected_provenance: WorkerProvenanceExpectations | None = None,
    ):
        if not command or any(not isinstance(value, str) or not value for value in command):
            raise ValueError("worker command must be a non-empty sequence of strings")
        self.command = tuple(command)
        self.attested_worker_script_path = None
        if expected_provenance is not None:
            self.attested_worker_script_path = _validate_worker_script_in_command(
                self.command, expected_provenance.worker_script_digest
            )
        self._process = subprocess.Popen(
            list(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            ready = self._read_json_line(startup_timeout_s)
            if ready.get("schema") != READY_SCHEMA or ready.get("status") != "ready":
                raise LiveVlmProtocolError(
                    f"worker did not send a valid ready record: {ready}"
                )
            if expected_provenance is not None:
                ready = validate_worker_ready_record(ready, expected_provenance)
        except Exception:
            self.close(force=True)
            raise
        self.ready_record = ready

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    def _read_json_line(self, timeout_s: float) -> dict[str, object]:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        stdout = self._process.stdout
        if stdout is None:
            raise LiveVlmProtocolError("worker stdout is unavailable")
        ready, _, _ = select.select([stdout.fileno()], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"worker did not respond within {timeout_s:.3f}s")
        line = stdout.readline()
        if not line:
            code = self._process.poll()
            raise LiveVlmProtocolError(f"worker stdout closed unexpectedly (returncode={code})")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise LiveVlmProtocolError(f"worker emitted non-JSON stdout: {line[:200]!r}") from error
        if not isinstance(value, dict):
            raise LiveVlmProtocolError("worker JSON line must be an object")
        return value

    def request(self, packet: Mapping[str, object], *, timeout_s: float = 600.0) -> dict[str, object]:
        if packet.get("schema") not in REQUEST_SCHEMAS:
            raise ValueError("cannot send a packet with an unknown schema")
        stdin = self._process.stdin
        if stdin is None or self._process.poll() is not None:
            raise LiveVlmProtocolError("worker is not running")
        command = {"schema": COMMAND_SCHEMA, "type": "infer", "packet": dict(packet)}
        stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        stdin.flush()
        return self._read_json_line(timeout_s)


    def request_batch(
        self,
        packets: Sequence[Mapping[str, object]],
        *,
        batch_id: int,
        timeout_s: float = 600.0,
    ) -> dict[str, object]:
        """Run one true worker microbatch and validate its ordered cardinality."""
        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 0:
            raise ValueError("batch_id must be a non-negative integer")
        if not packets:
            raise ValueError("packets must be non-empty")
        packet_copies: list[dict[str, object]] = []
        expected_tokens: list[dict[str, int]] = []
        for packet in packets:
            if packet.get("schema") not in REQUEST_SCHEMAS:
                raise ValueError("cannot send a packet with an unknown schema")
            packet_copies.append(dict(packet))
            expected_tokens.append(normalize_token(packet.get("token", {})))
        token_keys = [tuple(token[name] for name in TOKEN_FIELDS) for token in expected_tokens]
        if len(set(token_keys)) != len(token_keys):
            raise ValueError("a worker batch cannot contain duplicate candidate tokens")
        stdin = self._process.stdin
        if stdin is None or self._process.poll() is not None:
            raise LiveVlmProtocolError("worker is not running")
        command = {
            "schema": COMMAND_SCHEMA,
            "type": "infer_batch",
            "batch_id": batch_id,
            "packets": packet_copies,
        }
        stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        stdin.flush()
        response = self._read_json_line(timeout_s)
        if response.get("schema") != BATCH_RESULT_SCHEMA:
            raise LiveVlmProtocolError(
                f"unexpected batch result schema: {response.get('schema')!r}"
            )
        if response.get("batch_id") != batch_id:
            raise LiveVlmProtocolError("worker batch_id does not match the request")
        if response.get("error") or response.get("error_type"):
            raise LiveVlmProtocolError(
                f"worker batch error: {response.get('error_type')}: {response.get('error')}"
            )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(expected_tokens):
            raise LiveVlmProtocolError("worker batch result cardinality mismatch")
        if response.get("batch_size") != len(expected_tokens):
            raise LiveVlmProtocolError("worker batch_size does not match the request")
        if response.get("processor_calls") != 1 or response.get("model_generate_calls") != 1:
            raise LiveVlmProtocolError("worker did not prove one real processor/model batch call")
        for result, expected_token in zip(results, expected_tokens):
            if not isinstance(result, Mapping):
                raise LiveVlmProtocolError("worker batch contains a non-object result")
            validate_worker_result(result, expected_token)
        return response

    def close(self, *, force: bool = False) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        if not force and process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps({"schema": COMMAND_SCHEMA, "type": "shutdown"}) + "\n"
                )
                process.stdin.flush()
                self._read_json_line(5.0)
            except (BrokenPipeError, LiveVlmProtocolError, TimeoutError):
                force = True
        if force and process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()


@dataclass
class GateTransitionMonitor:
    """Tie one accepted/rejected visual result to later physical state updates."""

    token: dict[str, int]
    prediction: str
    apply_accepted: bool
    apply_reason: str
    max_post_result_steps: int = 32
    observations: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.token = normalize_token(self.token)
        if self.prediction not in ("success", "failure", "unknown"):
            raise ValueError("invalid prediction")
        if self.max_post_result_steps < 1:
            raise ValueError("max_post_result_steps must be positive")

    def observe(
        self,
        *,
        global_step: int,
        episode_id: int,
        stage_after_step: int,
        strict_candidate: bool,
        stable_count_before_step: int,
        stable_count_after_step: int,
        visual_confirmation_before_step: bool,
        loose_pre_gate_after_step: bool,
    ) -> dict[str, object] | None:
        expected_stage = self.token["stage_id"]
        expected_episode = self.token["episode_id"]
        transitioned = episode_id == expected_episode and stage_after_step == expected_stage + 1
        dwell_after_candidate = stable_count_before_step + 1 if strict_candidate else 0
        observation = {
            "global_step": int(global_step),
            "episode_id": int(episode_id),
            "stage_after_step": int(stage_after_step),
            "strict_candidate": bool(strict_candidate),
            "stable_count_before_step": int(stable_count_before_step),
            "stable_count_after_step": int(stable_count_after_step),
            "strict_dwell_after_candidate": int(dwell_after_candidate),
            "visual_confirmation_before_step": bool(visual_confirmation_before_step),
            "loose_pre_gate_after_step": bool(loose_pre_gate_after_step),
            "actual_stage_transition": bool(transitioned),
        }
        self.observations.append(observation)

        reason: str | None = None
        if episode_id != expected_episode:
            reason = "episode_reset_before_transition"
        elif transitioned:
            reason = "strict_candidate_dwell_transition"
        elif stage_after_step != expected_stage:
            reason = "unexpected_stage_change"
        elif not self.apply_accepted:
            reason = "vlm_result_not_applied"
        elif self.prediction != "success":
            reason = "vlm_not_confirmed_gate_blocked"
        elif not loose_pre_gate_after_step:
            reason = "candidate_window_broke_before_transition"
        elif len(self.observations) >= self.max_post_result_steps:
            reason = "transition_observation_limit"
        if reason is None:
            return None
        return self.finalize(reason)

    def finalize(self, reason: str) -> dict[str, object]:
        transitioned = any(bool(row["actual_stage_transition"]) for row in self.observations)
        transition_step = next(
            (int(row["global_step"]) for row in self.observations if row["actual_stage_transition"]),
            None,
        )
        return {
            "event": "vlm_gate_transition_outcome",
            "token": dict(self.token),
            "prediction": self.prediction,
            "apply_accepted": bool(self.apply_accepted),
            "apply_reason": self.apply_reason,
            "resolution_reason": str(reason),
            "actual_stage_transition": transitioned,
            "actual_transition_global_step": transition_step,
            "post_result_observations": list(self.observations),
        }


class JsonlProofLog:
    """Flush every proof event so a failed simulator run still leaves evidence."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        self.records_written = 0

    def write(self, record: Mapping[str, object]) -> None:
        self._handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        self._handle.flush()
        self.records_written += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

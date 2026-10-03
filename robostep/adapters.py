"""Framework-neutral adapter hooks for benchmark environments.

Adapters keep benchmark-native observations/actions and expose only the small
state API required by the reward compiler.  This is where a local checkout of
Meta-World, ManiSkill, or SoftGym should be connected; no benchmark import is
required to use the core package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .gates import GateVLM
from .reward import ActiveStageReward, RewardStep
from .schema import RewardProgram
from .vlm import FramePacket, GateRequest


class BenchmarkAdapter(Protocol):
    task_id: str
    benchmark: str

    def reset(self, seed: int | None = None) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, Mapping[str, Any]]: ...
    def reward_inputs(self, observation: np.ndarray, info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def official_success(self, observation: np.ndarray, info: Mapping[str, Any]) -> bool: ...
    def frame_packet(self, observation: np.ndarray, info: Mapping[str, Any], *, reference: bool = False) -> Any: ...
    def close(self) -> None: ...


def normalize_reset(result: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Accept Gymnasium `(obs, info)` and legacy Gym `obs` reset returns."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return np.asarray(result[0]), result[1]
    return np.asarray(result), {}


def normalize_step(result: Any) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
    """Accept both four- and five-element Gym step APIs."""
    if not isinstance(result, tuple):
        raise TypeError("environment step must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return np.asarray(observation), float(reward), bool(terminated), bool(truncated), info
    if len(result) == 4:
        observation, reward, done, info = result
        return np.asarray(observation), float(reward), bool(done), False, info
    raise ValueError(f"unsupported step return length: {len(result)}")


@dataclass
class CallableBenchmarkAdapter:
    """Minimal adapter assembled from benchmark-specific callbacks.

    `reward_inputs_fn` is the only task-specific part: it maps native state/API
    data to `(potentials[K], candidates[K], maintenance[K], safety)`.  This
    keeps the benchmark's official success API separate from reward shaping.
    """

    benchmark: str
    task_id: str
    reset_fn: Callable[[int | None], Any]
    step_fn: Callable[[np.ndarray], Any]
    reward_inputs_fn: Callable[[np.ndarray, Mapping[str, Any]], tuple[Any, Any, Any, Any]]
    success_fn: Callable[[np.ndarray, Mapping[str, Any]], bool]
    frame_fn: Callable[[np.ndarray, Mapping[str, Any], bool], Any] | None = None
    close_fn: Callable[[], None] | None = None
    _last_info: Mapping[str, Any] = field(default_factory=dict)

    def reset(self, seed: int | None = None) -> np.ndarray:
        observation, info = normalize_reset(self.reset_fn(seed))
        self._last_info = info
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, Mapping[str, Any]]:
        observation, _, terminated, truncated, info = normalize_step(self.step_fn(action))
        merged = dict(info)
        merged["terminated"] = terminated
        merged["truncated"] = truncated
        self._last_info = merged
        return observation, merged

    def reward_inputs(self, observation: np.ndarray, info: Mapping[str, Any]):
        return self.reward_inputs_fn(observation, info)

    def official_success(self, observation: np.ndarray, info: Mapping[str, Any]) -> bool:
        return bool(self.success_fn(observation, info))

    def frame_packet(self, observation: np.ndarray, info: Mapping[str, Any], *, reference: bool = False) -> Any:
        if self.frame_fn is None:
            return None
        return self.frame_fn(observation, info, reference)

    def close(self) -> None:
        if self.close_fn is not None:
            self.close_fn()


@dataclass
class EpisodeTrace:
    task_id: str
    seed: int | None
    rewards: list[float] = field(default_factory=list)
    stages: list[int] = field(default_factory=list)
    transitions: list[int] = field(default_factory=list)
    official_success: bool = False
    terminated: bool = False
    truncated: bool = False

    @property
    def return_sum(self) -> float:
        return float(np.sum(self.rewards))


class StageAwareRunner:
    """Glue a native adapter to `ActiveStageReward` and an optional GateVLM."""

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        program: RewardProgram,
        *,
        gate_vlm: GateVLM | None = None,
        reward_mode: str = "staged_reset",
        task_description: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.program = program
        self.gate_vlm = gate_vlm
        self.reward_mode = reward_mode
        self.task_description = task_description or program.task_description
        self.engine = ActiveStageReward(program, batch_size=1)
        self.episode_id = 0
        self.step_index = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        observation = self.adapter.reset(seed)
        potentials, _, _, _ = self.adapter.reward_inputs(observation, {})
        self.engine.reset(np.asarray(potentials, dtype=np.float64).reshape(1, -1))
        self.episode_id += 1
        self.step_index = 0
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, Mapping[str, Any]]:
        observation, info = self.adapter.step(action)
        potentials, candidates, maintenance, safety = self.adapter.reward_inputs(observation, info)
        potentials = np.asarray(potentials, dtype=np.float64).reshape(1, -1)
        candidates = np.asarray(candidates, dtype=bool).reshape(1, -1)
        maintenance = np.asarray(maintenance, dtype=np.float64).reshape(1, -1)
        safety = np.asarray(safety, dtype=np.float64).reshape(1)
        authorization = None
        if self.gate_vlm is not None:
            stage_index = int(self.engine.stage[0])
            candidate = bool(self.engine.candidate_mask(candidates)[0])
            contract_stage = min(stage_index, self.program.num_stages - 1)
            request_id = f"{self.episode_id}:{stage_index}:{self.step_index}"
            request = GateRequest(
                request_id=request_id,
                task_id=self.adapter.task_id,
                episode_id=self.episode_id,
                stage_index=stage_index,
                request_step=self.step_index,
                task_description=self.task_description,
                stage_contract=asdict(self.program.stages[contract_stage]),
                reference_frames=_frame_tuple(_adapter_frame_packet(self.adapter, observation, info, reference=True)) if candidate else (),
                candidate_frames=_frame_tuple(_adapter_frame_packet(self.adapter, observation, info, reference=False)) if candidate else (),
            )
            self.gate_vlm.request_if_candidate(request, candidate)
            decisions = self.gate_vlm.drain(episode_id=self.episode_id, stage_index=int(self.engine.stage[0]), request_step=self.step_index)
            if request_id in decisions:
                authorization = np.array([decisions[request_id]], dtype=bool)
        success = bool(self.adapter.official_success(observation, info))
        if self.gate_vlm is not None and authorization is None:
            # A pending/abstaining VLM must block the transition; it must never
            # silently fall back to the privileged Rule gate.
            authorization = np.array([False], dtype=bool)
        result = self.engine.step(potentials, candidates, maintenance=maintenance, safety_penalty=safety, terminal_success=np.array([success]), transition_authorization=authorization, reward_mode=self.reward_mode, transition_authority="rule_candidate" if self.gate_vlm is None else "rule_candidate_and_external_authorization")
        done = bool(info.get("terminated", False) or info.get("truncated", False) or success or self.engine.stage[0] >= self.program.num_stages)
        diagnostics = result.as_dict()
        diagnostics.update({"official_success": success, "episode_id": self.episode_id, "step": self.step_index})
        self.step_index += 1
        return observation, float(result.reward[0]), done, diagnostics

    def run_scripted(self, actions: list[np.ndarray], *, seed: int | None = None) -> EpisodeTrace:
        self.reset(seed)
        trace = EpisodeTrace(self.adapter.task_id, seed)
        for action in actions:
            _, reward, done, diagnostics = self.step(action)
            trace.rewards.append(reward)
            trace.stages.append(int(diagnostics["stage_after"][0]))
            trace.transitions.append(int(diagnostics["transition"][0]))
            trace.official_success = bool(diagnostics["official_success"])
            trace.terminated = done
            if done:
                break
        return trace


def _frame_tuple(value: Any) -> tuple[FramePacket, ...]:
    """Normalize an adapter frame callback to the GateRequest wire format."""

    if value is None:
        return ()
    if isinstance(value, FramePacket):
        return (value,)
    if isinstance(value, (list, tuple)):
        frames = tuple(value)
    else:
        frames = (value,)
    if not all(isinstance(frame, FramePacket) for frame in frames):
        raise TypeError("frame_packet must return FramePacket or a sequence of FramePacket")
    return frames


def _adapter_frame_packet(adapter: BenchmarkAdapter, observation: np.ndarray, info: Mapping[str, Any], *, reference: bool) -> Any:
    """Call an optional frame hook without making GateRule adapters implement it."""

    callback = getattr(adapter, "frame_packet", None)
    if not callable(callback):
        return None
    return callback(observation, info, reference=reference)

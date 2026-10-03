"""Vector environment for one-stage target-reaching CORE tasks in MT50."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .reward_form_modes import uses_benchmark_native_dense
import torch
from tensordict import TensorDict

from metaworld import MT1
from rsl_rl.env import VecEnv
from stage_policy.live_vlm_gate import FrozenRgbFrame, RgbFrameRing

from .qwen_gate import GateCandidate
from .suite_qwen_gate import MandatoryQwenStageGate
from .target_reach_core import (
    STAGE_COUNT,
    TERMINAL_STAGE,
    TargetReachConfig,
    TargetReachState,
    apply_qwen_decision,
    quality,
    step,
)


GateMode = Literal["rule", "qwen"]


class MetaWorldTargetReachVecEnv(VecEnv):
    """Single shared actor with a Qwen-authorized target-reaching stage code."""

    def __init__(
        self,
        *,
        task: str,
        stage_instruction: str,
        num_envs: int,
        seed: int,
        gate_mode: GateMode,
        qwen_gate: MandatoryQwenStageGate | None = None,
        config: TargetReachConfig | None = None,
        device: str = "cpu",
        max_episode_length: int = 200,
        render_width: int = 320,
        render_height: int = 320,
        camera_name: str = "corner3",
        ring_frame_count: int = 8,
        ring_capture_period_s: float = 0.10,
    ) -> None:
        if gate_mode not in ("rule", "qwen"):
            raise ValueError("gate_mode must be rule or qwen")
        if (gate_mode == "qwen") != (qwen_gate is not None):
            raise ValueError("Qwen service must be present exactly in qwen mode")
        self.task = str(task)
        self.stage_instruction = str(stage_instruction)
        self.num_envs = int(num_envs)
        self.num_actions = 3
        self.device = torch.device(device)
        self.gate_mode = gate_mode
        self.qwen_gate = qwen_gate
        self.config = config or TargetReachConfig()
        self.max_episode_length = int(max_episode_length)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.camera_name = str(camera_name)
        self.ring_capture_period_s = float(ring_capture_period_s)
        self.cfg = {
            "task": self.task,
            "method": "PredStageActor",
            "gate_mode": self.gate_mode,
            "actor_observation": "metaworld_state39_plus_qwen_stage_onehot2",
            "critic_observation": "same_as_actor",
            "action": "delta_eef_xyz_actor_plus_fixed_neutral_gripper",
            "max_episode_length": self.max_episode_length,
            "qwen_camera": self.camera_name if gate_mode == "qwen" else None,
            "reward_config": vars(self.config),
        }
        benchmark = MT1(self.task, seed=int(seed))
        self._tasks = tuple(benchmark.train_tasks)
        env_class = benchmark.train_classes[self.task]
        render_kwargs = (
            {
                "render_mode": "rgb_array",
                "camera_name": self.camera_name,
                "width": self.render_width,
                "height": self.render_height,
            }
            if gate_mode == "qwen"
            else {}
        )
        self._envs = [env_class(**render_kwargs) for _ in range(self.num_envs)]
        for env_id, env in enumerate(self._envs):
            env.seed(int(seed) + env_id)
        self._obs = np.zeros((self.num_envs, 39), dtype=np.float32)
        self._states = [TargetReachState() for _ in range(self.num_envs)]
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._returns = np.zeros(self.num_envs, dtype=np.float64)
        self._raw_success_seen = np.zeros(self.num_envs, dtype=bool)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._global_step = 0
        self._source_frame_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._next_ring_render_time_s = np.zeros(self.num_envs, dtype=np.float64)
        self._rings: list[RgbFrameRing] = []
        self._references: list[FrozenRgbFrame | None] = [None] * self.num_envs
        if gate_mode == "qwen":
            self._rings = [
                RgbFrameRing(
                    frame_count=ring_frame_count,
                    capture_period_s=self.ring_capture_period_s,
                )
                for _ in range(self.num_envs)
            ]
        self.completed_episodes = 0
        self.raw_success_episodes = 0
        self.mandatory_success_episodes = 0
        self.stage_entries = np.zeros(STAGE_COUNT, dtype=np.int64)
        self.gate_requests = 0
        self.gate_accepts = 0
        self.gate_predictions = {"success": 0, "failure": 0, "unknown": 0}
        self._last_episode_log: dict[str, float] = {}
        for env_id in range(self.num_envs):
            self._reset_env(env_id, initial=True)

    @property
    def unwrapped(self) -> "MetaWorldTargetReachVecEnv":
        return self

    def _task_index(self, env_id: int) -> int:
        return int(
            (env_id + self._episode_ids[env_id] * self.num_envs) % len(self._tasks)
        )

    def _distance(self, env_id: int) -> float:
        tcp = np.asarray(self._envs[env_id].tcp_center, dtype=np.float64)
        target = np.asarray(self._envs[env_id]._target_pos, dtype=np.float64) + np.asarray(
            self.config.target_offset_xyz, dtype=np.float64
        )
        return float(np.linalg.norm(tcp - target))

    def _render_frame(self, env_id: int) -> FrozenRgbFrame:
        rgb = np.asarray(self._envs[env_id].render(), dtype=np.uint8)
        if rgb.shape != (self.render_height, self.render_width, 3):
            raise RuntimeError(f"unexpected render shape {rgb.shape}")
        frame = FrozenRgbFrame(
            rgb=np.ascontiguousarray(rgb).tobytes(),
            width=self.render_width,
            height=self.render_height,
            global_step=int(self._global_step),
            sim_time_s=float(
                self.episode_length_buf[env_id].item() * self._envs[env_id].dt
            ),
            source_frame_id=int(self._source_frame_ids[env_id]),
        )
        self._source_frame_ids[env_id] += 1
        return frame

    def _reset_env(self, env_id: int, *, initial: bool = False) -> None:
        if not initial:
            self._episode_ids[env_id] += 1
        env = self._envs[env_id]
        env.set_task(self._tasks[self._task_index(env_id)])
        obs, _ = env.reset()
        self._obs[env_id] = np.asarray(obs, dtype=np.float32)
        self._states[env_id].reset(quality(self._distance(env_id), self.config))
        self._returns[env_id] = 0.0
        self._raw_success_seen[env_id] = False
        self.episode_length_buf[env_id] = 0
        self.stage_entries[0] += 1
        if self.gate_mode == "qwen":
            reference = self._render_frame(env_id)
            self._references[env_id] = reference
            self._rings[env_id].reset(reference)
            self._next_ring_render_time_s[env_id] = (
                reference.sim_time_s + self.ring_capture_period_s
            )

    def _capture_due_ring_frames(self) -> None:
        if self.gate_mode != "qwen":
            return
        for env_id in range(self.num_envs):
            now = float(self.episode_length_buf[env_id].item() * self._envs[env_id].dt)
            if now + 1.0e-9 < self._next_ring_render_time_s[env_id]:
                continue
            while self._next_ring_render_time_s[env_id] <= now + 1.0e-9:
                self._next_ring_render_time_s[env_id] += self.ring_capture_period_s
            self._rings[env_id].maybe_append(self._render_frame(env_id))

    def _candidate(self, env_id: int) -> GateCandidate:
        current = self._render_frame(env_id)
        ring = self._rings[env_id]
        ring.maybe_append(current)
        reference = self._references[env_id]
        if reference is None:
            raise RuntimeError("candidate is missing its reference image")
        state = self._states[env_id]
        return GateCandidate(
            env_id=env_id,
            episode_id=int(self._episode_ids[env_id]),
            stage_id=0,
            camera_name=self.camera_name,
            candidate_epoch=int(state.candidate_epoch),
            request_step=int(self._global_step),
            candidate_start_step=int(self._global_step - state.stable_count + 1),
            reference=reference,
            ring_frames=ring.frames,
            current=current,
            ring_nominal_fps=ring.nominal_fps,
            source_update_period_s=float(self._envs[env_id].dt),
            target_samples_due_since_reset=ring.target_samples_due_since_reset,
            duplicates_skipped_since_reset=ring.pixel_identical_duplicates_skipped_since_reset,
        )

    def _observations(self) -> TensorDict:
        stages = np.fromiter(
            (state.stage for state in self._states), dtype=np.int64, count=self.num_envs
        )
        values = np.concatenate(
            (self._obs, np.eye(STAGE_COUNT, dtype=np.float32)[stages]), axis=1
        )
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return TensorDict(
            {"policy": tensor, "critic": tensor.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        action_np = np.clip(
            actions.detach().to("cpu", dtype=torch.float32).numpy(), -1.0, 1.0
        )
        if action_np.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"unexpected action shape {action_np.shape}")
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        native_rewards = np.zeros(self.num_envs, dtype=np.float32)
        timeouts = np.zeros(self.num_envs, dtype=bool)
        due: list[int] = []
        raw_success = np.zeros(self.num_envs, dtype=bool)
        self._global_step += 1
        for env_id, env in enumerate(self._envs):
            full_action = np.concatenate((action_np[env_id], np.array([0.0]))).astype(
                np.float32
            )
            obs, native_reward, _, native_timeout, info = env.step(full_action)
            native_rewards[env_id] = float(native_reward)
            self._obs[env_id] = np.asarray(obs, dtype=np.float32)
            self.episode_length_buf[env_id] += 1
            result = step(
                self._states[env_id],
                distance_m=self._distance(env_id),
                action=action_np[env_id],
                config=self.config,
            )
            rewards[env_id] = result.reward
            if result.request_due:
                due.append(env_id)
            raw_success[env_id] = bool(info["success"])
            self._raw_success_seen[env_id] |= raw_success[env_id]
            timeouts[env_id] = bool(
                native_timeout
                or self.episode_length_buf[env_id].item() >= self.max_episode_length
            )
        if self.gate_mode == "qwen":
            self._capture_due_ring_frames()
        if due:
            self.gate_requests += len(due)
            if self.gate_mode == "rule":
                predictions = ["success"] * len(due)
                decisions = [True] * len(due)
            else:
                assert self.qwen_gate is not None
                returned = self.qwen_gate.decide_batch([self._candidate(i) for i in due])
                predictions = [decision.prediction for decision in returned]
                decisions = [decision.accepted for decision in returned]
            for env_id, prediction, decision in zip(due, predictions, decisions, strict=True):
                self.gate_predictions[prediction] += 1
                bonus = apply_qwen_decision(self._states[env_id], decision, self.config)
                rewards[env_id] += bonus
                self.gate_accepts += int(decision)
                if decision:
                    self.stage_entries[TERMINAL_STAGE] += 1
        terminal = np.fromiter(
            (state.stage == TERMINAL_STAGE for state in self._states),
            dtype=bool,
            count=self.num_envs,
        )
        dones = np.logical_or(terminal, timeouts)
        if uses_benchmark_native_dense(self.config):
            rewards = native_rewards
        self._returns += rewards
        for env_id in np.flatnonzero(dones):
            self.completed_episodes += 1
            raw = bool(self._raw_success_seen[env_id])
            mandatory = bool(terminal[env_id] and raw)
            self.raw_success_episodes += int(raw)
            self.mandatory_success_episodes += int(mandatory)
            self._last_episode_log = {
                "return": float(self._returns[env_id]),
                "length": float(self.episode_length_buf[env_id].item()),
                "raw_success": float(raw),
                "mandatory_success": float(mandatory),
            }
            self._reset_env(int(env_id))
        return (
            self._observations(),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.long, device=self.device),
            {
                "time_outs": torch.as_tensor(timeouts, dtype=torch.bool, device=self.device),
                "log": {
                    "/Gate/candidate_accept_rate": self.gate_accepts / max(self.gate_requests, 1),
                    "/Success/raw": self.raw_success_episodes / max(self.completed_episodes, 1),
                    "/Success/mandatory_qwen": self.mandatory_success_episodes / max(self.completed_episodes, 1),
                },
            },
        )

    def statistics(self) -> dict[str, object]:
        return {
            "task": self.task,
            "gate_mode": self.gate_mode,
            "completed_episodes": self.completed_episodes,
            "raw_success_episodes": self.raw_success_episodes,
            "raw_success_rate": self.raw_success_episodes / max(self.completed_episodes, 1),
            "mandatory_success_episodes": self.mandatory_success_episodes,
            "mandatory_success_rate": self.mandatory_success_episodes / max(self.completed_episodes, 1),
            "stage_entries": self.stage_entries.tolist(),
            "gate_requests": self.gate_requests,
            "gate_accepts": self.gate_accepts,
            "gate_candidate_accept_rate": self.gate_accepts / max(self.gate_requests, 1),
            "gate_predictions": dict(self.gate_predictions),
            "qwen_protocol_errors": (
                self.qwen_gate.protocol_errors if self.qwen_gate is not None else None
            ),
            "qwen_accuracy": None,
            "qwen_accuracy_reason": "runtime candidates lack independent visual labels",
        }

    def close(self) -> None:
        for env in self._envs:
            env.close()
        if self.qwen_gate is not None:
            self.qwen_gate.close()
            self.qwen_gate = None

"""RSL-RL vector wrapper for staged MetaWorld drawer opening."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .reward_form_modes import uses_benchmark_native_dense
import torch
from tensordict import TensorDict

from metaworld import MT1
from rsl_rl.env import VecEnv
from stage_policy.live_vlm_gate import FrozenRgbFrame, RgbFrameRing

from .drawer_open_core import (
    STAGE_COUNT,
    DrawerFeatures,
    DrawerRewardConfig,
    DrawerStageState,
    apply_due_qwen_decision,
    drawer_features,
    drawer_stage_step,
    stage_quality,
)
from .qwen_gate import GateCandidate, MandatoryQwenGate


GateMode = Literal["rule", "qwen"]


class MetaWorldDrawerVecEnv(VecEnv):
    """Synchronous vector env with a single shared PredStageActor.

    The three-dimensional actor action controls only delta-EEF. MetaWorld's
    documented action mask fixes the drawer-task gripper command supplied by
    the experiment configuration.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        gate_mode: GateMode,
        reward_config: DrawerRewardConfig | None = None,
        qwen_gate: MandatoryQwenGate | None = None,
        task_name: str = "drawer-open-v3",
        gripper_command: float = -1.0,
        device: str = "cpu",
        max_episode_length: int = 200,
        render_width: int = 320,
        render_height: int = 320,
        camera_name: str = "corner3",
        ring_frame_count: int = 8,
        ring_capture_period_s: float = 0.10,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if gate_mode not in ("rule", "qwen"):
            raise ValueError("gate_mode must be rule or qwen")
        if gate_mode == "qwen" and qwen_gate is None:
            raise ValueError("qwen mode requires a real MandatoryQwenGate")
        if gate_mode == "rule" and qwen_gate is not None:
            raise ValueError("rule control must not carry a Qwen service")
        if task_name not in ("drawer-open-v3", "drawer-close-v3"):
            raise ValueError("unsupported drawer task")
        if not -1.0 <= float(gripper_command) <= 1.0:
            raise ValueError("gripper command must be normalized")

        self.num_envs = int(num_envs)
        self.num_actions = 3
        self.device = torch.device(device)
        self.max_episode_length = int(max_episode_length)
        self.gate_mode = gate_mode
        self.reward_config = reward_config or DrawerRewardConfig()
        self.qwen_gate = qwen_gate
        self.task_name = str(task_name)
        self.gripper_command = float(gripper_command)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.camera_name = str(camera_name)
        self.ring_capture_period_s = float(ring_capture_period_s)
        self.cfg = {
            "task": self.task_name,
            "method": "PredStageActor",
            "gate_mode": gate_mode,
            "actor_observation": "metaworld_state39_plus_qwen_stage_onehot4",
            "critic_observation": "same_as_actor",
            "action": "delta_eef_xyz_actor_plus_fixed_gripper",
            "fixed_gripper_command": self.gripper_command,
            "max_episode_length": self.max_episode_length,
            "qwen_camera": self.camera_name if gate_mode == "qwen" else None,
            "reward_config": vars(self.reward_config),
        }

        benchmark = MT1(self.task_name, seed=int(seed))
        self._tasks = tuple(benchmark.train_tasks)
        env_class = benchmark.train_classes[self.task_name]
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
        self._states = [DrawerStageState() for _ in range(self.num_envs)]
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._raw_success_seen = np.zeros(self.num_envs, dtype=bool)
        self._episode_transition_masks = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._global_step = 0
        self._source_frame_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._next_ring_render_time_s = np.zeros(
            self.num_envs, dtype=np.float64
        )
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
    def unwrapped(self) -> "MetaWorldDrawerVecEnv":
        return self

    def _task_index(self, env_id: int) -> int:
        return int(
            (env_id + self._episode_ids[env_id] * self.num_envs)
            % len(self._tasks)
        )

    def _features(self, env_id: int) -> DrawerFeatures:
        env = self._envs[env_id]
        target = np.asarray(env._target_pos, dtype=np.float64)
        return drawer_features(
            self._obs[env_id, :3],
            self._obs[env_id, 4:7],
            target,
            max_open_distance_m=float(env.maxDist),
            active_waypoint_offset_xyz=self.reward_config.affordance_offset_xyz,
        )

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
                self.episode_length_buf[env_id].item()
                * self._envs[env_id].dt
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
        features = self._features(env_id)
        self._states[env_id].reset(stage_quality(0, features))
        self._episode_returns[env_id] = 0.0
        self._raw_success_seen[env_id] = False
        self._episode_transition_masks[env_id] = 0
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
            current_time_s = float(
                self.episode_length_buf[env_id].item() * self._envs[env_id].dt
            )
            epsilon = max(1.0e-9, self.ring_capture_period_s * 1.0e-6)
            if current_time_s + epsilon < self._next_ring_render_time_s[env_id]:
                continue
            while self._next_ring_render_time_s[env_id] <= current_time_s + epsilon:
                self._next_ring_render_time_s[env_id] += (
                    self.ring_capture_period_s
                )
            frame = self._render_frame(env_id)
            self._rings[env_id].maybe_append(frame)

    def _actor_observations(self) -> np.ndarray:
        stages = np.fromiter(
            (state.stage for state in self._states),
            dtype=np.int64,
            count=self.num_envs,
        )
        one_hot = np.eye(STAGE_COUNT, dtype=np.float32)[stages]
        return np.concatenate((self._obs, one_hot), axis=1, dtype=np.float32)

    def get_observations(self) -> TensorDict:
        values = torch.as_tensor(
            self._actor_observations(), dtype=torch.float32, device=self.device
        )
        return TensorDict(
            {"policy": values, "critic": values.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _candidate_evidence(
        self, env_id: int, stage_id: int
    ) -> GateCandidate:
        ring = self._rings[env_id]
        current = self._render_frame(env_id)
        ring.maybe_append(current)
        reference = self._references[env_id]
        if reference is None:
            raise RuntimeError("Qwen candidate is missing its reference frame")
        return GateCandidate(
            env_id=env_id,
            episode_id=int(self._episode_ids[env_id]),
            stage_id=stage_id,
            camera_name=self.camera_name,
            candidate_epoch=int(self._states[env_id].candidate_epoch),
            request_step=int(self._global_step),
            candidate_start_step=int(
                self._global_step - self._states[env_id].stable_count + 1
            ),
            reference=reference,
            ring_frames=ring.frames,
            current=current,
            ring_nominal_fps=ring.nominal_fps,
            source_update_period_s=float(self._envs[env_id].dt),
            target_samples_due_since_reset=(
                ring.target_samples_due_since_reset
            ),
            duplicates_skipped_since_reset=(
                ring.pixel_identical_duplicates_skipped_since_reset
            ),
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        action_np = actions.detach().to("cpu", dtype=torch.float32).numpy()
        if action_np.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"unexpected action shape {action_np.shape}")
        if not np.all(np.isfinite(action_np)):
            raise FloatingPointError("policy emitted a non-finite action")
        action_np = np.clip(action_np, -1.0, 1.0)

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        native_rewards = np.zeros(self.num_envs, dtype=np.float32)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        native_infos: list[dict[str, object]] = []
        features_after: list[DrawerFeatures] = []
        due_ids: list[int] = []
        self._global_step += 1

        for env_id, env in enumerate(self._envs):
            full_action = np.concatenate(
                (
                    action_np[env_id],
                    np.array([self.gripper_command], dtype=np.float32),
                )
            ).astype(np.float32)
            obs, native_reward, _, native_timeout, info = env.step(full_action)
            native_rewards[env_id] = float(native_reward)
            self._obs[env_id] = np.asarray(obs, dtype=np.float32)
            self.episode_length_buf[env_id] += 1
            features = self._features(env_id)
            result = drawer_stage_step(
                self._states[env_id],
                features,
                action_np[env_id],
                qwen_decision=None,
                config=self.reward_config,
            )
            rewards[env_id] = result.reward
            if result.request_due:
                due_ids.append(env_id)
            native_infos.append(info)
            features_after.append(features)
            self._raw_success_seen[env_id] |= bool(info["success"])
            time_outs[env_id] = bool(
                native_timeout
                or self.episode_length_buf[env_id].item()
                >= self.max_episode_length
            )

        if self.gate_mode == "qwen":
            self._capture_due_ring_frames()

        if due_ids:
            self.gate_requests += len(due_ids)
            if self.gate_mode == "rule":
                predictions = ["success"] * len(due_ids)
                accepted = [True] * len(due_ids)
            else:
                assert self.qwen_gate is not None
                candidates = [
                    self._candidate_evidence(
                        env_id, self._states[env_id].stage
                    )
                    for env_id in due_ids
                ]
                decisions = self.qwen_gate.decide_batch(candidates)
                predictions = [decision.prediction for decision in decisions]
                accepted = [decision.accepted for decision in decisions]
            for env_id, prediction, decision in zip(
                due_ids, predictions, accepted, strict=True
            ):
                self.gate_predictions[prediction] += 1
                transition = apply_due_qwen_decision(
                    self._states[env_id],
                    features_after[env_id],
                    decision,
                    self.reward_config,
                )
                self.gate_accepts += int(transition.accepted)
                rewards[env_id] += transition.transition_reward
                if transition.transitioned:
                    self._episode_transition_masks[env_id] |= (
                        1 << transition.old_stage
                    )
                    self.stage_entries[transition.new_stage] += 1

        terminal = np.fromiter(
            (state.stage == STAGE_COUNT - 1 for state in self._states),
            dtype=bool,
            count=self.num_envs,
        )
        dones_np = np.logical_or(terminal, time_outs)
        if uses_benchmark_native_dense(self.config):
            rewards = native_rewards
        self._episode_returns += rewards

        for env_id in np.flatnonzero(dones_np):
            self.completed_episodes += 1
            raw_success = bool(self._raw_success_seen[env_id])
            mandatory_success = bool(
                terminal[env_id]
                and self._episode_transition_masks[env_id] == 0b111
                and raw_success
            )
            self.raw_success_episodes += int(raw_success)
            self.mandatory_success_episodes += int(mandatory_success)
            self._last_episode_log = {
                "return": float(self._episode_returns[env_id]),
                "length": float(self.episode_length_buf[env_id].item()),
                "raw_success": float(raw_success),
                "mandatory_success": float(mandatory_success),
            }
            self._reset_env(int(env_id))

        gate_accept_rate = (
            self.gate_accepts / self.gate_requests
            if self.gate_requests
            else 0.0
        )
        raw_success_rate = (
            self.raw_success_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        mandatory_success_rate = (
            self.mandatory_success_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        extras = {
            "time_outs": torch.as_tensor(
                time_outs, dtype=torch.bool, device=self.device
            ),
            "log": {
                "/Stage/mean": float(
                    np.mean([state.stage for state in self._states])
                ),
                "/Gate/candidate_accept_rate": float(gate_accept_rate),
                "/Success/official_reach_once_rate": float(raw_success_rate),
                "/Success/mandatory_qwen_rate": float(
                    mandatory_success_rate
                ),
            },
        }
        return (
            self.get_observations(),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones_np, dtype=torch.long, device=self.device),
            extras,
        )

    def statistics(self) -> dict[str, object]:
        return {
            "task": self.task_name,
            "gate_mode": self.gate_mode,
            "completed_episodes": self.completed_episodes,
            "raw_success_episodes": self.raw_success_episodes,
            "raw_success_rate": (
                self.raw_success_episodes / self.completed_episodes
                if self.completed_episodes
                else 0.0
            ),
            "mandatory_success_episodes": self.mandatory_success_episodes,
            "mandatory_success_rate": (
                self.mandatory_success_episodes / self.completed_episodes
                if self.completed_episodes
                else 0.0
            ),
            "stage_entries": self.stage_entries.tolist(),
            "gate_requests": self.gate_requests,
            "gate_accepts": self.gate_accepts,
            "gate_candidate_accept_rate": (
                self.gate_accepts / self.gate_requests
                if self.gate_requests
                else 0.0
            ),
            "gate_predictions": dict(self.gate_predictions),
            "qwen_protocol_errors": (
                self.qwen_gate.protocol_errors
                if self.qwen_gate is not None
                else None
            ),
            "qwen_accuracy": None,
            "qwen_accuracy_reason": (
                "training candidates have no independent visual ground-truth "
                "labels; candidate acceptance is not reported as accuracy"
            ),
        }

    def close(self) -> None:
        for env in self._envs:
            env.close()
        if self.qwen_gate is not None:
            self.qwen_gate.close()
            self.qwen_gate = None


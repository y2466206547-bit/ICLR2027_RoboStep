"""Vector environment for object-move/push CORE tasks."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .reward_form_modes import uses_benchmark_native_dense
import torch
from tensordict import TensorDict

from metaworld import MT1
from rsl_rl.env import VecEnv
from stage_policy.live_vlm_gate import FrozenRgbFrame, RgbFrameRing

from .object_move_core import (
    ObjectMoveConfig,
    ObjectMoveState,
    active_stage_count,
    apply_qwen,
    features,
    stage_count,
    stage_quality,
    step,
    terminal_stage,
    validate_stage_config,
)
from .qwen_gate import GateCandidate
from .suite_qwen_gate import MandatoryQwenStageGate
from .transition_frequency_runtime import (
    bypass_rule_candidate,
    close_synthetic_gate_log,
    configure_transition_schedule,
    decide_qwen_with_repeated_accepts,
    decide_synthetic_gate,
    mark_transition_queries,
    record_stage_transition,
    record_synthetic_gate_outcome,
    reset_transition_schedule,
    select_synthetic_gate_opportunities,
    select_transition_due,
    synthetic_gate_enabled,
    synthetic_gate_summary,
)

GateMode = Literal['rule', 'qwen', 'synthetic']


class MetaWorldObjectMoveVecEnv(VecEnv):
    """Single shared stage-conditioned actor for observable object motion."""

    def __init__(self, *, task: str, stage_instructions: tuple[str, ...], num_envs: int, seed: int, gate_mode: GateMode, qwen_gate: MandatoryQwenStageGate | None = None, config: ObjectMoveConfig | None = None, device: str = 'cpu', max_episode_length: int = 200, gripper_command: float = 0.0, render_width: int = 320, render_height: int = 320, camera_name: str = 'corner3', ring_frame_count: int = 8, ring_capture_period_s: float = 0.10) -> None:
        if gate_mode not in ('rule', 'qwen', 'synthetic'):
            raise ValueError('unsupported gate mode')
        if gate_mode == 'qwen' and qwen_gate is None:
            raise ValueError('Qwen service must be present in qwen mode')
        if gate_mode != 'qwen' and qwen_gate is not None:
            raise ValueError('Qwen service is only valid in qwen mode')
        self.task = str(task)
        self.stage_instructions = tuple(stage_instructions)
        self.config, self.max_episode_length = config or ObjectMoveConfig(), int(max_episode_length)
        self.actor_controls_gripper = bool(self.config.actor_controls_gripper)
        self.num_envs = int(num_envs)
        self.num_actions = 4 if self.actor_controls_gripper else 3
        self.device, self.gate_mode, self.qwen_gate = torch.device(device), gate_mode, qwen_gate
        self.gripper_command = float(gripper_command)
        self.render_width, self.render_height, self.camera_name = int(render_width), int(render_height), str(camera_name)
        self.ring_capture_period_s = float(ring_capture_period_s)
        validate_stage_config(self.config)
        self.active_stage_count = active_stage_count(self.config)
        self.stage_count = stage_count(self.config)
        self.terminal_stage = terminal_stage(self.config)
        if len(self.stage_instructions) != self.active_stage_count:
            raise ValueError(
                f"expected {self.active_stage_count} stage instructions")
        self.cfg = {
            'task': self.task,
            'method': 'PredStageActor',
            'gate_mode': gate_mode,
            'actor_observation': f'metaworld_state39_plus_qwen_stage_onehot{self.stage_count}',
            'critic_observation': 'same_as_actor',
            'action': 'standard_delta_eef_xyz_plus_actor_gripper' if self.actor_controls_gripper else 'delta_eef_xyz_actor_plus_fixed_gripper',
            'fixed_gripper_command': None if self.actor_controls_gripper else self.gripper_command,
            'max_episode_length': self.max_episode_length,
            'qwen_camera': self.camera_name if gate_mode == 'qwen' else None,
            'reward_config': vars(self.config),
        }
        benchmark = MT1(self.task, seed=int(seed))
        self._tasks = tuple(benchmark.train_tasks)
        env_class = benchmark.train_classes[self.task]
        kwargs = {'render_mode': 'rgb_array', 'camera_name': self.camera_name, 'width': self.render_width, 'height': self.render_height} if gate_mode == 'qwen' else {}
        self._envs = [env_class(**kwargs) for _ in range(self.num_envs)]
        for env_id, env in enumerate(self._envs):
            env.seed(int(seed) + env_id)
        self._obs = np.zeros((self.num_envs, 39), dtype=np.float32)
        self._states = [ObjectMoveState() for _ in range(self.num_envs)]
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._returns = np.zeros(self.num_envs, dtype=np.float64)
        self._raw_success_seen = np.zeros(self.num_envs, dtype=bool)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._global_step = 0
        self._source_frame_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._next_ring_render_time_s = np.zeros(self.num_envs, dtype=np.float64)
        self._rings = [RgbFrameRing(frame_count=ring_frame_count, capture_period_s=self.ring_capture_period_s) for _ in range(self.num_envs)] if gate_mode == 'qwen' else []
        self._references: list[FrozenRgbFrame | None] = [None] * self.num_envs
        self.completed_episodes = self.raw_success_episodes = self.mandatory_success_episodes = 0
        self.stage_entries = np.zeros(self.stage_count, dtype=np.int64)
        self.gate_requests = self.gate_accepts = 0
        self.gate_predictions = {'success': 0, 'failure': 0, 'unknown': 0}
        self._last_episode_log: dict[str, float] = {}
        configure_transition_schedule(self, self.cfg)
        for env_id in range(self.num_envs):
            self._reset_env(env_id, initial=True)

    @property
    def unwrapped(self) -> 'MetaWorldObjectMoveVecEnv':
        return self

    def _task_index(self, env_id: int) -> int:
        return int((env_id + self._episode_ids[env_id] * self.num_envs) % len(self._tasks))

    def _target(self, env_id: int) -> np.ndarray:
        env = self._envs[env_id]
        target = getattr(env, '_target_pos', None)
        if target is None:
            target = getattr(env, 'goal')
        return np.asarray(target, dtype=np.float64)

    def _object(self, env_id: int) -> np.ndarray:
        env = self._envs[env_id]
        try:
            return np.asarray(env._get_pos_objects(), dtype=np.float64)
        except Exception:
            return np.asarray(self._obs[env_id, 4:7], dtype=np.float64)

    def _features(self, env_id: int):
        env = self._envs[env_id]
        return features(
            env.tcp_center,
            self._object(env_id),
            self._target(env_id),
            self.config,
            gripper_opening=float(self._obs[env_id, 3]),
        )

    def _render(self, env_id: int) -> FrozenRgbFrame:
        rgb = np.asarray(self._envs[env_id].render(), dtype=np.uint8)
        if rgb.shape != (self.render_height, self.render_width, 3):
            raise RuntimeError('unexpected camera shape')
        frame = FrozenRgbFrame(rgb=np.ascontiguousarray(rgb).tobytes(), width=self.render_width, height=self.render_height, global_step=int(self._global_step), sim_time_s=float(self.episode_length_buf[env_id].item() * self._envs[env_id].dt), source_frame_id=int(self._source_frame_ids[env_id]))
        self._source_frame_ids[env_id] += 1
        return frame

    def _reset_env(self, env_id: int, *, initial: bool = False) -> None:
        if not initial:
            self._episode_ids[env_id] += 1
        env = self._envs[env_id]
        env.set_task(self._tasks[self._task_index(env_id)])
        obs, _ = env.reset()
        self._obs[env_id] = np.asarray(obs, dtype=np.float32)
        self._states[env_id].reset(stage_quality(0, self._features(env_id), self.config))
        self._returns[env_id], self._raw_success_seen[env_id], self.episode_length_buf[env_id] = 0.0, False, 0
        self.stage_entries[0] += 1
        if self.gate_mode == 'qwen':
            frame = self._render(env_id)
            self._references[env_id] = frame
            self._rings[env_id].reset(frame)
            self._next_ring_render_time_s[env_id] = frame.sim_time_s + self.ring_capture_period_s
        reset_transition_schedule(self, env_id)

    def _capture(self) -> None:
        if self.gate_mode != 'qwen':
            return
        for env_id in range(self.num_envs):
            now = float(self.episode_length_buf[env_id].item() * self._envs[env_id].dt)
            if now + 1e-9 < self._next_ring_render_time_s[env_id]:
                continue
            while self._next_ring_render_time_s[env_id] <= now + 1e-9:
                self._next_ring_render_time_s[env_id] += self.ring_capture_period_s
            self._rings[env_id].maybe_append(self._render(env_id))

    def _candidate(self, env_id: int) -> GateCandidate:
        current = self._render(env_id)
        ring = self._rings[env_id]
        ring.maybe_append(current)
        reference = self._references[env_id]
        if reference is None:
            raise RuntimeError('missing reference image')
        state = self._states[env_id]
        return GateCandidate(env_id=env_id, episode_id=int(self._episode_ids[env_id]), stage_id=int(state.stage), camera_name=self.camera_name, candidate_epoch=int(state.candidate_epoch), request_step=int(self._global_step), candidate_start_step=int(self._global_step - state.stable_count + 1), reference=reference, ring_frames=ring.frames, current=current, ring_nominal_fps=ring.nominal_fps, source_update_period_s=float(self._envs[env_id].dt), target_samples_due_since_reset=ring.target_samples_due_since_reset, duplicates_skipped_since_reset=ring.pixel_identical_duplicates_skipped_since_reset)

    def _observations(self) -> TensorDict:
        stages = np.fromiter((s.stage for s in self._states), dtype=np.int64, count=self.num_envs)
        values = np.concatenate((self._obs, np.eye(self.stage_count, dtype=np.float32)[stages]), axis=1)
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return TensorDict({'policy': tensor, 'critic': tensor.clone()}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:
        return self._observations()

    def step(self, actions: torch.Tensor):
        action_np = np.clip(actions.detach().to('cpu', dtype=torch.float32).numpy(), -1.0, 1.0)
        if action_np.shape != (self.num_envs, self.num_actions):
            raise ValueError(f'invalid policy action shape {action_np.shape}')
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        native_rewards = np.zeros(self.num_envs, dtype=np.float32)
        timeouts, due = np.zeros(self.num_envs, dtype=bool), []
        self._global_step += 1
        for env_id, env in enumerate(self._envs):
            xyz_action = action_np[env_id, :3]
            full_action = (
                action_np[env_id]
                if self.actor_controls_gripper
                else np.concatenate((xyz_action, np.array([self.gripper_command], dtype=np.float32))).astype(np.float32)
            )
            obs, native_reward, _, native_timeout, info = env.step(full_action)
            native_rewards[env_id] = float(native_reward)
            self._obs[env_id] = np.asarray(obs, dtype=np.float32)
            self.episode_length_buf[env_id] += 1
            reward_action = (
                full_action if self.config.smooth_actor_gripper else xyz_action
            )
            result = step(
                self._states[env_id],
                self._features(env_id),
                reward_action,
                self.config,
            )
            rewards[env_id] = result.reward
            if result.request_due:
                due.append(env_id)
            self._raw_success_seen[env_id] |= bool(info['success'])
            timeouts[env_id] = bool(native_timeout or self.episode_length_buf[env_id].item() >= self.max_episode_length)
        self._capture()
        if synthetic_gate_enabled(self):
            opportunities = select_synthetic_gate_opportunities(self, due)
            due = [opportunity.env_id for opportunity in opportunities]
        else:
            opportunities = []
            due = select_transition_due(self, due)
        if due:
            mark_transition_queries(self, due)
            if self.gate_mode == 'rule':
                self.gate_requests += len(due)
                predictions, decisions = ['success'] * len(due), [True] * len(due)
                for env_id, prediction, decision in zip(due, predictions, decisions, strict=True):
                    self.gate_predictions[prediction] += 1
                    old_stage = int(self._states[env_id].stage)
                    value = self._features(env_id)
                    rewards[env_id] += apply_qwen(
                        self._states[env_id],
                        value,
                        decision,
                        self.config,
                        bypass_candidate=bypass_rule_candidate(self, env_id),
                    )
                    self.gate_accepts += int(decision)
                    if decision:
                        if int(self._states[env_id].stage) > old_stage:
                            record_stage_transition(self, env_id)
                        self.stage_entries[self._states[env_id].stage] += 1
            elif synthetic_gate_enabled(self):
                synthetic_decisions = decide_synthetic_gate(self, opportunities)
                self.gate_requests += len(synthetic_decisions)
                for synthetic_decision in synthetic_decisions:
                    env_id = synthetic_decision.env_id
                    self.gate_predictions[synthetic_decision.prediction] += 1
                    old_stage = int(self._states[env_id].stage)
                    new_stage = old_stage
                    transitioned = False
                    if synthetic_decision.gt_complete or synthetic_decision.accepted:
                        value = self._features(env_id)
                        rewards[env_id] += apply_qwen(
                            self._states[env_id],
                            value,
                            synthetic_decision.accepted,
                            self.config,
                            bypass_candidate=not synthetic_decision.gt_complete,
                        )
                        new_stage = int(self._states[env_id].stage)
                        transitioned = new_stage > old_stage
                    self.gate_accepts += int(synthetic_decision.accepted)
                    if synthetic_decision.accepted:
                        if transitioned:
                            record_stage_transition(self, env_id)
                        self.stage_entries[self._states[env_id].stage] += 1
                    record_synthetic_gate_outcome(
                        self,
                        synthetic_decision,
                        old_stage=old_stage,
                        new_stage=new_stage,
                        transitioned=transitioned,
                    )
            else:
                predictions, decisions, qwen_requests = decide_qwen_with_repeated_accepts(
                    self, [self._candidate(i) for i in due]
                )
                self.gate_requests += qwen_requests
                for env_id, prediction, decision in zip(due, predictions, decisions, strict=True):
                    self.gate_predictions[prediction] += 1
                    old_stage = int(self._states[env_id].stage)
                    value = self._features(env_id)
                    rewards[env_id] += apply_qwen(
                        self._states[env_id],
                        value,
                        decision,
                        self.config,
                        bypass_candidate=bypass_rule_candidate(self, env_id),
                    )
                    self.gate_accepts += int(decision)
                    if decision:
                        if int(self._states[env_id].stage) > old_stage:
                            record_stage_transition(self, env_id)
                        self.stage_entries[self._states[env_id].stage] += 1
        terminal = np.fromiter((s.stage == self.terminal_stage for s in self._states), dtype=bool, count=self.num_envs)
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
            self._last_episode_log = {'return': float(self._returns[env_id]), 'length': float(self.episode_length_buf[env_id].item()), 'raw_success': float(raw), 'mandatory_success': float(mandatory)}
            self._reset_env(int(env_id))
        return self._observations(), torch.as_tensor(rewards, dtype=torch.float32, device=self.device), torch.as_tensor(dones, dtype=torch.long, device=self.device), {'time_outs': torch.as_tensor(timeouts, dtype=torch.bool, device=self.device), 'log': {'/Gate/candidate_accept_rate': self.gate_accepts / max(self.gate_requests, 1), '/Success/raw': self.raw_success_episodes / max(self.completed_episodes, 1), '/Success/mandatory_qwen': self.mandatory_success_episodes / max(self.completed_episodes, 1)}}

    def statistics(self) -> dict[str, object]:
        payload = {'task': self.task, 'gate_mode': self.gate_mode, 'completed_episodes': self.completed_episodes, 'raw_success_episodes': self.raw_success_episodes, 'raw_success_rate': self.raw_success_episodes / max(self.completed_episodes, 1), 'mandatory_success_episodes': self.mandatory_success_episodes, 'mandatory_success_rate': self.mandatory_success_episodes / max(self.completed_episodes, 1), 'stage_entries': self.stage_entries.tolist(), 'gate_requests': self.gate_requests, 'gate_accepts': self.gate_accepts, 'gate_candidate_accept_rate': self.gate_accepts / max(self.gate_requests, 1), 'gate_predictions': dict(self.gate_predictions), 'qwen_protocol_errors': self.qwen_gate.protocol_errors if self.qwen_gate else None, 'qwen_accuracy': None, 'qwen_accuracy_reason': 'runtime candidates lack independent visual labels'}
        synthetic = synthetic_gate_summary(self)
        if synthetic is not None:
            payload['synthetic_gate'] = synthetic
        return payload

    def close(self) -> None:
        for env in self._envs:
            env.close()
        if self.qwen_gate is not None:
            self.qwen_gate.close()
            self.qwen_gate = None
        close_synthetic_gate_log(self)

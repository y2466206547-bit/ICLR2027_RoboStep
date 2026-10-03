"""Same-scene rule-gated A/B/C pilot: pick-place, push, three-point reach.

All modules execute in a single pick-place-v3 MuJoCo episode. This pilot is
not the proposed drawer/button compound-scene experiment.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from tensordict import TensorDict

from .grasp_insert_core import (
    GraspInsertState,
    apply_qwen_decision as accept_pick,
    stage_quality as pick_quality,
    step as pick_step,
)
from .grasp_insert_vec_env import MetaWorldGraspInsertVecEnv
from .object_move_core import (
    ObjectMoveConfig,
    ObjectMoveState,
    apply_qwen as accept_push,
    features as push_features,
    stage_quality as push_quality,
    step as push_step,
)
from .target_reach_core import (
    TargetReachConfig,
    TargetReachState,
    apply_qwen_decision as accept_reach,
    quality as reach_quality,
    step as reach_step,
)
from .reward_form_modes import ALL_DENSE_WITH_STAGE_ACTOR
from .transition_frequency_runtime import (
    mark_transition_queries,
    record_stage_transition,
    select_transition_due,
)

REACH_WAYPOINTS_XY = ((-0.08, 0.76), (0.08, 0.76), (0.0, 0.88))


class ForwardMixedPickPlaceVecEnv(MetaWorldGraspInsertVecEnv):
    """A=pick-place, B=push, C=three waypoint reach."""

    def __init__(self, *, module_sequence: str, non_a_gripper_command: float = 1.0, non_a_gripper_commands: dict[str, float] | None = None, reward_protocol: str = "stage_machine", policy_context: str = "stage_onehot", stage_id_size: int = 10, **kwargs: object) -> None:
        if not 1 <= len(module_sequence) <= 5:
            raise ValueError("module compositions must contain between one and five macros")
        if set(module_sequence) - {"A", "B", "C"}:
            raise ValueError("unknown module")
        if kwargs.get("gate_mode") != "rule":
            raise ValueError("pilot currently supports rule gate only")
        if non_a_gripper_command not in (-1.0, 1.0):
            raise ValueError("non-A gripper command must be -1 (open) or +1 (closed)")
        if reward_protocol not in ("stage_machine", "flat_dense"):
            raise ValueError("unknown reward protocol")
        if policy_context not in ("stage_onehot", "state_goal_only"):
            raise ValueError("unknown policy context")
        commands = {"B": float(non_a_gripper_command), "C": float(non_a_gripper_command)}
        if non_a_gripper_commands is not None:
            if set(non_a_gripper_commands) - {"B", "C"}:
                raise ValueError("only B and C may override the gripper command")
            commands.update(non_a_gripper_commands)
        if any(value not in (-1.0, 1.0) for value in commands.values()):
            raise ValueError("module gripper commands must be -1 or +1")
        self.module_sequence = module_sequence
        self.reward_protocol = reward_protocol
        self.policy_context = policy_context
        self.stage_id_size = int(stage_id_size)
        if self.stage_id_size < 3 * len(module_sequence) + 1:
            raise ValueError("stage_id_size must cover all atomic stages and terminal")
        self.non_a_gripper_command = float(non_a_gripper_command)
        self.non_a_gripper_commands = commands
        self.push_config = ObjectMoveConfig()
        self.reach_config = TargetReachConfig()
        self._local_stages = np.zeros(int(kwargs["num_envs"]), dtype=np.int64)
        self._strict_prefix_counts = np.zeros(len(module_sequence), dtype=np.int64)
        self._failed_macro_counts = np.zeros(len(module_sequence), dtype=np.int64)
        super().__init__(macro_count=len(module_sequence), stage_id_size=self.stage_id_size, **kwargs)
        self._flat_pick_config = replace(
            self.config, reward_form_mode=ALL_DENSE_WITH_STAGE_ACTOR
        )
        self._flat_push_config = replace(
            self.push_config, reward_form_mode=ALL_DENSE_WITH_STAGE_ACTOR
        )
        self._flat_reach_config = replace(
            self.reach_config, reward_form_mode=ALL_DENSE_WITH_STAGE_ACTOR
        )
        self.cfg.update({
            "composition_protocol": "same_scene_pick_push_reach_pilot_v1",
            "module_sequence": module_sequence,
            "stage_id_size": self.stage_id_size,
            "push_config": vars(self.push_config),
            "reach_config": vars(self.reach_config),
            "reach_waypoints_xy": [list(xy) for xy in REACH_WAYPOINTS_XY],
            "non_a_gripper_command": self.non_a_gripper_command,
            "non_a_gripper_commands": self.non_a_gripper_commands,
            "gripper_open_for_push_and_reach": self.non_a_gripper_command < 0.0,
            "native_success_not_used_for_mixed_task": True,
            "reward_protocol": reward_protocol,
            "policy_context": policy_context,
            "flat_dense_reward_definition": (
                "all_stage_quality_potential_difference_no_transition_bonus"
                if reward_protocol == "flat_dense" else None
            ),
        })

    def get_observations(self) -> TensorDict:
        if self.policy_context == "stage_onehot":
            return super().get_observations()
        tensor = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
        return TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[self.num_envs], device=self.device)

    def _module_reward_config(self, module: str):
        if self.reward_protocol != "flat_dense":
            return self.config if module == "A" else self.push_config if module == "B" else self.reach_config
        return self._flat_pick_config if module == "A" else self._flat_push_config if module == "B" else self._flat_reach_config

    def _global_stage(self, env_id: int) -> int:
        return 3 * int(self._macro_indices[env_id]) + int(self._local_stages[env_id])

    def _module(self, env_id: int) -> str:
        return self.module_sequence[int(self._macro_indices[env_id])]

    def _push_features(self, env_id: int):
        observation = self._obs[env_id]
        return push_features(
            observation[:3],
            observation[4:7],
            self._envs[env_id]._target_pos,
            self.push_config,
            gripper_opening=float(observation[3]),
        )

    def _reach_target(self, env_id: int) -> None:
        local_stage = int(self._local_stages[env_id])
        if local_stage >= 3:
            return
        xy = REACH_WAYPOINTS_XY[local_stage]
        object_z = float(self._object_position(env_id)[2])
        target = np.array((xy[0], xy[1], object_z + 0.16), dtype=np.float64)
        env = self._envs[env_id]
        env._target_pos = target
        env.model.site("goal").pos = target
        self._obs[env_id] = np.asarray(env._get_obs(), dtype=np.float32)

    def _start_module(self, env_id: int) -> None:
        self._local_stages[env_id] = 0
        module = self._module(env_id)
        if module == "A":
            state = GraspInsertState()
            state.reset(pick_quality(0, self._features(env_id), self.config))
        elif module == "B":
            state = ObjectMoveState()
            state.reset(push_quality(0, self._push_features(env_id), self.push_config))
        else:
            self._reach_target(env_id)
            state = TargetReachState()
            distance = float(np.linalg.norm(self._obs[env_id, :3] - self._envs[env_id]._target_pos))
            state.reset(reach_quality(distance, self.reach_config))
        self._states[env_id] = state

    def _reset_env(self, env_id: int, *, initial: bool = False) -> None:
        self._local_stages[env_id] = 0
        if hasattr(self, "_states"):
            self._states[env_id] = GraspInsertState()
        super()._reset_env(env_id, initial=initial)
        if self.module_sequence[0] != "A":
            self._start_module(env_id)

    def _advance_macro(self, env_id: int) -> None:
        macro = int(self._macro_indices[env_id])
        if macro >= self.macro_count - 1:
            return
        self._macro_indices[env_id] += 1
        self._macro_step_counts[env_id] = 0
        self._set_macro_target(env_id)
        self._initial_objects[env_id] = self._object_position(env_id)
        self._start_module(env_id)

    def step(self, actions: torch.Tensor):
        action_np = np.clip(actions.detach().to("cpu", dtype=torch.float32).numpy(), -1.0, 1.0)
        if action_np.shape != (self.num_envs, 4):
            raise ValueError(f"invalid action shape {action_np.shape}")
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        timeouts = np.zeros(self.num_envs, dtype=bool)
        due: list[int] = []
        self._global_step += 1
        for env_id, env in enumerate(self._envs):
            module = self._module(env_id)
            full_action = action_np[env_id].copy()
            if module != "A":
                full_action[3] = self.non_a_gripper_commands[module]
            observation, _, _, native_timeout, _ = env.step(full_action)
            self._obs[env_id] = np.asarray(observation, dtype=np.float32)
            self.episode_length_buf[env_id] += 1
            self._macro_step_counts[env_id] += 1
            state = self._states[env_id]
            reward_config = self._module_reward_config(module)
            if module == "A":
                result = pick_step(state, self._features(env_id), full_action, reward_config)
            elif module == "B":
                result = push_step(state, self._push_features(env_id), full_action[:3], reward_config)
            else:
                distance = float(np.linalg.norm(self._obs[env_id, :3] - env._target_pos))
                result = reach_step(
                    state, distance_m=distance, action=full_action[:3], config=reward_config
                )
            rewards[env_id] = result.reward
            if result.request_due:
                due.append(env_id)
            timeouts[env_id] = bool(
                native_timeout or self.episode_length_buf[env_id].item() >= self.max_episode_length
            )
        due = select_transition_due(self, due)
        if due:
            mark_transition_queries(self, due)
            self.gate_requests += len(due)
            for env_id in due:
                module = self._module(env_id)
                state = self._states[env_id]
                old_global = self._global_stage(env_id)
                if module == "A":
                    bonus = accept_pick(state, True, self.config)
                elif module == "B":
                    bonus = accept_push(state, self._push_features(env_id), True, self.push_config)
                else:
                    bonus = accept_reach(state, True, self.reach_config)
                if self.reward_protocol == "stage_machine":
                    rewards[env_id] += bonus
                self.gate_accepts += 1
                self.gate_predictions["success"] += 1
                self._local_stages[env_id] += 1
                self._transition_masks[env_id] |= 1 << old_global
                self.stage_entries[old_global + 1] += 1
                record_stage_transition(self, env_id)
                if module == "C" and self._local_stages[env_id] < 3:
                    self._reach_target(env_id)
                    reach_state = TargetReachState()
                    distance = float(np.linalg.norm(
                        self._obs[env_id, :3] - self._envs[env_id]._target_pos
                    ))
                    reach_state.reset(reach_quality(distance, self.reach_config))
                    self._states[env_id] = reach_state
                if self._local_stages[env_id] == 3 and self._macro_indices[env_id] < self.macro_count - 1:
                    self._advance_macro(env_id)

        terminal = np.array([
            self._macro_indices[i] == self.macro_count - 1 and self._local_stages[i] == 3
            for i in range(self.num_envs)
        ], dtype=bool)
        timeouts |= self._macro_step_counts >= self.max_macro_length
        timeouts[terminal] = False
        dones = terminal | timeouts
        self._returns += rewards
        for env_id in np.flatnonzero(dones):
            self.completed_episodes += 1
            mask = int(self._transition_masks[env_id])
            for prefix in range(1, self.macro_count + 1):
                required = (1 << (3 * prefix)) - 1
                self._strict_prefix_counts[prefix - 1] += int(mask & required == required)
            mandatory = bool(terminal[env_id] and mask == (1 << (3 * self.macro_count)) - 1)
            self.mandatory_success_episodes += int(mandatory)
            self._failed_macro_counts[int(self._macro_indices[env_id])] += int(not mandatory)
            self._reset_env(int(env_id))
        extras = {
            "time_outs": torch.as_tensor(timeouts, dtype=torch.bool, device=self.device),
            "log": {
                "/Gate/candidate_accept_rate": self.gate_accepts / max(self.gate_requests, 1),
                "/Success/mandatory_qwen": self.mandatory_success_episodes / max(self.completed_episodes, 1),
            },
        }
        return (
            self.get_observations(),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.long, device=self.device),
            extras,
        )

    def statistics(self) -> dict[str, object]:
        result = super().statistics()
        result["native_success_not_applicable"] = True
        result["module_sequence"] = self.module_sequence
        result["strict_gated_prefix_counts"] = self._strict_prefix_counts.tolist()
        result["strict_gated_prefix_rates"] = (
            self._strict_prefix_counts / max(self.completed_episodes, 1)
        ).tolist()
        result["failed_macro_counts"] = self._failed_macro_counts.tolist()
        if self.completed_episodes and self._strict_prefix_counts[-1] != self.mandatory_success_episodes:
            raise AssertionError("last strict prefix differs from mandatory success")
        return result

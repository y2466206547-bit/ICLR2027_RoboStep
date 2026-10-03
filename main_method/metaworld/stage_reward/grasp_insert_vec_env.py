"""Vector environment for the reusable grasp-and-insert CORE."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .reward_form_modes import uses_benchmark_native_dense
import torch
from tensordict import TensorDict

from metaworld import MT1
from rsl_rl.env import VecEnv
from stage_policy.live_vlm_gate import FrozenRgbFrame, RgbFrameRing

from .grasp_insert_core import (
    STAGE_COUNT,
    TERMINAL_STAGE,
    GraspInsertConfig,
    GraspInsertState,
    apply_qwen_decision,
    features,
    stage_quality,
    step,
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

GeometryProtocol = Literal[
    "native_forward_v0",
    "tabletop_supported_chain_v1",
    "tabletop_bidirectional_v2",
    "tabletop_balanced_chain_v2",
    "tabletop_square_chain_v3",
    "tabletop_square_chain_v4",
    "tabletop_square_chain_v5",
    "tabletop_unique_chain_v6",
    "tabletop_unique_chain_v8",
    "tabletop_unique_chain_v9",
    "tabletop_unique_chain_v10",
    "tabletop_reset_v1",
]

GateMode = Literal["rule", "qwen", "synthetic"]


class MetaWorldGraspInsertVecEnv(VecEnv):
    """One shared stage-conditioned actor; Qwen alone authorizes transitions."""

    def __init__(
        self,
        *,
        task: str,
        stage_instructions: tuple[str, ...],
        num_envs: int,
        seed: int,
        gate_mode: GateMode,
        qwen_gate: MandatoryQwenStageGate | None = None,
        config: GraspInsertConfig | None = None,
        device: str = "cpu",
        max_episode_length: int = 250,
        max_macro_length: int | None = None,
        render_width: int = 320,
        render_height: int = 320,
        camera_name: str = "corner3",
        ring_frame_count: int = 8,
        ring_capture_period_s: float = 0.10,
        macro_count: int = 1,
        stage_id_size: int | None = None,
        target_chain: tuple[tuple[float, float, float], ...] | None = None,
        source_macro_index: int = 0,
        geometry_protocol: GeometryProtocol = "tabletop_supported_chain_v1",
        reset_between_macros: bool = False,
        reset_arm_between_macros: bool = False,
        render_for_video: bool = False,
    ) -> None:
        if gate_mode not in ("rule", "qwen", "synthetic"):
            raise ValueError("gate_mode must be rule, qwen, or synthetic")
        if gate_mode == "qwen" and qwen_gate is None:
            raise ValueError("Qwen service must be present in qwen mode")
        if gate_mode != "qwen" and qwen_gate is not None:
            raise ValueError("Qwen service is only valid in qwen mode")
        self.task, self.num_envs = str(task), int(num_envs)
        self.num_actions = 4
        self.device = torch.device(device)
        self.gate_mode, self.qwen_gate = gate_mode, qwen_gate
        self.config = config or GraspInsertConfig()
        self.stage_instructions = tuple(stage_instructions)
        self.macro_count = int(macro_count)
        if self.macro_count < 1:
            raise ValueError("macro_count must be positive")
        self.source_macro_index = int(source_macro_index)
        if self.source_macro_index < 0:
            raise ValueError("source_macro_index must be non-negative")
        self.active_stage_count = TERMINAL_STAGE * self.macro_count
        if geometry_protocol not in (
            "native_forward_v0",
            "tabletop_supported_chain_v1",
            "tabletop_bidirectional_v2",
            "tabletop_balanced_chain_v2",
            "tabletop_square_chain_v3",
            "tabletop_square_chain_v4",
            "tabletop_square_chain_v5",
            "tabletop_unique_chain_v6",
            "tabletop_unique_chain_v8",
            "tabletop_unique_chain_v9",
            "tabletop_unique_chain_v10",
            "tabletop_reset_v1",
        ):
            raise ValueError(f"unknown geometry_protocol: {geometry_protocol}")
        self.geometry_protocol: GeometryProtocol = geometry_protocol
        self.reset_between_macros = bool(reset_between_macros)
        self.reset_arm_between_macros = bool(reset_arm_between_macros)
        if self.reset_between_macros and self.reset_arm_between_macros:
            raise ValueError("choose full environment reset or arm-only reset, not both")
        self.stage_id_size = int(stage_id_size or (self.active_stage_count + 1))
        if self.stage_id_size < self.active_stage_count + 1:
            raise ValueError("stage_id_size must cover all active stages and terminal")
        if len(self.stage_instructions) != self.active_stage_count:
            raise ValueError(
                "one Qwen instruction is required per atomic stage in every macro"
            )
        if target_chain is not None and len(target_chain) != self.macro_count:
            raise ValueError("target_chain must have macro_count targets")
        self._target_chain_template = (
            tuple(tuple(float(v) for v in target) for target in target_chain)
            if target_chain is not None
            else None
        )
        self.max_episode_length = int(max_episode_length)
        self.max_macro_length = int(max_macro_length or max_episode_length)
        if self.max_macro_length <= 0:
            raise ValueError("max_macro_length must be positive")
        self.render_width, self.render_height = int(render_width), int(render_height)
        self.camera_name, self.ring_capture_period_s = str(camera_name), float(ring_capture_period_s)
        self.cfg = {
            "task": self.task,
            "method": "PredStageActor",
            "gate_mode": self.gate_mode,
            "actor_observation": f"metaworld_state39_plus_qwen_stage_onehot{self.stage_id_size}",
            "critic_observation": "same_as_actor",
            "action": "standard_delta_eef_xyz_plus_gripper",
            "max_episode_length": self.max_episode_length,
            "max_macro_length": self.max_macro_length,
            "qwen_camera": self.camera_name if gate_mode == "qwen" else None,
            "reward_config": vars(self.config),
            "source_macro_index": self.source_macro_index,
            "geometry_protocol": self.geometry_protocol,
            "reset_between_macros": self.reset_between_macros,
            "reset_arm_between_macros": self.reset_arm_between_macros,
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
            if gate_mode == "qwen" or render_for_video
            else {}
        )
        self._envs = [env_class(**render_kwargs) for _ in range(self.num_envs)]
        for env_id, env in enumerate(self._envs):
            # MetaWorld defaults to a 500-step native horizon. For continuous
            # composition the wrapper owns the total horizon (M * base horizon).
            env.max_path_length = self.max_episode_length
            env.seed(int(seed) + env_id)
        self._obs = np.zeros((self.num_envs, 39), dtype=np.float32)
        self._initial_objects = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._states = [GraspInsertState() for _ in range(self.num_envs)]
        self._macro_indices = np.zeros(self.num_envs, dtype=np.int64)
        self._macro_step_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._macro_success_seen = np.zeros(self.num_envs, dtype=bool)
        self._macro_success_masks = np.zeros(self.num_envs, dtype=np.int64)
        self._macro_success_episodes = np.zeros(self.macro_count, dtype=np.int64)
        self._prefix_success_episodes = np.zeros(self.macro_count, dtype=np.int64)
        self._target_chains: list[tuple[np.ndarray, ...]] = [tuple() for _ in range(self.num_envs)]
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._raw_success_seen = np.zeros(self.num_envs, dtype=bool)
        self._transition_masks = np.zeros(self.num_envs, dtype=np.int64)
        self._returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._global_step = 0
        self._source_frame_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._next_ring_time_s = np.zeros(self.num_envs, dtype=np.float64)
        self._rings: list[RgbFrameRing] = []
        self._references: list[FrozenRgbFrame | None] = [None] * self.num_envs
        if gate_mode == "qwen":
            self._rings = [
                RgbFrameRing(frame_count=ring_frame_count, capture_period_s=self.ring_capture_period_s)
                for _ in range(self.num_envs)
            ]
        self._canonical_objects = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.completed_episodes = 0
        self.raw_success_episodes = 0
        self.mandatory_success_episodes = 0
        self.stage_entries = np.zeros(self.stage_id_size, dtype=np.int64)
        self.gate_requests = self.gate_accepts = 0
        self.gate_predictions = {"success": 0, "failure": 0, "unknown": 0}
        configure_transition_schedule(self, self.cfg)
        for env_id in range(self.num_envs):
            self._reset_env(env_id, initial=True)

    @property
    def unwrapped(self) -> "MetaWorldGraspInsertVecEnv":
        return self

    def _task_index(self, env_id: int) -> int:
        return int((env_id + self._episode_ids[env_id] * self.num_envs) % len(self._tasks))

    def _features(self, env_id: int):
        env = self._envs[env_id]
        target = np.asarray(env._target_pos, dtype=np.float64).copy()
        if self.config.target_z_override_m is not None:
            target[2] = float(self.config.target_z_override_m)
        return features(
            self._obs[env_id, :3],
            self._obs[env_id, 4:7],
            target,
            self._initial_objects[env_id],
            self.config,
            float(self._obs[env_id, 3]),
        )

    def _global_stage(self, env_id: int) -> int:
        return int(
            self._macro_indices[env_id] * TERMINAL_STAGE
            + self._states[env_id].stage
        )

    @staticmethod
    def _default_target_chain(
        initial_target: np.ndarray,
        initial_object: np.ndarray,
        macro_count: int,
        *,
        geometry_protocol: GeometryProtocol,
    ) -> tuple[np.ndarray, ...]:
        """Create versioned destinations while retaining old-run reproducibility."""
        first = np.asarray(initial_target, dtype=np.float64).copy()
        source = np.asarray(initial_object, dtype=np.float64).copy()
        if first.shape != (3,) or not np.all(np.isfinite(first)):
            raise ValueError("initial target must be a finite three-vector")
        if source.shape != (3,) or not np.all(np.isfinite(source)):
            raise ValueError("initial object must be a finite three-vector")
        if geometry_protocol == "tabletop_bidirectional_v2":
            # Every macro traverses the exact same per-episode displacement,
            # alternating direction. This removes v1's unequal task lengths
            # while keeping every macro handoff on the support surface.
            first[2] = source[2]
            return tuple(
                (first if index % 2 == 0 else source).copy()
                for index in range(macro_count)
            )
        if geometry_protocol == "tabletop_reset_v1":
            first[2] = source[2]
            return tuple(first.copy() for _ in range(macro_count))
        if geometry_protocol == "tabletop_square_chain_v3":
            # Four distinct legal tabletop positions on a square path.
            # Every macro remains the same object-to-target primitive, while
            # avoiding a two-point ping-pong sequence.
            z = float(source[2])
            square = (
                np.array([-0.04, 0.815, z], dtype=np.float64),
                np.array([0.04, 0.815, z], dtype=np.float64),
                np.array([0.04, 0.895, z], dtype=np.float64),
                np.array([-0.04, 0.895, z], dtype=np.float64),
            )
            return tuple(square[index % len(square)].copy() for index in range(macro_count))
        if geometry_protocol == "tabletop_square_chain_v4":
            # Safer four-position square: top y=0.88 stays off the workspace
            # edge, while all adjacent displacements exceed the 7.5 cm place
            # threshold so a macro cannot pass without moving the object.
            z = float(source[2])
            square = (
                np.array([-0.075, 0.800, z], dtype=np.float64),
                np.array([0.075, 0.800, z], dtype=np.float64),
                np.array([0.075, 0.880, z], dtype=np.float64),
                np.array([-0.075, 0.880, z], dtype=np.float64),
            )
            return tuple(square[index % len(square)].copy() for index in range(macro_count))
        if geometry_protocol == "tabletop_square_chain_v5":
            # Keep every object start in the lower safe band.  The four points
            # are still distinct and every adjacent move is a large diagonal
            # or horizontal displacement, so no macro is a two-point ping-pong.
            z = float(source[2])
            square = (
                np.array([-0.080, 0.800, z], dtype=np.float64),
                np.array([0.080, 0.800, z], dtype=np.float64),
                np.array([-0.080, 0.840, z], dtype=np.float64),
                np.array([0.080, 0.840, z], dtype=np.float64),
            )
            return tuple(square[index % len(square)].copy() for index in range(macro_count))
        if geometry_protocol == "tabletop_unique_chain_v6":
            # Preserve the exact A--D prefix from v5, then append a fifth safe
            # tabletop destination rather than cycling back to A.  The D->E
            # displacement is comparable to B->C and remains well above the
            # 7.5 cm place threshold.
            z = float(source[2])
            unique = (
                np.array([-0.080, 0.800, z], dtype=np.float64),
                np.array([0.080, 0.800, z], dtype=np.float64),
                np.array([-0.080, 0.840, z], dtype=np.float64),
                np.array([0.080, 0.840, z], dtype=np.float64),
                np.array([-0.080, 0.880, z], dtype=np.float64),
            )
            if macro_count > len(unique):
                raise ValueError(
                    "tabletop_unique_chain_v6 supports at most five distinct targets"
                )
            return tuple(target.copy() for target in unique[:macro_count])
        if geometry_protocol == "tabletop_unique_chain_v8":
            # Ten distinct safe targets.  A--E exactly preserve v6; later
            # points continue alternating across the table so every handoff
            # remains a substantial move rather than a near-duplicate target.
            z = float(source[2])
            unique = (
                np.array([-0.080, 0.800, z], dtype=np.float64),
                np.array([0.080, 0.800, z], dtype=np.float64),
                np.array([-0.080, 0.840, z], dtype=np.float64),
                np.array([0.080, 0.840, z], dtype=np.float64),
                np.array([-0.080, 0.880, z], dtype=np.float64),
                np.array([0.080, 0.880, z], dtype=np.float64),
                np.array([-0.080, 0.820, z], dtype=np.float64),
                np.array([0.080, 0.820, z], dtype=np.float64),
                np.array([-0.080, 0.860, z], dtype=np.float64),
                np.array([0.080, 0.860, z], dtype=np.float64),
            )
            if macro_count > len(unique):
                raise ValueError(
                    "tabletop_unique_chain_v8 supports at most ten distinct targets"
                )
            return tuple(target.copy() for target in unique[:macro_count])
        if geometry_protocol == "tabletop_unique_chain_v9":
            # M11 extension of v8: preserve the first ten destinations exactly
            # and append one new central lower-band destination.  The eleventh
            # point is distinct from all v8 points and the final handoff is a
            # substantial move, so M11 does not silently repeat a target.
            z = float(source[2])
            unique = (
                np.array([-0.080, 0.800, z], dtype=np.float64),
                np.array([0.080, 0.800, z], dtype=np.float64),
                np.array([-0.080, 0.840, z], dtype=np.float64),
                np.array([0.080, 0.840, z], dtype=np.float64),
                np.array([-0.080, 0.880, z], dtype=np.float64),
                np.array([0.080, 0.880, z], dtype=np.float64),
                np.array([-0.080, 0.820, z], dtype=np.float64),
                np.array([0.080, 0.820, z], dtype=np.float64),
                np.array([-0.080, 0.860, z], dtype=np.float64),
                np.array([0.080, 0.860, z], dtype=np.float64),
                np.array([0.000, 0.800, z], dtype=np.float64),
            )
            if macro_count > len(unique):
                raise ValueError(
                    "tabletop_unique_chain_v9 supports at most eleven distinct targets"
                )
            return tuple(target.copy() for target in unique[:macro_count])
        if geometry_protocol == "tabletop_unique_chain_v10":
            # M12--M15 extension: preserve the complete v9 prefix and append
            # four new tabletop destinations.  The points remain inside the
            # central legal goal region and are all distinct from the first
            # eleven targets, so longer runs never silently revisit a goal.
            z = float(source[2])
            unique = (
                np.array([-0.080, 0.800, z], dtype=np.float64),
                np.array([0.080, 0.800, z], dtype=np.float64),
                np.array([-0.080, 0.840, z], dtype=np.float64),
                np.array([0.080, 0.840, z], dtype=np.float64),
                np.array([-0.080, 0.880, z], dtype=np.float64),
                np.array([0.080, 0.880, z], dtype=np.float64),
                np.array([-0.080, 0.820, z], dtype=np.float64),
                np.array([0.080, 0.820, z], dtype=np.float64),
                np.array([-0.080, 0.860, z], dtype=np.float64),
                np.array([0.080, 0.860, z], dtype=np.float64),
                np.array([0.000, 0.800, z], dtype=np.float64),
                np.array([0.000, 0.880, z], dtype=np.float64),
                np.array([0.000, 0.840, z], dtype=np.float64),
                np.array([-0.040, 0.800, z], dtype=np.float64),
                np.array([0.040, 0.880, z], dtype=np.float64),
            )
            if macro_count > len(unique):
                raise ValueError(
                    "tabletop_unique_chain_v10 supports at most fifteen distinct targets"
                )
            return tuple(target.copy() for target in unique[:macro_count])
        if geometry_protocol == "tabletop_balanced_chain_v2":
            # Keep every appended destination inside the central goal support
            # region.  The old supported-chain protocol sent B/C to +/- 9cm
            # workspace edges, making the suffix tasks materially unequal.
            first[2] = source[2]
            left = np.array([-0.06, float(first[1]), float(first[2])], dtype=np.float64)
            right = np.array([0.06, float(first[1]), float(first[2])], dtype=np.float64)
            targets = [first.copy()]
            for index in range(1, macro_count):
                targets.append((right if index % 2 == 1 else left).copy())
            return tuple(targets)
        if geometry_protocol == "tabletop_supported_chain_v1":
            # v1 is retained so its existing checkpoints remain evaluable.
            first[2] = source[2]
        elif geometry_protocol != "native_forward_v0":
            raise ValueError(f"unknown geometry_protocol: {geometry_protocol}")
        # The later destinations stay inside the native pick-place goal box and
        # are separated enough to be distinct under the 7 cm success radius.
        offsets = ((0.14, 0.0), (-0.14, 0.0), (0.0, 0.075), (0.0, -0.075))
        targets = [first]
        base_x = float(np.clip(first[0], -0.04, 0.04))
        base_y = float(np.clip(first[1], 0.825, 0.875))
        for index in range(1, macro_count):
            dx, dy = offsets[(index - 1) % len(offsets)]
            targets.append(
                np.array(
                    [
                        np.clip(base_x + dx, -0.09, 0.09),
                        np.clip(base_y + dy, 0.81, 0.89),
                        first[2],
                    ],
                    dtype=np.float64,
                )
            )
        return tuple(targets)

    def _object_position(self, env_id: int) -> np.ndarray:
        return np.asarray(
            self._envs[env_id]._get_pos_objects(), dtype=np.float64
        ).copy()

    def _set_macro_target(self, env_id: int) -> None:
        env = self._envs[env_id]
        target = self._target_chains[env_id][int(self._macro_indices[env_id])]
        env._target_pos = np.asarray(target, dtype=np.float64).copy()
        # Some native MetaWorld XMLs (notably assembly/bin-picking) do not
        # expose a visual goal site.  The target is still carried by
        # env._target_pos and the observation; the marker update is optional.
        try:
            env.model.site("goal").pos = env._target_pos
        except (KeyError, ValueError):
            pass
        # The native observation stores the goal in its final three entries.
        # Refresh it after every target switch; otherwise a suffix policy sees
        # the previous target while reward/candidate logic uses the new one.
        self._obs[env_id] = np.asarray(env._get_obs(), dtype=np.float32)

    def _reset_qwen_reference(self, env_id: int) -> None:
        if self.gate_mode != "qwen":
            return
        frame = self._render_frame(env_id)
        self._references[env_id] = frame
        self._rings[env_id].reset(frame)
        self._next_ring_time_s[env_id] = frame.sim_time_s + self.ring_capture_period_s

    def _advance_macro(self, env_id: int) -> None:
        """Start the next 3-atomic-stage macro without resetting the episode."""
        macro = int(self._macro_indices[env_id])
        if macro >= self.macro_count - 1:
            return
        self._macro_success_masks[env_id] |= int(self._macro_success_seen[env_id]) << macro
        self._macro_success_seen[env_id] = False
        self._macro_indices[env_id] += 1
        self._macro_step_counts[env_id] = 0
        if self.reset_between_macros:
            env = self._envs[env_id]
            obs, _ = env.reset()
            env._set_obj_xyz(self._canonical_objects[env_id])
            env.obj_init_pos = self._canonical_objects[env_id].copy()
            self._obs[env_id] = np.asarray(env._get_obs(), dtype=np.float32)
            # Full reset mode restarts both robot and object for an isolated macro.
            self._set_macro_target(env_id)
            self._initial_objects[env_id] = self._canonical_objects[env_id].copy()
            self._reset_qwen_reference(env_id)
            reset_transition_schedule(self, env_id)
        elif self.reset_arm_between_macros:
            # Arm-only reset keeps the object at the previous macro destination,
            # while returning the Sawyer TCP/gripper to its native initial pose.
            env = self._envs[env_id]
            env._reset_hand()
            self._obs[env_id] = np.asarray(env._get_obs(), dtype=np.float32)
            self._set_macro_target(env_id)
            self._initial_objects[env_id] = self._object_position(env_id)
            self._reset_qwen_reference(env_id)
            reset_transition_schedule(self, env_id)
        else:
            self._set_macro_target(env_id)
            self._initial_objects[env_id] = self._object_position(env_id)
        self._states[env_id].reset(stage_quality(0, self._features(env_id), self.config))

    def _record_transition(self, env_id: int, old_global_stage: int) -> None:
        self._transition_masks[env_id] |= 1 << old_global_stage
        self.stage_entries[old_global_stage + 1] += 1
        # A macro boundary exists only after its local terminal atomic stage.
        # Earlier atomic transitions advance the stage index within the same
        # macro and must not change the target or reset the local state.
        if old_global_stage % TERMINAL_STAGE == TERMINAL_STAGE - 1:
            self._advance_macro(env_id)


    def _render_frame(self, env_id: int) -> FrozenRgbFrame:
        rgb = np.asarray(self._envs[env_id].render(), dtype=np.uint8)
        if rgb.shape != (self.render_height, self.render_width, 3):
            raise RuntimeError(f"unexpected render shape {rgb.shape}")
        frame = FrozenRgbFrame(
            rgb=np.ascontiguousarray(rgb).tobytes(),
            width=self.render_width,
            height=self.render_height,
            global_step=int(self._global_step),
            sim_time_s=float(self.episode_length_buf[env_id].item() * self._envs[env_id].dt),
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
        native_target = np.asarray(env._target_pos, dtype=np.float64).copy()
        if self._target_chain_template is not None:
            if self.source_macro_index:
                raise ValueError("source_macro_index cannot be combined with an explicit target_chain")
            active_chain = tuple(np.asarray(target, dtype=np.float64).copy() for target in self._target_chain_template)
        else:
            full_chain = self._default_target_chain(
                native_target,
                np.asarray(env.obj_init_pos, dtype=np.float64),
                self.source_macro_index + self.macro_count,
                geometry_protocol=self.geometry_protocol,
            )
            active_chain = full_chain[self.source_macro_index : self.source_macro_index + self.macro_count]
            if self.source_macro_index:
                env._set_obj_xyz(full_chain[self.source_macro_index - 1])
                env.obj_init_pos = np.asarray(full_chain[self.source_macro_index - 1], dtype=np.float64).copy()
                obs = env._get_obs()
        self._obs[env_id] = np.asarray(obs, dtype=np.float32)
        self._target_chains[env_id] = active_chain
        self._canonical_objects[env_id] = np.asarray(env.obj_init_pos, dtype=np.float64).copy()
        self._macro_indices[env_id] = 0
        self._macro_step_counts[env_id] = 0
        self._macro_success_seen[env_id] = False
        self._macro_success_masks[env_id] = 0
        self._set_macro_target(env_id)
        self._initial_objects[env_id] = np.asarray(env.obj_init_pos, dtype=np.float64)
        value = self._features(env_id)
        self._states[env_id].reset(stage_quality(0, value, self.config))
        self._raw_success_seen[env_id] = False
        self._transition_masks[env_id] = 0
        self._returns[env_id] = 0.0
        self.episode_length_buf[env_id] = 0
        self.stage_entries[0] += 1
        self._reset_qwen_reference(env_id)
        reset_transition_schedule(self, env_id)

    def _capture_due_frames(self) -> None:
        if self.gate_mode != "qwen":
            return
        for env_id, env in enumerate(self._envs):
            now = float(self.episode_length_buf[env_id].item() * env.dt)
            if now + 1.0e-9 < self._next_ring_time_s[env_id]:
                continue
            while self._next_ring_time_s[env_id] <= now + 1.0e-9:
                self._next_ring_time_s[env_id] += self.ring_capture_period_s
            self._rings[env_id].maybe_append(self._render_frame(env_id))

    def _candidate(self, env_id: int) -> GateCandidate:
        state = self._states[env_id]
        ring = self._rings[env_id]
        current = self._render_frame(env_id)
        ring.maybe_append(current)
        reference = self._references[env_id]
        if reference is None:
            raise RuntimeError("candidate has no reference frame")
        return GateCandidate(
            env_id=env_id,
            episode_id=int(self._episode_ids[env_id]),
            stage_id=self._global_stage(env_id),
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

    def get_observations(self) -> TensorDict:
        stages = np.fromiter((self._global_stage(i) for i in range(self.num_envs)), dtype=np.int64, count=self.num_envs)
        values = np.concatenate((self._obs, np.eye(self.stage_id_size, dtype=np.float32)[stages]), axis=1)
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        action_np = np.clip(actions.detach().to("cpu", dtype=torch.float32).numpy(), -1.0, 1.0)
        if action_np.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"unexpected action shape {action_np.shape}")
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        native_rewards = np.zeros(self.num_envs, dtype=np.float32)
        timeouts = np.zeros(self.num_envs, dtype=bool)
        due: list[int] = []
        values = []
        self._global_step += 1
        for env_id, env in enumerate(self._envs):
            obs, native_reward, _, native_timeout, info = env.step(action_np[env_id])
            native_rewards[env_id] = float(native_reward)
            self._obs[env_id] = np.asarray(obs, dtype=np.float32)
            self.episode_length_buf[env_id] += 1
            self._macro_step_counts[env_id] += 1
            value = self._features(env_id)
            values.append(value)
            result = step(self._states[env_id], value, action_np[env_id], self.config)
            rewards[env_id] = result.reward
            if result.request_due:
                due.append(env_id)
            self._macro_success_seen[env_id] |= bool(info["success"])
            timeouts[env_id] = bool(native_timeout or self.episode_length_buf[env_id].item() >= self.max_episode_length)
        self._capture_due_frames()
        if synthetic_gate_enabled(self):
            opportunities = select_synthetic_gate_opportunities(self, due)
            due = [opportunity.env_id for opportunity in opportunities]
        else:
            opportunities = []
            due = select_transition_due(self, due)
        if due:
            mark_transition_queries(self, due)
            if self.gate_mode == "rule":
                self.gate_requests += len(due)
                predictions, decisions = ["success"] * len(due), [True] * len(due)
                for env_id, prediction, decision in zip(due, predictions, decisions, strict=True):
                    self.gate_predictions[prediction] += 1
                    old_stage = self._states[env_id].stage
                    old_global_stage = self._global_stage(env_id)
                    reward = apply_qwen_decision(
                        self._states[env_id],
                        decision,
                        self.config,
                        bypass_candidate=bypass_rule_candidate(self, env_id),
                    )
                    rewards[env_id] += reward
                    if decision:
                        self.gate_accepts += 1
                        if self._states[env_id].stage > old_stage:
                            record_stage_transition(self, env_id)
                        self._record_transition(env_id, old_global_stage)
            elif synthetic_gate_enabled(self):
                synthetic_decisions = decide_synthetic_gate(self, opportunities)
                self.gate_requests += len(synthetic_decisions)
                for synthetic_decision in synthetic_decisions:
                    env_id = synthetic_decision.env_id
                    self.gate_predictions[synthetic_decision.prediction] += 1
                    old_stage = self._states[env_id].stage
                    old_global_stage = self._global_stage(env_id)
                    new_stage = old_stage
                    transitioned = False
                    if synthetic_decision.gt_complete or synthetic_decision.accepted:
                        rewards[env_id] += apply_qwen_decision(
                            self._states[env_id],
                            synthetic_decision.accepted,
                            self.config,
                            bypass_candidate=not synthetic_decision.gt_complete,
                        )
                        new_stage = self._states[env_id].stage
                        transitioned = new_stage > old_stage
                    if synthetic_decision.accepted:
                        self.gate_accepts += 1
                        if transitioned:
                            record_stage_transition(self, env_id)
                        self._record_transition(env_id, old_global_stage)
                    record_synthetic_gate_outcome(
                        self,
                        synthetic_decision,
                        old_stage=int(old_stage),
                        new_stage=int(new_stage),
                        transitioned=transitioned,
                    )
            else:
                predictions, decisions, qwen_requests = decide_qwen_with_repeated_accepts(
                    self, [self._candidate(i) for i in due]
                )
                self.gate_requests += qwen_requests
                for env_id, prediction, decision in zip(due, predictions, decisions, strict=True):
                    self.gate_predictions[prediction] += 1
                    old_stage = self._states[env_id].stage
                    old_global_stage = self._global_stage(env_id)
                    reward = apply_qwen_decision(
                        self._states[env_id],
                        decision,
                        self.config,
                        bypass_candidate=bypass_rule_candidate(self, env_id),
                    )
                    rewards[env_id] += reward
                    if decision:
                        self.gate_accepts += 1
                        if self._states[env_id].stage > old_stage:
                            record_stage_transition(self, env_id)
                        self._record_transition(env_id, old_global_stage)
        terminal = np.fromiter(
            (
                self._macro_indices[i] == self.macro_count - 1
                and self._states[i].stage == TERMINAL_STAGE
                for i in range(self.num_envs)
            ),
            dtype=bool,
            count=self.num_envs,
        )
        # Each composed macro receives the same local horizon as its isolated
        # counterpart. A successful boundary resets only this counter; robot
        # and object state remain physically continuous.
        macro_timeouts = self._macro_step_counts >= self.max_macro_length
        timeouts = np.logical_or(timeouts, macro_timeouts)
        # A terminal success on the last allowed step is a task termination,
        # not a timeout bootstrap.
        timeouts[terminal] = False
        dones = np.logical_or(terminal, timeouts)
        if uses_benchmark_native_dense(self.config):
            rewards = native_rewards
        self._returns += rewards
        for env_id in np.flatnonzero(dones):
            self.completed_episodes += 1
            complete_mask = int(self._macro_success_masks[env_id])
            current_macro = int(self._macro_indices[env_id])
            complete_mask |= int(self._macro_success_seen[env_id]) << current_macro
            full_macro_mask = (1 << self.macro_count) - 1
            for macro_index in range(self.macro_count):
                self._macro_success_episodes[macro_index] += (complete_mask >> macro_index) & 1
                prefix_mask = (1 << (macro_index + 1)) - 1
                self._prefix_success_episodes[macro_index] += int(complete_mask & prefix_mask == prefix_mask)
            raw = bool(complete_mask == full_macro_mask)
            # `_global_stage` is relative to the active composition, including
            # for an isolated suffix run (`source_macro_index > 0`), so its
            # transition mask remains local to the active stages.
            full_stage_mask = (1 << self.active_stage_count) - 1
            mandatory = bool(terminal[env_id] and self._transition_masks[env_id] == full_stage_mask and raw)
            self.raw_success_episodes += int(raw)
            self.mandatory_success_episodes += int(mandatory)
            self._reset_env(int(env_id))
        gate_rate = self.gate_accepts / self.gate_requests if self.gate_requests else 0.0
        raw_rate = self.raw_success_episodes / self.completed_episodes if self.completed_episodes else 0.0
        mandatory_rate = self.mandatory_success_episodes / self.completed_episodes if self.completed_episodes else 0.0
        extras = {
            "time_outs": torch.as_tensor(timeouts, dtype=torch.bool, device=self.device),
            "log": {
                "/Gate/candidate_accept_rate": float(gate_rate),
                "/Success/raw": float(raw_rate),
                "/Success/mandatory_qwen": float(mandatory_rate),
            },
        }
        return self.get_observations(), torch.as_tensor(rewards, dtype=torch.float32, device=self.device), torch.as_tensor(dones, dtype=torch.long, device=self.device), extras

    def statistics(self) -> dict[str, object]:
        payload = {
            "task": self.task,
            "gate_mode": self.gate_mode,
            "macro_count": self.macro_count,
            "active_atomic_stage_count": self.active_stage_count,
            "stage_id_size": self.stage_id_size,
            "completed_episodes": self.completed_episodes,
            "raw_success_episodes": self.raw_success_episodes,
            "raw_success_rate": self.raw_success_episodes / self.completed_episodes if self.completed_episodes else 0.0,
            "mandatory_success_episodes": self.mandatory_success_episodes,
            "mandatory_success_rate": self.mandatory_success_episodes / self.completed_episodes if self.completed_episodes else 0.0,
            "macro_success_episodes": self._macro_success_episodes.tolist(),
            "macro_success_rates": (self._macro_success_episodes / self.completed_episodes).tolist() if self.completed_episodes else [0.0] * self.macro_count,
            "prefix_success_episodes": self._prefix_success_episodes.tolist(),
            "prefix_success_rates": (self._prefix_success_episodes / self.completed_episodes).tolist() if self.completed_episodes else [0.0] * self.macro_count,
            "stage_entries": self.stage_entries.tolist(),
            "gate_requests": self.gate_requests,
            "gate_accepts": self.gate_accepts,
            "gate_candidate_accept_rate": self.gate_accepts / self.gate_requests if self.gate_requests else 0.0,
            "gate_predictions": dict(self.gate_predictions),
            "qwen_protocol_errors": self.qwen_gate.protocol_errors if self.qwen_gate else None,
            "qwen_accuracy": None,
            "qwen_accuracy_reason": "runtime candidates lack independent visual labels",
        }
        synthetic = synthetic_gate_summary(self)
        if synthetic is not None:
            payload["synthetic_gate"] = synthetic
        return payload

    def close(self) -> None:
        for env in self._envs:
            env.close()
        if self.qwen_gate is not None:
            self.qwen_gate.close()
            self.qwen_gate = None
        close_synthetic_gate_log(self)

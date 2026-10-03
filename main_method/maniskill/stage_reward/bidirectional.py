"""Bidirectional extension of the staged reward machine.

The training machine is forward-only by default.  This module is used by the
controlled recovery experiment and leaves the training implementation
unchanged.  A persistent prerequisite predicate can move the active stage
back by one step.  Positive downstream credit is suppressed as soon as the
prerequisite becomes invalid, including during rollback hysteresis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from stage_reward.core import BatchedStageReward, StageRewardConfig


PrerequisiteFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BidirectionalBatchedStageReward(BatchedStageReward):
    """A staged reward machine with one-step prerequisite rollback.

    ``prerequisite_fn`` returns a boolean matrix of shape
    ``(num_envs, num_stages + 1)``.  Column ``j`` says whether the state still
    satisfies the prerequisite for being in stage ``j``.  The extra column is
    the prerequisite for the terminal state.
    """

    def __init__(
        self,
        num_envs: int,
        num_stages: int,
        device: torch.device | str,
        config: StageRewardConfig,
        prerequisite_fn: PrerequisiteFn,
        rollback_dwell_steps: int = 2,
    ) -> None:
        super().__init__(num_envs, num_stages, device, config)
        if rollback_dwell_steps <= 0:
            raise ValueError("rollback_dwell_steps must be positive")
        self.prerequisite_fn = prerequisite_fn
        self.rollback_dwell_steps = int(rollback_dwell_steps)
        self.rollback_enabled = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.rollback_dwell = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(
        self, potentials: torch.Tensor, env_idx: torch.Tensor | None = None
    ) -> None:
        super().reset(potentials, env_idx=env_idx)
        if env_idx is None:
            self.rollback_dwell.zero_()
        else:
            env_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            )
            self.rollback_dwell[env_idx] = 0

    def step(
        self,
        potentials: torch.Tensor,
        gates: torch.Tensor,
        maintenance: torch.Tensor | None = None,
        safety_penalty: torch.Tensor | None = None,
        transition_authorization: torch.Tensor | None = None,
        reward_form_mode: str = "staged_reset",
        transition_authority_mode: str = "auto",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        stage_before = self.stage.clone()
        prerequisites = self.prerequisite_fn(potentials, gates).bool()
        expected_shape = (self.num_envs, self.num_stages + 1)
        if prerequisites.shape != expected_shape:
            raise ValueError(
                "prerequisites must have shape "
                f"{expected_shape}, found {tuple(prerequisites.shape)}"
            )

        rows = torch.arange(self.num_envs, device=self.device)
        prerequisite_valid = prerequisites[rows, stage_before]
        invalid_downstream = (
            self.rollback_enabled & (stage_before > 0) & (~prerequisite_valid)
        )
        self.rollback_dwell = torch.where(
            invalid_downstream,
            self.rollback_dwell + 1,
            torch.zeros_like(self.rollback_dwell),
        )
        rollback = invalid_downstream & (
            self.rollback_dwell >= self.rollback_dwell_steps
        )

        # A regressed state must not also advance on an instantaneous gate bit.
        effective_gates = gates.clone()
        active_invalid = invalid_downstream & (stage_before < self.num_stages)
        if active_invalid.any():
            effective_gates[
                rows[active_invalid], stage_before[active_invalid]
            ] = False

        reward, diagnostics = super().step(
            potentials=potentials,
            gates=effective_gates,
            maintenance=maintenance,
            safety_penalty=safety_penalty,
            transition_authorization=transition_authorization,
            reward_form_mode=reward_form_mode,
            transition_authority_mode=transition_authority_mode,
        )

        # Keep genuine negative/safety feedback, but never issue positive
        # downstream credit while its prerequisite is known to be invalid.
        reward = torch.where(
            invalid_downstream,
            torch.minimum(reward, torch.zeros_like(reward)),
            reward,
        )

        rollback_target = (stage_before - 1).clamp(min=0)
        self.stage = torch.where(rollback, rollback_target, self.stage)
        if rollback.any():
            rollback_rows = rows[rollback]
            rollback_cols = self.stage[rollback_rows].clamp(
                max=self.num_stages - 1
            )
            self.prev_potential[rollback_rows] = potentials[
                rollback_rows, rollback_cols
            ]
            self.dwell[rollback_rows] = 0
            self.rollback_dwell[rollback_rows] = 0

        diagnostics.update(
            {
                "stage_after": self.stage.clone(),
                "prerequisite_valid": prerequisite_valid,
                "invalid_downstream": invalid_downstream,
                "rollback": rollback,
                "rollback_from_stage": torch.where(
                    rollback, stage_before, torch.full_like(stage_before, -1)
                ),
                "rollback_to_stage": torch.where(
                    rollback, rollback_target, torch.full_like(stage_before, -1)
                ),
                "rollback_dwell": self.rollback_dwell.clone(),
            }
        )
        return reward, diagnostics


def make_task_prerequisite_fn(env_id: str, env: Any | None = None) -> PrerequisiteFn:
    """Return frozen prerequisite logic derived from Frame-VLM stage gates."""

    if env_id == "PickCube-v1":

        def pick_cube(
            potentials: torch.Tensor, gates: torch.Tensor
        ) -> torch.Tensor:
            del potentials
            always = torch.ones_like(gates[:, 0], dtype=torch.bool)
            # Stage 2 requires a maintained grasp unless placement is already
            # complete.  Terminal stage 3 requires the placement predicate.
            stage2 = gates[:, 1] | gates[:, 2]
            return torch.stack(
                (always, gates[:, 0] | gates[:, 1] | gates[:, 2], stage2, gates[:, 2]), dim=1
            )

        return pick_cube

    if env_id == "PlaceSphere-v1":

        def place_sphere(
            potentials: torch.Tensor, gates: torch.Tensor
        ) -> torch.Tensor:
            always = torch.ones_like(gates[:, 0], dtype=torch.bool)
            stage1 = (potentials[:, 0] >= 0.55) | gates[:, 1] | gates[:, 3]
            stage2 = gates[:, 1] | (potentials[:, 0] >= 0.65) | gates[:, 3]
            stage3 = gates[:, 2] | gates[:, 3]
            return torch.stack(
                (always, stage1, stage2, stage3, gates[:, 3]), dim=1
            )

        return place_sphere

    if env_id == "RotateValveLevel2-v1":
        if env is None:
            raise ValueError("RotateValve prerequisite requires the simulator env")

        def rotate_valve(
            potentials: torch.Tensor, gates: torch.Tensor
        ) -> torch.Tensor:
            del potentials
            always = torch.ones_like(gates[:, 0], dtype=torch.bool)
            signed_rotation = (
                (env.valve.qpos - env.rest_qpos)[:, 0] * env.rotate_direction
            )
            turn_started = (
                signed_rotation >= 0.02 * float(env.success_threshold)
            ) | gates[:, 1]
            return torch.stack((always, turn_started, gates[:, 1]), dim=1)

        return rotate_valve

    raise KeyError(f"No backward prerequisite definition for {env_id}")

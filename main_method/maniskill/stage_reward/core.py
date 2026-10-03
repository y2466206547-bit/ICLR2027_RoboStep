"""Vectorized active-stage potential reward state machine.

For the active stage i, this implements

    d_i(t) = s_i clip(gamma * phi_i(x_t) - phi_i(x_{t-1}), -delta, delta)
    r_t = d_i(t) + b_i e_i(t) + m_i(t) - p_safety(t)

Only the pre-transition stage contributes on a transition step. The next
stage's potential is initialized at the transition state and becomes active on
the following step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class StageRewardConfig:
    gamma: float = 0.99
    delta_max: float = 0.1
    dense_scales: Sequence[float] = (1.0,)
    transition_bonuses: Sequence[float] = (0.5,)
    dwell_steps: Sequence[int] = (1,)


class BatchedStageReward:
    """Stateful reward calculator for vectorized environments."""

    def __init__(
        self,
        num_envs: int,
        num_stages: int,
        device: torch.device | str,
        config: StageRewardConfig,
    ) -> None:
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if len(config.dense_scales) != num_stages:
            raise ValueError("dense_scales must have one value per stage")
        if len(config.transition_bonuses) != num_stages:
            raise ValueError("transition_bonuses must have one value per stage")
        if len(config.dwell_steps) != num_stages:
            raise ValueError("dwell_steps must have one value per stage")
        if any(value <= 0 for value in config.dwell_steps):
            raise ValueError("dwell_steps values must be positive")

        self.num_envs = int(num_envs)
        self.num_stages = int(num_stages)
        requested_device = torch.device(device)
        if (
            requested_device.type == "cuda"
            and requested_device.index is None
        ):
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.device = requested_device
        self.config = config
        self.stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.dwell = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.prev_potential = torch.zeros(self.num_envs, device=self.device)
        self.form_prev_potentials = torch.zeros(
            (self.num_envs, self.num_stages), device=self.device
        )
        self.form_included = torch.zeros(
            (self.num_envs, self.num_stages),
            dtype=torch.bool,
            device=self.device,
        )
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.dense_scales = torch.as_tensor(
            config.dense_scales, device=self.device
        )
        self.transition_bonuses = torch.as_tensor(
            config.transition_bonuses, device=self.device
        )
        self.dwell_steps = torch.as_tensor(
            config.dwell_steps, dtype=torch.long, device=self.device
        )

    def reset(
        self, potentials: torch.Tensor, env_idx: torch.Tensor | None = None
    ) -> None:
        self._validate_matrix("potentials", potentials)
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        else:
            env_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            )
        self.stage[env_idx] = 0
        self.dwell[env_idx] = 0
        self.prev_potential[env_idx] = potentials[env_idx, 0]
        self.form_prev_potentials[env_idx] = potentials[env_idx]
        self.form_included[env_idx] = False
        self.form_included[env_idx, 0] = True
        self.initialized[env_idx] = True

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
        self._validate_matrix("potentials", potentials)
        self._validate_matrix("gates", gates)
        if not torch.all(self.initialized):
            missing = torch.nonzero(~self.initialized).flatten().tolist()
            raise RuntimeError(f"Stage reward used before reset for envs {missing}")
        if maintenance is None:
            maintenance = torch.zeros_like(potentials)
        self._validate_matrix("maintenance", maintenance)
        if safety_penalty is None:
            safety_penalty = torch.zeros(
                self.num_envs, device=self.device
            )
        if safety_penalty.shape != (self.num_envs,):
            raise ValueError(
                f"safety_penalty must have shape {(self.num_envs,)}, "
                f"found {tuple(safety_penalty.shape)}"
            )
        if transition_authorization is not None:
            if transition_authorization.shape != (self.num_envs,):
                raise ValueError(
                    "transition_authorization must have shape "
                    f"{(self.num_envs,)}, found "
                    f"{tuple(transition_authorization.shape)}"
                )
            transition_authorization = transition_authorization.bool()

        stage_before = self.stage.clone()
        active = stage_before < self.num_stages
        gather_stage = stage_before.clamp(max=self.num_stages - 1)
        row = torch.arange(self.num_envs, device=self.device)

        phi = potentials[row, gather_stage]
        dense_raw = self.config.gamma * phi - self.prev_potential
        dense = self.dense_scales[gather_stage] * torch.clamp(
            dense_raw, -self.config.delta_max, self.config.delta_max
        )
        dense = torch.where(active, dense, torch.zeros_like(dense))
        staged_reset_dense = dense

        active_gate = gates[row, gather_stage].bool() & active
        self.dwell = torch.where(
            active_gate, self.dwell + 1, torch.zeros_like(self.dwell)
        )
        rule_candidate = active_gate & (
            self.dwell >= self.dwell_steps[gather_stage]
        )
        authority_mode = self._normalize_transition_authority_mode(
            transition_authority_mode
        )
        if authority_mode == "auto":
            authority_mode = (
                "rule_candidate_and_external_authorization"
                if transition_authorization is not None
                else "rule_candidate"
            )
        if transition_authorization is None:
            if authority_mode != "rule_candidate":
                raise ValueError(
                    "external transition authority requires "
                    "transition_authorization"
                )
            transition = rule_candidate
        elif authority_mode == "rule_candidate_and_external_authorization":
            # Original visual-gate behavior: Qwen is a mandatory semantic
            # verifier of a persistent rule candidate.
            transition = rule_candidate & transition_authorization
        elif authority_mode == "external_authorization":
            # Fixed-frequency visual-gate ablation: query schedule is no longer
            # rule-candidate-triggered, and an accepted visual judgment is the
            # transition authority. Rule candidates remain diagnostic only.
            transition = active & transition_authorization
        else:
            transition = rule_candidate
        transition_bonus = torch.where(
            transition,
            self.transition_bonuses[gather_stage],
            torch.zeros_like(dense),
        )
        active_maintenance = torch.where(
            active,
            maintenance[row, gather_stage],
            torch.zeros_like(dense),
        )
        mode = self._normalize_reward_form_mode(reward_form_mode)
        if mode != "staged_reset":
            dense = self._reward_form_dense(
                mode=mode,
                potentials=potentials,
                stage_before=stage_before,
                active=active,
            )
        reward = (
            dense
            + transition_bonus
            + active_maintenance
            - safety_penalty
        )

        self.stage = stage_before + transition.long()
        self.dwell = torch.where(
            transition, torch.zeros_like(self.dwell), self.dwell
        )

        next_stage = self.stage.clamp(max=self.num_stages - 1)
        entry_potential = potentials[row, next_stage]
        self.prev_potential = torch.where(
            transition, entry_potential, phi
        )
        self.prev_potential = torch.where(
            self.stage < self.num_stages,
            self.prev_potential,
            torch.zeros_like(self.prev_potential),
        )

        event_stage = torch.where(
            transition,
            stage_before,
            torch.full_like(stage_before, -1),
        )
        if mode != "staged_reset":
            valid_next = transition & (self.stage < self.num_stages)
            if valid_next.any():
                next_rows = torch.nonzero(valid_next).flatten()
                next_cols = self.stage[next_rows]
                self.form_prev_potentials[next_rows, next_cols] = potentials[
                    next_rows, next_cols
                ]
                self.form_included[next_rows, next_cols] = True
        diagnostics = {
            "stage_before": stage_before,
            "stage_after": self.stage.clone(),
            "dense": dense,
            "staged_reset_dense": staged_reset_dense,
            "rule_candidate": rule_candidate,
            "transition": transition,
            "event_stage": event_stage,
            "transition_bonus": transition_bonus,
            "maintenance": active_maintenance,
            "safety_penalty": safety_penalty,
            "transition_source": (
                authority_mode
                if transition_authorization is not None
                else "rule_dwell"
            ),
            "transition_authority_mode": authority_mode,
            "reward_form_mode": mode,
        }
        return reward, diagnostics

    def rule_candidate(self, gates: torch.Tensor) -> torch.Tensor:
        """Return the candidate mask that the next step will evaluate.

        This is side-effect free. The wrapper uses it to schedule a slow VLM
        only when the active fast rule will satisfy its dwell requirement on
        the current transition.
        """
        self._validate_matrix("gates", gates)
        if not torch.all(self.initialized):
            missing = torch.nonzero(~self.initialized).flatten().tolist()
            raise RuntimeError(f"Stage reward used before reset for envs {missing}")
        active = self.stage < self.num_stages
        gather_stage = self.stage.clamp(max=self.num_stages - 1)
        row = torch.arange(self.num_envs, device=self.device)
        active_gate = gates[row, gather_stage].bool() & active
        next_dwell = torch.where(
            active_gate, self.dwell + 1, torch.zeros_like(self.dwell)
        )
        return active_gate & (next_dwell >= self.dwell_steps[gather_stage])

    @staticmethod
    def _normalize_reward_form_mode(mode: str) -> str:
        if mode in {"ours_rule", "main", "active_only", "reset"}:
            return "staged_reset"
        if mode in {"completed", "cumulative", "completed_cumsum"}:
            return "completed_cumulative"
        if mode in {"all_dense", "dense_no_stage", "dense_no_stage_actor_stage"}:
            return "all_dense_with_stage_actor"
        if mode in {
            "staged_reset",
            "completed_cumulative",
            "all_dense_with_stage_actor",
        }:
            return mode
        raise ValueError(f"unknown reward_form_mode: {mode}")

    @staticmethod
    def _normalize_transition_authority_mode(mode: str) -> str:
        if mode in {"auto", "default"}:
            return "auto"
        if mode in {"rule", "rule_candidate", "rule_dwell"}:
            return "rule_candidate"
        if mode in {
            "candidate_visual",
            "rule_candidate_and_external_authorization",
            "candidate_and_external",
        }:
            return "rule_candidate_and_external_authorization"
        if mode in {
            "fixed_frequency_visual",
            "external_authorization",
            "visual_only",
        }:
            return "external_authorization"
        raise ValueError(f"unknown transition_authority_mode: {mode}")

    def _reward_form_dense(
        self,
        *,
        mode: str,
        potentials: torch.Tensor,
        stage_before: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        stage_ids = torch.arange(self.num_stages, device=self.device).view(1, -1)
        current = stage_before.view(-1, 1)
        if mode == "completed_cumulative":
            include = (stage_ids <= current) & active.view(-1, 1)
        elif mode == "all_dense_with_stage_actor":
            include = active.view(-1, 1).expand(-1, self.num_stages)
        else:
            raise ValueError(f"unexpected reward_form_mode: {mode}")

        previous = torch.where(
            self.form_included,
            self.form_prev_potentials,
            potentials,
        )
        dense_raw = self.config.gamma * potentials - previous
        dense = self.dense_scales.view(1, -1) * torch.clamp(
            dense_raw, -self.config.delta_max, self.config.delta_max
        )
        dense = torch.where(include, dense, torch.zeros_like(dense))
        self.form_prev_potentials = torch.where(
            include,
            potentials,
            self.form_prev_potentials,
        )
        self.form_included |= include
        return dense.sum(dim=1)

    def _validate_matrix(self, name: str, value: torch.Tensor) -> None:
        expected = (self.num_envs, self.num_stages)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}, "
                f"found {tuple(value.shape)}"
            )
        if value.device != self.device:
            raise ValueError(
                f"{name} is on {value.device}, expected {self.device}"
            )

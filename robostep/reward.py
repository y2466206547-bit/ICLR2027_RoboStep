"""Numpy implementation of the active-stage reward state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import RewardProgram


@dataclass
class RewardStep:
    reward: np.ndarray
    stage_before: np.ndarray
    stage_after: np.ndarray
    dense: np.ndarray
    staged_reset_dense: np.ndarray
    transition_bonus: np.ndarray
    maintenance: np.ndarray
    safety_penalty: np.ndarray
    rule_candidate: np.ndarray
    transition: np.ndarray
    event_stage: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ActiveStageReward:
    """Stateful vectorized reward engine.

    Inputs use batch-first arrays.  `potentials` and `candidates` have shape
    ``(batch, K)``; maintenance has the same shape and safety has shape
    ``(batch,)``.  A scalar batch works as a one-row array.  The engine is
    deliberately independent of a simulator so benchmark adapters can keep
    their native state API and only provide numeric terms.
    """

    def __init__(
        self,
        program: RewardProgram,
        *,
        batch_size: int = 1,
        shaping_gamma: float = 0.995,
        delta_max: float = 0.30,
    ) -> None:
        self.program = program
        self.batch_size = int(batch_size)
        self.k = program.num_stages
        self.shaping_gamma = float(shaping_gamma)
        self.delta_max = float(delta_max)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.stage = np.zeros(self.batch_size, dtype=np.int64)
        self.dwell = np.zeros(self.batch_size, dtype=np.int64)
        self.prev_potential = np.zeros(self.batch_size, dtype=np.float64)
        # The non-reset reward-form controls in the benchmark core keep one
        # previous potential per stage.  A scalar previous potential is only
        # correct for the active-stage staged-reset form.
        self.form_prev_potentials = np.zeros((self.batch_size, self.k), dtype=np.float64)
        self.form_included = np.zeros((self.batch_size, self.k), dtype=bool)
        self.initialized = np.zeros(self.batch_size, dtype=bool)

    def reset(self, potentials: np.ndarray, env_indices: np.ndarray | None = None) -> None:
        values = self._matrix("potentials", potentials)
        indices = np.arange(self.batch_size) if env_indices is None else np.asarray(env_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.batch_size):
            raise IndexError("env_indices out of range")
        self.stage[indices] = 0
        self.dwell[indices] = 0
        self.prev_potential[indices] = values[indices, 0]
        self.form_prev_potentials[indices] = values[indices]
        self.form_included[indices] = False
        self.form_included[indices, 0] = True
        self.initialized[indices] = True

    def candidate_mask(self, candidates: np.ndarray) -> np.ndarray:
        values = self._matrix("candidates", candidates).astype(bool)
        self._require_reset()
        rows = np.arange(self.batch_size)
        active = self.stage < self.k
        col = np.minimum(self.stage, self.k - 1)
        return active & values[rows, col] & ((self.dwell + 1) >= self._dwell_for(col))

    def step(
        self,
        potentials: np.ndarray,
        candidates: np.ndarray,
        *,
        maintenance: np.ndarray | None = None,
        safety_penalty: np.ndarray | None = None,
        transition_authorization: np.ndarray | None = None,
        terminal_success: np.ndarray | None = None,
        reward_mode: str = "staged_reset",
        transition_authority: str = "auto",
    ) -> RewardStep:
        phi = self._matrix("potentials", potentials).astype(np.float64)
        cand = self._matrix("candidates", candidates).astype(bool)
        self._require_reset()
        maint = np.zeros_like(phi) if maintenance is None else self._matrix("maintenance", maintenance).astype(np.float64)
        safety = np.zeros(self.batch_size, dtype=np.float64) if safety_penalty is None else np.asarray(safety_penalty, dtype=np.float64)
        if safety.shape != (self.batch_size,):
            raise ValueError(f"safety_penalty must have shape {(self.batch_size,)}")
        auth = None if transition_authorization is None else np.asarray(transition_authorization, dtype=bool)
        if auth is not None and auth.shape != (self.batch_size,):
            raise ValueError(f"transition_authorization must have shape {(self.batch_size,)}")
        terminal = None if terminal_success is None else np.asarray(terminal_success, dtype=bool)
        if terminal is not None and terminal.shape != (self.batch_size,):
            raise ValueError(f"terminal_success must have shape {(self.batch_size,)}")
        reward_mode = self._normalize_reward_mode(reward_mode)
        if reward_mode not in {"staged_reset", "completed_cumulative", "all_dense_with_stage_actor", "terminal_only", "per_subtask"}:
            raise ValueError(f"unsupported reward_mode: {reward_mode}")
        rows = np.arange(self.batch_size)
        before = self.stage.copy()
        active = before < self.k
        col = np.minimum(before, self.k - 1)
        current = phi[rows, col]
        raw = self.shaping_gamma * current - self.prev_potential
        staged_dense = np.where(active, self._dense_scale(col) * np.clip(raw, -self.delta_max, self.delta_max), 0.0)

        active_candidate = active & cand[rows, col]
        self.dwell = np.where(active_candidate, self.dwell + 1, 0)
        rule_candidate = active_candidate & (self.dwell >= self._dwell_for(col))
        if transition_authority == "auto":
            transition_authority = "rule_candidate" if auth is None else "rule_candidate_and_external_authorization"
        if transition_authority == "rule_candidate":
            transition = rule_candidate
        elif transition_authority == "rule_candidate_and_external_authorization":
            if auth is None:
                raise ValueError("external authorization is required")
            transition = rule_candidate & auth
        elif transition_authority == "external_authorization":
            if auth is None:
                raise ValueError("external authorization is required")
            transition = active & auth
        else:
            raise ValueError(f"unsupported transition_authority: {transition_authority}")

        bonus = np.where(transition, self._transition_bonus(col), 0.0)
        active_maintenance = np.where(active, maint[rows, col], 0.0)
        dense = self._dense_for_mode(phi, before, staged_dense, reward_mode)
        transition_bonus = bonus.copy()
        if reward_mode == "terminal_only":
            dense = np.zeros_like(dense)
            transition_bonus = np.zeros_like(bonus)
        elif reward_mode == "per_subtask":
            dense = np.zeros_like(dense)
            transition_bonus = transition.astype(np.float64)
        reward = dense + bonus + active_maintenance - safety
        # The terminal-only dummy baseline is independent of stage candidates:
        # its sole reward is the benchmark's official episode success event.
        # This input is an evaluation signal supplied by the adapter at the
        # episode boundary, never a dense potential or a VLM decision.
        if reward_mode == "terminal_only":
            reward = (terminal.astype(np.float64) if terminal is not None else np.zeros(self.batch_size, dtype=np.float64))
        elif reward_mode == "per_subtask":
            # This is the benchmark control definition: one unit for each
            # accepted transition, with dense shaping and penalties removed.
            reward = transition_bonus.copy()

        self.stage = before + transition.astype(np.int64)
        self.dwell = np.where(transition, 0, self.dwell)
        next_col = np.minimum(self.stage, self.k - 1)
        entry = phi[rows, next_col]
        self.prev_potential = np.where(transition & (self.stage < self.k), entry, current)
        self.prev_potential = np.where(self.stage < self.k, self.prev_potential, 0.0)
        if reward_mode != "staged_reset":
            valid_next = transition & (self.stage < self.k)
            if np.any(valid_next):
                rows_next = np.flatnonzero(valid_next)
                cols_next = self.stage[rows_next]
                self.form_prev_potentials[rows_next, cols_next] = phi[rows_next, cols_next]
                self.form_included[rows_next, cols_next] = True
        return RewardStep(
            reward=reward,
            stage_before=before,
            stage_after=self.stage.copy(),
            dense=dense,
            staged_reset_dense=staged_dense,
            transition_bonus=transition_bonus,
            maintenance=active_maintenance,
            safety_penalty=safety,
            rule_candidate=rule_candidate,
            transition=transition,
            event_stage=np.where(transition, before, -1),
        )

    def _dense_for_mode(self, phi: np.ndarray, before: np.ndarray, staged: np.ndarray, mode: str) -> np.ndarray:
        if mode == "staged_reset" or mode in {"terminal_only", "per_subtask"}:
            return staged
        rows = np.arange(self.batch_size)
        stage_ids = np.arange(self.k)[None, :]
        current = before[:, None]
        if mode == "completed_cumulative":
            include = (stage_ids <= current) & (before[:, None] < self.k)
        elif mode == "all_dense_with_stage_actor":
            include = np.broadcast_to(before[:, None] < self.k, phi.shape)
        else:
            raise ValueError(f"unexpected reward_mode: {mode}")
        previous = np.where(self.form_included, self.form_prev_potentials, phi)
        raw = self.shaping_gamma * phi - previous
        scales = np.asarray([s.dense_scale for s in self.program.stages], dtype=np.float64)
        dense = scales[None, :] * np.clip(raw, -self.delta_max, self.delta_max)
        dense = np.where(include, dense, 0.0)
        self.form_prev_potentials = np.where(include, phi, self.form_prev_potentials)
        self.form_included |= include
        return np.sum(dense, axis=1)

    @staticmethod
    def _normalize_reward_mode(mode: str) -> str:
        aliases = {
            "main": "staged_reset",
            "ours_rule": "staged_reset",
            "active_only": "staged_reset",
            "reset": "staged_reset",
            "completed": "completed_cumulative",
            "cumulative": "completed_cumulative",
            "completed_cumsum": "completed_cumulative",
            "all_dense": "all_dense_with_stage_actor",
            "dense_no_stage": "all_dense_with_stage_actor",
            "dense_no_stage_actor_stage": "all_dense_with_stage_actor",
        }
        return aliases.get(str(mode), str(mode))

    def _dense_scale(self, col: np.ndarray) -> np.ndarray:
        return np.asarray([self.program.stages[int(i)].dense_scale for i in np.asarray(col).reshape(-1)], dtype=np.float64)

    def _transition_bonus(self, col: np.ndarray) -> np.ndarray:
        return np.asarray([self.program.stages[int(i)].transition_bonus for i in np.asarray(col).reshape(-1)], dtype=np.float64)

    def _dwell_for(self, col: np.ndarray) -> np.ndarray:
        return np.asarray([self.program.stages[int(i)].dwell_steps for i in np.asarray(col).reshape(-1)], dtype=np.int64)

    def _matrix(self, name: str, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values)
        if result.shape != (self.batch_size, self.k):
            raise ValueError(f"{name} must have shape {(self.batch_size, self.k)}, got {result.shape}")
        return result

    def _require_reset(self) -> None:
        if not np.all(self.initialized):
            raise RuntimeError("ActiveStageReward.step() called before reset()")

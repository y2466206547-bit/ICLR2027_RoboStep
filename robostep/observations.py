"""Matched policy observation layouts and input ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ABALATIONS = {
    "ours",
    "actor_blind_stage",
    "no_stage_obs",
    "rgb_policy",
    "rgb_policy_actor_blind_stage",
    "dense_no_stage",
}


@dataclass(frozen=True)
class PolicyInputConfig:
    base_state_dim: int
    stage_count: int
    ablation: str = "ours"
    actor_rgb_dim: int = 0
    critic_rgb_dim: int = 0

    def __post_init__(self) -> None:
        if self.base_state_dim < 1 or self.stage_count < 1:
            raise ValueError("base_state_dim and stage_count must be positive")
        if self.ablation not in ABALATIONS:
            raise ValueError(f"unknown policy ablation: {self.ablation}")
        if self.actor_rgb_dim < 0 or self.critic_rgb_dim < 0:
            raise ValueError("RGB feature dimensions cannot be negative")

    @property
    def stage_width(self) -> int:
        return self.stage_count + 1

    @property
    def actor_stage(self) -> bool:
        return self.ablation in {"ours", "rgb_policy", "dense_no_stage"}

    @property
    def critic_stage(self) -> bool:
        # DenseNoStage is a reward-form ablation; it retains the matched
        # actor/critic observation block.  Removing the stage code is the
        # separate Ours-NoStageObs input ablation.
        return self.ablation != "no_stage_obs"

    @property
    def actor_dim(self) -> int:
        return self.base_state_dim + (self.stage_width if self.actor_stage else 0) + self.actor_rgb_dim

    @property
    def critic_dim(self) -> int:
        return self.base_state_dim + (self.stage_width if self.critic_stage else 0) + self.critic_rgb_dim


def ablation_config(name: str, *, base_state_dim: int, stage_count: int, rgb_dim: int = 0) -> PolicyInputConfig:
    aliases = {"Ours": "ours", "Ours-ActorBlindStage": "actor_blind_stage", "Ours-NoStageObs": "no_stage_obs", "Ours-RGB": "rgb_policy", "Ours-RGB-ActorBlindStage": "rgb_policy_actor_blind_stage", "DenseNoStage": "dense_no_stage"}
    normalized = aliases.get(name, name.lower().replace("-", "_"))
    return PolicyInputConfig(base_state_dim, stage_count, normalized, actor_rgb_dim=rgb_dim if normalized.startswith("rgb") else 0)


def pack_policy_observation(
    state: np.ndarray,
    active_stage: np.ndarray | int,
    config: PolicyInputConfig,
    *,
    rgb_features: np.ndarray | None = None,
    role: Literal["actor", "critic"] = "actor",
) -> np.ndarray:
    """Concatenate base state, optional RGB features, and explicit stage code."""
    values = np.asarray(state, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != config.base_state_dim:
        raise ValueError(f"state has width {values.shape[1]}, expected {config.base_state_dim}")
    stage = np.asarray(active_stage, dtype=np.int64).reshape(-1)
    if stage.size == 1 and values.shape[0] > 1:
        stage = np.repeat(stage, values.shape[0])
    if stage.size != values.shape[0] or np.any(stage < 0) or np.any(stage > config.stage_count):
        raise ValueError("active_stage shape/value is incompatible with state")
    include_stage = config.actor_stage if role == "actor" else config.critic_stage
    pieces = [values]
    if rgb_features is not None:
        rgb = np.asarray(rgb_features, dtype=np.float32)
        if rgb.ndim == 1:
            rgb = rgb[None, :]
        expected = config.actor_rgb_dim if role == "actor" else config.critic_rgb_dim
        if rgb.shape != (values.shape[0], expected):
            raise ValueError(f"rgb_features has shape {rgb.shape}, expected {(values.shape[0], expected)}")
        pieces.append(rgb)
    elif (config.actor_rgb_dim if role == "actor" else config.critic_rgb_dim) > 0:
        raise ValueError("RGB features are required by this policy configuration")
    if include_stage:
        one_hot = np.zeros((values.shape[0], config.stage_width), dtype=np.float32)
        one_hot[np.arange(values.shape[0]), stage] = 1.0
        pieces.append(one_hot)
    return np.concatenate(pieces, axis=1)

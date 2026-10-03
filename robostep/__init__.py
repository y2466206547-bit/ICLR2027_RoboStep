"""Compact reference implementation of the RoboStep reward/gate protocol."""

from .schema import PotentialTerm, RewardProgram, StageSpec
from .reward import ActiveStageReward, RewardStep
from .gates import GateDecision, GateRule, GateVLM
from .observations import PolicyInputConfig, ablation_config, pack_policy_observation
from .adapters import BenchmarkAdapter, EpisodeTrace, StageAwareRunner
from .compilation import CompilationResult, RewardCompilationLoop
from .task_machine import load_reward_machine

__all__ = [
    "PotentialTerm",
    "RewardProgram",
    "StageSpec",
    "ActiveStageReward",
    "RewardStep",
    "GateDecision",
    "GateRule",
    "GateVLM",
    "PolicyInputConfig",
    "ablation_config",
    "pack_policy_observation",
    "BenchmarkAdapter",
    "EpisodeTrace",
    "StageAwareRunner",
    "CompilationResult",
    "RewardCompilationLoop",
    "load_reward_machine",
]

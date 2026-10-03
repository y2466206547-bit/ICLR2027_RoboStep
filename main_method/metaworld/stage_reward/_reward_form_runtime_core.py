"""Runtime adapter for staged reward-form ablations."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner as _OnPolicyRunner

from .reward_form_modes import (
    BENCHMARK_NATIVE_DENSE,
    STAGED_RESET,
    is_sparse_reward_mode,
    normalize_reward_form_mode,
)


ENV_MODE = "STAGE_REWARD_FORM_MODE"
ABLATION_NAME = "StageRewardForm"


def _mode_for_env(env: VecEnv, explicit: str | None = None) -> str:
    if explicit:
        return normalize_reward_form_mode(explicit)
    env_mode = os.environ.get(ENV_MODE)
    if env_mode:
        return normalize_reward_form_mode(env_mode)
    config = getattr(env, "config", None)
    return normalize_reward_form_mode(getattr(config, "reward_form_mode", STAGED_RESET))


def _replace_config_mode(config: Any, mode: str) -> Any:
    if config is None:
        # Some benchmark-native VecEnvs (notably the drawer wrapper) keep
        # their reward dataclass in ``env.reward_config`` and expose only the
        # serializable dictionary through ``env.cfg``.  The ablation metadata
        # below is sufficient for these environments; do not require a
        # synthetic ``env.config`` object just to switch the reward-form tag.
        # A tiny namespace is still installed because a few VecEnv methods
        # consult ``self.config.reward_form_mode`` when deciding whether to
        # return the benchmark-native reward.
        return SimpleNamespace(reward_form_mode=mode)
    if is_dataclass(config):
        return replace(config, reward_form_mode=mode)
    setattr(config, "reward_form_mode", mode)
    return config


def make_reward_form_in_place(env: VecEnv, mode: str | None = None) -> VecEnv:
    """Patch one VecEnv instance with the requested reward-form mode."""

    canonical = _mode_for_env(env, mode)
    sparse = is_sparse_reward_mode(canonical)
    native = canonical == BENCHMARK_NATIVE_DENSE
    if getattr(env, "_stage_reward_form_mode", None) == canonical:
        return env
    config = _replace_config_mode(getattr(env, "config", None), canonical)
    if config is not None:
        env.config = config
    env._stage_reward_form_mode = canonical
    if isinstance(getattr(env, "cfg", None), dict):
        reward_config = dict(env.cfg.get("reward_config") or {})
        reward_config["reward_form_mode"] = canonical
        env.cfg.update(
            {
                "ablation": ABLATION_NAME,
                "reward_form_mode": canonical,
                "reward_config": reward_config,
                "actor_receives_stage_id": True,
                "critic_receives_stage_id": True,
                "stage_fsm_unchanged": True,
                "transition_bonuses_unchanged": not sparse and not native,
                "benchmark_native_dense": native,
                "native_environment_reward_returned_directly": native,
                "sparse_reward_definition": (
                    "terminal_transition_only"
                    if canonical == "terminal_only_sparse"
                    else "unit_reward_per_accepted_transition"
                    if sparse
                    else None
                ),
            }
        )
    return env


def annotate_run_config(log_dir: str | None, env: VecEnv) -> None:
    if log_dir is None:
        return
    mode = getattr(env, "_stage_reward_form_mode", None)
    if mode is None:
        return
    sparse = is_sparse_reward_mode(mode)
    native = mode == BENCHMARK_NATIVE_DENSE
    config_path = Path(log_dir) / "config.json"
    if not config_path.is_file():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    environment = payload.setdefault("environment", {})
    if isinstance(environment, dict):
        reward_config = dict(environment.get("reward_config") or {})
        reward_config["reward_form_mode"] = mode
        environment.update(
            {
                "ablation": ABLATION_NAME,
                "reward_form_mode": mode,
                "reward_config": reward_config,
                "actor_receives_stage_id": True,
                "critic_receives_stage_id": True,
                "stage_fsm_unchanged": True,
                "transition_bonuses_unchanged": not sparse and not native,
                "benchmark_native_dense": native,
                "native_environment_reward_returned_directly": native,
                "sparse_reward_definition": (
                    "terminal_transition_only"
                    if mode == "terminal_only_sparse"
                    else "unit_reward_per_accepted_transition"
                    if sparse
                    else None
                ),
            }
        )
    payload["ablation"] = {
        "name": ABLATION_NAME,
        "reward_form_mode": mode,
        "actor_receives_stage_id": True,
        "critic_receives_stage_id": True,
        "stage_fsm_unchanged": True,
        "transition_bonuses_unchanged": not sparse and not native,
        "benchmark_native_dense": native,
        "native_environment_reward_returned_directly": native,
        "sparse_reward_definition": (
            "terminal_transition_only"
            if mode == "terminal_only_sparse"
            else "unit_reward_per_accepted_transition"
            if sparse
            else None
        ),
    }
    payload["native_reward_used_for_ppo"] = native
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class StageRewardFormOnPolicyRunner(_OnPolicyRunner):
    """Drop-in runner that injects a reward-form mode before training/eval."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        patched_env = make_reward_form_in_place(env)
        super().__init__(patched_env, train_cfg, log_dir=log_dir, device=device)
        annotate_run_config(log_dir, patched_env)

"""Optional PyTorch PPO pieces used by the native task runner and adapters.

The environment, reward engine, and gate are framework agnostic.  This module
only supplies the matched actor/critic configuration and a conventional PPO
update so benchmark wrappers can keep their native rollout code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .benchmarks import PPO_DEFAULTS


@dataclass(frozen=True)
class PPOConfig:
    hidden_sizes: tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"
    rollout_steps: int = 256
    rollout_envs: int = 1
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    target_kl: float = 0.01
    entropy_coef: float = 0.0
    value_coef: float = 0.5

    @classmethod
    def for_benchmark(cls, benchmark: str) -> "PPOConfig":
        defaults = PPO_DEFAULTS[benchmark]
        hidden_text = defaults["actor_critic"].split()[0]
        if hidden_text.startswith("3x"):
            hidden = (int(hidden_text.split("x", 1)[1]),) * 3
        else:
            hidden = tuple(int(x) for x in hidden_text.replace("x", "-").split("-") if x.isdigit())
        rollout = defaults["rollout"]
        if "steps" in rollout:
            envs, steps = 1, 256
        else:
            envs, steps = (int(x) for x in rollout.split("x"))
        return cls(hidden_sizes=hidden, activation="tanh" if "Tanh" in defaults["actor_critic"] else "elu", rollout_steps=steps, rollout_envs=envs, epochs=defaults["epochs"], minibatches=defaults["minibatches"], learning_rate=float(defaults.get("lr", 3e-4)), lr_schedule=str(defaults.get("lr_schedule", "constant")), gamma=defaults["gamma"], gae_lambda=defaults["gae_lambda"], clip_ratio=defaults["clip"])


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, *, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if values.shape != rewards.shape:
        raise ValueError("values must have the same shape as rewards")
    advantages = np.zeros_like(rewards)
    running = 0.0
    next_value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        running = delta + gamma * gae_lambda * (1.0 - dones[t]) * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


class PPOTrainer:
    """Small continuous-action PPO implementation (PyTorch optional)."""

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig, *, seed: int = 0) -> None:
        try:
            import torch
            import torch.nn as nn
            from torch.distributions import Normal
        except ImportError as exc:  # pragma: no cover - depends on optional torch
            raise ImportError("PPOTrainer requires torch; install robostep-supplement[torch]") from exc
        torch.manual_seed(seed)
        self.torch = torch
        self.Normal = Normal
        self.config = config

        def activation(name: str):
            return nn.Tanh() if name.lower() == "tanh" else nn.ELU()

        def mlp(out_dim: int):
            layers = []
            width = obs_dim
            for hidden in config.hidden_sizes:
                layers.extend([nn.Linear(width, hidden), activation(config.activation)])
                width = hidden
            layers.append(nn.Linear(width, out_dim))
            return nn.Sequential(*layers)

        class ActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = mlp(action_dim)
                self.critic = mlp(1)
                self.log_std = nn.Parameter(torch.zeros(action_dim))

            def distribution(self, obs):
                return Normal(self.actor(obs), self.log_std.exp())

            def value(self, obs):
                return self.critic(obs).squeeze(-1)

        self.model = ActorCritic()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32))
        if obs.ndim == 1:
            obs = obs[None, :]
        with torch.no_grad():
            dist = self.model.distribution(obs)
            action = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            value = self.model.value(obs)
        return action.numpy(), log_prob.numpy(), value.numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        torch = self.torch
        required = {"obs", "actions", "old_log_prob", "returns", "advantages"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"PPO batch missing {sorted(missing)}")
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        old_log = torch.as_tensor(batch["old_log_prob"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
        adv = torch.as_tensor(batch["advantages"], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        order = np.arange(n)
        actor_loss = value_loss = entropy = approx_kl = 0.0
        updates = 0
        for _ in range(self.config.epochs):
            np.random.shuffle(order)
            for indices in np.array_split(order, self.config.minibatches):
                idx = torch.as_tensor(indices, dtype=torch.long)
                dist = self.model.distribution(obs[idx])
                log_prob = dist.log_prob(actions[idx]).sum(-1)
                approx_kl += float((old_log[idx] - log_prob).mean().detach())
                ratio = (log_prob - old_log[idx]).exp()
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv[idx]
                loss_actor = -torch.minimum(unclipped, clipped).mean()
                value = self.model.value(obs[idx])
                loss_value = (value - returns[idx]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = loss_actor + self.config.value_coef * loss_value - self.config.entropy_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                actor_loss += float(loss_actor.detach())
                value_loss += float(loss_value.detach())
                entropy += float(ent.detach())
                updates += 1
        approx_kl /= max(updates, 1)
        if self.config.lr_schedule == "adaptive_kl":
            # Match the benchmark's adaptive-KL intent while keeping the
            # trainer lightweight.  The native benchmark PPO can replace this
            # policy update but should preserve the recorded schedule.
            factor = 0.5 if approx_kl > 1.5 * self.config.target_kl else 2.0 if approx_kl < self.config.target_kl / 1.5 else 1.0
            if factor != 1.0:
                for group in self.optimizer.param_groups:
                    group["lr"] = float(np.clip(group["lr"] * factor, 1e-6, 1e-2))
        return {"actor_loss": actor_loss / max(updates, 1), "value_loss": value_loss / max(updates, 1), "entropy": entropy / max(updates, 1), "approx_kl": approx_kl, "learning_rate": float(self.optimizer.param_groups[0]["lr"])}

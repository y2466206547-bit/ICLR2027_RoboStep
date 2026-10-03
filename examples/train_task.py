"""Train one real benchmark task with its frozen Frame-VLM reward machine.

This is the reproducibility entry point for a single task. It uses the native
benchmark environment and the release PPO implementation; no synthetic
environment is involved. The selected task must expose the feature-adapter
contract documented in ``robostep.native_envs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robostep.native_envs import feature_packet, make_environment, observation_array
from robostep.observations import ablation_config, pack_policy_observation
from robostep.ppo import PPOConfig, PPOTrainer, compute_gae
from task_registry import load_reward_machine


def _reset(env, seed: int):
    value = env.reset(seed=seed)
    if isinstance(value, tuple) and len(value) == 2:
        return observation_array(value[0]), dict(value[1])
    return observation_array(value), {}


def _step(env, action):
    value = env.step(action)
    if len(value) == 5:
        obs, _, terminated, truncated, info = value
        return observation_array(obs), bool(terminated), bool(truncated), dict(info)
    obs, _, done, info = value
    return observation_array(obs), bool(done), False, dict(info)


def _success(env, info: dict) -> bool:
    if "success" in info:
        value = np.asarray(info["success"])
        return bool(value.reshape(-1)[0])
    evaluate = getattr(getattr(env, "unwrapped", env), "evaluate", None)
    if callable(evaluate):
        result = evaluate()
        if isinstance(result, dict) and "success" in result:
            value = result["success"]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return bool(np.asarray(value).reshape(-1)[0])
    return False


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.stage_observer != "rule":
        raise SystemExit(
            "The one-task release runner uses GateRule. For GateVLM, first "
            "freeze a gate cache and connect the benchmark frame callback in "
            "a native adapter; the reward machine API is unchanged."
        )
    machine = load_reward_machine(args.task, args.benchmark)
    env = make_environment(args.benchmark, args.task, seed=args.seed, render=False)
    state, info = _reset(env, args.seed)
    features, candidates = feature_packet(env, info, args.task)
    machine.reset(features)
    policy_config = PPOConfig.for_benchmark(machine.benchmark)
    ablation = ablation_config("ours", base_state_dim=state.size, stage_count=machine.stage_count)
    trainer = PPOTrainer(ablation.actor_dim, int(np.prod(env.action_space.shape)), policy_config, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    total_steps = 0
    episode_successes = 0
    episode_count = 0
    updates = []
    rollout_steps = max(1, min(policy_config.rollout_steps, args.rollout_steps))
    while total_steps < args.total_timesteps:
        observations = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        for _ in range(rollout_steps):
            policy_obs = pack_policy_observation(state, int(machine.engine.stage[0]), ablation)
            action, log_prob, value = trainer.act(policy_obs)
            action = np.asarray(action[0], dtype=np.float32)
            if hasattr(env.action_space, "low"):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, terminated, truncated, next_info = _step(env, action)
            next_features, next_candidates = feature_packet(env, next_info, args.task)
            result = machine.step(next_features, next_candidates, terminal_success=np.asarray([_success(env, next_info)]))
            done = bool(terminated or truncated or _success(env, next_info) or machine.engine.stage[0] >= machine.stage_count)
            observations.append(policy_obs[0])
            actions.append(action)
            log_probs.append(float(log_prob[0]))
            values.append(float(value[0]))
            rewards.append(float(result.reward[0]))
            dones.append(float(done))
            total_steps += 1
            if done:
                episode_count += 1
                episode_successes += int(_success(env, next_info))
                state, info = _reset(env, args.seed + episode_count)
                features, candidates = feature_packet(env, info, args.task)
                machine.reset(features)
            else:
                state = next_state
                info = next_info
            if total_steps >= args.total_timesteps:
                break
        reward_array = np.asarray(rewards, dtype=np.float32)
        value_array = np.asarray(values, dtype=np.float32)
        done_array = np.asarray(dones, dtype=np.float32)
        advantages, returns = compute_gae(reward_array, value_array, done_array, gamma=policy_config.gamma, gae_lambda=policy_config.gae_lambda)
        stats = trainer.update({
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "old_log_prob": np.asarray(log_probs, dtype=np.float32),
            "returns": returns,
            "advantages": advantages,
        })
        updates.append(stats)
        print(json.dumps({"task": args.task, "steps": total_steps, "episodes": episode_count, "successes": episode_successes, **stats}))
    env.close()
    result = {
        "benchmark": machine.benchmark,
        "task": machine.task_id,
        "seed": args.seed,
        "stage_observer": args.stage_observer,
        "reward_program_source": machine.program.source,
        "total_timesteps": total_steps,
        "episodes": episode_count,
        "successes": episode_successes,
        "success_rate": episode_successes / max(episode_count, 1),
        "updates": updates,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=("Meta-World", "ManiSkill", "SoftGym", "metaworld", "maniskill", "softgym"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--stage-observer", choices=("rule", "vlm"), default="rule")
    parser.add_argument("--output", default="runs/one_task_result.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

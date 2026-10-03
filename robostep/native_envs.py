"""Native benchmark constructors and the small feature-adapter contract.

The simulators stay external, but the construction and observation conventions
used by the release are defined here.  A benchmark wrapper may expose
``env.unwrapped.get_robostep_features()`` and
``env.unwrapped.get_robostep_candidates()``; otherwise the adapter reads the
corresponding dictionaries from ``info``.  The ManiSkill PickCube example is
wired directly so the one-command example is a real native environment run.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def make_environment(benchmark: str, task_id: str, *, seed: int, render: bool = False) -> Any:
    name = benchmark.lower().replace("-", "")
    if name == "maniskill":
        import gymnasium as gym

        env = gym.make(
            task_id,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array" if render else None,
        )
        env.reset(seed=seed)
        return env
    if name == "metaworld":
        import metaworld

        benchmark_suite = metaworld.MT1(task_id, seed=seed)
        env = benchmark_suite.train_classes[task_id]()
        env.set_task(benchmark_suite.train_tasks[0])
        env.reset(seed=seed)
        return env
    if name == "softgym":
        try:
            from softgym.registered_env import SOFTGYM_ENVS
        except ImportError as exc:
            raise RuntimeError(
                "Install the pinned SoftGym/PyFlex checkout and expose its "
                "registered_env module before running SoftGym tasks."
            ) from exc
        factory = SOFTGYM_ENVS.get(task_id)
        if factory is None:
            raise KeyError(f"SoftGym task is not registered: {task_id}")
        env = factory(headless=not render)
        env.reset()
        return env
    raise ValueError(f"unknown benchmark: {benchmark}")


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return float(value.reshape(-1)[0])


def _phi(distance: Any, gain: float = 5.0) -> float:
    return float(np.clip(1.0 - np.tanh(gain * max(_as_float(distance), 0.0)), 0.0, 1.0))


def feature_packet(env: Any, info: Mapping[str, Any], task_id: str) -> tuple[dict[str, float], np.ndarray]:
    """Return normalized named features and stage candidates for one task.

    Full benchmark wrappers should expose the two methods documented above.
    The direct PickCube path makes the ManiSkill example executable without a
    second local checkout of RoboStep.
    """

    base = getattr(env, "unwrapped", env)
    feature_fn = getattr(base, "get_robostep_features", None)
    candidate_fn = getattr(base, "get_robostep_candidates", None)
    if callable(feature_fn) and callable(candidate_fn):
        return dict(feature_fn()), np.asarray(candidate_fn(), dtype=bool)
    if "robostep_features" in info and "robostep_candidates" in info:
        return dict(info["robostep_features"]), np.asarray(info["robostep_candidates"], dtype=bool)

    if task_id == "PickCube-v1" and hasattr(base, "agent") and hasattr(base, "cube"):
        tcp = base.agent.tcp.pose.p
        cube = base.cube.pose.p
        goal = base.goal_site.pose.p
        tcp_cube = _as_float(np.linalg.norm(np.asarray(tcp.detach().cpu()) - np.asarray(cube.detach().cpu())))
        cube_goal = _as_float(np.linalg.norm(np.asarray(cube.detach().cpu()) - np.asarray(goal.detach().cpu())))
        grasped = bool(_as_float(base.agent.is_grasping(base.cube)))
        success = bool(info.get("success", False))
        features = {
            "tcp_obj": float(np.clip(tcp_cube, 0.0, 1.0)),
            "grasped": float(grasped),
            "obj_goal": float(np.clip(cube_goal, 0.0, 1.0)),
            "success": float(success),
        }
        candidates = np.asarray([tcp_cube <= 0.05, grasped, success], dtype=bool)
        return features, candidates

    raise RuntimeError(
        f"{task_id} needs a native feature adapter. Expose "
        "get_robostep_features()/get_robostep_candidates() or add the task "
        "mapping in robostep/native_envs.py."
    )


def observation_array(observation: Any) -> np.ndarray:
    """Flatten native state observations while preserving their numeric values."""

    if isinstance(observation, Mapping):
        parts = [observation[key] for key in sorted(observation)]
        return np.concatenate([observation_array(part) for part in parts]).astype(np.float32)
    return np.asarray(observation, dtype=np.float32).reshape(-1)

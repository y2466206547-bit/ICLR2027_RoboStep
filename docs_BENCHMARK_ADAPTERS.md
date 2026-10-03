# Benchmark adapter contract

The release keeps simulator construction outside the reward machine, but the
adapter contract is fixed so a main-table run is reproducible. Each adapter
must provide:

```python
get_robostep_features() -> dict[str, float]
get_robostep_candidates() -> array[bool]
official_success(info) -> bool
frame_packet(reference, ordered_frames) -> object  # GateVLM only
```

Feature values are normalized to `[0, 1]` using the task recipe. A distance,
error, speed, or spill term is a normalized cost; a `decrease` term is inverted
inside `TaskRewardMachine`. All other terms are progress values. The official
success signal is never used to construct these features.

## Meta-World

Install `metaworld`, construct the MT1 task, preserve its native state and
action spaces, and expose TCP/object/target fields through the adapter. The
main policy receives native state plus the active-stage one-hot. Rendered RGB
is sent only to the GateVLM callback. `success` is read only for evaluation or
the terminal-only control.

## ManiSkill

Install the pinned ManiSkill 3 package and configure SAPIEN/Vulkan according to
the upstream installation guide. Use `obs_mode="state"` and
`control_mode="pd_joint_delta_pos"`; request `render_mode="rgb_array"` only
when collecting GateVLM frames. The 19 package-local reward machines name the
TCP/object/goal/grasp/articulation features required by each task. The supplied
PickCube adapter demonstrates the complete path from native simulator state to
the reward machine.

## SoftGym

Install the pinned SoftGym/PyFlex environment and its assets from the upstream
project. Keep particle, cup, rope, and action state native. The six adapters
must expose the named particle/cup/rope features in their JSON machine. The
reported ClothFold coverage metric and the other five task metrics remain
evaluation-only; they are not dense reward inputs.

## Wrapper skeleton

```python
from task_registry import load_reward_machine

machine = load_reward_machine(task_id, benchmark)
features, candidates = adapter.get_robostep_features(), adapter.get_robostep_candidates()
machine.reset(features)
for action in rollout:
    observation, info = env.step(action)
    features = adapter.get_robostep_features()
    candidates = adapter.get_robostep_candidates()
    result = machine.step(
        features,
        candidates,
        terminal_success=[adapter.official_success(info)],
    )
```

The same wrapper accepts a frozen GateVLM authorization array without changing
the PPO policy or reward program. `StageAwareRunner` wires this online: when
the active-stage candidate is true it builds a `GateRequest` from the current
stage contract and the adapter's reference/candidate frames, submits it to
`GateVLM`, and passes only a matching accept/reject response to the reward
machine. A missing or stale response is an abstention and blocks the
transition; it never falls back to the privileged rule gate. The environment
may continue stepping while the worker is answering, but the stage cannot
advance until a matching response is available.

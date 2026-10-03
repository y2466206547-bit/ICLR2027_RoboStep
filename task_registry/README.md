# 75-task Frame-VLM reward machines

This directory is self-contained. It contains the frozen reward-machine
program used for the main-table task suite:

- 50 Meta-World tasks;
- 19 ManiSkill tasks;
- 6 SoftGym tasks.

Every JSON file under `reward_machines/` is a complete `RewardProgram` record:
ordered stage names, dense feature terms, term directions, stage completion
predicates, transition policy, and the task state interface expected from the
native benchmark adapter. The programs are marked
`frame_vlm_main_table_round5` so they cannot be confused with a native dense
reward or an environment success function.

The runnable state machine is `reward_machine.py`:

```python
from task_registry import load_reward_machine

machine = load_reward_machine("PickCube-v1", "ManiSkill")
machine.reset({"tcp_obj": 0.8, "grasped": 0.0, "obj_goal": 1.0})
result = machine.step(
    {"tcp_obj": 0.6, "grasped": 1.0, "obj_goal": 1.0},
    candidates=[True, False, False],
)
```

Adapters supply normalized `[0, 1]` feature values and one candidate boolean
per stage from the native simulator. The machine itself owns the active-stage
potential difference, dwell counter, transition bonus, maintenance/safety
terms, optional VLM authorization, and terminal-only control. Official success
is passed only as the separate terminal signal.

`manifest.json` is the full machine-readable record (including each task's
stage metadata); `task_manifest.csv` is only the compact 75-row lookup table.
Its columns are benchmark, task id, coarse task type, long-horizon flag,
stage count (K), transition count (K-1), horizon, package-relative machine
file, and frozen program source. It is for inventory, scripts, and audits; it
does not implement reward logic and is not another recipe source.

The active-stage state machine is the release implementation used by the main
method and is kept in sync with the release-local
`main_method/maniskill/stage_reward/core.py::BatchedStageReward`. Task-specific
native feature extraction is also shipped under `main_method/`; the simulator
adapter supplies only the benchmark state/RGB interface. The compositional and
backward-recovery drivers are shipped under `experiments/` because they add
sequence handoffs and disturbance protocols beyond the 75-task main suite.

Use the verifier after changing a machine:

```bash
python tools/verify_task_registry.py
```

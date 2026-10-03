# Backward Regression and Recovery

## Claim

Completing stage `k` does not make its prerequisite permanently true. After a
controlled disturbance makes `A_k(x_t)=0`, a bidirectional reward machine may
move from stage `k+1` back to `k` and continue rolling back if multiple
prerequisites were invalidated. Positive reward from an invalid downstream
stage is suppressed immediately; stage rollback uses three-step hysteresis.

## Matched comparison

- Methods: the existing forward-only FSM and RoboStep bidirectional FSM.
- Policy: the same frozen checkpoint for each task and method.
- Actor input: state plus active-stage one-hot in both methods.
- Before intervention: rollback is disarmed, so both methods execute the same
  forward-only trajectory. It is armed only when the disturbance starts.
- Reward: the existing Frame-VLM-authored `rule_specs.py` potentials and gates.
  Official dense reward is never called.
- Privileged simulator state: used to inject disturbances and score prerequisite
  validity. Official task success is used only for final Recovery SR.
- Sampling: 50 intervention-qualified episodes per method, task, and severity;
  evaluation seed 1042; deterministic actor inference.

## Tasks and interventions

| Category | Task | Regression | Severities | Stagnation control |
|---|---|---|---|---|
| Grasp | `PickCube-v1` | teleport a held cube onto the table away from the TCP | 0.02, 0.04, 0.06 m | block arm deltas for 4, 8, 12 steps while preserving the gripper command |
| Articulation | `RotateValveLevel2-v1` | after successful half-turn, set the valve behind its initial angle in the closing direction | 0.10, 0.30, 0.50 rad | block policy motion for 8, 16, 24 steps |
| Placement | `PlaceSphere-v1` | move a successfully placed sphere from the bin to the table | 0.02, 0.04, 0.08 m | block arm deltas for 4, 8, 12 steps while preserving the gripper command |

The regression is injected only after the policy reaches the specified
downstream stage. Grasp regression starts in transport; articulation and
placement regression start after official task completion. Stagnation preserves
the active-stage prerequisite and should not cause rollback. Any attempted
stagnation rollout in which that prerequisite physically breaks is excluded and
resampled rather than mislabeled as a negative control.

## Metrics

- **Rollback Accuracy**: fraction of effective regressions producing the first
  correct stage decrement within eight policy steps of prerequisite failure.
- **False Rollback**: fraction of stagnation episodes with rollback during the
  blocked-motion window.
- **Recovery SR**: fraction of disturbed episodes that later satisfy official
  task success after intervention or block release.
- **Recovery Time**: steps from disturbance to the first re-established
  prerequisite. For stagnation, steps from block release to task success.
- **IRR**: fraction of steps receiving positive reward while the machine still
  occupies a downstream stage whose prerequisite is false. Valid reward after
  rollback in an earlier recovery stage is not counted as invalid.

## Frozen checkpoints

- `PickCube-v1`: seed-1 Rule checkpoint, checkpoint-selection SR 0.95.
- `PlaceSphere-v1`: seed-1 Rule checkpoint, checkpoint-selection SR 1.00.
- `RotateValveLevel2-v1`: seed-42 Rule checkpoint, checkpoint-selection SR 1.00.

Checkpoint-selection rates describe undisturbed validation and are not
substituted for controlled-intervention results.

## Outputs

- `raw/*.json`: condition protocol, summary, episode records, and traces.
- `logs/*.log`: complete simulator stdout/stderr.
- `summary.json`, `table.csv`, `exp-Q4.tex`: aggregated results.
- `protocol_diagram.{png,pdf}`: experiment timeline and rollback schematic.
- `severity_curves.{png,pdf}`: severity versus Recovery SR and rollback latency.
- `recovery_traces.{png,pdf}`: representative reward/stage trajectories.

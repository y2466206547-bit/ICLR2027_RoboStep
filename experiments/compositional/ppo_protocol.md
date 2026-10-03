# Ten-task reward-form ablation protocol

## Purpose

This is a focused, matched ablation of the stage-conditioned MetaWorld method.
It isolates reward activation from the information supplied to the actor.  It
does **not** compare against the existing one-shot no-stage VLM reward
experiment under `ablation/reward_form/vlm_no_stage`: that experiment removes
both the stage FSM and the stage one-hot, so it cannot answer whether stage
context alone is sufficient.

## Frozen reference method

The reference is the best per-task Rule/Frame-VLM-stage recipe used in the
fair three-seed MT50 run.  Each ablation copies the selected recipe's task
specific reward configuration, PPO settings, curriculum settings, stage FSM,
and stage-transition rule.  The stage ID is computed by the same state-rule
FSM as the reference and is supplied to both actor and critic as the same
one-hot suffix.  Native MetaWorld reward and official success remain excluded
from PPO reward.

The historical actor-blind control establishes why retaining that suffix is
important: on MT50, stage-visible Rule scored 61.41 +/- 4.68% whereas the
otherwise identical actor-blind control scored 48.84 +/- 1.12% (50 tasks,
three train seeds, 50 held-out episodes per task/seed).

## Task subset

The subset spans simple articulated control, object transport/pushing, and
multi-stage grasp/transport.  The reference difficulty estimate is the
stage-visible Rule mean on the historical 50-episode held-out evaluation;
it is used only to stratify the subset, never for checkpoint selection.

| stratum | task | family | reference mean success |
|---|---|---|---:|
| easy | button-press-v3 | button | 100.0% |
| easy | button-press-wall-v3 | button | 98.0% |
| medium-easy | dial-turn-v3 | button | 74.0% |
| medium | coffee-pull-v3 | sequential | 64.0% |
| medium | coffee-push-v3 | object | 54.0% |
| medium | push-back-v3 | object | 49.3% |
| medium-hard | assembly-v3 | sequential | 47.3% |
| hard | hand-insert-v3 | sequential | 30.7% |
| hard | peg-insert-side-v3 | sequential | 8.7% |
| very hard | push-wall-v3 | object | 5.3% |

## Conditions

1. `staged_reset` (reference): only the unfinished active stage provides its
   dense potential reward.  On a confirmed transition, the completed stage's
   dense term becomes exactly zero from the next step; its one-time transition
   bonus is unchanged.
2. `completed_cumulative`: all completed stages and the current stage retain
   their own original potential-difference terms.  A transition bonus remains
   a first-occurrence event; it is never repeatedly paid.  This specifically
   tests the anti-camping rationale for clearing completed-stage reward.
3. `all_dense_with_stage_actor`: every original stage potential is active from
   reset and summed every step, while the actor still receives the unchanged
   rule-derived stage one-hot.  The FSM continues to advance and provides the
   same actor context, but it no longer selects which dense term is active.

Potential baselines are initialized independently for each newly activated
term, so a stage entry cannot receive an artificial reward caused by comparing
against zero.  Action/safety penalties are included exactly once per step in
all conditions.  This makes conditions 2--3 reward-form ablations, not
changes in reward scale through duplicated penalties.

## Runnable implementation

The implementation lives under `benchmark/Metaworld/stage_reward/`:
`reward_form_modes.py` defines the dense activation modes,
`reward_form_runtime_entrypoint.py` injects the selected mode into train/eval
subprocesses, and `run_reward_form_ablation_queue.py` reuses the final frozen Rule recipes
for exactly these ten tasks.  Strict-original tasks come from the immutable
frozen checkpoint manifest; `hand-insert-v3`, `peg-insert-side-v3`, and
`push-wall-v3` replay their reward-iteration replacement warm-start chains.
The queue also relocates legacy audit paths from
`benchmark/Metaworld/results/...` to the currently present
`benchmark/Metaworld/results/ours/...` source configs.

Primary commands:

```bash
python -m stage_reward.run_reward_form_ablation_queue --reward-form-mode completed_cumulative
python -m stage_reward.run_reward_form_ablation_queue --reward-form-mode all_dense_with_stage_actor
```

Optional matched reference rerun:

```bash
python -m stage_reward.run_reward_form_ablation_queue --reward-form-mode staged_reset
```

Each invocation defaults to `--gate rule --train-seeds 42 43 44`, validation
seed `2042`, test seed `1042`, and 50 held-out test episodes.  Outputs are
separated by mode under
`benchmark/Metaworld/results/reward_form_ablation_10task_bestrecipe_20260825/<mode>/`
and status logs under
`reward_model/reward_form_ablation_10task_bestrecipe_20260825/<mode>/`.

After queues finish, summarize with:

```bash
python -m stage_reward.summarize_reward_form_ablation
```

This writes `task_seed_results.csv`, `task_means.csv`, and `aggregate.json`
under `reward_model/reward_form_ablation_10task_bestrecipe_20260825/summary/`.

## Fairness and reporting

- Train from scratch with train seeds `42, 43, 44`.
- Preserve each reference task's PPO hyperparameters, horizon, number of
  environments, curriculum, checkpoint cadence, and task-specific reward
  parameters.
- Choose a checkpoint using only deterministic validation seed `2042`; never
  inspect test outcomes while selecting it.
- Evaluate the selected checkpoint once on untouched seed `1042`, with 50
  episodes per task/seed.  Report task means/SEMs and the 10-task macro
  mean/SEM over the three training seeds.
- Store the source config path, exact reward-mode code hash, selected
  checkpoint ranking, per-episode test records, and stage occupancy/transition
  statistics for every run.

The expected failure mode of `completed_cumulative` is a behavioral question,
not a built-in outcome: potential-difference shaping may instead yield a
negative drift near a saturated old stage.  We will report measured dwell,
transition rates, and success rather than assert reward camping in advance.

## 2026-08-26 broader expansion

MetaWorld hard10 extension uses the same best-recipe replay protocol as the original 10-task reward-form ablation. It changes only the task subset and output/status roots via wrapper scripts; checkpoint selection remains validation seed `2042`, final test remains seed `1042` with 50 episodes.

MetaWorld hard10 task subset:
`handle-press-v3`, `handle-pull-v3`, `plate-slide-side-v3`, `handle-press-side-v3`, `lever-pull-v3`, `plate-slide-v3`, `sweep-into-v3`, `handle-pull-side-v3`, `pick-out-of-hole-v3`, `sweep-v3`.

ManiSkill best5 extension uses the existing 20M PPO recipe from `maniskill19_rule20m_parallel_seed1` and changes only the scalar stage reward backend:

- `completed_cumulative`: same stage FSM and transition bonus, dense reward sums completed stages through the current stage.
- `all_dense_with_stage_actor`: same stage FSM and transition bonus, dense reward sums all stage potentials while actor/critic still receive active stage one-hot.

ManiSkill best5 task subset:
`StackCube-v1`, `PushCube-v1`, `PullCube-v1`, `PlaceSphere-v1`, `PickCube-v1`.

ManiSkill train seeds for the two ablation modes are `42,43,44`. Historical `ours_rule` ManiSkill baseline is only seed `1`; treat it as orientation unless a same-seed baseline is later launched.


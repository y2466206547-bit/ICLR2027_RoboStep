# Forward Composition of Stepwise Rewards: PnP Protocol and Extensions

**Update (2026-09-22 20:05 UTC):** Dense M10 has now finished both strict held-out evaluations: **0/256 on seed 1042 and 0/256 on seed 2042** (0/512 pooled), with every mandatory prefix also zero. This is a stronger completed dense long-horizon failure result, but it remains the intentional no-stage baseline rather than a reward-only causal comparison. The M9 continuation to iteration 6999 has since completed at 358/512 pooled strict success; the corrected M11 scratch run is now active. Matched no-stage dense M2 and M3 runs have also been queued with the same geometry, no-reset protocol, 700M budget scaling, PPO settings, and reward scales as the main runs, differing only by removal of stage ID.

**Update (2026-09-22 20:08 UTC):** The M9 extension has completed its extra 700 iterations and strict evaluation: **178/256** (seed 1042) and **180/256** (seed 2042), pooled **358/512 = 69.92%**. This is a substantial recovery from the earlier 18.32% continuation result, but it is not yet solved. The first M11 launch exposed that `tabletop_unique_chain_v8` was capped at ten targets; that run produced no evidence. A versioned `tabletop_unique_chain_v9` now preserves the first ten positions and adds an eleventh distinct target, and the corrected M11 run is active. The failed launch is excluded from results.

**Closeout update (2026-09-23 01:28 UTC):** The matched flat-dense/no-stage M2 control completed both strict held-out evaluations at **0/256 + 0/256 = 0/512**, with mandatory prefixes `[0,0]` on both seeds. The matched dense M3 run and corrected M11 run were stopped for time; M3 stopped before its final checkpoint/EVAL and M11 stopped at the saved model_1650.pt checkpoint, so neither receives an SR. The reportable homogeneous appendix is therefore intentionally capped at M10; M11 and beyond are documented as unmeasured rather than treated as failures.

**Final recovery-evaluation update (2026-09-22 11:14 UTC):** The width-31 recovery continuations were evaluated with the strict gated evaluator on seeds 1042 and 2042. M8 reached **491/513 = 95.71%** pooled end-to-end success; M9 reached only **94/513 = 18.32%**; M10 reached **482/512 = 94.14%**. M10 is therefore practically trained in this continuation, but M9 is not. These are checkpoint continuations (not uninterrupted scratch policies), so the non-monotone M9-to-M10 pattern must not be presented as a clean length law or breakpoint. The uninterrupted scratch result remains the width-25 M8 result (1024/1024); M10 is supporting evidence that a longer policy can be recovered under the continuation protocol, not a replacement for a matched scratch row. Dense M10 is now complete at 0/512 pooled strict success; matched dense M2/M3 are queued separately.

Last audited: 2026-09-23 01:28 UTC. This file records the **already-run homogeneous pick-and-place (PnP) experiment**, its current evidence, and the three follow-up branches. The authoritative raw runs and evaluations live in `release-root/runs/compositional/`; code lives in `main_method/metaworld/stage_reward/`. A proposed experiment is not a reported result.

**Extension audit (2026-10-02 22:47 UTC):** The corrected M11 scratch run is still live at approximately `2058/7700` iterations, with the latest training-rollout rates around raw `0.464` and mandatory `0.453`; these are not held-out SRs. To isolate the fixed-stage-width contract, a post-extension queue has also been launched for matched-width scratch controls `M9` and `M10` using the same v10 target chain, 46-way raw stage one-hot, no-reset handoff, and `700M` PPO scaling. It waits for the M11--M15 and heterogeneous queues and will not overwrite any existing result. Until those controls and the final M11--M15 evaluations exist, no length breakpoint is claimed.

**Queue-rescue audit (2026-10-02 22:59 UTC):** GPU3 also contains a long-lived Qwen stage worker configured with `--single-gpu-device 0`. The original queue's conservative PID guard would therefore block every post-M11 job indefinitely even though the active M11 PPO process is healthy. A rescue queue, `benchmark/Metaworld/queue_composition_extension_v6_rescue.sh`, now waits for M11's actual `training_summary.json`, retires only the stale v3/v4/v5 shell wrappers, completes M11's held-out evaluation, and continues M12--M15, plate-slide M1--M5, and all 23 predeclared heterogeneous sequences. It ignores only that explicitly identified worker and never terminates it; no duplicate training is started while M11 is live.

**Closeout status (2026-09-23 01:28 UTC):** The remaining M11 and matched dense M3 jobs were stopped after preserving their latest checkpoints and logs. M11 has no final EVAL; matched dense M3 has no final EVAL. They are excluded from the reportable M1--M10 table.
**Update (2026-09-21 21:50 UTC):** The original no-stage dense v1 M2 endpoint is now complete at **0/256** strict success with strict prefixes `[0,0]` (raw/native prefixes `[46,0]`). The direct dense v1 M5 screen is also complete at **0/256**, with strict prefixes `[0,0,0,0,0]` (raw/native prefixes `[51,0,0,0,0]`). Together with dense M1=256/256, this gives a consistent preliminary pattern: one macro is learnable, while no-stage dense composition fails already at M2 and remains failed at M5 under the tested budgets. This is still not a reward-only causal claim because the baseline removes stage information as well as changing reward activation.


At 22:20 UTC, dedicated monitor sessions were attached to all three recovery jobs. Each waits for the target checkpoint and `training_summary.json`, then runs strict gated evaluations on seeds 1042 and 2042 (256 episodes each) and writes the summaries under the continuation run directory. The dense M10 monitor was corrected after an initial missing-`PYTHONPATH` attempt; both dense seed evaluations are now complete and strict-zero.

At 21:53 UTC, recovery continuations were started from the last optimizer checkpoints (M8 model 5050, M9 model 4125, M10 model 4075) to the original fixed budgets. These are explicitly labeled checkpoint continuations with reinitialized environments/RNG, not new uninterrupted scratch rows. Final strict evaluations will be recorded separately and cannot retroactively turn the interrupted scratch rows into a breakpoint claim.

## Current conclusions for the main text (2026-09-21)

**Update (2026-09-21 14:30 UTC):** The dense v1 M1 endpoint is complete at **256/256** strict success (700 PPO iterations, 256-episode held-out evaluation). The tuned no-stage dense M2 endpoint (scales `(4,20,32)`, 1400 iterations) is complete at **0/256** end-to-end: its strict first-macro prefix is **246/256**, while the raw/native prefix is 256/256 and only 25/256 episodes complete the second macro natively. Thus the current empirical trend is strong: the flat dense policy learns one transfer but drops sharply at the next no-reset handoff. The original unmodified dense v1 M2 run is still finishing and should be reported separately; the M2-v2 result is an exploratory scale variant, not a reward-only causal comparison because the policy also lacks stage ID.

**Implementation audit (14:40 UTC):** No execution/evaluation mismatch was found. Dense training and evaluation use the shared `MetaWorldGraspInsertVecEnv` stage FSM, the same `tabletop_unique_chain_v8`, `max_macro_length=250`, no reset flags, and the same strict prefix evaluator. In `grasp_insert_core.step`, the all-dense mode explicitly calls `dense_sum(..., active_stage_count=3)`, so it sums local atomic functions `(0,1,2)` for the current macro only; it does not leak future macro targets. The handoff calls `state.reset(...)`, clearing per-stage potential baselines. Saved M1/M2 actor matrices and normalizers have input shape 39, confirming no stage one-hot reached the policy. The M2-v2 summary's strict fields are `mandatory_prefix_success_episodes=[246,0]`, `mandatory_success_episodes=0`; its raw/native prefix `[256,25]` must not be reported as strict success. The remaining confound is intentional but important: this baseline changes both reward form and stage observation. The unmodified v1 M2 endpoint and a stage-ID-retained all-dense control are the clean next checks.

**Update (2026-09-21 08:52 UTC):** The unchanged summed-dense/no-stage M1 policy reached 58/64 strict held-out successes at its nonfinal iteration-50 checkpoint, while M2 was 0/64 at iteration 100; full-budget results are pending. Direct dense M5/M10 screens are now running. This is an early optimization trend, not proof that dense cannot solve longer chains. This baseline changes both reward activation and stage observation, so any gap is not reward-only. The same-width-31 RoboStep M8/M9/M10 scratch series is also pending; successful width-25 M8 does not guarantee M10 success.

**No breakpoint is itself a positive, bounded result.** In the continuous homogeneous PnP protocol, one object is moved through distinct targets without resetting the arm, object, or scene between macros. Each transfer has three gated atomic stages, so M8 means eight transfers and K24 stages. The scratch-trained width-25 M8 policy passed 1024/1024 strict gated held-out episodes; its same-width M7 control passed 1003/1031 (97.28%). The separate matched width-22 M6/M7 pair passed 768/768 at both lengths. The width-19 M5/M6 pair passed 1024/1024 and 1030/1042 (98.85%). Thus completed matched-length evidence shows no robust monotonic decreasing law or confirmed length-induced breakpoint through M8. This demonstrates reach to M8; it does **not** prove that composition never degrades. These evaluations pool several environment seeds for **one training seed per row**, not independent trained-policy replications.

A separate width-28 seed-42 scratch M8/M9 pair both scored 0/1024, despite the successful width-25 M8 run. Because the same-width *shorter* control already fails, M9's zero is **not** an M8-to-M9 length breakpoint. The result exposes training/configuration sensitivity but does not by itself identify input width as the cause. A new same-width-31 M8/M9/M10 scratch series and width-28 seed controls were pending at the latest machine digest (2026-09-21 06:53 UTC); pending is not zero, and interim checkpoints are not final results. Do not select only the successful width when discussing robustness or only the failed width when arguing for a decreasing law.

The fair claim is method capability in this controlled setting, **not** that “ordinary RL cannot solve long PnP” or that composed rewards alone caused the result. This task repeats one semantic primitive with controlled geometry and reward template; stage ID is provided, targets are fixed, and PPO training budget grows as `700M` iterations. To compare with conventional RL, run matched PPO baselines using the same scene, target sequence, no-reset handoff, observation/stage signal, architecture, horizon, training seeds, evaluation seeds, and `B(K)`, changing only the reward formulation (for example terminal-only and a noncomposed dense reward). Until then, “ordinary RL is harder” remains a hypothesis, not an experimental finding. Suggested wording: “Under a no-reset, homogeneous pick-and-place protocol, composed stepwise rewards support eight consecutive object transfers (24 atomic stages) at 1024/1024 strict gated evaluation success; completed matched-length comparisons reveal no reproducible length-induced breakpoint in the tested range.” Add the one-training-seed limitation next to the claim.

The other branches currently answer different questions. Continuous plate-slide A and canonical-reset isolated B each passed 256/256, but AB passed 0/256 with gated macro prefixes `[256,0]`; isolated B trained/evaluated on real no-reset A-terminal states also passed 0/256. This implicates handoff-distribution learnability and cannot establish an intrinsic reward-composition law. In the *compound-scene* heterogeneous AAA/AAB/ABC pilot, strict end-to-end rates were 7/256, 0/259, and 0/256, respectively, but original isolated C was itself 0/256. Changing only C's first-stage dense scale from 4 to 12 raised isolated C to 256/256; revised ABC scratch remained pending at the 06:58 UTC digest. Do not use the unsolved original C in an SR* comparison or conflate these compound-scene rows with the earlier same-scene semantic pilot below.

For the paper table, enter completed PnP `M,K,SR` and strict gated `q_prefix`; leave `SR*` and `SR/SR*` unset until isolated `p_i` are measured on a clearly defined distribution, preferably including real predecessor-terminal handoffs. A publishable “no detected breakpoint up to M8” claim must disclose the failed width-28 screen, training-seed uncertainty, and the absence of a matched ordinary-RL baseline. Extending M9/M10 tightens the tested bound; it is not necessary to manufacture a failure to make the result meaningful.

## New flat/no-stage control and long-task tuning (2026-09-21 07:41 UTC)

The requested **pure summed-dense/no-stage** PnP baseline is implemented in `main_method/metaworld/stage_reward/run_forward_dense_no_stage.py`. At each step of the *currently active* object-to-target macro, it sums the three existing atomic potential-difference rewards with unchanged scales `(4,10,32)` and gamma `0.99`, but zero transition bonuses. Actor and critic receive only the MetaWorld 39-D state (including the current target), not stage one-hot; a smoke test confirmed both first layers have 39 inputs. The internal rule gate remains only for macro target switching and identical strict gated evaluation. Arm/object/scene do not reset between macros. We do **not** sum future-target potentials simultaneously, because that would reward mutually incompatible locations. This changes both reward form *and* stage observation, so any gap cannot be attributed to rewards alone.

Formal seed-42 M1/M2 scratch controls started at 07:29 UTC under `release-root/runs/compositional/additional_result/`, with the same `128 envs × 64 rollout steps × 700M PPO iterations`, target chain, 250-step-per-macro horizon, hidden architecture, and strict evaluator as the staged series. Their final 256-episode scores are **pending**. If M1 itself is not solved at its full budget, tune that primitive before interpreting M2 as a length effect; keep the untouched direct-sum v1 as a separate baseline. The inherited `task_definition.stage_context_normalization` string in these two generated configs incorrectly mentions stage one-hot. The authoritative `environment.actor_observation=metaworld_state39_without_stage_id`, `policy_stage_information=false`, and the saved actor/critic input shape of 39 confirm that stage information is absent.

“Width-28 M8” means retraining M8 with a 28-slot global stage one-hot, the minimum for M9's 27 active stages plus a terminal slot, so M8 and M9 have the same policy input dimension. Both width-28 M8 and M9 scored 0/1024, despite successful width-25 M8; therefore M9's zero is not a length breakpoint. The new width-31 M8/M9/M10 runs are still below their fixed 5600/6300/7000-iteration budgets. Nonfinal 64-episode strict-prefix diagnostics are M8 at iteration 3500: `[64,61,52,0,0,0,0,0]`; M9 at 2800: `[64,0,0,0,0,0,0,0,0]`; M10 at 2700: `[64,0,0,0,0,0,0,0,0,0]`. These are optimization snapshots, **not** final SR or a demonstrated breakpoint.

A guarded queue waits for each fixed final scratch evaluation. Only if gated SR is below 95%, it continues that M8/M9/M10 checkpoint from `700M` to `1050M` total iterations with PPO optimizer restored and evaluates strict prefixes. The M8 same-width control is extended too. This is explicitly a checkpoint continuation with simulator/RNG reinitialized, not an uninterrupted scratch replicate or a main equal-budget row.

A nonfinal checkpoint-25 probe on 64 seed-1042 episodes found M1 native object-goal success 43/64 but **strict three-stage success 0/64**; M2 had first-macro native success 36/64 and strict prefix `[0,0]`. Both policies triggered the pregrasp transition in all 64 episodes but never the capture/lift transition. This is too early to call either baseline unsolved, yet it exposes a potentially important mismatch between optimizing the flat object-goal potential and completing a prescribed PnP sequence. Report both physical/native and strict gated diagnostics; do not treat the native M1 count as full composed-task SR.

A separately named v2 tuning run started at 07:54 UTC on GPU 2 for M1 only, using summed dense scales `(4,20,32)` instead of v1's `(4,10,32)` to strengthen the unsolved capture/lift behavior; the function family, no-stage 39-D policy observation, zero transition bonuses, scene, optimizer, and `700M` budget are unchanged. This is an exploratory hyperparameter revision. Its result is pending and will be compared against the preserved v1 checkpoint at the same fixed budget; it is not silently substituted for the unmodified baseline.

Further **nonfinal** held-out seed-1042 probes show the original `(4,10,32)` pure-dense/no-stage M1 checkpoint at iteration 50 reached 58/64 (90.63%) strict gated completion, versus 0/64 at iteration 25. This confirms the single macro can be learned with the unmodified flat reward. M2 at iteration 75 remained 0/64 full and 0/64 strict first-macro prefix, despite 56/64 native first-target proximity; its actor had not completed capture/lift. Because the two policies were trained from scratch for different budgets and neither is final, these snapshots **cannot** establish a dense-reward length breakpoint. Continue the fixed endpoints and inspect whether M2's strict first prefix recovers before tuning the longer baseline.

At the same nonfinal iteration 25, tuned v2 M1 with scales `(4,20,32)` reached 9/64 strict gated success versus v1's 0/64, and entered capture/lift in 46 of the 64 episodes; that is early optimization evidence, not a final superiority claim. Tuned v2 M2 also started at 08:04 UTC with the same no-stage contract and full 1400-iteration budget. A guarded v1 length-search queue waits for the M1/M2 fixed final evaluations: only if both reach at least 95% strict SR will it train M4, then M8 and (if solved) M9/M10; if M4/M8 fail it trains an intermediate M3 or M5–M7 control. Every row remains an independent scratch run, and a candidate drop still needs fresh evaluation/training seeds before a paper breakpoint claim.

At 08:42 UTC, additional **direct skip-length screens** started for the unchanged v1 pure-dense/no-stage reward: scratch M10 on GPU 1 (`7000` PPO iterations) and scratch M5 on GPU 2 (`3500` iterations), both seed 42 and scheduled for 256 held-out episodes at seed 1042. They run concurrently to shorten wall time; the M1/M2 controls continue. No M5/M10 scores are available yet. Treat these as candidate points for an adaptive length search, not as a binary-search proof of a monotone breakpoint: stochastic training can fail at one length and succeed at another. After final evaluations, bracket with intermediate lengths and matched training-seed replications; if M1/M2 remain unsolved, first repair the primitive baseline before claiming a composition-length effect. The old M4 `tabletop_square_chain_v5` final evaluation used 128 episodes, while the later same-protocol M5 and M8 results each pool four approximately 256-episode evaluation seeds into 1024 episodes for **one policy trained at seed 42**.

Queue correction (08:50 UTC): the earlier M4-then-M8 guarded schedule described above was replaced by an adaptive midpoint queue. It waits for the final M1/M2 and M5/M10 screens; only if M1/M2 pass does it train the next midpoint (`M3/M4` if M5 fails, or `M7/M8/M9` as needed if M5 passes and M10 fails). This is a screening order, not a monotonic-law assertion.

## Question and unit of composition

The question is whether a policy trained from composed *stepwise reward programs* degrades as independently learnable macro-subtasks are chained. A macro-task is one object-to-target transfer; in this benchmark each macro has three gated RoboStep atomic stages:

\[
P_i=[\text{pregrasp}\rightarrow\text{capture and lift}\rightarrow\text{place and release}],\qquad
T_M=P_1\circ P_2\circ\cdots\circ P_M,\qquad K=3M.
\]

Report both `M` (macro-subtasks) and `K` (atomic stages). A new suffix adds a *complete* macro and three reward stages; it does not rewrite the reward program or constants of earlier macros. Only the active target, stage routing/index, and termination logic change. Training is **from scratch for every `M`**; suffix fine-tuning and physical resetting at a macro boundary are different ablations, not the primary comparison.

## Existing PnP benchmark: physically continuous homogeneous composition

Environment: MetaWorld `pick-place-v3`, one object and one arm. The object is moved through a fixed sequence of **distinct** tabletop destinations (`tabletop_unique_chain_v8`): `(-0.08,0.80), (0.08,0.80), (-0.08,0.84), (0.08,0.84), (-0.08,0.88), (0.08,0.88), (-0.08,0.82), (0.08,0.82), (-0.08,0.86), (0.08,0.86)` in XY, with the environment's object-height Z. The initial object location and first destination define `P_1`; subsequent entries are suffix destinations. These ten locations support at most ten macros without extending the geometry protocol; no destination repeats through `M=10`.

The arm, object, and MuJoCo scene **do not reset between macros**. After the third gated stage of `P_i` completes, the environment switches the visible target to the next destination and advances to `P_{i+1}` in the same physical state. A canonical scene reset happens only at the next episode. This retains handoff pose, object placement error, and accumulated control error as genuine composition factors. A fixed active-stage reward program supplies PPO's reward; MetaWorld native success is an evaluation diagnostic, not a replacement training reward. Earlier macro reward code/configuration remain frozen as the chain grows.

Current matched-run recipe: 128 parallel environments; 64 rollout steps per environment per PPO iteration; `700 M` iterations (equivalently `700/3` iterations per atomic stage); per-macro horizon 250 environment steps; seed 42; gate=`rule`; learning rate `5e-4`; entropy coefficient `0.003`; initial action noise SD `0.45`; deterministic held-out evaluation. Policy actor/critic hidden widths are `256,256,128`. State observation has 39 components; the global stage one-hot is left raw while the continuous state is running-normalized. **Input width must be held fixed within every adjacent-length causal comparison**: e.g. M5/M6 both stage-ID width 19, and current M6/M7 both width 22. If width must increase to fit a longer chain, retrain the shorter control at that width; do not attribute a width-induced failure to chain length. The general paper protocol should choose one predeclared `K_max` large enough for the whole tested range.

Primary outcome `SR(M)` is the fraction of episodes completing **all gated stages through `M`**, not merely putting the object near the final target. For each run, save the cumulative gated prefix rate

\[
q_{1:j}(M)=\Pr(\text{all gated transitions of }P_1,\ldots,P_j\text{ complete in }T_M),
\]

plus the handoff-conditional rate `q_j(M)=q_{1:j}(M)/q_{1:j-1}(M)` for `j>1`; `q_{1:1}` is the first-macro success. The last cumulative entry **must equal** `SR(M)`. For shared-prefix interference, compare `q_{1:1}(1),q_{1:1}(2),...,q_{1:1}(M)` with the same geometry, architecture, seed policy, and held-out evaluation protocol. Earlier raw `prefix_success_rates` recorded target proximity/native success and can be larger than actual gated completion: do not use them for the paper's `q_prefix`. The strict audit is in `appendix/forward_composition_v10/` and generated by `eval_forward_prefix_v10.py` / `collect_forward_gated_prefix_v10.py`.

Independently solve each `P_i` to obtain isolated success `p_i` and compute `SR* = \prod_i p_i` and `SR/SR*`. **Define the isolation distribution.** A canonical-reset standalone `p_i` is a learnability check, but is not necessarily the same as the distribution of arm/object states delivered by `P_{i-1}`. For a meaningful product baseline, additionally measure handoff-matched isolated `p_i` from saved predecessor terminal states (without resetting the arm); report canonical and handoff-matched products separately. A near-100% canonical rate does not itself explain a low composed rate. When `SR*=0`, ratio is undefined, not zero.

### Verified evidence, not a claimed breakpoint

### Complete homogeneous result ledger (including evaluation episode counts)

The previously omitted early rows are now listed explicitly. `Eval episodes` is the
actual number of completed evaluation episodes pooled over the reported held-out
seeds (not merely the requested batch size). M1--M4 use the earlier
`tabletop_square_chain_v5` scratch protocol; M5--M8 use the continuous
`tabletop_unique_chain_v8` scratch protocol; M9--M10 are the explicitly labelled
width-31 optimizer-restored continuation diagnostics. The geometry/protocol change
means this is a complete ledger, not a single clean causal length curve.

| Composition | M | K | Protocol / run family | Eval episodes | Mandatory gated SR | Gated cumulative prefix counts |
|---|---:|---:|---|---:|---:|---|
| PnP `P_1` | 1 | 3 | square-v5 scratch | 128 | 128/128 (100.00%) | `[128]` |
| PnP `P_1...P_2` | 2 | 6 | square-v5 scratch | 128 | 128/128 (100.00%) | `[128,128]` |
| PnP `P_1...P_3` | 3 | 9 | square-v5 scratch | 130 | 129/130 (99.23%) | `[130,130,129]` |
| PnP `P_1...P_4` | 4 | 12 | square-v5 scratch | 128 | 128/128 (100.00%) | `[128,128,128,128]` |
| PnP `P_1...P_5` | 5 | 15 | unique-v8 scratch, width 19 | 1024 | 1024/1024 (100.00%) | `[1024,1024,1024,1024,1024]` |
| PnP `P_1...P_6` | 6 | 18 | unique-v8 scratch, width 19 | 1042 | 1030/1042 (98.85%) | `[1042,1042,1042,1042,1042,1038]` |
| PnP `P_1...P_7` | 7 | 21 | unique-v8 scratch, width 25 | 1031 | 1003/1031 (97.28%) | `[1031,1020,1020,1018,1018,1015,1003]` |
| PnP `P_1...P_8` | 8 | 24 | unique-v8 scratch, width 25 | 1024 | 1024/1024 (100.00%) | `[1024,1024,1024,1024,1024,1024,1024,1024]` |
| PnP `P_1...P_9` | 9 | 27 | unique-v8 continuation, width 31 | 512 | 358/512 (69.92%) | `[512,512,493,493,493,493,488,469,440]` |
| PnP `P_1...P_{10}` | 10 | 30 | unique-v8 continuation, width 31 | 512 | 482/512 (94.14%) | `[512,497,497,497,492,482,482,482,482,482]` |

The M3 row has 130 completed episodes because vectorized evaluation finished a
parallel batch beyond the requested 128; this is why all SRs are reported as
numerator/denominator rather than rounded percentages alone. M9 and M10 prefix
counts above are pooled from their two 256-episode held-out evaluations. M11 and
all lengths beyond M10 were not included in the reportable table because time
expired before a final EVAL; they are unmeasured, not negative results.

The cleanest completed adjacent-length control so far uses the *same* width-19 stage input, same target chain, same PPO recipe, and training seed 42. Four held-out evaluation seeds test **one trained policy per length**, not four independently trained seeds.

| Composition | M | K | Mandatory gated SR, pooled held-out episodes | Gated cumulative prefix counts |
|---|---:|---:|---:|---|
| PnP `P_1...P_5` | 5 | 15 | 1024/1024 = 100.00% | `[1024,1024,1024,1024,1024]` |
| PnP `P_1...P_6` | 6 | 18 | 1030/1042 = 98.85% | `[1042,1042,1042,1042,1038,1030]` |

The M6 prefixes show a small loss beginning at the fifth/sixth handoff, but **not** a sharp, confirmed optimization breakpoint. Different evaluation seeds have different realized episode counts; pool numerators and denominators, not just percentages. Earlier `M6=0` with a narrower `K_max=10` is **confounded** because the matching M5 width-10 control also failed. Older reset/append/warm-start runs and different geometry or input widths are not interchangeable with the scratch, continuous, matched-control protocol. Do not fill a decreasing-law table by selecting incompatible checkpoints.

Audit update: the width-22 M6/M7 and width-25 M8 scratch jobs were interrupted before their planned budgets, leaving checkpoints at iterations 1800/1825/1700, respectively; none has a final training summary or held-out final score. Mid-training success is not a composition breakpoint. The next runs must either finish comparable full scratch jobs or explicitly label optimizer-restored checkpoint continuations as such. If M8 drops, run an **M7 width-25** control before attributing the drop to length. Confirm a candidate knee on several held-out evaluation seeds, then at least one additional **training seed** for the decisive adjacent pair. Save gated `q_prefix`, per-macro conditional success, failure videos/state traces, and the exact reward-code/config hashes. If all well-controlled lengths stay near ceiling, report that fact and a lower bound on the breakpoint; never manufacture a decline.

## Follow-up branch 1 — find or bound the PnP breakpoint

Complete the interrupted M6/M7 matched-width pair and M8 screen without treating intermediate checkpoints as final. Audit strict prefixes and isolate whether failure is an early-prefix regression, a particular new suffix, or a horizon cap. Extend to M9/M10 only if the preceding matched comparison remains at ceiling, using distinct remaining destinations. Keep `B(K)\propto K`, physical continuity, reward templates, and architecture fixed. For every new stage-ID width, rerun the shorter-length control. Beyond M10 first design a larger valid nonrepeating workspace trajectory and test standalone reachability; do not silently reuse target positions or move them outside MetaWorld's feasible workspace. A “breakpoint” means a reproducible downward change under matched controls, not a single seed's stochastic failure.

## Follow-up branch 2 — heterogeneous three-macro composition

Target rows are `AAA`, `AAB`, `ABC`, each with `M=3`, `K=9`, and identical budget `2100` PPO iterations, rollout size, actor/critic architecture, fixed global stage-ID width, horizon `750`, and evaluation protocol. Intended semantic modules are `A=pick-and-place`, `B=articulation` (drawer), `C=contact/press` (button). Compile/validate each three-stage reward module once, freeze its logic and constants, and compose by switching only the active module and its target/state at a **gated** boundary. Keep physical arm and scene continuous. Measure isolated `p_A,p_B,p_C`, strict per-macro prefixes, full `SR`, `SR*`, and ratio.

Important implementation constraint: the existing MetaWorld MT1 `pick-place-v3`, `drawer-open-v3`, and `button-press-v3` wrappers instantiate *different MuJoCo scenes*; PnP exposes four action dimensions whereas current drawer/button training wrappers expose three with fixed gripper. Sequentially loading those stock environments would reset/change the scene and **does not implement the primary continuous `AAA/AAB/ABC` experiment**. First build and validate one compound scene with all three manipulanda, a shared 4D action/observation contract, and individually solvable module starts (or explicitly label a same-scene semantic-pilot as such). Drawer/button standalone runs were planned but had not started at the latest audit; when run, they are feasibility baselines only, not composition results. If a scene-reset control is later run, put it in a separate ablation, never in the main heterogeneous row.

## Follow-up branch 3 — another homogeneous task class

Use **nonprehensile `plate-slide-v3`** as the first non-PnP class: its three existing atomic stages are `align_pre_push -> establish_contact -> move_object_to_target`. It is genuinely different from grasp-and-lift PnP yet supports tabletop translations in the reachable central workspace. Reuse the same unique XY-chain idea after confirming plate-slide's puck/goal bounds and that every transition is individually reachable; keep one puck and one arm in one scene with no between-macro reset. Build a continuous goal-switching wrapper, freeze the three-stage slide reward template for every macro, and scratch-train M1, M2, ... with `K=3M`, `700M` iterations, fixed architecture and matched stage-ID width. Validate isolated locations and handoffs before interpreting composition drops. A continuous plate-slide M1 scratch run has completed at 256/256 mandatory held-out success with stage-ID width 7. This validates the local primitive; a matched-width M2 is still required before any slide-chain breakpoint can be claimed. `push-v3` and `sweep-v3` may be later sensitivity checks, but are not substituted silently into the primary slide series.

## Reporting discipline

For each branch record source-code revision/hash, scene/geometry version, reward-module version, stage-ID width, train seed, held-out evaluation seeds, checkpoint choice, total environment transitions, wall-clock cost, `M`, `K`, `SR`, the definition and value of `SR*`, `SR/SR*`, every gated cumulative/conditional prefix, and videos of typical success/failure. Distinguish a **task-length effect** from a changed input dimension, new geometry, undertrained suffix, unsolved isolated module, horizon cap, or incompatible physical handoff. A clean negative result (no breakpoint up to tested `M`) is valid evidence.

## Current execution root and gates

New follow-up outputs belong under release-root/runs/compositional/additional_result/. For every PnP continuation record the source checkpoint iteration, whether optimizer state was restored, new environment/RNG reset, and exact additional PPO iterations. Such a resumed run is not an uninterrupted scratch replicate. For heterogeneous rows, do not count separately instantiated MT1 scenes as a continuous AAA/AAB/ABC comparison. Validate a single shared scene, action/observation interface, individual modules, and their handoff states first. Report every screened condition, including negative and infeasible ones.

## Execution update (2026-09-19)

The PnP M6/M7 width-22 optimizer-restored continuations are active in tmux (`comp_pnp_M6`, `comp_pnp_M7`). Their source iterations are 1800/1825 and target budgets are 4200/4900; the environment and RNG are reinitialized, so these are not uninterrupted scratch replicates. M8 width-25 continuation and a fully scratch M7/M8 width-25 control pair are queued. Strict gated prefix evaluations on four held-out seeds follow training automatically. Do not declare a knee by comparing M8 continuation against M7 scratch without also checking the continuation asymmetry.

The heterogeneous **same-scene semantic pilot** was queued as `AAA`, `AAB`, `ABC`, each at 2100 PPO iterations with 128 environments, 64 rollout steps, 250 steps per macro, seed 42, a 4D action interface, and a fixed 10-way raw global-stage one-hot. A is the frozen three-stage pick-place reward; B is the frozen three-stage object-push reward; C is three successive one-stage TCP waypoint-reach rewards. All modules run in the same `pick-place-v3` MuJoCo scene with no macro-boundary reset. The strict nine-transition gated success is the score; native PnP success is inapplicable to B/C. This is a semantic pilot, **not** the intended drawer/button compound-scene result. The original B/C runs were intended to open the gripper, but the `+1` action actually closes it in MetaWorld; see the later audit below. Isolated B/C runs initialize the object at a predecessor destination but reset the arm, so their success is not handoff-matched `SR*`. Low mixed scores cannot yet establish a compositional breakpoint. The separate drawer/button single-task probes are only feasibility checks.

The other homogeneous class is continuous `plate-slide-v3`: M1 width-7 finished at 256/256 mandatory held-out success; M2 width-7 is active, and a same-width-10 M2/M3 pair is queued. Those later runs have no final scores yet. The earlier scheduler was `main_method/metaworld/stage_reward/queue_compositional_followup_v1.sh`; the current extension scheduler is `experiments/compositional/run_m10.sh`. All new raw results and logs go under `release-root/runs/compositional/`.

A **nonfinal diagnostic** at plate-slide M2 iteration 475 used 64 deterministic seed-1042 episodes: the first macro's strict gated prefix completed 64/64, but the second macro's first gated stage completed 0/64 (stage-entry counts `[80,64,64,64,0,0,0]`). This is a candidate suffix failure, not a final M2 score or a proven composition breakpoint. A second-target standalone control is queued with the puck initialized at the first target and the arm reset to its native start, using the same frozen three-stage slide reward, 700 iterations, and width-7 input. It tests local suffix learnability but is **not** handoff-matched: a successful standalone plus failed chain still requires direct predecessor-terminal-state evaluation to separate exploration/interference from unfavorable arm handoff.

## Audit update (2026-09-20 UTC): what the current evidence does and does not show

The homogeneous PnP scratch, matched-width-25 M7/M8 pair is complete: M7 reached 1003/1031 (97.28%) and M8 reached 1024/1024 (100.00%) over four held-out evaluation seeds. There is no PnP length breakpoint through M8 in this pair. These are **one training seed each**; M8/M9 matched-width-28 scratch runs and further train seeds remain pending. The width-22 and width-25 continuation runs are diagnostics and should not be merged into a scratch-only trend.

The legacy heterogeneous pilot at held-out seed 1042 yielded AAA 256/256, AAB 61/258, ABC 0/256. The original B/C action override was `+1`, although the code called it "open"; MetaWorld interprets `+1` as gripper closure and `-1` as opening. Thus those rows are a *closed-gripper v1 protocol* and cannot support a claim that heterogeneity alone caused the drop. Merely changing v1 AAB's evaluation-time gripper command to `-1` reduced success to 7/256: actor training and evaluation must use the same command. The v1 AAB training curve was still rising near its 2100-iteration budget, so undertraining is also plausible.

A corrected **v2 open-gripper protocol** now trains isolated B and the matched AAA/AAB/ABC scratch rows from seed 42 with `-1` for B/C. The first B checkpoint at iteration 125 already scored 64/64 on its canonical standalone held-out states. A separate actor-swap diagnostic used the old AAB actor for the first two A macros, then the v2 B actor at the *real*, unreset AA-to-B handoff, remapping global stage IDs 6--8 to B-local 0--2. It scored only 5/256, mostly failing B's first alignment stage. That shows canonical isolated B success does not imply handoff-matched success; it is not the one-actor main result. V1 AAB/ABC optimizer-restored continuations to iteration 3500 run separately as training-budget diagnostics, with environment/RNG reinitialized. Their scores and the final v2 scores are pending. Do not attribute any remaining AAB/ABC gap to incompatible action distributions until the v2 rows, learning curves, handoff distributions, and continuation diagnostics are examined together.

The exploratory v1 AAB continuation checkpoint at iteration 2350 reached 210/256 and 186/256 on evaluation seeds 1042 and 2042, respectively: pooled 396/512 (77.34%), with both A prefixes at 512/512. This is a large gain over v1's iteration-2099 result of 61/258, showing strong budget sensitivity without any arm or scene reset at the macro boundary. The continuation did reinitialize the environment/RNG, and iteration 2350 was inspected after training started, so this is **not** a preselected main-table endpoint. At the same continuation iteration v1 ABC remained 0/64, failing before completing its first A macro. A separate ABC warm-start from the AAA actor and a v1-to-v2 AAB gripper adaptation are running; both are explicitly non-scratch diagnostics.

The fixed final v2 B-alone checkpoint at iteration 699 scored 256/256 on each of two canonical held-out evaluation seeds (512/512 pooled), but the **same B actor** scored 0/256 on each seed when switched in at the real, unreset AA-to-B handoff from the frozen v1 AAB prefix actor. The two A prefixes still completed in 251/256 and 250/256 episodes. Thus canonical isolated B learnability is not evidence for handoff compatibility. A separate handoff-matched B diagnostic now replays full simulator states collected at actual AA-to-B transitions for isolated B training, with different environment seeds reserved for evaluation. This uses privileged MuJoCo states solely to study the distribution shift; it is neither one-actor composition nor a main-table RoboStep result.

## Handoff audit (2026-09-21 UTC; diagnostic, not main-table SR)

At the real no-reset AB-to-C handoff, the `ABC` warm-start actor at iteration 1000 completes the first two gated macros in 250/258 and 254/258 episodes on held-out seeds 1042 and 2042. Replacing only its C-stage actions with a proportional controller using the already-visible TCP and goal coordinates yields complete nine-stage success of 250/258 and 254/258 (504/516 pooled, 97.67%) under the corrected v2 open-gripper protocol. Every episode that reaches C completes C under that controller. The proportional controller is a **scripted reachability upper bound**, not learned RoboStep actor performance or a main-table method. These runs show that C's target sequence, stage transitions, and physical handoff are viable; a learned-policy failure should be investigated as optimization/generalization at the C handoff, not automatically called a task-composition law. Counts may slightly exceed the requested 256 episodes per seed because parallel environments finish an evaluation batch together.

At the predeclared fixed continuation endpoint (iteration 3499), v1 AAB reached 257/260 and 251/257 on untouched evaluation seeds 1042 and 2042: 508/517 = 98.26% pooled, with both A prefixes 517/517. This is a single actor under continuous macro handoff. Relative to 61/258 at iteration 2099, the earlier AAB drop was strongly budget-sensitive; the continuation restored the PPO optimizer but reinitialized simulator and RNG, so it is not an uninterrupted scratch training replicate. The same v1 ABC continuation still scored 0/512 at the fixed endpoint, failing before the first A macro. This does not isolate task heterogeneity as the cause of ABC failure.

The v2 scratch AAA/AAB/ABC rows and a separate predeclared ABC warm-start from the corrected-gripper AAB final checkpoint remain pending. The warm-start is an optimization diagnostic, not a scratch comparator. Report the fixed final checkpoints on both held-out seeds before a paper conclusion is drawn.

### Matched v2 C transfer audit

The fixed-final v2 isolated C policy (open gripper) scored 256/256 on each of the two held-out seeds from canonical reset states. Swapping this *same* final C actor into the real, no-reset AB-to-C handoff of the fixed-final AAA-initialized ABC prefix yielded 0/256 on each seed. The prefix completes AB in most episodes, so the 0/512 is a C policy transfer failure, not merely failed access to C. On the same corrected open-gripper handoff, the visible-state proportional controller completed 504/516 end-to-end episodes. The controller is a scripted reachability diagnostic, never learned-policy performance. Canonical C success therefore cannot be used as handoff-matched `p_C`; the gap is evidence of state-distribution mismatch for this isolated policy, while the scene and target remain reachable.

The final v1-to-v2 AAB gripper-adaptation diagnostic reached 256/256 and 241/257 on the two test seeds, 497/513 = 96.88% pooled, with both A prefixes at 513/513. It is a single actor with physical continuity between macros, but a cross-protocol continuation, not a v2 scratch row. The fixed-final AAA-initialized ABC actor reached A in 512/512 and AB in 492/512, yet finished ABC in 0/512, never completing C's first waypoint stage. The AAB-initialized ABC diagnostic and v2 scratch seed-43/44 replicas remain pending; these are necessary before attributing the remaining drop to an irreducible action-distribution conflict.

### Fair v2 scratch endpoint, training seed 42

At the predeclared iteration-2099 checkpoint with identical PPO budget, architecture, state/stage input, corrected open-gripper protocol, and no macro-boundary reset, two held-out evaluation seeds give AAA 512/512 (100.0%), AAB 509/514 (99.03%), and ABC 0/512 (0.0%) mandatory nine-stage completion. Strict gated prefix counts are `[512,512,512]`, `[514,514,509]`, and `[0,0,0]`, respectively. The ABC failure already occurs in the **first A macro**, before B or C is executed at evaluation; it is not direct evidence that the B-to-C physical handoff failed. The result is a sharp optimization failure for this training seed, not the gradual compositional decline originally hypothesized. Do not report it as a smooth action-distribution law.

AAB and ABC training configs differ only in sequence, generated stage labels/instructions, and run name. Their saved policy/normalizer state tensors are bitwise identical through iteration 75 and diverge by iteration 100. At iteration 75 both frozen policies score 0/64 on the first A macro in a held-out seed-1042 probe. This establishes a shared initial training trajectory and later divergence, but does not by itself identify the gradient event responsible. Seed-43/44 scratch replicas and the adapted-AAB-to-ABC warm-start are still running; all screened outcomes must be reported, including failures.

### Frozen-checkpoint fresh test and extension queue

Because seeds 1042/2042 were used repeatedly for development diagnostics, they are not called untouched test. Before running any new evaluations, environment seeds 11042/12042 were fixed for the final scratch checkpoints and are not used for model selection. For training seed 42, fresh-test AAA is 508/512 (99.22%), AAB is 503/516 (97.48%), and ABC is 0/512. Their strict prefix counts are `[508,508,508]`, `[516,512,503]`, and `[0,0,0]`. These reproduce the development ordering. Main reporting should average each training seed's test SR over training seeds 42/43/44, not pretend that multiple environment seeds are independent trained policies.

Additional same-scene screens `AAC`, `ABA`, and `ABB` were queued before seeing their results, with the same 2100-iteration scratch recipe, fixed 10-way stage input, v2 open-gripper protocol, and continuous arm/scene. They test whether the collapse is specific to the early B or B-to-C ordering. These screens are exploratory and do not become a selectively chosen replacement for the original AAA/AAB/ABC triplet; report every outcome. None of these pilots is the originally intended drawer/button compound scene.

### Curriculum diagnostics and stage semantics

The first adapted-AAB-to-ABC weight initialization was semantically mismatched: source AAB uses global stage IDs 6--8 for B, while target ABC uses IDs 3--5 for B. Its fixed-final result `[512,0,0]` must not be presented as a clean skill-transfer failure. An explicit target-to-source first-layer stage-column remap (`3:6,4:7,5:8`) was implemented and recorded in config. Its first saved checkpoint preserves A at 64/64 but still has B at 0/64. That remaining gap is also physically meaningful: source AAB learned B after an AA handoff, whereas target ABC requires B after only A. A stage-index remap cannot remove this state-distribution change.

Therefore a handoff-matched curriculum trains `AB` from scratch for 1400 iterations and then initializes `ABC` from that fixed AB endpoint without a stage remap; A and B occupy the same global IDs 0--5 and the same physical A-to-B prefix. A separate continuation extends the successful AAA-initialized ABC actor from 1400 to 2800 iterations. Both are optimization diagnostics, not equal-budget scratch rows. The original naive transfer, corrected column-remap transfer, matched-AB curriculum, and all continuations remain in the result ledger regardless of outcome.

### Three training seeds on frozen fresh test

For the fair 2100-iteration v2 scratch triplet, environment seeds 11042/12042 were fixed and not used for model selection. Each cell pools those two evaluation seeds for **one trained policy**; mean and sample standard deviation are then taken over training seeds 42/43/44, not over episodes.

| Sequence | Train seed 42 | Train seed 43 | Train seed 44 | Mean +/- SD (3 train seeds) |
|---|---:|---:|---:|---:|
| AAA | 508/512 (99.22%) | 0/512 (0%) | 512/512 (100%) | 66.41 +/- 57.51% |
| AAB | 503/516 (97.48%) | 0/512 (0%) | 512/512 (100%) | 65.83 +/- 57.02% |
| ABC | 0/512 (0%) | 0/530 (0%) | 512/512 (100%) | 33.33 +/- 57.74% |

This is severe training-seed sensitivity, not a stable smooth decrease: seed 43 fails even homogeneous AAA, while seed 44 solves fully heterogeneous ABC on all 512 fresh-test episodes. On seed 42, ABC fails its first macro; on seed 43, ABC completes its first macro in 130/530 but fails its second; on seed 44, all three macros complete. Canonical isolated C and the real AB-to-C handoff are learnable, as seed 44 also demonstrates with a single end-to-end actor. With only three training seeds and such high variance, do not claim a monotonic heterogeneity law, a statistically resolved gap, or an intrinsic ABC impossibility. The 3500-iteration seed-42 ABC scratch continuation reaches A in 512/512 but still completes AB in 0/512, confirming that training budget changes *where* its failure occurs; this continuation is not a matched scratch main-table row.

### Additional fixed screens

The predeclared seed-42 `AAC`, `ABA`, and `ABB` scratch screens all scored 0/512 on development evaluation seeds 1042/2042. Their strict prefixes were respectively `[512,512,0]`, `[0,0,0]`, and `[0,0,0]`. AAC retains its A/A prefix and fails when C is added; ABA/ABB fail before reaching the changed suffix. None supports a stable smooth decline, and all remain in the ledger.

For homogeneous PnP, the matched width-28 seed-42 M8 and M9 scratch rows are both 0/1024 over four held-out evaluation seeds, whereas M8 width-25 scratch was 1024/1024. Thus M9=0 is not an M8-to-M9 length breakpoint: its same-width M8 control already fails. Width-28 training seeds 43/44 are queued to test whether this is a repeatable width/input effect or initialization sensitivity. Do not infer either cause before those controls finish.

### Six-training-seed extension and flat-dense comparator (queued 2026-09-21)

Freeze the existing v2 scratch protocol for AAA/AAB/ABC and add training seeds 45, 46, and 47. These are nine new 2100-iteration runs, not extra evaluation seeds or checkpoint continuations. Evaluate each fixed final checkpoint (iteration 2099) on predeclared environment seeds 11042/12042, 256 episodes requested per seed, using the same strict gated-completion metric. Report all six train seeds (42--47), per-seed rates, mean and sample SD; do not select training seeds or checkpoints by validation score.

For each sequence, the flat_dense comparator uses the same scene, 4D action, goal coordinates in the native 39D observation, PPO architecture/hyperparameters, train seed 42, 2100 iterations, no inter-macro reset, goal schedule, and success criterion. Its actor and critic receive only the 39D state, with **no 10D stage one-hot**. Its training reward has **no transition bonus** and no local-stage-dependent dense term: for A/B it is -10 ||object-goal|| - 2 ||tcp-object||; for C it is -10 ||tcp-goal||, each per step. The internal rule predicates remain solely to advance the sequential task/goal and to score strict completion. Thus this is a goal-conditioned flat-dense comparator, **not** a completely stage-free environment and **not** a single-factor stage-ID ablation; the reward shape and policy input both differ. The task-specific choice of controlled entity (object for A/B, TCP for C) is fixed before seeing results.

### Causal 2x2 diagnostic (queued 2026-09-21 14:25 UTC)

The fixed flat-dense result (stage ID absent, transition bonus absent) is now complete at 0/512 for each of AAA, AAB, and ABC on fresh seeds 11042/12042, but it uses only training seed 42. To separate the two missing ingredients, six additional fixed-budget runs are queued at train seed 42: `stage ID + flat dense` and `no stage ID + transition bonus` for each of AAA/AAB/ABC. Both retain the same scene, PPO architecture, 2100-iteration budget, no-reset handoff, goal schedule, action convention, and final evaluation seeds. The runner and collector store these as separate diagnostic rows; they are not replacements for the original staged-vs-flat main comparison. The intended interpretation is a 2x2 factor screen, not a claim based on one seed: the prior staged condition is stage ID + transition bonus, and the completed flat-dense condition is no stage ID + no transition bonus.

The flat-dense iteration-700 checkpoint and the existing staged seed-42 iteration-700 checkpoint each get a matched development-only 128-episode seed-1042 probe for fast diagnosis. These are not final results and will not be used to choose checkpoints. The final comparison uses iteration 2099 and seeds 11042/12042, exactly as above. Runs live in persistent comp_compare_* tmux sessions, with code in run_forward_mixed_seed_compare.sh, watch_forward_mixed_flat_probe.sh, and run_forward_mixed_stage_probe.sh; raw data go under additional_result/heterogeneous/same_scene_pilot/.


## Consolidated reportable conclusion (2026-09-22)

The strongest completed homogeneous evidence is a capability bound, not a monotone failure law. Under the continuous no-reset protocol, matched scratch controls reach M5/M6 and M6/M7 with approximately 99--100% strict gated success, and the width-25 M8 scratch policy reaches **1024/1024** (K=24 atomic stages). Width-31 continuation checks reach M8 **491/513 = 95.71%**, M9 **358/512 = 69.92% after an extra 700-iteration continuation**, and M10 **482/512 = 94.14%**. Because M8--M10 are checkpoint continuations rather than uninterrupted scratch policies, the M9 dip and M10 recovery are not a clean length-induced breakpoint.

The completed no-stage flat-dense control is much weaker: dense M1 reaches **256/256**, original and tuned dense M2 both reach **0/256**, the matched dense M2 control reaches **0/512** across two held-out seeds, dense M5 reaches **0/256**, and dense M10 reaches **0/512** with all mandatory prefixes zero. The matched dense M3 run was stopped before final EVAL. This supports the conclusion that the present flat-dense/no-stage configuration does not learn long no-reset compositions under the tested budgets. It is not a reward-only causal claim because removing stage ID changes the policy input as well as reward activation.

Paper-safe wording: **stepwise stage-conditioned rewards support long homogeneous compositions (at least M8/24 atomic stages in the clean scratch protocol), whereas the tested flat-dense/no-stage baseline collapses at M2 and remains unsuccessful at M5/M10.** Do not claim that ordinary RL cannot solve long PnP or a monotone decreasing law until matched-length scratch replications or a stage-ID-retained flat-dense control are available.

The **real checkpoint-inference staircase** is saved as `compotional_homo.svg` and `compotional_homo.pdf` (exact requested filenames); correctly spelled copies `compositional_homo.svg` and `compositional_homo.pdf` are also retained. It uses frames captured at each strict macro handoff from the successful M10 continuation checkpoint `model_6999.pt`; panel M is a cumulative prefix formed from frame 1 plus frames 2--M at 60% opacity, so M10 contains all ten handoff frames. It is not a plot asserting that every M1--M10 row has a final SR.

The v8 target chain is **unique but deliberately narrow**: it uses two x-columns (`-0.08,+0.08`) and five y-levels (`0.80,0.82,0.84,0.86,0.88`). Thus M10 is not only two repeated points, but it is a zig-zag through a compact 2-by-5 layout; pairs such as M1/M7 and M2/M8 are close (2 cm in y) and can look nearly coincident in the `corner3` camera. The revised staircase now overlays the true goal-site locations as explicit colored rings on the real inference frames. This is a visualization aid, not a change to the policy or target geometry. It should not be described as ten widely separated arbitrary destinations; a more spatially diverse chain would require a new geometry protocol and retraining.

### Explicit M9 continuation record

The previously missing M9 row is now recorded explicitly: `unique10_v11_cont_M9_stage31_seed42_base700_to7000_recovery_v2`, continued from `model_6299.pt` to `model_6999.pt` with PPO optimizer restored and environments/RNG reinitialized. Strict gated evaluation gives **178/256** on seed 1042 and **180/256** on seed 2042, pooled **358/512 = 69.92%**; mandatory prefix vectors are `[256,243,243,243,243,243,243,218,178]` and `[256,250,250,250,250,250,240,231,180]`. This is an admissible continuation diagnostic, not an uninterrupted scratch M9 row.

### Composition extension queue (2026-10-02)

**Live continuation status (2026-10-02):** The persistent extension queue is
running the corrected scratch PnP `M=11` row under
`pick-place-v3/M11/rule/unique15_v10_scratch_M11_stage46_seed42_base700_env128/`.
At the latest poll it had reached iteration `1698/7700` (latest saved
checkpoint `model_1700.pt`); the training-rollout rates were raw `0.3547` and
mandatory `0.3414`.  These are
optimization diagnostics, not a held-out SR.
The queue
has not started M12 yet, and it will not infer a breakpoint from this early
checkpoint.  M12--M15 remain scheduled with the same no-reset protocol,
distinct-target geometry, fixed stage width 46, and `700M` PPO budget.

The alternate homogeneous macro is **plate-slide-v3** rather than another
pick-and-place variant.  It uses one puck and one arm on the same tabletop;
each macro is `align/pre-push -> establish contact -> push to target`, and the
next macro changes only the target and global stage index.  The arm, puck, and
scene remain physically continuous at the boundary (`reset_between_macros =
false`).  The v2 waypoint list keeps all M1--M5 destinations distinct and in
the central support region.  This preserves the key no-reset composition
question while changing the semantic primitive and action mode.

For the heterogeneous length-five pilot, exhaustive enumeration would require
`3^5=243` sequences.  The queue instead uses a predeclared coverage screen:
three homogeneous controls (`AAAAA`, `BBBBB`, `CCCCC`), ten single-substitution
position probes (one `B` or `C` at each of the five positions), three block or
alternation probes (`AABBB`, `BBBAA`, `ABABA`), and two mixed-type order probes
(`ABCAB`, `ABCBA`).  This is 18 sequences, not a post-hoc selection.  It
separates within-type length effects, position/order sensitivity, and
multi-type interaction at a fraction of the full factorial cost.  All rows use
the same-scene continuous protocol, fixed stage width 16, 3,500 iterations
(`700 x 5`), open-gripper command `-1.0` for B/C, and no macro-boundary reset.
If this screen reveals an order effect, the next expansion should add the
reverse/cyclic ABC orders rather than silently treating the 18-row screen as
an exhaustive law.

**Pre-registered order-coverage extension:** because the adjacent ordered-pair
counts in the 18-row screen are not perfectly balanced, a separate persistent
queue `benchmark/Metaworld/queue_composition_extension_v4_hetero.sh` waits for
the v3 queue to finish and then adds five non-overlapping probes:
`ACBAC`, `BCABC`, `CABCA`, `BACBA`, and `CBACB`.  These cyclic/reverse strings
are not selected from their outcomes; they complete the intended order probe
set while keeping the total at 23 length-five sequences rather than 243.  They
use exactly the same seed, stage width, budget, no-reset handoff, open-gripper
protocol, and held-out evaluation as v3.

Diagnostic note: an old incomplete M11-v9 checkpoint was inspected while the v10 queue waits for a GPU. At iteration 1650 (only 1650/7700 planned PPO iterations), it reached 4/129 = 3.10% mandatory completion on 128 requested seed-1042 episodes. Prefix rates were `[1.00, 1.00, 1.00, 0.829, 0.829, 0.829, 0.147, 0.047, 0.031, 0.031, 0.031]`; the failure begins around macro 7 and is consistent with undertraining at this checkpoint, not evidence against the new M11--M15 protocol.

The new plate-slide v2 implementation passed a CPU smoke run (M1, 700 iterations, 16 environments) with a 128-episode deterministic seed-1042 diagnostic at **128/128** mandatory success. This is a protocol/implementation smoke result, not the planned 128-environment M1--M5 main run; its purpose was to verify that the central target chain and fixed 16-way stage context are trainable before spending the GPU budget.

During the CPU M2 screen, a predeclared diagnostic at iteration 225 already completed the first slide macro in **64/64** held-out episodes, while the second macro completed **0/64** (`strict_gated_prefix_rates=[1.0, 0.0]`). This localizes the current failure to the second-macro handoff/learning stage rather than the v2 target-chain implementation or M1 learnability. It remains an intermediate checkpoint, not the final M2 result; training continues to iteration 1400.

The same M2 screen completed its full 1400-iteration budget. The independent deterministic
evaluation of the final `model_1399.pt` used 256 episodes (64 vectorized environments),
the fixed seed 1042, the v2 target chain, and the 16-way stage context. It achieved
**256/256 (100%)** for the first macro but **0/256** full two-macro completions;
the strict prefix rates were `[1.0, 0.0]`, with failure counts `[0, 256]`.
The gate accepted all 1280 rule candidates, so this is not a gate rejection or target
protocol mismatch. This remains a CPU screen rather than the planned 128-environment
formal row, but it confirms that the second slide macro is the current learnability/
handoff bottleneck under this setting.

This low M2 screen is not evidence that the second v2 target is geometrically
unreachable. Existing controls for the same target `(0.08, 0.88)` include a
canonical isolated suffix policy at stage width 7 with seed 42 reaching **256/256**,
and a continuous v2 M2 policy at stage width 7 with training seed 43 reaching
**256/256**; the corresponding training-seed-42 M2 row reached 0/256. Thus the
failure is seed/width/handoff-sensitive and must be debugged with matched-width
replicates before being interpreted as a slide-composition breakpoint.

As a matched-width debug, the v2 second-target suffix was trained from its
predecessor target state with `source_macro_index=1`, stage width 16, seed 42,
and the same 700-iteration recipe. Its independent 256-episode evaluation
completed **256/256**, with strict prefix `[1.0]` and all 768 gate candidates
accepted. The target `(0.08, 0.88)` is therefore learnable under the fixed
extension input width; the M2 failure is specifically a continuous two-macro
optimization/handoff issue, not a malformed target or an unusable stage-ID
width.

To extend the evidence rather than select a convenient endpoint, a persistent queue `experiments/compositional/run_m10.sh` is running in tmux session `comp_ext_v3`. It waits for an available GPU and then executes three predeclared blocks: (i) homogeneous pick-and-place M11--M15 with a fixed 46-way stage one-hot, 700 iterations per macro, and a new 15-point `tabletop_unique_chain_v10`; (ii) homogeneous `plate-slide-v3` M1--M5 using a fixed 16-way stage one-hot and a central `target_protocol=v2` waypoint chain; and (iii) length-five same-scene heterogeneous sequences with a fixed 16-way stage one-hot and 3500 PPO iterations. All blocks preserve physical continuity and do not reset the MuJoCo scene at macro boundaries. The heterogeneous screen contains **18 sequences**: three homogeneous controls, five single-B substitutions, five single-C substitutions, and five multi-type order probes. This is a representative screen rather than an exhaustive enumeration of all (3^5=243) strings. Every final checkpoint is evaluated for 256 deterministic episodes on seed 1042; raw logs and summaries remain under the bidirectional experiment root. Results are pending and will be appended without replacing earlier rows.

An early M11 diagnostic is available at iteration 100 (32 deterministic
seed-1042 episodes, not a reportable endpoint). It completed the first macro in
23/32 episodes (71.875%) but completed no second macro; strict prefix counts
were `[23,0,0,0,0,0,0,0,0,0,0]`. This is consistent with a policy still
learning the first long-horizon prefix and is not evidence of an M11 boundary.

At iteration 125, the same 32-episode diagnostic improved to 32/32 for the
first macro and 1/32 for the first two macros; strict prefix counts were
`[32,1,0,0,0,0,0,0,0,0,0]`. The improvement in the shared prefix while the
second handoff is only beginning to appear is further evidence that the M11
training is progressing, rather than having already reached a composition
failure boundary.

The next checkpoint, iteration 150, scored 28/32 on the first macro and 0/32
on the first two (`[28,0,0,0,0,0,0,0,0,0,0]`). This small-sample fluctuation is
not monotone and is retained as a diagnostic only; it reinforces that early
checkpoint rates are too noisy to define an M11 breakpoint.

At iteration 200, the first macro was again completed in 32/32 episodes, while
the first two remained 0/32 (`[32,0,0,0,0,0,0,0,0,0,0]`). The repeated perfect
first-prefix result with no full composition is an optimization/handoff
diagnostic during training, not a final M11 result; the run is still far below
its predeclared 7700-iteration budget.

At iteration 225, the diagnostic remained `[32,1,0,0,0,0,0,0,0,0,0]`.
The stable first prefix and rare second-prefix success are useful for tracking
the suffix-learning phase, but this 32-episode screen is still not a final
composition score.

At iteration 250, the same diagnostic improved to
`[32,13,1,0,0,0,0,0,0,0,0]`: 32/32 completed the first macro, 13/32 reached
the second, and 1/32 reached the third. This is the first clear mid-training
expansion of the learned prefix and further confirms that the long M11 run is
still learning rather than exhibiting a settled boundary.

At iteration 300, the prefix expanded further to
`[31,23,0,0,0,0,0,0,0,0,0]`: 31/32 episodes completed the first macro and
23/32 completed the first two. No episode had reached macro 3 in this small
screen, but the second-prefix rate is substantially higher than at iteration
250; this is still an intermediate optimization diagnostic, not a final SR.

At iteration 325, the prefix reached macro 3 in 6/33 and macro 4 in 2/33
episodes, with cumulative counts `[30,29,6,2,0,0,0,0,0,0,0]`. The vectorized
evaluation completed 33 episodes rather than the requested 32; this is another
intermediate checkpoint and not a reportable M11 success rate, but it shows the
learned prefix continuing to expand during training.

At iteration 350, all 32 diagnostic episodes completed the first two macros;
2/32 reached macro 3 and 2/32 reached macro 4, giving cumulative counts
`[32,32,2,2,0,0,0,0,0,0,0]`. This confirms that the first handoff has become
reliably learned at this checkpoint, while later suffixes are still being
optimized.

At iteration 400, the prefix expanded to
`[32,32,16,10,1,0,0,0,0,0,0]`: all episodes completed the first two macros,
16/32 reached the third, 10/32 the fourth, and 1/32 the fifth. This continued
expansion is still an intermediate diagnostic, but it provides strong evidence
that the full-budget M11 run is learning successive handoffs rather than being
stuck at the second macro.

At iteration 450, the diagnostic was `[31,29,25,20,1,0,0,0,0,0,0]`.
Although the first two prefix counts fluctuate slightly, the third and fourth
prefixes continue to strengthen (25/32 and 20/32), while one episode still
reaches the fifth. This nonmonotone small-sample curve is retained only as a
learning trace, not as a final success-rate estimate.

At iteration 500, the same 32-episode CPU diagnostic reached
`[32,31,31,26,2,0,0,0,0,0,0]` (prefix rates
`[1.00,0.96875,0.96875,0.8125,0.0625,0,0,0,0,0,0]`). The first four
macros therefore continue to become reachable while the later suffix is still
being learned. This is an intermediate checkpoint only: M11 remains at
498/7700 training iterations in the live run, and no final M11 SR or
breakpoint claim is made from this diagnostic.

At iteration 550, the diagnostic prefix was
`[32,32,31,24,5,3,1,1,0,0,0]`. Relative to iteration 500, the learned
prefix now reaches macro 8 in one of 32 episodes. This further supports the
interpretation that the scratch policy is progressively acquiring later
handoffs; it is not a final M11 evaluation and remains excluded from the
reportable SR table.

At the first macro-budget checkpoint, `model_700.pt`, the 32-episode probe
reached prefix counts `[32,31,30,29,25,16,10,5,2,1,1]`. One episode completed
all eleven macros (`mandatory_success_rate=1/32`), while 16/32 reached macro
6 and 5/32 reached macro 8. This is still only an intermediate diagnostic,
but it shows that the full M11 chain is beginning to execute end-to-end before
the remaining ten macro budgets are trained.

At `model_725.pt`, the same probe reached prefix counts
`[32,32,32,28,28,19,15,10,3,2,2]`. The first three macros were completed in
all 32 episodes, and 2/32 episodes completed the full M11 chain. The deeper
prefix therefore continued to improve after iteration 700; this remains an
intermediate learning trace rather than the final M11 SR.

At `model_750.pt`, the probe reached prefix counts
`[30,30,29,28,27,25,23,18,6,6,5]`. Five of 32 episodes reached the full
eleven-macro prefix and four satisfied the mandatory full-stage metric. The
small first-prefix fluctuation is sampling noise, while the later-prefix and
full-chain rates continue to improve; this is still an intermediate probe.

At `model_775.pt`, prefix counts were
`[32,32,32,31,30,27,22,15,5,2,2]` (2/32 full mandatory episodes, 1/32
under the strict mandatory counter). The full-chain count fluctuates relative
to model 750 because this is only a 32-episode probe, while the learned prefix
still reaches macro 8 in 15/32 episodes. This nonmonotonicity is retained as
an optimization trace and is not interpreted as a length breakpoint.

At `model_800.pt`, prefix counts were
`[31,31,29,28,27,25,22,21,9,5,3]`; the learned prefix reached macro 8 in
21/32 episodes, with 3/32 raw and 2/32 mandatory full-chain completions. The
continued deep-prefix coverage is consistent with ongoing suffix learning,
despite small-sample fluctuations in the first macros.

At `model_825.pt`, prefix counts were
`[32,32,32,31,30,28,25,22,8,3,1]`. Macro 8 remained reachable in 22/32
episodes, while the full-chain count fluctuated to 1/32 (also 1/32 mandatory).
This is consistent with finite-sample variance around an otherwise stable,
deep learned prefix, not a detected M11 breakpoint.

At `model_850.pt`, prefix counts were
`[30,29,28,28,27,26,24,17,11,7,7]`. Seven of 32 episodes completed the
full M11 composition and six satisfied the mandatory metric; the final macro
itself was reached in 7/32 episodes. This is a substantial later-suffix gain
over the preceding probes and further confirms that the M11 run is still
learning rather than exhibiting a settled boundary.

To debug a possible low M11 endpoint before interpreting it as a composition
effect, an isolated canonical suffix control is also running on CPU:
`tabletop_unique_chain_v10`, `source_macro_index=10` (the eleventh target),
one macro, fixed stage width 46, seed 42, and 700 iterations. It starts from
the predecessor-target initialization but is not a no-reset handoff result;
its only purpose is to test whether the newly appended target is locally
learnable. Its held-out 256-episode evaluation will be kept separate from the
main M11--M15 rows.

The suffix control reached its fixed endpoint (`model_699.pt`) and was then
evaluated independently for 256 seed-1042 episodes. It obtained **256/256
native raw success**, but only one gated transition per episode: the stage
entry counts were `[320,256,0,0,...]`, `gate_requests=256`, and the strict
three-atomic-stage mandatory rate was **0/256**. Thus the eleventh target is
physically/native-reward reachable, but this canonical suffix actor did not
learn the full pregrasp--capture--place reward protocol; raw success can be
achieved without the later stage predicates. This is a useful debug result,
not evidence that the suffix stagewise task is solved and not a main-table
composition score. A low M11 mandatory SR must therefore first be checked
against stage-transition/pick-and-place reward learning rather than being
attributed to target geometry alone.

At `model_875.pt`, the same 32-episode diagnostic reached prefix counts
`[30,30,30,29,29,28,25,24,12,7,4]`. Four episodes completed the full M11
chain and satisfied the strict mandatory metric (`raw=4/32`,
`mandatory=4/32`). The first eight macros remain mostly reliable, while the
last three are still the active optimization bottleneck; this is a later
checkpoint trace, not a final M11 estimate or a detected length breakpoint.

At `model_900.pt`, the prefix counts were
`[32,31,31,31,30,30,28,22,11,7,6]`; raw and mandatory full-chain completion
were 6/32 and 5/32, respectively. The first six macros are now at or above
30/32, while the final suffix continues to improve gradually. This remains a
small diagnostic sample during the live 7700-iteration run, not a reportable
endpoint.

Protocol-count correction: the queued heterogeneous screen contains **18
sequences**, not 16 (three homogeneous controls, five single-B substitutions,
five single-C substitutions, and five multi-type order probes). It remains a
representative screen of the 243 possible length-five strings rather than an
exhaustive enumeration.

Before the formal heterogeneous queue starts, a CPU smoke run of `ABCAB` with
8 environments and 100 iterations completed without protocol errors: the
fixed 16-way observation, `-1` non-A gripper command, no-reset scene, and
stage-machine transitions all executed as intended. As expected for only 100
iterations, it learned the first gated transition (85 accepted requests) but
had `0/128` full episodes; this is an implementation smoke result, not a
heterogeneous score.

At `model_950.pt`, the 32-episode prefix counts were
`[32,32,32,31,31,30,30,26,14,13,12]`; 12/32 raw and 11/32 strict mandatory
episodes completed all eleven macros. This is statistically consistent with
the preceding model-925 probe and still shows a substantial unresolved
late-suffix bottleneck, so the formal run continues toward its full budget.

At `model_925.pt`, the M11 diagnostic prefix was
`[32,32,32,32,32,31,31,29,19,14,12]`; 12/32 episodes completed the full
chain under both the raw and strict mandatory counters. The first five macros
were reached in all 32 episodes, and the final suffix rose from 6/32 at
iteration 900 to 12/32 here. The live run is therefore still improving and
has not exposed a stable M11 boundary.

At `model_975.pt`, the diagnostic prefix was
`[32,32,32,32,32,32,32,26,16,13,12]`; the strict mandatory full-chain rate
was `11/32`, with all first seven macros reached in every episode. The
late-suffix rate is temporarily flat relative to model 950, but the run is
still only about 13% of the planned M11 budget, so this is not a breakpoint
estimate.

At `model_1000.pt`, the prefix counts were
`[32,32,32,32,32,32,32,28,22,18,18]`; raw completion was `18/32` and strict
mandatory completion `17/32` (`53.125%`). The first seven macros were reached
in every diagnostic episode, and the final macro improved from 12/32 at
model 975 to 18/32. This confirms that the M11 suffix is still learning, not
that a length boundary has appeared.

At `model_1025.pt`, prefix counts were
`[32,32,32,32,32,32,32,29,17,11,11]`; raw completion was `11/32` and strict
mandatory completion `9/32`. The first seven macros stayed perfect while the
small suffix sample fluctuated, so this remains an optimization trace rather
than evidence of a breakpoint.

At `model_1050.pt`, prefix counts were
`[32,32,32,32,32,32,32,27,21,20,19]`; raw completion was `19/32` and strict
mandatory completion `17/32` (`53.125%`). The seventh macro remained perfect,
while the final four suffixes continued to gain relative to model 1025. M11
is still learning under the long budget and remains unsuitable for a boundary
claim at this stage.

At `model_1150.pt`, the prefix counts were
`[32,32,32,32,32,32,32,30,25,24,24]`; raw completion was `24/32` and strict
mandatory completion `23/32` (`71.875%`). The final macro was reached in
26/32 episodes. Although this is below model 1125 on the small diagnostic
sample, the late suffix remains highly learnable and does not indicate a
stable M11 boundary.

At `model_1175.pt`, the prefix counts were
`[32,32,32,32,32,32,32,31,30,30,30]`; raw completion was `30/32` and strict
mandatory completion `26/32` (`81.25%`). The final macro was reached in
32/32 episodes, while the full strict chain reached 30/32 under the raw
prefix counter. This is a strong late-suffix recovery and further rules out
calling the current M11 training state a composition breakpoint.

At `model_1100.pt`, the prefix counts were
`[32,31,31,31,30,30,30,26,22,21,21]`; raw completion was `21/32` and strict
mandatory completion `19/32` (`59.375%`). The final macro was reached in
21/32 episodes, up from 16/32 at model 1075. The suffix continues to improve,
so M11 is still not a breakpoint candidate.

At `model_1075.pt`, the 32-episode prefix was
`[32,32,32,32,32,31,31,29,22,20,16]`; raw completion was `16/32` and strict
mandatory completion `15/32`. This fluctuates below model 1050 but retains a
50% full-prefix rate and the same strong first-seven-macro behavior. The
variation is treated as finite-sample/optimization noise during training, not
as a breakpoint.

At `model_1125.pt`, the prefix counts were
`[32,32,32,32,32,32,32,31,27,26,26]`; raw completion was `26/32` and strict
mandatory completion `25/32` (`78.125%`). The final macro was reached in
30/32 episodes. This large improvement over model 1100 confirms that M11 is
still optimizing successfully; no length boundary is visible at this stage.

At `model_1200.pt`, the prefix counts were
`[32,32,32,32,32,32,32,27,25,23,23]`; raw and strict mandatory completion
were both `23/32` (`71.875%`). The final macro itself was reached in 31/32
episodes. Relative to model 1175 this is a small-sample fluctuation, while
the first seven macros remain fully learned; no stable M11 boundary is
indicated.

At `model_1250.pt`, all 32 diagnostic episodes reached every macro
(`prefix=[32,32,32,32,32,32,32,32,32,32,32]`). Raw completion was `32/32`
and strict mandatory completion `31/32` (`96.875%`). This is the first
near-solved M11 checkpoint; it strongly confirms that the earlier low suffix
rates were undertraining rather than an intrinsic composition boundary.

At `model_1225.pt`, the prefix counts were
`[32,32,32,32,32,32,32,32,30,30,30]`; raw completion was `30/32` and strict
mandatory completion `27/32` (`84.375%`). The final macro was reached in
32/32 episodes. M11 is now close to a solved diagnostic endpoint, so any
future decrease must be evaluated only after the full planned training and
256-episode held-out evaluation; no breakpoint is present here.

An intermediate 256-episode held-out evaluation of `model_1250.pt` confirms
that the small diagnostic was not an artifact: raw full-chain success was
`251/256 = 98.05%`, strict mandatory success was `245/256 = 95.70%`, and the
prefix counts were `[256,256,256,256,256,256,256,256,251,251,251]`. This is
still an intermediate checkpoint rather than the final M11 result, but it
establishes that the long suffix is genuinely learnable.

**Live extension update (2026-10-02 23:18 UTC):** The corrected PnP M11
scratch run remains healthy at `2268/7700` iterations. Its latest training
rollout rates are raw `0.5073` and mandatory `0.4970`; no held-out SR or
breakpoint is assigned before the final checkpoint and evaluation. The M11
rescue queue is still waiting for `training_summary.json` and will then run
the predeclared M12--M15 rows, their held-out evaluations, and the
heterogeneous screen. In parallel, the missing plate-slide-v2 M1 scratch row
has now been launched in its own directory on GPU2 with the same `700M`
budget scaling and no-reset protocol; this does not overwrite the earlier
plate-slide diagnostics and will be followed by M2--M5 if the lane remains
healthy. The persistent Qwen worker on GPU2 is left untouched.

**Queue correctness audit (2026-10-02 23:21 UTC):** Before the first new
held-out evaluation, the rescue and parallel queue scripts were corrected so
they no longer pre-create the evaluator's output directory. The evaluators
create that directory exclusively (`exist_ok=False`); removing the redundant
`mkdir -p` prevents a completed training row from losing its EVAL to an
already-existing empty directory. The waiting rescue shell was restarted
after this script fix; active training processes were not interrupted.

The predeclared heterogeneous set currently has 23 length-five sequences. A
machine audit confirms that every position contains all three primitive
labels (`A/B/C`), and that all nine ordered adjacent pairs
`{AA,AB,AC,BA,BB,BC,CA,CB,CC}` occur at least once. Seven sequences contain
all three primitive types (`ABCAB`, `ABCBA`, `ACBAC`, `BCABC`, `CABCA`,
`BACBA`, `CBACB`). Thus the screen is not exhaustive over `3^5=243` strings,
but it covers positional, within-type, order, and mixed-type effects with a
fixed predeclared set.

**Live poll (2026-10-02 23:25 UTC):** M11 has reached checkpoint
`model_2300.pt` while continuing toward `7700` iterations; the parallel
plate-slide M1 lane has reached `model_25.pt`. Both processes are healthy and
GPU2 remains below its memory limit despite the pre-existing VLM worker.

**Plate-slide implementation probe (2026-10-02 23:27 UTC):** A development
probe of plate-slide M1 at `model_50.pt` on 64 episodes reached raw and
mandatory success `64/64`. Its strict prefix was `[64]`, and stage entries
were `64` for each of the three atomic stages (with no entries beyond the
single macro). This rules out an immediate target-coordinate or stage-gate
deadlock in the new macro wrapper; the 700-iteration endpoint remains the
formal result.

**Live poll (2026-10-02 23:28 UTC):** PnP M11 is at `2328/7700` with
mandatory training rollout `0.5081`; plate-slide M1 is at `68/700` with raw
training rollout `0.7608`. Neither row has reached its formal endpoint or
held-out evaluation yet, and the rescue queues remain live.

**Plate-slide debug guard:** Prior v2 plate-slide runs show that a low M2/M3
number can be optimization-sensitive rather than a handoff failure: the
same continuous M2 target profile had seed-42 endpoints near zero under
`entropy_coef=0.003`, while a pre-existing matched run with the same scene,
targets, budget, and no-reset contract but `init_noise_std=0.60` and
`entropy_coef=0.02` reached about `98%` training success. If the new formal
M2--M5 rows are low, the first follow-up is therefore a matched exploration
diagnostic (with prefix/stage-entry reporting), not a breakpoint claim.

**Live poll (2026-10-02 23:31 UTC):** M11 has advanced to `2348/7700`
(mandatory rollout `0.5113`), and plate-slide M1 to `88/700` (raw rollout
`0.8402`). Both jobs remain healthy; no formal SR has changed yet.

## Open-ended continuation protocol (2026-10-02)

The extension is intentionally kept open beyond the original M=10 report.  The
first lane continues the same homogeneous pick-and-place construction through
M=11,12,13,14,15.  Each row is a fresh scratch policy, uses the same native
39-D observation, the fixed maximum stage one-hot of width 46, the same rule
gate and no physical reset at macro boundaries, and receives `700*M` PPO
iterations (`K=3M` atomic stages).  The target protocol is
`tabletop_unique_chain_v10`, so the appended destinations are distinct rather
than a repeated back-and-forth pair.  Every endpoint is evaluated on 256
held-out seed-1042 episodes with both native and strict mandatory metrics.

For a second homogeneous macro primitive, the continuation uses
MetaWorld `plate-slide-v3`.  It is a suitable no-reset analogue because one
puck remains in the scene while the controller successively changes its goal
to distinct waypoints; `_advance_macro()` changes only the target and reward
stage state, never calling `env.reset()`.  The frozen primitive has three
atomic stages (align/pre-push, establish contact, push to target), giving
M=1,...,5, `K=3M`, stage-one-hot width 16, target protocol `v2`, and the same
`700*M` budget scaling.  A 64-episode model-50 probe already reached 64/64
raw and mandatory success with all three stage entries, so a low endpoint will
first be diagnosed through prefix/stage-entry counts rather than labelled a
composition breakpoint.

The heterogeneous lane uses fixed length-five sequences from `{A,B,C}` rather
than enumerating all 243 strings.  The 23 predeclared rows include homogeneous
controls, each positional single-substitution family, block/reverse and
alternating orders, and seven fully mixed cyclic/order probes.  Collectively
they cover all five positions and all nine ordered adjacent pairs, while every
row keeps the same 15-atomic-stage budget, fixed width-16 stage context, no
macro-boundary reset, and 3,500 PPO iterations.  A low heterogeneous SR is
accepted as a composition result only after checking native-vs-mandatory SR,
prefix counts, and per-stage entries for an implementation or undertraining
failure.

The corrected rescue queue waits for an existing plate-slide directory's
authoritative `training_summary.json` instead of skipping it, preventing a
race between the GPU-2 parallel lane and the GPU-3 continuation lane.

**Live poll (2026-10-02 23:44 UTC):** The scratch PnP M11 run is healthy at
`2438/7700` iterations (latest training-rollout raw `0.5370`, mandatory
`0.5268`). The formal plate-slide M1 run is at `168/700` iterations (latest
training-rollout raw `0.9475`, mandatory `0.9458`). These are optimization
telemetry only; neither row has a final held-out SR yet. The corrected rescue
queue is live and waiting for M11 completion.

**Fixed-width audit (2026-10-02 23:47 UTC):** The historical M9/M10 rows use
stage-ID width 31, whereas the M11--M15 extension uses the predeclared
`K_max=46` width. Their hidden layers and optimizer are identical, but the
input dimensions are not, so the historical M10 score is not by itself a
strict matched control for the new extension. A persistent post-M15 queue
`benchmark/Metaworld/queue_fixed_width_controls_v1.sh` is now waiting to
train fresh M9 and M10 scratch controls with width 46, the v10 unique-target
chain, no reset, and the same `700M` budget. These controls are kept separate
from the earlier width-31 results and will be used for any width-matched
length comparison.

The v10 geometry audit finds exactly 15 unique `(x,y)` destinations; the
minimum pairwise separation is 2 cm and all points remain in the central
tabletop support region. Thus M11--M15 do not silently recycle a target,
although the chain is deliberately compact for reachability.

The plate-slide v2 waypoint audit likewise finds 11 unique destinations with
2 cm minimum separation; the formal M1--M5 rows therefore use distinct
targets throughout the tested five-macro range.

**Queue handoff fix (2026-10-02 23:57 UTC):** The rescue wrapper's
plate-slide handoff now calls its dedicated evaluator after waiting for the
parallel M1--M5 owner to write `training_summary.json`.  The wrapper was
restarted without touching the active M11 or plate-slide training processes;
this removes a shell-function failure that would otherwise have occurred only
at the first parallel handoff.

**Live poll (2026-10-02 23:58 UTC):** M11 is at approximately `2530/7700`
iterations; its latest training-rollout rates are raw `0.5509` and mandatory
`0.5409`.  Plate-slide M1 is at approximately `239/700`, with raw `0.9691`
and mandatory `0.9681`.  Both jobs remain CPU/GPU-active.  These are rollout
telemetry, not held-out success rates; the formal evaluations are still
pending and the continuation queues remain live.

**Live poll (2026-10-03 00:00 UTC):** M11 has advanced to `2549/7700`
iterations (rollout raw `0.5535`, mandatory `0.5436`), and plate-slide M1 to
`259/700` (raw `0.9723`, mandatory `0.9714`).  No authoritative training
summary has appeared yet; both jobs are still making progress.

**Live poll (2026-10-03 00:04 UTC):** New checkpoints confirm continued
progress: PnP M11 has reached `model_2575.pt` and plate-slide M1 has reached
`model_275.pt`.  Both training processes remain active; the rescue queues are
still waiting for their authoritative summaries before launching the next
rows.

**Live poll (2026-10-03 00:07 UTC):** PnP M11 has advanced to
`model_2600.pt`; plate-slide M1 remains active between checkpoints after
`model_275.pt`.  Neither formal endpoint has finished yet.

**Live poll (2026-10-03 00:09 UTC):** Plate-slide M1 has now reached
`model_300.pt`; PnP M11 remains active after `model_2600.pt`.  Both jobs are
still progressing toward their full budgets, with no formal endpoint summary
yet.

**Live poll (2026-10-03 00:11 UTC):** PnP M11 has reached `model_2625.pt`.
Plate-slide M1 remains active after `model_300.pt`; neither endpoint has yet
produced its authoritative training summary.

**Live poll (2026-10-03 00:13 UTC):** Plate-slide M1 has advanced to
`model_325.pt`; PnP M11 remains active after `model_2625.pt`.  Both endpoint
summaries are still pending.

**Live poll (2026-10-03 00:14 UTC):** PnP M11 has reached `model_2650.pt`.
Plate-slide M1 remains active after `model_325.pt`; neither endpoint has
written a formal training summary.

**Live poll (2026-10-03 00:18 UTC):** Both active jobs crossed another
checkpoint: PnP M11 is at `model_2675.pt` and plate-slide M1 at
`model_350.pt`.  Neither endpoint summary has appeared yet.

**Live poll (2026-10-03 00:21 UTC):** PnP M11 has advanced to
`model_2700.pt`; plate-slide M1 continues after `model_350.pt`.  Both formal
training summaries remain pending.

**Live poll (2026-10-03 00:24 UTC):** Plate-slide M1 has reached
`model_375.pt`; PnP M11 remains active after `model_2700.pt`.  The endpoint
summaries are still pending.

**Live poll (2026-10-03 00:25 UTC):** PnP M11 has reached `model_2725.pt`.
Plate-slide M1 remains active after `model_375.pt`; neither formal endpoint
summary has appeared.

**Live poll (2026-10-03 00:28 UTC):** Both runs crossed another checkpoint:
PnP M11 is at `model_2750.pt`, and plate-slide M1 is at `model_400.pt`.
Neither formal endpoint summary has appeared yet.

**Live poll (2026-10-03 00:31 UTC):** PnP M11 has advanced to
`model_2775.pt`; plate-slide M1 remains active after `model_400.pt`.  Formal
endpoint summaries are still pending.

**Live poll (2026-10-03 00:32 UTC):** Plate-slide M1 has reached
`model_425.pt`; PnP M11 remains active after `model_2775.pt`.  No endpoint
summary has appeared yet.

**Live poll (2026-10-03 00:35 UTC):** PnP M11 has reached `model_2800.pt`.
Plate-slide M1 remains active after `model_425.pt`; neither endpoint has
produced a formal summary.

**Live poll (2026-10-03 00:38 UTC):** Both active jobs crossed another
checkpoint: PnP M11 is at `model_2825.pt`, and plate-slide M1 is at
`model_450.pt`.  The formal endpoint summaries are still pending.

**Live poll (2026-10-03 00:42 UTC):** PnP M11 has reached `model_2850.pt`;
plate-slide M1 remains active after `model_450.pt`.  No formal endpoint
summary has appeared yet.

**Live poll (2026-10-03 00:43 UTC):** Plate-slide M1 has advanced to
`model_475.pt`; PnP M11 remains active after `model_2850.pt`.  Both endpoint
summaries are still pending.

**Live poll (2026-10-03 00:45 UTC):** PnP M11 has reached `model_2875.pt`.
Plate-slide M1 remains active after `model_475.pt`; formal endpoint summaries
are still pending.

**Live poll (2026-10-03 00:47 UTC):** Plate-slide M1 has reached
`model_500.pt`; PnP M11 remains active after `model_2875.pt`.  Both endpoint
summaries are still pending.

**Live poll (2026-10-03 00:49 UTC):** PnP M11 has reached `model_2900.pt`;
plate-slide M1 remains active after `model_500.pt`.  Formal endpoint summaries
are still pending.

**Live poll (2026-10-03 00:52 UTC):** PnP M11 has reached `model_2925.pt`;
plate-slide M1 remains active after `model_500.pt`.  Neither formal endpoint
summary has appeared.

**Live poll (2026-10-03 00:53 UTC):** Plate-slide M1 has reached
`model_525.pt`; PnP M11 remains active after `model_2925.pt`.  Formal endpoint
summaries are still pending.

**Live poll (2026-10-03 00:56 UTC):** PnP M11 has reached `model_2950.pt`;
plate-slide M1 remains active after `model_525.pt`.  Both formal endpoint
summaries are still pending.

**Live poll (2026-10-03 00:57 UTC):** Plate-slide M1 has reached
`model_550.pt`; PnP M11 remains active after `model_2950.pt`.  Formal endpoint
summaries are still pending.

**Live poll (2026-10-03 00:59 UTC):** PnP M11 has reached `model_2975.pt`;
plate-slide M1 remains active after `model_550.pt`.  Neither endpoint has
written its formal summary yet.

**Live poll (2026-10-03 01:03 UTC):** Both active jobs crossed another
checkpoint: PnP M11 is at `model_3000.pt`, and plate-slide M1 is at
`model_575.pt`.  Formal endpoint summaries are still pending.

**Live poll (2026-10-03 01:06 UTC):** PnP M11 has reached `model_3025.pt`;
plate-slide M1 remains active after `model_575.pt`.  Neither endpoint has
produced its formal summary yet.

**Live poll (2026-10-03 01:07 UTC):** Plate-slide M1 has reached
`model_600.pt`; PnP M11 remains active after `model_3025.pt`.  The temporary
sleeping state observed between PnP iterations is normal; CPU utilization and
checkpoint progression remain healthy.  Formal endpoint summaries are still
pending.

**Live poll (2026-10-03 01:10 UTC):** PnP M11 has reached `model_3050.pt`;
plate-slide M1 remains active after `model_600.pt`.  Neither formal endpoint
summary has appeared yet.

**Live poll (2026-10-03 01:13 UTC):** Both active jobs crossed another
checkpoint: PnP M11 is at `model_3075.pt`, and plate-slide M1 is at
`model_625.pt`.  Formal endpoint summaries are still pending.

**Live poll (2026-10-03 01:17 UTC):** PnP M11 has reached `model_3100.pt`;
plate-slide M1 remains active after `model_625.pt`.  Formal endpoint summaries
are still pending.

**Live poll (2026-10-03 01:18 UTC):** Plate-slide M1 has reached
`model_650.pt`; PnP M11 remains active after `model_3100.pt`.  Formal endpoint
summaries are still pending.

**Live poll (2026-10-03 01:20 UTC):** PnP M11 has reached `model_3125.pt`;
plate-slide M1 remains active after `model_650.pt`.  Formal endpoint summaries
are still pending.

**Live poll (2026-10-03 01:23 UTC):** Plate-slide M1 has reached
`model_675.pt`; PnP M11 remains active after `model_3125.pt`.  Formal endpoint
summaries are still pending.

**Live poll (2026-10-03 01:24 UTC):** PnP M11 has reached `model_3150.pt`;
plate-slide M1 remains active after `model_675.pt`.  Formal endpoint summaries
are still pending.

**Live poll (2026-10-03 01:27 UTC):** PnP M11 has reached `model_3175.pt`;
plate-slide M1 remains active after `model_675.pt`.  Formal endpoint summaries
are still pending.

**Verified transition (2026-10-03 01:27--01:32 UTC):** The formal plate-slide
M1 scratch run completed its full 700 iterations and wrote `model_699.pt`.
Its training summary reports 125,180 completed episodes and a mandatory gated
success rate of 0.99154 (raw 0.99180), with healthy stage-entry counts and no
deadlock.  The first held-out evaluator invocation failed only because an
empty destination directory had been left by the queue (`FileExistsError`),
not because of the policy or environment.  The evaluator now reuses empty
destinations while refusing to overwrite an existing summary, and the
authoritative 256-episode M1 evaluation has been relaunched.  Plate-slide M2
has started independently at the full 1,400-iteration budget; at iteration
18/1,400 its zero endpoint success is an early-training observation and is not
interpreted as a breakpoint.  PnP M11 remains healthy at `model_3175.pt` and
continues toward 7,700 iterations.

**Queue-integrity correction (2026-10-03 01:35 UTC):** The original v3 shell
was already resident when the GPU-worker exemption was corrected, so it could
not load the new `wait_for_gpu` function.  A single v6 continuation owner was
restarted to wait for M11, retire only the stale wrapper shells after M11
finishes, and then own M12--M15 plus the remaining plate-slide and
heterogeneous rows.  The active M11 trainer and the independent plate-slide
M2 trainer were not touched.  The exemption is restricted to the known
`run_qwen_stage_live_worker.py --single-gpu-device 0` command; no training PID
is exempted.

**Held-out verification (2026-10-03 01:37 UTC):** The repaired formal
plate-slide M1 evaluator completed all 256 requested episodes.  It reports
raw SR = 1.000 and mandatory gated SR = 1.000, with strict prefix rate 1.000,
stage entries `[320, 256, 256, 256]`, and zero macro failures.  Thus the
plate-slide replacement has a verified single-macro baseline before M2 is
interpreted.  M2 is currently at 40/1,400 iterations (about 3% of its
budget), so its current endpoint statistics remain non-diagnostic.

**Live poll (2026-10-03 01:40 UTC):** PnP M11 is at `3268/7700`
iterations (latest training-rollout mandatory SR `0.6285`) and remains
healthy.  Plate-slide M2 is at `58/1400` iterations (latest rollout
mandatory SR `0.0164`); this is only 4.1% of its allocated budget and is
explicitly retained as an early optimization diagnostic, not an endpoint or
breakpoint result.  The corrected v6 owner is waiting for M11 completion.

**Live poll (2026-10-03 01:46 UTC):** A short one-minute gap after plate-slide
M2 iteration 88 was checked: the process remained running at 100% CPU and
resumed through iteration 98, so this was normal rollout/output variability,
not a hang.  Its training-rollout mandatory SR has risen to `0.0925` and raw
SR to `0.5860`; M11 remains active.

**Live poll (2026-10-03 01:49 UTC):** After a longer interval, both jobs have
continued advancing.  PnP M11 reached `3328/7700` (rollout mandatory SR
`0.6332`), while plate-slide M2 reached `108/1400` with raw SR `0.6357` and
mandatory SR `0.1766`.  The rising M2 prefix performance reinforces that its
initial low value was an optimization transient, not evidence of a broken
no-reset transition.

**Coverage audit (2026-10-03 01:41 UTC):** The fixed heterogeneous manifest
contains exactly 23 length-five sequences.  Every position contains A, B, and
C across the manifest, all nine ordered adjacent pairs `{AA, AB, AC, BA,
BB, BC, CA, CB, CC}` occur, and seven rows are fully mixed (each containing
all three module types).  This confirms the representative screen is not
biased to one position or one transition direction while avoiding the full
243-row Cartesian enumeration.

**Live poll (2026-10-03 01:44 UTC):** PnP M11 has advanced to
`3298/7700` iterations with latest training-rollout mandatory SR `0.6307`.
Plate-slide M2 is at `88/1400` with rollout mandatory SR `0.0389`; both
processes are making normal progress and neither has a formal endpoint yet.

**Live poll (2026-10-03 01:42 UTC):** No new formal endpoint has appeared.
PnP M11 and plate-slide M2 remain the only active trainers, while the
continuation and heterogeneous queues are correctly waiting rather than
launching duplicate rows.  Existing historical plate-slide M2 summaries show
why a low endpoint will be debugged first: the same no-reset construction has
both near-zero and approximately 0.98 training SR under different exploration
conditions, with the decisive evidence visible in the stage-2 prefix entry
count.  The formal run is therefore retained at its full budget before any
diagnostic branch is started.

**Live poll (2026-10-03 01:50 UTC):** PnP M11 reached `3338/7700` with
rollout mandatory SR `0.6340`.  Plate-slide M2 reached `118/1400`; its raw
and mandatory rollout SR rose to `0.6992` and `0.2936`, respectively.  Both
active trainers remain healthy and no formal endpoint has appeared.

**Live poll (2026-10-03 01:54 UTC):** PnP M11 reached `3368/7700` with
rollout mandatory SR `0.6365`.  Plate-slide M2 reached `138/1400`; raw SR is
`0.7789` and mandatory SR `0.4719`.  The second macro continues its expected
optimization recovery; no endpoint or breakpoint claim is made yet.

**Live poll (2026-10-03 01:55 UTC):** PnP M11 reached `3378/7700` with
rollout mandatory SR `0.6373`.  Plate-slide M2 reached `148/1400`; raw SR is
`0.8125` and mandatory SR `0.5504`.  The M2 recovery is now clear in the
training trace, while the full-budget endpoint remains pending.

**Live poll (2026-10-03 01:57 UTC):** The apparent M2 pause was checked
directly: the log had advanced to `158/1400`, the process remained `Rl+` at
~100% CPU, and GPU memory was stable.  Its rollout raw SR is `0.8392` and
mandatory SR `0.6140`; this is normal PPO/logging variability, not a stall.
PnP M11 remains active at `3388/7700` with mandatory SR `0.6381`.

**Live poll (2026-10-03 02:00 UTC):** PnP M11 reached `3408/7700` with
rollout mandatory SR `0.6400`.  Plate-slide M2 reached `168/1400`; raw SR is
`0.8598` and mandatory SR `0.6630`.  Both active jobs continue normally and
no formal endpoint has appeared.

**Live poll (2026-10-03 02:01 UTC):** PnP M11 reached `3418/7700` with
rollout mandatory SR `0.6407`.  Plate-slide M2 reached `178/1400`; raw SR is
`0.8761` and mandatory SR `0.7018`.  No formal endpoint has appeared yet.

**Live poll (2026-10-03 02:03 UTC):** PnP M11 reached `3428/7700` with
rollout mandatory SR `0.6414`.  Plate-slide M2 reached `188/1400`; raw SR is
`0.8890` and mandatory SR `0.7321`.  Both runs remain healthy and are still
inside their allocated budgets.

**Live poll (2026-10-03 02:05 UTC):** PnP M11 reached `3448/7700` with
rollout mandatory SR `0.6432`.  Plate-slide M2 reached `198/1400`; raw SR is
`0.8993` and mandatory SR `0.7571`.  The queue is still waiting for the full
M11/M2 endpoints before starting downstream rows.

**Live poll (2026-10-03 02:08 UTC):** A two-minute M2 logging gap was
diagnosed: GPU utilization stayed at 100%, the main process stayed in the
running state, and it subsequently advanced to `208/1400`.  Its rollout raw
SR is `0.9076` and mandatory SR `0.7774`; the gap was slow computation, not a
hang.

**Live poll (2026-10-03 02:10 UTC):** PnP M11 reached `3478/7700` with
rollout mandatory SR `0.6457`.  Plate-slide M2 reached `218/1400`; raw SR is
`0.9148` and mandatory SR `0.7945`.  The active runs remain healthy; no
formal endpoint or downstream handoff has occurred.

**Live poll (2026-10-03 02:11 UTC):** PnP M11 reached `3488/7700` with
rollout mandatory SR `0.6465`.  Plate-slide M2 reached `228/1400`; raw SR is
`0.9210` and mandatory SR `0.8096`.  Both remain inside their full budgets.

**Live poll (2026-10-03 02:12 UTC):** PnP M11 reached `3498/7700` with
rollout mandatory SR `0.6472`.  Plate-slide M2 reached `238/1400`; raw SR is
`0.9262` and mandatory SR `0.8221`.  No formal endpoint has appeared.

**Live poll (2026-10-03 02:15 UTC):** PnP M11 reached `3518/7700` with
rollout mandatory SR `0.6486`.  Plate-slide M2 reached `248/1400`; raw SR is
`0.9309` and mandatory SR `0.8336`.  Both remain healthy and within budget.

**Live poll (2026-10-03 02:16 UTC):** PnP M11 reached `3528/7700` with
rollout mandatory SR `0.6495`.  Plate-slide M2 reached `258/1400`; raw SR is
`0.9349` and mandatory SR `0.8434`.  No formal endpoint has appeared yet.

**Live poll (2026-10-03 01:52 UTC):** PnP M11 reached `3358/7700` with
rollout mandatory SR `0.6356`.  Plate-slide M2 reached `128/1400`; raw SR is
`0.7390` and mandatory SR `0.3790`, continuing the recovery of the second
macro without any code or queue intervention.

**Live poll (2026-10-03 02:20 UTC):** PnP M11 reached `3559/7700` with
rollout mandatory SR `0.6515`; plate-slide M2 reached `279/1400` with raw SR
`0.9419` and mandatory SR `0.8600`.
These are training-rollout diagnostics, not held-out endpoints.  The only
formal result currently available for the alternate macro is plate-slide M1;
M2 and the PnP M11--M15 continuation remain in progress.

**Live poll (2026-10-03 02:22 UTC):** PnP M11 advanced to `3569/7700` with
rollout mandatory SR `0.6522`; plate-slide M2 advanced to `289/1400` with
rollout raw SR `0.9447` and mandatory SR `0.8671`.  Both trainers remain in
the running state.  The rescue queue is still waiting for M11's formal
training summary, so no downstream job has started and no breakpoint is
claimed.

**Live poll (2026-10-03 02:23 UTC):** PnP M11 reached `3579/7700` with
rollout mandatory SR `0.6528`; plate-slide M2 had not emitted a newer log
checkpoint yet and remains active at `289/1400` (latest rollout mandatory SR
`0.8671`).  No formal endpoint or stage-entry debug trigger is available.

**Live poll (2026-10-03 02:25 UTC):** PnP M11 advanced to `3589/7700` with
rollout mandatory SR `0.6534`.  Plate-slide M2 remains alive at its latest
logged `289/1400` checkpoint (`0.8733` mandatory rollout); its process is
running and has not produced a formal summary yet.  The rescue queue still
waits for M11.

**Live poll (2026-10-03 02:27 UTC):** PnP M11 advanced to `3599/7700` with
rollout mandatory SR `0.6541`.  Plate-slide M2 advanced to `309/1400`; raw
rollout SR is `0.9496` and mandatory rollout SR is `0.8788`.  Both processes
remain active; no formal summary or downstream handoff is available yet.

**Live poll (2026-10-03 02:27 UTC, second check):** PnP M11 is at
`3609/7700` with rollout mandatory SR `0.6548`; plate-slide M2 remains at
`309/1400` in the latest emitted checkpoint with mandatory rollout SR
`0.8788`.  Both trainer processes are still live, and the rescue queue still
waits for M11's formal summary.

**Live poll (2026-10-03 02:28 UTC):** Direct directory inspection found no
formal `training_summary.json` for either active row.  PnP M11 is still
running at `3609/7700`; plate-slide M2 has advanced to `319/1400` with
rollout mandatory SR `0.8834` (latest saved checkpoint `model_300.pt`).  The
absence of a summary is therefore expected in-progress state, not a failed
job.

**Live poll (2026-10-03 02:29 UTC):** PnP M11 advanced to `3619/7700` with
rollout mandatory SR `0.6556`; plate-slide M2 advanced to `329/1400` with raw
rollout SR `0.9535` and mandatory rollout SR `0.8882`.  Both jobs remain
healthy and the rescue queue continues to wait for M11's formal summary.

**Live poll (2026-10-03 02:30 UTC):** PnP M11 reached `3629/7700` with
rollout mandatory SR `0.6563`; plate-slide M2 remains at its latest emitted
`329/1400` checkpoint (`0.8882` mandatory rollout).  Both target directories
still lack a formal training summary, so downstream execution remains
correctly deferred.

**Checkpoint poll (2026-10-03 02:31 UTC):** Direct file timestamps confirm
both trainers are writing checkpoints: PnP M11 has `model_3625.pt`, and
plate-slide M2 has `model_325.pt`.  The logs are only a few minutes behind
the live processes; neither row has reached its formal endpoint.

**Live poll (2026-10-03 02:32 UTC):** PnP M11 reached `3639/7700` with
rollout mandatory SR `0.6571`; plate-slide M2 reached `339/1400` with raw
rollout SR `0.9553` and mandatory rollout SR `0.8926`.  Both trainers remain
healthy, and M11's downstream queue is still waiting for its final summary.

**Live poll (2026-10-03 02:33 UTC):** PnP M11 advanced to `3649/7700` with
rollout mandatory SR `0.6578`; plate-slide M2's latest emitted checkpoint is
`339/1400` with mandatory rollout SR `0.8926`.  No formal summary has appeared
and the downstream queue remains correctly deferred.

**Live poll (2026-10-03 02:34 UTC):** PnP M11 has not emitted a newer
checkpoint yet and remains at `3649/7700`; plate-slide M2 advanced to
`349/1400`, with raw rollout SR `0.9569` and mandatory rollout SR `0.8965`.
Both target summaries are still pending and both processes remain live.

**Live poll (2026-10-03 02:35 UTC):** PnP M11 advanced to `3659/7700` with
rollout mandatory SR `0.6585`.  Plate-slide M2 has not emitted a newer
checkpoint than `349/1400` (`0.8965` mandatory rollout), but its trainer
process remains live; no formal summary or downstream handoff is available.

**Live poll (2026-10-03 02:36 UTC):** PnP M11 advanced to `3669/7700` with
rollout mandatory SR `0.6593`; plate-slide M2 advanced to `359/1400` with raw
rollout SR `0.9585` and mandatory rollout SR `0.9003`.  Both are still
training from scratch; formal summaries remain pending.

**Live poll (2026-10-03 02:37 UTC):** PnP M11 reached `3679/7700` with
rollout mandatory SR `0.6599`; plate-slide M2 reached `369/1400` with raw
rollout SR `0.9598` and mandatory rollout SR `0.9037`.  Both processes remain
healthy and the M11-dependent queue is still waiting.

**Live poll (2026-10-03 02:39 UTC):** PnP M11 advanced to `3689/7700` with
rollout mandatory SR `0.6607`.  Plate-slide M2 has not emitted a newer
checkpoint than `369/1400` (`0.9037` mandatory rollout), while its process
remains live.  No formal endpoint or downstream handoff is available yet.

**Live poll (2026-10-03 02:40 UTC):** PnP M11 reached `3699/7700` with
rollout mandatory SR `0.6615`; plate-slide M2 reached `379/1400` with raw
rollout SR `0.9611` and mandatory rollout SR `0.9070`.  Both are still
training; no final summary has appeared.

**Live poll (2026-10-03 02:42 UTC):** PnP M11 reached `3709/7700` with
rollout mandatory SR `0.6623`; plate-slide M2 reached `389/1400` with raw
rollout SR `0.9623` and mandatory rollout SR `0.9099`.  Both jobs remain
healthy, while the M11-dependent queue is still waiting for the formal
summary.

**Live poll (2026-10-03 02:43 UTC):** PnP M11 advanced to `3719/7700` with
rollout mandatory SR `0.6631`; plate-slide M2 remains at `389/1400` with
mandatory rollout SR `0.9099`.  No formal summary or downstream handoff has
appeared.

**Live poll (2026-10-03 02:43 UTC, second check):** PnP M11's latest emitted
rollout remains `3719/7700` (`0.6631` mandatory), while plate-slide M2 has
advanced to `399/1400` with raw rollout SR `0.9635` and mandatory rollout SR
`0.9127`.  Both trainers are live; formal summaries are still pending.

**Live poll (2026-10-03 02:44 UTC):** PnP M11 advanced to `3729/7700` with
rollout mandatory SR `0.6639`; plate-slide M2 remains at `399/1400` with
mandatory rollout SR `0.9127`.  No formal endpoint or downstream handoff has
appeared.

**Live poll (2026-10-03 02:45 UTC):** PnP M11 reached `3739/7700` with
rollout mandatory SR `0.6646`; plate-slide M2 reached `409/1400` with raw
rollout SR `0.9645` and mandatory rollout SR `0.9151`.  Both processes remain
healthy and the downstream queue is still waiting for M11's formal summary.

**Live poll (2026-10-03 02:47 UTC):** PnP M11 advanced to `3749/7700` with
rollout mandatory SR `0.6653`; plate-slide M2 advanced to `419/1400` with raw
rollout SR `0.9654` and mandatory rollout SR `0.9174`.  No formal endpoint has
appeared, and both trainers remain active.

**Live poll (2026-10-03 02:48 UTC):** PnP M11 reached `3759/7700` with
rollout mandatory SR `0.6660`; plate-slide M2 remains at `419/1400` with
mandatory rollout SR `0.9174`.  Both jobs are live, and the M11-dependent
queue remains in its wait state.

**Live poll (2026-10-03 02:49 UTC):** PnP M11's latest emitted checkpoint
remains `3759/7700` (`0.6660` mandatory rollout); plate-slide M2 advanced to
`429/1400` with raw rollout SR `0.9664` and mandatory rollout SR `0.9197`.
Both processes remain active, with no formal summary yet.

**Live poll (2026-10-03 02:50 UTC):** PnP M11 advanced to `3769/7700` with
rollout mandatory SR `0.6668`; plate-slide M2 remains at `429/1400` with
mandatory rollout SR `0.9197`.  Both trainers remain live and no formal
endpoint has appeared.

**Live poll (2026-10-03 02:51 UTC):** PnP M11 reached `3779/7700` with
rollout mandatory SR `0.6675`; plate-slide M2 reached `439/1400` with raw
rollout SR `0.9673` and mandatory rollout SR `0.9218`.  Both remain healthy;
the M11-dependent queue is still waiting.

**Live poll (2026-10-03 02:53 UTC):** PnP M11 advanced to `3789/7700` with
rollout mandatory SR `0.6682`; plate-slide M2 advanced to `449/1400` with raw
rollout SR `0.9680` and mandatory rollout SR `0.9238`.  Both trainers remain
active; no formal summary or downstream handoff yet.

**Live poll (2026-10-03 02:54 UTC):** PnP M11 reached `3799/7700` with
rollout mandatory SR `0.6689`; plate-slide M2 remains at `449/1400` with
mandatory rollout SR `0.9238`.  Both jobs are still active and no formal
endpoint has appeared.

**Live poll (2026-10-03 02:55 UTC):** PnP M11 reached `3809/7700` with
rollout mandatory SR `0.6697`; plate-slide M2 reached `459/1400` with raw
rollout SR `0.9688` and mandatory rollout SR `0.9255`.  Both remain healthy;
the downstream queue is still waiting for M11's formal summary.

**Live poll (2026-10-03 02:56 UTC):** PnP M11 advanced to `3819/7700` with
rollout mandatory SR `0.6705`; plate-slide M2 remains at `459/1400` with
mandatory rollout SR `0.9255`.  Both trainer processes remain active and no
formal endpoint has appeared.

**Live poll (2026-10-03 02:58 UTC):** PnP M11 reached `3838/7700` with
rollout mandatory SR `0.6719`; plate-slide M2 reached `478/1400` with raw
rollout SR `0.9702` and mandatory rollout SR `0.9291`.  Both processes are
healthy, and the rescue queue remains correctly deferred until M11 writes its
formal training summary.

**Live poll (2026-10-03 03:00 UTC):** PnP M11 reached `3848/7700` with
rollout mandatory SR `0.6727`; plate-slide M2 reached `488/1400` with raw
rollout SR `0.9709` and mandatory rollout SR `0.9308`.  Both trainers remain
live; the downstream rescue queue is still waiting for M11's endpoint.

**Live poll (2026-10-03 03:01 UTC):** PnP M11 reached `3858/7700` with
rollout mandatory SR `0.6735`.  Plate-slide M2's process remains live; its
latest flushed log remains `488/1400` (`0.9308` mandatory).  No formal summary
has appeared, so no downstream job was started.

**Live poll (2026-10-03 03:03 UTC):** PnP M11 reached `3868/7700` with
rollout mandatory SR `0.6741`; plate-slide M2 reached `498/1400` with raw
rollout SR `0.9715` and mandatory rollout SR `0.9323`.  Both trainers remain
healthy and the M11-dependent queue is still waiting for the formal summary.

**Live poll (2026-10-03 03:04 UTC):** PnP M11 reached `3878/7700` with
rollout mandatory SR `0.6748`; plate-slide M2 reached `508/1400` with raw
rollout SR `0.9720` and mandatory rollout SR `0.9337`.  No new training
summary or held-out evaluation has appeared yet.

**Live poll (2026-10-03 15:55 UTC):** PnP M12 reached `1118/8400` with rollout
raw `0.0308`; old plate-slide v2 M4 reached `1938/2800` with rollout raw
`0.0000`; v3 M1 reached `688/700` with rollout raw SR `0.9917`.  M1 is still
finishing its last iterations; no final training or held-out evaluation summary
has appeared yet.

**Live poll (2026-10-03 15:16 UTC):** PnP M12 reached `848/8400` with rollout
raw `0.0019`; old plate-slide v2 M4 reached `1678/2800` with rollout raw
`0.0000`; v3 M1 reached `358/700` with rollout raw SR `0.9820`.  M1 remains
healthy and is still running to the predeclared endpoint.

**Live poll (2026-10-03 15:18 UTC):** PnP M12 reached `868/8400` with rollout
raw `0.0024`; old plate-slide v2 M4 reached `1688/2800` with rollout raw
`0.0000`; v3 M1 reached `378/700` with rollout raw SR `0.9832`.  M1 remains
stable under the full training budget.

**Live poll (2026-10-03 15:20 UTC):** PnP M12 reached `878/8400` with rollout
raw `0.0026`; old plate-slide v2 M4 reached `1708/2800` with rollout raw
`0.0000`; v3 M1 reached `398/700` with rollout raw SR `0.9842`.  M1 remains
stable past the halfway point.

**Live poll (2026-10-03 15:24 UTC):** PnP M12 reached `898/8400` with rollout
raw `0.0031`; old plate-slide v2 M4 reached `1728/2800` with rollout raw
`0.0000`; v3 M1 reached `428/700` with rollout raw SR `0.9855`.  M1 remains
stable in the second half of its full-budget run.

**Live poll (2026-10-03 15:27 UTC):** PnP M12 reached `928/8400` with rollout
raw `0.0043`; old plate-slide v2 M4 reached `1748/2800` with rollout raw
`0.0000`; v3 M1 reached `458/700` with rollout raw SR `0.9867`.  M1 remains
stable in the latter half of training.

**Live poll (2026-10-03 15:29 UTC):** PnP M12 reached `938/8400` with rollout
raw `0.0047`; old plate-slide v2 M4 reached `1758/2800` with rollout raw
`0.0000`; v3 M1 reached `478/700` with rollout raw SR `0.9873`.  M1 remains
stable and continues toward its endpoint.

**Live poll (2026-10-03 15:31 UTC):** PnP M12 reached `948/8400` with rollout
raw `0.0052`; old plate-slide v2 M4 reached `1768/2800` with rollout raw
`0.0000`; v3 M1 reached `488/700` with rollout raw SR `0.9876`.  M1 remains
stable with no endpoint summary yet.

**Live poll (2026-10-03 15:33 UTC):** PnP M12 reached `958/8400` with rollout
raw `0.0058`; old plate-slide v2 M4 reached `1778/2800` with rollout raw
`0.0000`; v3 M1 reached `498/700` with rollout raw SR `0.9879`.  M1 remains
stable; the queue has not yet started its final evaluation.

**Live poll (2026-10-03 15:35 UTC):** PnP M12 reached `968/8400` with rollout
raw `0.0066`; old plate-slide v2 M4 reached `1788/2800` with rollout raw
`0.0000`; v3 M1 reached `518/700` with rollout raw SR `0.9885`.  M1 remains
stable, with endpoint training still in progress.

**Live poll (2026-10-03 15:36 UTC):** PnP M12 reached `988/8400` with rollout
raw `0.0081`; old plate-slide v2 M4 reached `1808/2800` with rollout raw
`0.0000`; v3 M1 reached `528/700` with rollout raw SR `0.9887`.  M1 remains
stable in its final quarter.

**Live poll (2026-10-03 15:38 UTC):** PnP M12 reached `998/8400` with rollout
raw `0.0091`; old plate-slide v2 M4 reached `1818/2800` with rollout raw
`0.0000`; v3 M1 reached `548/700` with rollout raw SR `0.9892`.  M1 remains
stable in the last quarter of full-budget training.

**Live poll (2026-10-03 15:42 UTC):** PnP M12 reached `1018/8400` with rollout
raw `0.0106`; old plate-slide v2 M4 reached `1838/2800` with rollout raw
`0.0000`; v3 M1 reached `578/700` with rollout raw SR `0.9898`.  M1 remains
stable with roughly 120 iterations remaining.

**Live poll (2026-10-03 15:44 UTC):** PnP M12 reached `1038/8400` with rollout
raw `0.0134`; old plate-slide v2 M4 reached `1848/2800` with rollout raw
`0.0000`; v3 M1 reached `598/700` with rollout raw SR `0.9902`.  M1 remains
stable with about 100 iterations left.

**Live poll (2026-10-03 15:46 UTC):** PnP M12 reached `1048/8400` with rollout
raw `0.0151`; old plate-slide v2 M4 reached `1858/2800` with rollout raw
`0.0000`; v3 M1 reached `608/700` with rollout raw SR `0.9904`.  M1 remains
stable with fewer than 100 iterations remaining.

**Live poll (2026-10-03 15:47 UTC):** PnP M12 reached `1058/8400` with rollout
raw `0.0169`; old plate-slide v2 M4 reached `1878/2800` with rollout raw
`0.0000`; v3 M1 reached `618/700` with rollout raw SR `0.9906`.  M1 remains
stable with about 80 iterations remaining.

**Live poll (2026-10-03 15:49 UTC):** PnP M12 reached `1068/8400` with rollout
raw `0.0189`; old plate-slide v2 M4 reached `1888/2800` with rollout raw
`0.0000`; v3 M1 reached `638/700` with rollout raw SR `0.9909`.  M1 remains
stable with about 60 iterations remaining.

**Live poll (2026-10-03 15:51 UTC):** PnP M12 reached `1078/8400` with rollout
raw `0.0211`; old plate-slide v2 M4 reached `1898/2800` with rollout raw
`0.0000`; v3 M1 reached `648/700` with rollout raw SR `0.9911`.  M1 remains
stable with roughly 50 iterations remaining.

**Live poll (2026-10-03 15:40 UTC):** PnP M12 reached `1008/8400` with rollout
raw `0.0099`; old plate-slide v2 M4 reached `1828/2800` with rollout raw
`0.0000`; v3 M1 reached `558/700` with rollout raw SR `0.9894`.  M1 remains
stable with about 20% of training remaining.

**Live poll (2026-10-03 15:26 UTC):** PnP M12 reached `908/8400` with rollout
raw `0.0033`; old plate-slide v2 M4 reached `1738/2800` with rollout raw
`0.0000`; v3 M1 reached `438/700` with rollout raw SR `0.9859`.  M1 remains
stable and continues toward the planned endpoint.

**Live poll (2026-10-03 15:22 UTC):** PnP M12 reached `888/8400` with rollout
raw `0.0028`; old plate-slide v2 M4 reached `1718/2800` with rollout raw
`0.0000`; v3 M1 reached `418/700` with rollout raw SR `0.9851`.  The v3 M1
single-macro run remains stable with no endpoint summary yet.

**Live poll (2026-10-03 15:12 UTC):** PnP M12 reached `828/8400` with rollout
raw `0.0015`; old plate-slide v2 M4 reached `1648/2800` with rollout raw
`0.0000`; v3 M1 reached `328/700` with rollout raw SR `0.9799`.  Checkpoints
through `model_325.pt` are present.

**Live poll (2026-10-03 15:14 UTC):** PnP M12 reached `838/8400` with rollout
raw `0.0017`; old plate-slide v2 M4 reached `1668/2800` with rollout raw
`0.0000`; v3 M1 reached `348/700` with rollout raw SR `0.9813`.  M1 remains
healthy, with no terminal summary yet.

**Live poll (2026-10-03 15:00 UTC):** PnP M12 reached `758/8400` with rollout
raw `0.0005`; old plate-slide v2 M4 reached `1578/2800` with rollout raw
`0.0000`; v3 M1 reached `228/700` with rollout raw SR `0.9671`.  All remain
in training and no endpoint EVAL has been produced.

**Live poll (2026-10-03 15:04 UTC):** PnP M12 reached `778/8400` with rollout
raw `0.0007`; old plate-slide v2 M4 reached `1598/2800` with rollout raw
`0.0000`; v3 M1 reached `268/700` with rollout raw SR `0.9737`.  M1 remains
on a healthy full-budget trajectory.

**Live poll (2026-10-03 15:10 UTC):** PnP M12 reached `818/8400` with rollout
raw `0.0014`; old plate-slide v2 M4 reached `1638/2800` with rollout raw
`0.0000`; v3 M1 reached `308/700` with rollout raw SR `0.9782`.  M1 is still
training toward its endpoint.

**Live poll (2026-10-03 15:08 UTC):** PnP M12 reached `808/8400` with rollout
raw `0.0013`; old plate-slide v2 M4 reached `1628/2800` with rollout raw
`0.0000`; v3 M1 reached `298/700` with rollout raw SR `0.9772`.  No endpoint
summary or held-out EVAL exists yet.

**Live poll (2026-10-03 15:07 UTC):** PnP M12 reached `788/8400` with rollout
raw `0.0009`; old plate-slide v2 M4 reached `1618/2800` with rollout raw
`0.0000`; v3 M1 reached `278/700` with rollout raw SR `0.9750`.  The v3 run
continues normally, with no endpoint summary yet.

**Live poll (2026-10-03 15:03 UTC):** PnP M12 reached `768/8400` with rollout
raw `0.0006`; old plate-slide v2 M4 reached `1588/2800` with rollout raw
`0.0000`; v3 M1 reached `248/700` with rollout raw SR `0.9708`.  M1 remains
healthy and is still before its full endpoint.

**Live poll (2026-10-03 03:06 UTC):** PnP M11 reached `3888/7700` with
rollout mandatory SR `0.6755`; plate-slide M2 reached `518/1400` with raw
rollout SR `0.9726` and mandatory rollout SR `0.9350`.  Both live handles are
healthy; the rescue queue continues waiting for M11's formal endpoint.

**Diagnostic audit (2026-10-03):** An existing independent 256-episode eval of
the M11 checkpoint at iteration 1250 reports raw SR `0.9805` and mandatory SR
`0.9570`; its prefix rates stay at `1.0` through macro 8 and are `0.9805`
through macro 11.  This confirms that the lower training-rollout metric is not
by itself evidence of a composition failure; the final endpoint will still be
judged only from the scheduled held-out evaluation at the completed checkpoint.

**Live poll (2026-10-03 03:08 UTC):** PnP M11 reached `3898/7700` with
rollout mandatory SR `0.6762`.  Plate-slide M2 remains live at its latest
flushed `518/1400` checkpoint (`0.9350` mandatory rollout).  M11 has not yet
written `training_summary.json`, so the rescue queue remains in its explicit
wait state.

**Live poll (2026-10-03 03:10 UTC):** PnP M11 reached `3918/7700` with
rollout mandatory SR `0.6775`; plate-slide M2 reached `538/1400` with raw
rollout SR `0.9737` and mandatory rollout SR `0.9374`.  Both trainers are
still active and the M11-dependent queue remains in its wait state.

**Live poll (2026-10-03 03:11 UTC):** PnP M11 reached `3928/7700` with
rollout mandatory SR `0.6782`.  Plate-slide M2 remains live at its latest
flushed `538/1400` checkpoint (`0.9374` mandatory rollout); no formal summary
or downstream handoff has appeared.

**Live poll (2026-10-03 03:13 UTC):** PnP M11 reached `3938/7700` with
rollout mandatory SR `0.6788`; plate-slide M2 reached `548/1400` with raw
rollout SR `0.9742` and mandatory rollout SR `0.9384`.  Both trainers remain
healthy, and the M11-dependent queue is still waiting for its formal summary.

**Live poll (2026-10-03 03:14 UTC):** PnP M11 reached `3948/7700` with
rollout mandatory SR `0.6795`; plate-slide M2 reached `558/1400` with raw
rollout SR `0.9747` and mandatory rollout SR `0.9394`.  Both live handles are
healthy; M11's formal endpoint is still pending.

**Live poll (2026-10-03 03:16 UTC):** PnP M11 reached `3958/7700` with
rollout mandatory SR `0.6801`; plate-slide M2 reached `568/1400` with raw
rollout SR `0.9752` and mandatory rollout SR `0.9404`.  Both processes remain
healthy, while the M11-dependent queue continues waiting for its summary.

**Live poll (2026-10-03 03:18 UTC):** PnP M11 reached `3968/7700` with
rollout mandatory SR `0.6808`; plate-slide M2 remains live at its latest
flushed `568/1400` checkpoint (`0.9404` mandatory rollout).  No formal M11
summary has appeared and the downstream queue remains deferred.

**Live poll (2026-10-03 03:20 UTC):** PnP M11 reached `3978/7700` with
rollout mandatory SR `0.6814`; plate-slide M2 reached `578/1400` with raw
rollout SR `0.9757` and mandatory rollout SR `0.9415`.  Both trainers remain
healthy; the rescue queue is still waiting for M11's formal summary.

**Live poll (2026-10-03 03:21 UTC):** PnP M11 reached `3988/7700` with
rollout mandatory SR `0.6820`; plate-slide M2 reached `588/1400` with raw
rollout SR `0.9761` and mandatory rollout SR `0.9423`.  Both processes remain
healthy; M11's formal endpoint is still pending.

**Live poll (2026-10-03 03:22 UTC):** PnP M11 reached `4008/7700` with
rollout mandatory SR `0.6833`; plate-slide M2 reached `598/1400` with raw
rollout SR `0.9766` and mandatory rollout SR `0.9431`.  Both jobs remain
healthy and the downstream queue is still waiting for M11's summary.

**Live poll (2026-10-03 03:24 UTC):** PnP M11 reached `4018/7700` with
rollout mandatory SR `0.6838`; plate-slide M2 reached `608/1400` with raw
rollout SR `0.9770` and mandatory rollout SR `0.9438`.  Both trainers remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 03:25 UTC):** PnP M11 reached `4028/7700` with
rollout mandatory SR `0.6843`; plate-slide M2 remains live at its latest
flushed `608/1400` checkpoint (`0.9438` mandatory rollout).  No formal M11
summary or downstream handoff has appeared.

**Live poll (2026-10-03 03:26 UTC):** PnP M11 reached `4038/7700` with
rollout mandatory SR `0.6849`; plate-slide M2 reached `618/1400` with raw
rollout SR `0.9774` and mandatory rollout SR `0.9445`.  Both trainers remain
healthy; the M11-dependent queue is still waiting for its endpoint.

**Live poll (2026-10-03 03:28 UTC):** PnP M11 reached `4048/7700` with
rollout mandatory SR `0.6856`; plate-slide M2 reached `628/1400` with raw
rollout SR `0.9779` and mandatory rollout SR `0.9452`.  Both live processes
remain healthy; the downstream queue is still waiting for M11's summary.

**Live poll (2026-10-03 03:29 UTC):** PnP M11 reached `4058/7700` with
rollout mandatory SR `0.6863`; plate-slide M2 reached `638/1400` with raw
rollout SR `0.9782` and mandatory rollout SR `0.9458`.  Both trainers remain
healthy; M11 has not yet entered final evaluation.

**Live poll (2026-10-03 03:31 UTC):** PnP M11 reached `4068/7700` with
rollout mandatory SR `0.6869`; plate-slide M2 remains live at `638/1400` with
mandatory rollout SR `0.9458`.  No formal M11 summary has appeared; the
downstream queue remains correctly deferred.

**Live poll (2026-10-03 03:32 UTC):** PnP M11 reached `4078/7700` with
rollout mandatory SR `0.6875`; plate-slide M2 reached `648/1400` with raw
rollout SR `0.9786` and mandatory rollout SR `0.9465`.  Both jobs remain
healthy; M11's formal endpoint is still pending.

**Live poll (2026-10-03 03:34 UTC):** PnP M11 reached `4088/7700` with
rollout mandatory SR `0.6882`; plate-slide M2 reached `658/1400` with raw
rollout SR `0.9789` and mandatory rollout SR `0.9470`.  Both trainers remain
healthy; the downstream queue is still waiting for M11's summary.

**Live poll (2026-10-03 03:35 UTC):** PnP M11 reached `4098/7700` with
rollout mandatory SR `0.6888`; plate-slide M2 reached `668/1400` with raw
rollout SR `0.9792` and mandatory rollout SR `0.9477`.  Both jobs remain
healthy; M11's formal endpoint is still pending.

**Live poll (2026-10-03 03:37 UTC):** PnP M11 reached `4108/7700` with
rollout mandatory SR `0.6894`; plate-slide M2 reached `678/1400` with raw
rollout SR `0.9796` and mandatory rollout SR `0.9484`.  Both trainers remain
healthy; the M11-dependent queue is still waiting for its formal summary.

**Live poll (2026-10-03 03:38 UTC):** PnP M11 reached `4118/7700` with
rollout mandatory SR `0.6900`; plate-slide M2 remains live at its latest
flushed `678/1400` checkpoint (`0.9484` mandatory rollout).  Both processes
remain healthy; M11's formal endpoint is still pending.

**Live poll (2026-10-03 03:40 UTC):** PnP M11 reached `4138/7700` with
rollout mandatory SR `0.6912`; plate-slide M2 reached `688/1400` with raw
rollout SR `0.9798` and mandatory rollout SR `0.9489`.  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 03:42 UTC):** PnP M11 reached `4148/7700` with
rollout mandatory SR `0.6918`; plate-slide M2 reached `698/1400` with raw
rollout SR `0.9802` and mandatory rollout SR `0.9495`.  Both trainers remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 03:43 UTC):** PnP M11 reached `4158/7700` with
rollout mandatory SR `0.6923`; plate-slide M2 reached `708/1400` with raw
rollout SR `0.9805` and mandatory rollout SR `0.9501`.  Both jobs remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 03:44 UTC):** PnP M11 reached `4168/7700` with
rollout mandatory SR `0.6927`; plate-slide M2 reached `718/1400` with raw
rollout SR `0.9808` and mandatory rollout SR `0.9506`.  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 03:46 UTC):** PnP M11 reached `4178/7700` with
rollout mandatory SR `0.6933`; plate-slide M2 remains live at its latest
flushed `718/1400` checkpoint (`0.9506` mandatory rollout).  Both trainers
remain healthy; the M11-dependent queue is still waiting.

**Live poll (2026-10-03 03:48 UTC):** PnP M11 reached `4188/7700` with
rollout mandatory SR `0.6937`; plate-slide M2 reached `728/1400` with raw
rollout SR `0.9811` and mandatory rollout SR `0.9510`.  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 03:49 UTC):** PnP M11 reached `4198/7700` with
rollout mandatory SR `0.6943`; plate-slide M2 reached `738/1400` with raw
rollout SR `0.9814` and mandatory rollout SR `0.9515`.  Both trainers remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 03:51 UTC):** PnP M11 reached `4208/7700` with
rollout mandatory SR `0.6947`; plate-slide M2 reached `748/1400` with raw
rollout SR `0.9816` and mandatory rollout SR `0.9520`.  Both jobs remain
healthy; M11's formal endpoint is still pending.

**Live poll (2026-10-03 03:52 UTC):** PnP M11 reached `4218/7700` with
rollout mandatory SR `0.6952`; plate-slide M2 reached `758/1400` with raw
rollout SR `0.9819` and mandatory rollout SR `0.9526`.  Both trainers remain
healthy; the downstream queue is still waiting for M11's summary.

**Live poll (2026-10-03 03:54 UTC):** PnP M11 reached `4238/7700` with
rollout mandatory SR `0.6962`; plate-slide M2 remains live at its latest
flushed `758/1400` checkpoint (`0.9526` mandatory rollout).  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 03:55 UTC):** PnP M11 reached `4248/7700` with
rollout mandatory SR `0.6967`; plate-slide M2 reached `768/1400` with raw
rollout SR `0.9822` and mandatory rollout SR `0.9531`.  Both trainers remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 03:57 UTC):** PnP M11 reached `4258/7700` with
rollout mandatory SR `0.6973`; plate-slide M2 reached `778/1400` with raw
rollout SR `0.9824` and mandatory rollout SR `0.9536`.  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 03:58 UTC):** PnP M11 reached `4268/7700` with
rollout mandatory SR `0.6979`; plate-slide M2 reached `788/1400` with raw
rollout SR `0.9827` and mandatory rollout SR `0.9541`.  Both trainers remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 04:00 UTC):** PnP M11 reached `4278/7700` with
rollout mandatory SR `0.6985`; plate-slide M2 reached `798/1400` with raw
rollout SR `0.9829` and mandatory rollout SR `0.9545`.  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 04:01 UTC):** PnP M11 reached `4288/7700` with
rollout mandatory SR `0.6991`; plate-slide M2 reached `808/1400` with raw
rollout SR `0.9831` and mandatory rollout SR `0.9549`.  Both trainers remain
healthy; the downstream queue is still waiting for M11's summary.

**Live poll (2026-10-03 04:03 UTC):** PnP M11 reached `4298/7700` with
rollout mandatory SR `0.6996`; plate-slide M2 remains live at its latest
flushed `808/1400` checkpoint (`0.9549` mandatory rollout).  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 04:04 UTC):** PnP M11 reached `4308/7700` with
rollout mandatory SR `0.7002`; plate-slide M2 reached `818/1400` with raw
rollout SR `0.9834` and mandatory rollout SR `0.9553`.  Both trainers remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 04:06 UTC):** PnP M11 reached `4318/7700` with
rollout mandatory SR `0.7007`; plate-slide M2 reached `828/1400` with raw
rollout SR `0.9836` and mandatory rollout SR `0.9557`.  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 04:08 UTC):** PnP M11 reached `4338/7700` with
rollout mandatory SR `0.7018`; plate-slide M2 reached `838/1400` with raw
rollout SR `0.9838` and mandatory rollout SR `0.9561`.  Both trainers remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 04:09 UTC):** PnP M11 reached `4348/7700` with
rollout mandatory SR `0.7024`; plate-slide M2 reached `848/1400` with raw
rollout SR `0.9840` and mandatory rollout SR `0.9564`.  Both jobs remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 04:11 UTC):** PnP M11 reached `4358/7700` with
rollout mandatory SR `0.7029`; plate-slide M2 remains live at its latest
flushed `848/1400` checkpoint (`0.9564` mandatory rollout).  Both trainers
remain healthy; M11's formal endpoint is still pending.

**Live poll (2026-10-03 04:12 UTC):** PnP M11 reached `4368/7700` with
rollout mandatory SR `0.7034`; plate-slide M2 reached `858/1400` with raw
rollout SR `0.9842` and mandatory rollout SR `0.9568`.  Both jobs remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 04:14 UTC):** PnP M11 reached `4378/7700` with
rollout mandatory SR `0.7039`; plate-slide M2 reached `868/1400` with raw
rollout SR `0.9844` and mandatory rollout SR `0.9572`.  Both trainers remain
healthy; M11's formal summary is still pending.

**Live poll (2026-10-03 04:15 UTC):** PnP M11 reached `4388/7700` with
rollout mandatory SR `0.7044`; plate-slide M2 reached `878/1400` with raw
rollout SR `0.9846` and mandatory rollout SR `0.9575`.  Both jobs remain
healthy; the M11-dependent queue is still waiting for its summary.

**Live poll (2026-10-03 04:18 UTC):** PnP M11 reached `4408/7700` with
rollout mandatory SR `0.7056`.  Plate-slide M2 remains healthy at its latest
flushed `888/1400` checkpoint with raw rollout SR `0.9848` and mandatory
rollout SR `0.9579`; the M11-dependent rescue queue is still waiting for the
formal training summary.

**Live poll (2026-10-03 04:19 UTC):** PnP M11 reached `4418/7700` with
rollout raw SR `0.7140` and mandatory SR `0.7061`.  Plate-slide M2 reached
`898/1400` with raw rollout SR `0.9849` and mandatory rollout SR `0.9582`.
Both trainers are still live; no endpoint summary has appeared yet.

**Live poll (2026-10-03 04:20 UTC):** PnP M11 reached `4428/7700` with
rollout raw SR `0.7145` and mandatory SR `0.7066`; plate-slide M2 reached
`908/1400` with raw rollout SR `0.9851` and mandatory rollout SR `0.9586`.
Both processes remain healthy, while the rescue queue continues waiting for
M11's endpoint summary.

**Live poll (2026-10-03 04:21 UTC):** PnP M11 reached `4438/7700` with
rollout raw SR `0.7151` and mandatory SR `0.7072`; plate-slide M2 remains
healthy at its latest flushed `908/1400` checkpoint.  The rescue queue is
still correctly waiting for the M11 training summary before launching the
independent endpoint evaluation and subsequent jobs.

**Live poll (2026-10-03 04:22 UTC):** PnP M11 remains live at its latest
flushed `4438/7700` checkpoint (raw rollout SR `0.7151`, mandatory SR
`0.7072`); plate-slide M2 advanced to `918/1400` (raw `0.9853`, mandatory
`0.9589`).  No M11 summary exists yet, so no downstream job has been started.

**Queue hygiene (2026-10-03 04:23 UTC):** The legacy v4/v5 wrappers were
confirmed to have only sleeping wait children and no trainer processes.  They
were stopped to remove a possible post-M11 GPU/duplicate-launch race; active
M11, plate-slide M2, and the v6 rescue queue were left untouched.

**Live poll (2026-10-03 04:24 UTC):** PnP M11 reached `4448/7700` with raw
rollout SR `0.7156` and mandatory SR `0.7077`; plate-slide M2 reached
`918/1400` with raw rollout SR `0.9853` and mandatory SR `0.9589`.  Neither
run has written its endpoint summary yet, and v6 remains waiting for M11.

**Live poll (2026-10-03 04:24 UTC, GPU audit):** PnP M11 advanced to
`4458/7700` (raw rollout SR `0.7161`, mandatory SR `0.7082`); plate-slide M2
advanced to `928/1400` (raw `0.9855`, mandatory `0.9593`).  GPUs 0--2 are
occupied by unrelated live workloads and GPU 3 is occupied by M11/Qwen, so no
safe parallel lane is available; the declared queue remains the correct
choice.

**Live poll (2026-10-03 04:25 UTC):** No endpoint artifacts have appeared
since the previous poll.  PnP M11 remains at its latest flushed `4458/7700`
checkpoint (raw rollout SR `0.7161`, mandatory `0.7082`), and plate-slide M2
remains at `928/1400` (raw `0.9855`, mandatory `0.9593`).  Both trainers and
the v6 rescue process are alive; no downstream job has started yet.

**Live poll (2026-10-03 04:25 UTC, later flush):** PnP M11 advanced to
`4468/7700` (raw rollout SR `0.7166`, mandatory `0.7086`), confirming active
progress rather than a stall.  Plate-slide M2 is still at its latest flushed
`928/1400` checkpoint.  No endpoint summary or downstream evaluation exists
yet.

**Live poll (2026-10-03 04:26 UTC):** PnP M11 remains healthy at its latest
flushed `4468/7700` checkpoint; plate-slide M2 advanced to `938/1400` with raw
rollout SR `0.9856` and mandatory SR `0.9596`.  The only M11 artifacts beyond
the training log are pre-existing diagnostic evals; no final summary/eval has
been created and the rescue queue is still waiting.

**Live poll (2026-10-03 04:27 UTC):** After a verified wait, PnP M11
advanced to `4478/7700` with raw rollout SR `0.7170` and mandatory SR
`0.7090`; plate-slide M2 remains live at `938/1400`.  No endpoint summary has
appeared, and v6 is still waiting for M11.

**Live poll (2026-10-03 04:29 UTC):** PnP M11 advanced to `4488/7700` with
raw rollout SR `0.7174` and mandatory SR `0.7094`; plate-slide M2 advanced to
`948/1400` with raw rollout SR `0.9858` and mandatory SR `0.9600`.  All three
live processes remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 04:30 UTC):** PnP M11 advanced to `4498/7700` with
raw rollout SR `0.7177` and mandatory SR `0.7098`; plate-slide M2 remains live
at its latest flushed `948/1400` checkpoint.  The only matching summaries for
M2/M3 are older exploratory runs, not this declared queue, so they are not
used as results.

**Live poll (2026-10-03 04:31 UTC):** After another verified wait, PnP M11
advanced to `4508/7700` with raw rollout SR `0.7181` and mandatory SR
`0.7101`; plate-slide M2 advanced to `958/1400` with raw rollout SR `0.9860`
and mandatory SR `0.9603`.  Both trainers remain healthy and no endpoint
summary has appeared.

**Live poll (2026-10-03 04:32 UTC):** PnP M11 remains at its latest flushed
`4508/7700` checkpoint; plate-slide M2 advanced to `968/1400` with raw
rollout SR `0.9861` and mandatory SR `0.9606`.  M11, M2, and v6 are alive;
there is still no endpoint summary or downstream launch.

**Live poll (2026-10-03 04:33 UTC):** PnP M11 advanced to `4518/7700` with
raw rollout SR `0.7185` and mandatory SR `0.7105`; plate-slide M2 remains
healthy at its latest flushed `968/1400` checkpoint.  v6 continues waiting for
M11's formal training summary.

**Live poll (2026-10-03 04:34 UTC):** PnP M11 advanced to `4528/7700` with
raw rollout SR `0.7189` and mandatory SR `0.7109`; plate-slide M2 advanced to
`978/1400` with raw rollout SR `0.9863` and mandatory SR `0.9609`.  No final
summary or downstream launch has appeared.

**Live poll (2026-10-03 04:35 UTC):** PnP M11 advanced to `4538/7700` with
raw rollout SR `0.7192` and mandatory SR `0.7112`; plate-slide M2 remains
healthy at its latest flushed `978/1400` checkpoint.  No endpoint summary has
appeared; v6 continues waiting for M11.

**Live poll (2026-10-03 04:36 UTC):** PnP M11 is still healthy at its latest
flushed `4538/7700` checkpoint; plate-slide M2 advanced to `988/1400` with raw
rollout SR `0.9864` and mandatory SR `0.9612`.  No new summary/evaluation has
appeared, and v6 remains in its M11-summary wait state.

**Live poll (2026-10-03 04:37 UTC):** PnP M11 advanced to `4548/7700` with
raw rollout SR `0.7196` and mandatory SR `0.7116`; plate-slide M2 advanced to
`998/1400` with raw rollout SR `0.9866` and mandatory SR `0.9616`.  Both
trainers remain healthy, with no endpoint summary yet.

**Live poll (2026-10-03 04:39 UTC):** PnP M11 advanced to `4568/7700` with
raw rollout SR `0.7204` and mandatory SR `0.7123`; plate-slide M2 remains
healthy at its latest flushed `998/1400` checkpoint.  v6 is still waiting for
M11's formal summary; no downstream run has begun.

**Live poll (2026-10-03 04:41 UTC):** PnP M11 advanced to `4578/7700` with
raw rollout SR `0.7207` and mandatory SR `0.7127`; plate-slide M2 advanced to
`1008/1400` with raw rollout SR `0.9867` and mandatory SR `0.9619`.  Both
trainers remain healthy, and no endpoint summary has appeared.

**Live poll (2026-10-03 04:41 UTC, later flush):** PnP M11 remains healthy at
its latest flushed `4578/7700` checkpoint; plate-slide M2 advanced to
`1018/1400` with raw rollout SR `0.9869` and mandatory SR `0.9622`.  No new
summary or downstream launch has appeared.

**Live poll (2026-10-03 04:43 UTC):** PnP M11 advanced to `4588/7700` with
raw rollout SR `0.7210` and mandatory SR `0.7130`; plate-slide M2 remains
healthy at its latest flushed `1018/1400` checkpoint.  No endpoint summary is
present and v6 remains waiting for M11.

**Live poll (2026-10-03 04:44 UTC):** PnP M11 advanced to `4598/7700` with
raw rollout SR `0.7214` and mandatory SR `0.7134`; plate-slide M2 advanced to
`1028/1400` with raw rollout SR `0.9870` and mandatory SR `0.9625`.  Both
trainers remain healthy; no endpoint summary or downstream launch yet.

**Live poll (2026-10-03 04:46 UTC):** PnP M11 advanced to `4608/7700` with
raw rollout SR `0.7218` and mandatory SR `0.7138`; plate-slide M2 advanced to
`1038/1400` with raw rollout SR `0.9872` and mandatory SR `0.9628`.  Both
trainers remain healthy and v6 is still waiting for the M11 summary.

**Live poll (2026-10-03 04:47 UTC):** PnP M11 advanced to `4618/7700` with
raw rollout SR `0.7222` and mandatory SR `0.7142`; plate-slide M2 advanced to
`1048/1400` with raw rollout SR `0.9873` and mandatory SR `0.9631`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 04:49 UTC):** PnP M11 advanced to `4638/7700` with
raw rollout SR `0.7231` and mandatory SR `0.7151`; plate-slide M2 remains
healthy at its latest flushed `1048/1400` checkpoint.  No endpoint summary or
downstream launch has appeared yet.

**Live poll (2026-10-03 04:50 UTC):** PnP M11 advanced to `4648/7700` with
raw rollout SR `0.7235` and mandatory SR `0.7156`; plate-slide M2 advanced to
`1058/1400` with raw rollout SR `0.9874` and mandatory SR `0.9633`.  Both
trainers remain healthy; v6 still waits for M11's endpoint summary.

**Live poll (2026-10-03 04:52 UTC):** PnP M11 advanced to `4658/7700` with
raw rollout SR `0.7240` and mandatory SR `0.7160`; plate-slide M2 advanced to
`1068/1400` with raw rollout SR `0.9876` and mandatory SR `0.9636`.  Both
trainers remain healthy; no endpoint summary or downstream launch yet.

**Live poll (2026-10-03 04:54 UTC):** PnP M11 advanced to `4668/7700` with
raw rollout SR `0.7244` and mandatory SR `0.7164`; plate-slide M2 advanced to
`1078/1400` with raw rollout SR `0.9877` and mandatory SR `0.9640`.  Both
trainers remain healthy, with no endpoint summary yet.

**Live poll (2026-10-03 04:55 UTC):** PnP M11 advanced to `4678/7700` with
raw rollout SR `0.7248` and mandatory SR `0.7168`; plate-slide M2 advanced to
`1088/1400` with raw rollout SR `0.9878` and mandatory SR `0.9642`.  Both
trainers remain healthy; v6 is still waiting for M11's formal summary.

**Live poll (2026-10-03 04:57 UTC):** PnP M11 advanced to `4688/7700` with
raw rollout SR `0.7251` and mandatory SR `0.7172`; plate-slide M2 advanced to
`1098/1400` with raw rollout SR `0.9880` and mandatory SR `0.9645`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 04:58 UTC):** PnP M11 advanced to `4698/7700` with
raw rollout SR `0.7255` and mandatory SR `0.7176`; plate-slide M2 remains
healthy at its latest flushed `1098/1400` checkpoint.  v6 continues waiting
for M11's formal summary.

**Live poll (2026-10-03 05:00 UTC):** PnP M11 advanced to `4718/7700` with
raw rollout SR `0.7263` and mandatory SR `0.7184`; plate-slide M2 advanced to
`1108/1400` with raw rollout SR `0.9881` and mandatory SR `0.9647`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:02 UTC):** PnP M11 advanced to `4728/7700` with
raw rollout SR `0.7267` and mandatory SR `0.7188`; plate-slide M2 advanced to
`1118/1400` with raw rollout SR `0.9882` and mandatory SR `0.9650`.  Both
trainers remain healthy, and v6 is still waiting for M11's formal summary.

**Live poll (2026-10-03 05:03 UTC):** PnP M11 advanced to `4738/7700` with
raw rollout SR `0.7271` and mandatory SR `0.7192`; plate-slide M2 advanced to
`1128/1400` with raw rollout SR `0.9883` and mandatory SR `0.9652`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:05 UTC):** PnP M11 advanced to `4748/7700` with
raw rollout SR `0.7274` and mandatory SR `0.7196`; plate-slide M2 advanced to
`1138/1400` with raw rollout SR `0.9884` and mandatory SR `0.9654`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:06 UTC):** PnP M11 advanced to `4758/7700` with
raw rollout SR `0.7278` and mandatory SR `0.7199`; plate-slide M2 remains
healthy at its latest flushed `1138/1400` checkpoint.  No endpoint summary or
downstream launch has appeared.

**Live poll (2026-10-03 05:08 UTC):** PnP M11 advanced to `4768/7700` with
raw rollout SR `0.7282` and mandatory SR `0.7203`; plate-slide M2 advanced to
`1148/1400` with raw rollout SR `0.9885` and mandatory SR `0.9656`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:10 UTC):** PnP M11 advanced to `4778/7700` with
raw rollout SR `0.7285` and mandatory SR `0.7207`; plate-slide M2 advanced to
`1158/1400` with raw rollout SR `0.9886` and mandatory SR `0.9659`.  Both
trainers remain healthy, with no endpoint summary yet.

**Live poll (2026-10-03 05:11 UTC):** PnP M11 advanced to `4798/7700` with
raw rollout SR `0.7291` and mandatory SR `0.7213`; plate-slide M2 advanced to
`1168/1400` with raw rollout SR `0.9887` and mandatory SR `0.9660`.  Both
trainers remain healthy; v6 continues waiting for M11's formal summary.

**Live poll (2026-10-03 05:13 UTC):** PnP M11 advanced to `4808/7700` with
raw rollout SR `0.7295` and mandatory SR `0.7217`; plate-slide M2 advanced to
`1178/1400` with raw rollout SR `0.9889` and mandatory SR `0.9662`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:14 UTC):** PnP M11 advanced to `4818/7700` with
raw rollout SR `0.7298` and mandatory SR `0.7220`; plate-slide M2 advanced to
`1188/1400` with raw rollout SR `0.9890` and mandatory SR `0.9664`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:16 UTC):** PnP M11 advanced to `4828/7700` with
raw rollout SR `0.7302` and mandatory SR `0.7224`; plate-slide M2 advanced to
`1198/1400` with raw rollout SR `0.9891` and mandatory SR `0.9667`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:18 UTC):** PnP M11 advanced to `4838/7700` with
raw rollout SR `0.7304` and mandatory SR `0.7226`; plate-slide M2 remains
healthy at its latest flushed `1198/1400` checkpoint.  No endpoint summary or
downstream launch has appeared.

**Live poll (2026-10-03 05:19 UTC):** PnP M11 advanced to `4848/7700` with
raw rollout SR `0.7308` and mandatory SR `0.7230`; plate-slide M2 advanced to
`1208/1400` with raw rollout SR `0.9892` and mandatory SR `0.9669`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:21 UTC):** PnP M11 advanced to `4868/7700` with
raw rollout SR `0.7316` and mandatory SR `0.7238`; plate-slide M2 advanced to
`1218/1400` with raw rollout SR `0.9893` and mandatory SR `0.9671`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:22 UTC):** PnP M11 advanced to `4878/7700` with
raw rollout SR `0.7320` and mandatory SR `0.7242`; plate-slide M2 advanced to
`1228/1400` with raw rollout SR `0.9894` and mandatory SR `0.9673`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:24 UTC):** PnP M11 advanced to `4888/7700` with
raw rollout SR `0.7324` and mandatory SR `0.7246`; plate-slide M2 advanced to
`1238/1400` with raw rollout SR `0.9895` and mandatory SR `0.9676`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:25 UTC):** PnP M11 advanced to `4898/7700` with
raw rollout SR `0.7327` and mandatory SR `0.7250`; plate-slide M2 remains
healthy at its latest flushed `1238/1400` checkpoint.  No endpoint summary or
downstream launch has appeared.

**Live poll (2026-10-03 05:27 UTC):** PnP M11 advanced to `4908/7700` with
raw rollout SR `0.7332` and mandatory SR `0.7254`; plate-slide M2 advanced to
`1248/1400` with raw rollout SR `0.9895` and mandatory SR `0.9678`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:29 UTC):** PnP M11 advanced to `4918/7700` with
raw rollout SR `0.7336` and mandatory SR `0.7258`; plate-slide M2 advanced to
`1258/1400` with raw rollout SR `0.9896` and mandatory SR `0.9680`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:30 UTC):** PnP M11 advanced to `4928/7700` with
raw rollout SR `0.7340` and mandatory SR `0.7263`; plate-slide M2 advanced to
`1268/1400` with raw rollout SR `0.9897` and mandatory SR `0.9682`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:32 UTC):** PnP M11 advanced to `4938/7700` with
raw rollout SR `0.7344` and mandatory SR `0.7267`; plate-slide M2 advanced to
`1278/1400` with raw rollout SR `0.9898` and mandatory SR `0.9684`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:33 UTC):** PnP M11 advanced to `4948/7700` with
raw rollout SR `0.7348` and mandatory SR `0.7271`; plate-slide M2 remains
healthy at its latest flushed `1278/1400` checkpoint.  No endpoint summary or
downstream launch has appeared.

**Live poll (2026-10-03 05:35 UTC):** PnP M11 advanced to `4968/7700` with
raw rollout SR `0.7356` and mandatory SR `0.7279`; plate-slide M2 advanced to
`1288/1400` with raw rollout SR `0.9899` and mandatory SR `0.9687`.  Both
trainers remain healthy; v6 still waits for M11's formal summary.

**Live poll (2026-10-03 05:36 UTC):** PnP M11 advanced to `4978/7700` with
raw rollout SR `0.7360` and mandatory SR `0.7283`; plate-slide M2 advanced to
`1298/1400` with raw rollout SR `0.9900` and mandatory SR `0.9689`.  Both
trainers remain healthy; no endpoint summary has appeared.

**Live poll (2026-10-03 05:38 UTC):** PnP M11's latest flushed checkpoint is
`4988/7700`, with raw rollout SR `0.7364` and mandatory SR `0.7288`; the
plate-slide M2 trainer is at `1308/1400`, with raw rollout SR `0.9901` and
mandatory SR `0.9691`.  Both processes are still running; the rescue queue is
still waiting for M11's formal `training_summary.json`.

**Live poll (2026-10-03 05:39 UTC):** PnP M11 advanced to `4998/7700` with
raw rollout SR `0.7368` and mandatory SR `0.7292`; plate-slide M2 advanced to
`1318/1400` with raw rollout SR `0.9902` and mandatory SR `0.9692`.  Both
trainers remain healthy and no downstream job has started yet.

**Live poll (2026-10-03 05:41 UTC):** PnP M11 advanced to `5008/7700` with
raw rollout SR `0.7372` and mandatory SR `0.7296`.  Plate-slide M2 remains
active; its log is still flushing `1318/1400`, while the output directory has
already written `model_1325.pt`, so it is making progress rather than stalled.
The rescue queue continues to wait for the M11 formal summary.

**Live poll (2026-10-03 05:43 UTC):** PnP M11 advanced to `5028/7700` with
raw rollout SR `0.7379` and mandatory SR `0.7303`; plate-slide M2 advanced to
`1338/1400` with raw rollout SR `0.9903` and mandatory SR `0.9696`.  Both
trainers remain active; the rescue queue is still waiting for M11's endpoint
summary before launching downstream jobs.

**Live poll (2026-10-03 05:46 UTC):** PnP M11 reached `5048/7700` with raw
rollout SR `0.7387` and mandatory SR `0.7311`; plate-slide M2 remains active
around its `1348/1400` log flush.  The independent plate-slide queue is set to
evaluate M2 over 256 episodes and then launch M3--M5; no intervention is needed.
**Live poll (2026-10-03 05:45 UTC):** PnP M11 advanced to `5038/7700` with
raw rollout SR `0.7383` and mandatory SR `0.7307`; plate-slide M2 advanced to
`1348/1400` with raw rollout SR `0.9904` and mandatory SR `0.9698`.  The latest
saved checkpoints are `model_5025.pt` and `model_1325.pt`, respectively; no
formal summary or downstream launch has appeared yet.

**Live poll (2026-10-03 05:51 UTC):** PnP M11 advanced to `5078/7700` with
raw rollout SR `0.7397` and mandatory SR `0.7321`; plate-slide M2 advanced to
`1378/1400` with raw rollout SR `0.9906` and mandatory SR `0.9703`.  Both
trainers are healthy; M2 is within its final 22 iterations, but its formal
summary and held-out evaluation are not yet present.

**Completed result (2026-10-03 05:55 UTC):** Plate-slide M2 finished its
`1400`-iteration budget.  The held-out evaluation used `256` episodes and
achieved raw SR `1.0000` (`256/256`) and mandatory SR `1.0000` (`256/256`).
The queue immediately launched plate-slide M3; PnP M11 remains in training.

**Live poll (2026-10-03 05:58 UTC):** Plate-slide M3 has started from scratch
and is at `8/2100`; its zero early rollout SR is expected before learning.  PnP
M11 is healthy at `5128/7700` with raw rollout SR `0.7411` and mandatory SR
`0.7335`.

**Live poll (2026-10-03 06:00 UTC):** Plate-slide M3 is making normal
progress at `18/2100` (the initial rollout is still `0.0000`); its event file
and `model_25.pt` are being written.  PnP M11 advanced to `5148/7700` with raw
rollout SR `0.7418` and mandatory SR `0.7343`.  No process has exited.

**Live poll (2026-10-03 06:01 UTC):** Plate-slide M3 advanced to `28/2100`
with early rollout SR still `0.0000`; the process and event stream are active.
PnP M11 remains healthy at its latest flushed `5148/7700` checkpoint.  No new
formal result has appeared.

**Live poll (2026-10-03 06:04 UTC):** Plate-slide M3 advanced to `48/2100`
with early rollout SR `0.0000`; its event stream is still being updated and no
failure/exit occurred.  PnP M11 advanced to `5178/7700` with raw rollout SR
`0.7428` and mandatory SR `0.7353`.

**Live poll (2026-10-03 06:06 UTC):** Plate-slide M3 advanced to `58/2100`
with raw rollout SR `0.0000`; its `model_50.pt` checkpoint is present, so the
run is progressing normally.  PnP M11 advanced to `5188/7700` with raw
rollout SR `0.7432` and mandatory SR `0.7357`.

**Live poll (2026-10-03 06:03 UTC):** Plate-slide M3 advanced to `38/2100`
with early rollout SR `0.0000`; its process is active and checkpoints are
being written.  PnP M11 advanced to `5168/7700` with raw rollout SR `0.7424`
and mandatory SR `0.7349`.  This is still training-time evidence only.

**Live poll (2026-10-03 06:07 UTC):** Plate-slide M3 remains at the latest
flushed `58/2100`; raw rollout SR is `0.0000`, while mean reward is `40.10` and
mean episode length is `328.63`.  Checkpoints and event files are updating, so
this is still an early training point rather than a failure diagnosis.  PnP
M11 advanced to `5198/7700` with raw rollout SR `0.7436` and mandatory SR
`0.7361`.

**Live poll (2026-10-03 06:08 UTC):** Plate-slide M3 advanced to `68/2100`
with mean reward `50.12` and mean episode length `356.25`; raw/mandatory
rollout SR remains `0.0000`, but the training process is healthy.  PnP M11
advanced to `5208/7700` with raw rollout SR `0.7439` and mandatory SR `0.7365`.

**Live poll (2026-10-03 06:08 UTC):** M3's event stream and `model_50.pt`
continue updating, while its latest flushed training metric is `58/2100` with
raw/mandatory rollout SR `0.0000`.  M11 has written `model_5200.pt` and is at
its latest flushed `5198/7700` metric (raw `0.7436`, mandatory `0.7361`).  No
formal endpoint result is available yet.

**Live poll (2026-10-03 06:10 UTC):** Plate-slide M3 advanced to `78/2100`
with mean episode length `345.36`, rollout SR still `0.0000`, and latest
checkpoint `model_75.pt`; training remains active.  PnP M11 advanced to
`5218/7700` with raw rollout SR `0.7443` and mandatory SR `0.7369`.

**Live poll (2026-10-03 06:11 UTC):** PnP M11 advanced to `5228/7700` with
raw rollout SR `0.7446` and mandatory SR `0.7372`.  Plate-slide M3 remains at
its latest flushed `78/2100` metric with raw rollout SR `0.0000`; `model_75.pt`
is present and the process remains active, so no failure/debug conclusion is
drawn yet.

**Live poll (2026-10-03 06:13 UTC):** Plate-slide M3 advanced to `88/2100`
with rollout raw/mandatory SR still `0.0000`; its event stream is updating and
the next checkpoint is pending.  PnP M11 advanced to `5238/7700` with raw
rollout SR `0.7449` and mandatory SR `0.7375`.

**Live poll (2026-10-03 06:14 UTC):** PnP M11 advanced to `5248/7700` with
raw rollout SR `0.7452` and mandatory SR `0.7378`.  Plate-slide M3 is at its
latest flushed `88/2100`; rollout SR remains `0.0000`, with the process still
active and the next checkpoint pending.

**Live poll (2026-10-03 06:15 UTC):** Plate-slide M3 advanced to `98/2100`
and wrote `model_100.pt`; rollout raw/mandatory SR remains `0.0000`, with mean
episode length `296.66`.  PnP M11 wrote `model_5250.pt` and remains healthy at
its latest flushed `5248/7700` metric (raw `0.7452`, mandatory `0.7378`).

**Live poll (2026-10-03 06:17 UTC):** Plate-slide M3 advanced to `108/2100`
with raw/mandatory rollout SR `0.0000`; its event stream continues to update
after `model_100.pt`.  PnP M11 advanced to `5268/7700` with raw rollout SR
`0.7460` and mandatory SR `0.7386`.
**Live poll (2026-10-03 06:19 UTC):** Plate-slide M3 advanced to `118/2100`;
rollout SR is still `0.0000`, but mean reward is `53.61` and mean episode
length `353.65`, indicating ongoing learning.  PnP M11 advanced to `5278/7700`
with raw rollout SR `0.7464` and mandatory SR `0.7390`.
**Live poll (2026-10-03 06:20 UTC):** Plate-slide M3 advanced to `128/2100`
and wrote `model_125.pt`; mean reward is `57.70`, while rollout raw/mandatory
SR remains `0.0000`.  PnP M11 advanced to `5288/7700` with raw rollout SR
`0.7468` and mandatory SR `0.7394`.
**Live poll (2026-10-03 06:22 UTC):** Plate-slide M3 advanced to `138/2100`
with rollout raw/mandatory SR `0.0000`; it remains active in the first tenth
of its budget.  PnP M11 advanced to `5298/7700` with raw rollout SR `0.7471`
and mandatory SR `0.7398`.

**Live poll (2026-10-03 06:25 UTC):** Plate-slide M3 reached `148/2100` and
has now written `model_150.pt`; rollout raw/mandatory SR remains `0.0000`.
PnP M11 advanced to `5318/7700` with raw rollout SR `0.7479` and mandatory SR
`0.7406`.
**Live poll (2026-10-03 06:23 UTC):** Plate-slide M3 advanced to `138/2100`
with mean reward `62.64` and mean episode length `367.85`; rollout raw/
mandatory SR remains `0.0000`, and the process is active.  PnP M11 advanced to
`5308/7700` with raw rollout SR `0.7475` and mandatory SR `0.7402`.
**Live poll (2026-10-03 06:26 UTC):** Plate-slide M3 advanced to `158/2100`
with mean reward `68.98` and mean episode length `377.28`; rollout raw/
mandatory SR remains `0.0000`, and the process is active.  PnP M11 advanced to
`5328/7700` with raw rollout SR `0.7482` and mandatory SR `0.7409`.
**Live poll (2026-10-03 06:28 UTC):** Plate-slide M3 advanced to `168/2100`
with mean episode length `378.95`, while rollout raw/mandatory SR remains
`0.0000`; training remains active.  PnP M11 advanced to `5338/7700` with raw
rollout SR `0.7485` and mandatory SR `0.7412`.
**Live poll (2026-10-03 06:30 UTC):** Plate-slide M3 advanced to `178/2100`
with mean episode length `379.78`; rollout raw/mandatory SR remains `0.0000`.
PnP M11 advanced to `5358/7700` with raw rollout SR `0.7492` and mandatory SR
`0.7419`.
**Live poll (2026-10-03 06:32 UTC):** Plate-slide M3 advanced to `188/2100`
with mean reward `73.23` and mean episode length `384.86`; rollout raw/
mandatory SR remains `0.0000`.  PnP M11 advanced to `5368/7700` with raw
rollout SR `0.7494` and mandatory SR `0.7422`.
**Live poll (2026-10-03 06:34 UTC):** Plate-slide M3 reached `198/2100` and
wrote `model_200.pt`; mean reward is `76.99`, while rollout raw/mandatory SR
remains `0.0000`.  PnP M11 remains healthy at its latest flushed `5378/7700`
metric (raw `0.7497`, mandatory `0.7424`).
**Live poll (2026-10-03 06:34 UTC):** Plate-slide M3 advanced to `208/2100`
with mean reward `81.85` and mean episode length `403.06`; rollout raw/
mandatory SR remains `0.0000`.  PnP M11 advanced to `5388/7700` with raw
rollout SR `0.7500` and mandatory SR `0.7428`.
**Live poll (2026-10-03 06:36 UTC):** Plate-slide M3 advanced to `218/2100`
with mean reward `80.62` and mean episode length `392.01`; rollout raw/
mandatory SR remains `0.0000`.  PnP M11 advanced to `5398/7700` with raw
rollout SR `0.7503` and mandatory SR `0.7431`.
**Live poll (2026-10-03 06:39 UTC):** Plate-slide M3 advanced to `228/2100`
and wrote `model_225.pt`; mean episode length is `399.00`, while rollout
raw/mandatory SR remains `0.0000`.  PnP M11 advanced to `5418/7700` with raw
rollout SR `0.7509` and mandatory SR `0.7437`.
**Live poll (2026-10-03 06:42 UTC):** Plate-slide M3 advanced to `248/2100`
and wrote `model_250.pt`; mean episode length is `384.98`, while rollout
raw/mandatory SR remains `0.0000`.  PnP M11 advanced to `5448/7700` with raw
rollout SR `0.7518` and mandatory SR `0.7446`.
**Live poll (2026-10-03 06:42 UTC):** Plate-slide M3 is at `248/2100` with
mean reward `81.04` and mean episode length `384.98`; rollout raw/mandatory SR
remains `0.0000`.  PnP M11 advanced to `5448/7700` with raw rollout SR `0.7518`
and mandatory SR `0.7446`.
**Live poll (2026-10-03 06:43 UTC):** Plate-slide M3 advanced to `258/2100`
with mean reward `85.89` and mean episode length `397.77`; rollout raw/
mandatory SR remains `0.0000`.  PnP M11 remains healthy at `5448/7700` with
raw rollout SR `0.7518` and mandatory SR `0.7446`.
**Live poll (2026-10-03 06:45 UTC):** Plate-slide M3 advanced to `268/2100`
with mean episode length `399.19`; rollout raw/mandatory SR remains `0.0000`.
PnP M11 advanced to `5468/7700` with raw rollout SR `0.7522` and mandatory SR
`0.7450`.
**Live poll (2026-10-03 06:45 UTC):** Plate-slide M3 advanced to `268/2100`
and wrote `model_275.pt`; mean reward is `88.81`, while rollout raw/mandatory
SR remains `0.0000`.  PnP M11 remains healthy at `5468/7700` with raw rollout
SR `0.7522` and mandatory SR `0.7450`.
**Live poll (2026-10-03 06:47 UTC):** Plate-slide M3 remains active at its
latest flushed `278/2100` metric, with mean episode length `392.57` and
rollout raw/mandatory SR `0.0000`; PnP M11 remains active at `5478/7700` with
raw rollout SR `0.7525` and mandatory SR `0.7453`.
**Live poll (2026-10-03 06:50 UTC):** Plate-slide M3 advanced to `298/2100`
(the next saved checkpoint is pending), with mean episode length `383.31` and
rollout raw/mandatory SR `0.0000`.  PnP M11 advanced to `5508/7700` with raw
rollout SR `0.7533` and mandatory SR `0.7461`; both trainers remain healthy.
**Live poll (2026-10-03 06:55 UTC):** Plate-slide M3 advanced to `328/2100`
and has written `model_325.pt`; rollout raw/mandatory SR is still `0.0000`.
PnP M11 advanced to `5538/7700`, with raw rollout SR `0.7541` and mandatory
SR `0.7470`; `model_5525.pt` is present and both processes remain healthy.
**Live poll (2026-10-03 06:56 UTC):** Plate-slide M3 advanced to `338/2100`
with mean episode length `378.25`; its rollout raw/mandatory SR remains
`0.0000`.  PnP M11 advanced to `5548/7700` with raw rollout SR `0.7545` and
mandatory SR `0.7474`.  No formal endpoint summary has appeared yet.
**Live poll (2026-10-03 06:57 UTC):** PnP M11 advanced to `5558/7700`
with raw rollout SR `0.7547` and mandatory SR `0.7476`; plate-slide M3 remains
active at its latest flushed `338/2100` metric.  The rescue queue still waits
for the M11 training summary, so no M12--M15 or heterogeneous row has started.
**Live poll (2026-10-03 06:59 UTC):** PnP M11 advanced to `5568/7700` with
raw rollout SR `0.7550` and mandatory SR `0.7479`.  Plate-slide M3's process
remains active at its latest flushed `348/2100` metric with rollout SR `0.0000`;
no formal endpoint summary or downstream queue start is present yet.
**Live poll (2026-10-03 07:00 UTC):** Both trainers are advancing and writing
checkpoints: PnP M11 is at `5578/7700` (`model_5575.pt`), with raw rollout SR
`0.7552` and mandatory SR `0.7482`; plate-slide M3 is at `358/2100`
(`model_350.pt`) with rollout SR `0.0000`.  The low M3 rate is still an early
training observation, not a terminal result.
**Live poll (2026-10-03 07:01 UTC):** PnP M11 advanced to `5588/7700` with
raw rollout SR `0.7554` and mandatory SR `0.7483`; plate-slide M3 advanced to
`368/2100` with rollout raw/mandatory SR `0.0000`.  Both processes are alive,
and the downstream rescue queue remains waiting for M11's summary.
**Live poll (2026-10-03 07:03 UTC):** PnP M11 advanced to `5598/7700` with
raw rollout SR `0.7556` and mandatory SR `0.7486`; plate-slide M3 advanced to
`378/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no new formal summary or downstream start has appeared.
**Live poll (2026-10-03 07:04 UTC):** PnP M11 advanced to `5608/7700` with
raw rollout SR `0.7559` and mandatory SR `0.7488`; plate-slide M3 advanced to
`388/2100` with rollout raw/mandatory SR `0.0000`.  Both processes remain
healthy and the rescue queue continues waiting for M11's formal summary.
**Live poll (2026-10-03 07:06 UTC):** PnP M11 advanced to `5618/7700` with
raw rollout SR `0.7561` and mandatory SR `0.7491`; plate-slide M3 advanced to
`398/2100` with rollout raw/mandatory SR `0.0000`.  No endpoint summaries or
downstream starts have appeared.
**Live poll (2026-10-03 07:07 UTC):** PnP M11 advanced to `5628/7700` with
raw rollout SR `0.7563` and mandatory SR `0.7493`.  Plate-slide M3's process is
still alive and GPU-active; its latest flushed metric remains `398/2100` with
rollout SR `0.0000`, so this is not treated as an exit or endpoint.
**Live poll (2026-10-03 07:09 UTC):** PnP M11 advanced to `5638/7700` with
raw rollout SR `0.7565` and mandatory SR `0.7495`; plate-slide M3 advanced to
`408/2100` with rollout raw/mandatory SR `0.0000`.  Both processes remain
alive, and no downstream queue has started.
**Live poll (2026-10-03 07:10 UTC):** PnP M11 advanced to `5658/7700` with
raw rollout SR `0.7569` and mandatory SR `0.7499`; plate-slide M3 advanced to
`418/2100` with rollout raw/mandatory SR `0.0000`.  The queue is still waiting
for M11's formal training summary.
**Live poll (2026-10-03 07:12 UTC):** PnP M11 advanced to `5668/7700` with
raw rollout SR `0.7571` and mandatory SR `0.7501`; plate-slide M3 advanced to
`428/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy and the downstream queue is still waiting for M11 completion.
**Live poll (2026-10-03 07:14 UTC):** PnP M11 advanced to `5678/7700` with
raw rollout SR `0.7573` and mandatory SR `0.7503`; plate-slide M3 advanced to
`438/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; the rescue queue still waits for M11's formal summary.
**Live poll (2026-10-03 07:15 UTC):** PnP M11 advanced to `5688/7700` with
raw rollout SR `0.7575` and mandatory SR `0.7505`; plate-slide M3 advanced to
`448/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy, and the downstream queue has not yet started.
**Live poll (2026-10-03 07:17 UTC):** PnP M11 advanced to `5698/7700` with
raw rollout SR `0.7576` and mandatory SR `0.7507`; plate-slide M3 advanced to
`458/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs are still waiting for M11's formal summary.
**Live poll (2026-10-03 07:18 UTC):** PnP M11 advanced to `5708/7700` with
raw rollout SR `0.7578` and mandatory SR `0.7509`; plate-slide M3 advanced to
`468/2100` with rollout raw/mandatory SR `0.0000`.  Both processes remain
healthy and the downstream queue has not started.
**Live poll (2026-10-03 07:20 UTC):** PnP M11 advanced to `5718/7700` with
raw rollout SR `0.7580` and mandatory SR `0.7510`; plate-slide M3 advanced to
`478/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy and the downstream queue continues waiting for M11's summary.
**Live poll (2026-10-03 07:21 UTC):** PnP M11 advanced to `5728/7700` with
raw rollout SR `0.7581` and mandatory SR `0.7512`; plate-slide M3 advanced to
`488/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs have not started.
**Live poll (2026-10-03 07:51 UTC):** PnP M11 advanced to `5948/7700` with
raw rollout SR `0.7624` and mandatory SR `0.7557`; plate-slide M3 advanced to
`688/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 07:53 UTC):** PnP M11 advanced to `5968/7700` with
raw rollout SR `0.7628` and mandatory SR `0.7561`; plate-slide M3 advanced to
`698/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs are still waiting for M11's formal summary.
**Live poll (2026-10-03 07:55 UTC):** PnP M11 advanced to `5978/7700` with
raw rollout SR `0.7630` and mandatory SR `0.7563`; plate-slide M3 advanced to
`708/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 07:56 UTC):** PnP M11 advanced to `5988/7700` with
raw rollout SR `0.7632` and mandatory SR `0.7565`; plate-slide M3 advanced to
`718/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs are still waiting for M11's summary.
**Live poll (2026-10-03 07:58 UTC):** PnP M11 advanced to `6008/7700` with
raw rollout SR `0.7635` and mandatory SR `0.7569`; plate-slide M3 advanced to
`728/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 08:00 UTC):** PnP M11 advanced to `6018/7700` with
raw rollout SR `0.7638` and mandatory SR `0.7571`; plate-slide M3 advanced to
`748/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy, and the downstream queue is still waiting for M11's summary.
**Live poll (2026-10-03 08:01 UTC):** PnP M11 advanced to `6028/7700` with
raw rollout SR `0.7639` and mandatory SR `0.7573`; plate-slide M3 advanced to
`758/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 08:03 UTC):** PnP M11 advanced to `6038/7700` with
raw rollout SR `0.7640` and mandatory SR `0.7574`; plate-slide M3 advanced to
`768/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs are still waiting for M11's formal summary.
**Live poll (2026-10-03 08:05 UTC):** PnP M11 advanced to `6048/7700` with
raw rollout SR `0.7642` and mandatory SR `0.7576`; plate-slide M3 advanced to
`778/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 08:06 UTC):** PnP M11 advanced to `6068/7700` with
raw rollout SR `0.7644` and mandatory SR `0.7578`; plate-slide M3 advanced to
`788/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy, and the downstream queue is still waiting for M11's summary.
**Live poll (2026-10-03 08:08 UTC):** PnP M11 advanced to `6078/7700` with
raw rollout SR `0.7645` and mandatory SR `0.7579`; plate-slide M3 advanced to
`798/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 08:09 UTC):** PnP M11 advanced to `6088/7700` with
raw rollout SR `0.7647` and mandatory SR `0.7581`; plate-slide M3 advanced to
`808/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs are still waiting for M11's summary.
**Live poll (2026-10-03 07:23 UTC):** PnP M11 advanced to `5748/7700` with
raw rollout SR `0.7584` and mandatory SR `0.7515`; plate-slide M3 advanced to
`498/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream job or formal endpoint summary has appeared.
**Live poll (2026-10-03 07:24 UTC):** PnP M11 advanced to `5758/7700` with
raw rollout SR `0.7586` and mandatory SR `0.7517`; plate-slide M3 advanced to
`508/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs are still waiting for M11's formal summary.
**Live poll (2026-10-03 07:26 UTC):** PnP M11 advanced to `5768/7700` with
raw rollout SR `0.7587` and mandatory SR `0.7519`; plate-slide M3 advanced to
`518/2100` with rollout raw/mandatory SR `0.0000`.  Both processes remain
healthy and no downstream job has started.
**Live poll (2026-10-03 07:27 UTC):** PnP M11 advanced to `5778/7700` with
raw rollout SR `0.7590` and mandatory SR `0.7521`; plate-slide M3 advanced to
`528/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy and the downstream queue is still waiting for M11's summary.
**Live poll (2026-10-03 07:29 UTC):** PnP M11 advanced to `5788/7700` with
raw rollout SR `0.7592` and mandatory SR `0.7523`; plate-slide M3 advanced to
`538/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 07:31 UTC):** PnP M11 advanced to `5798/7700` with
raw rollout SR `0.7594` and mandatory SR `0.7526`; plate-slide M3 advanced to
`548/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; the rescue queue still waits for M11's formal summary.
**Live poll (2026-10-03 07:32 UTC):** PnP M11 advanced to `5808/7700` with
raw rollout SR `0.7596` and mandatory SR `0.7528`; plate-slide M3 advanced to
`558/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream start or endpoint summary has appeared.
**Live poll (2026-10-03 07:34 UTC):** PnP M11 advanced to `5828/7700` with
raw rollout SR `0.7599` and mandatory SR `0.7531`; plate-slide M3 advanced to
`568/2100` with rollout raw/mandatory SR `0.0000`.  Both processes remain
healthy; the rescue queue continues waiting for M11's formal summary.
**Live poll (2026-10-03 07:35 UTC):** PnP M11 advanced to `5838/7700` with
raw rollout SR `0.7601` and mandatory SR `0.7533`; plate-slide M3 advanced to
`578/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy, and M12--M15 have not started yet.
**Live poll (2026-10-03 07:37 UTC):** PnP M11 advanced to `5848/7700` with
raw rollout SR `0.7603` and mandatory SR `0.7535`; plate-slide M3 advanced to
`588/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 07:39 UTC):** PnP M11 advanced to `5858/7700` with
raw rollout SR `0.7605` and mandatory SR `0.7538`; plate-slide M3 advanced to
`598/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs continue waiting for M11's formal summary.
**Live poll (2026-10-03 07:40 UTC):** PnP M11 advanced to `5868/7700` with
raw rollout SR `0.7607` and mandatory SR `0.7539`; plate-slide M3 advanced to
`608/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy and the rescue queue is still waiting for M11 completion.
**Live poll (2026-10-03 07:42 UTC):** PnP M11 advanced to `5878/7700` with
raw rollout SR `0.7609` and mandatory SR `0.7541`; plate-slide M3 advanced to
`618/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs have not started.
**Live poll (2026-10-03 07:43 UTC):** PnP M11 advanced to `5898/7700` with
raw rollout SR `0.7613` and mandatory SR `0.7545`; plate-slide M3 advanced to
`628/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy, and the downstream queue is still waiting for M11's summary.
**Live poll (2026-10-03 07:45 UTC):** PnP M11 advanced to `5908/7700` with
raw rollout SR `0.7615` and mandatory SR `0.7548`; plate-slide M3 advanced to
`638/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; no downstream queue start or formal endpoint summary has appeared.
**Live poll (2026-10-03 07:46 UTC):** PnP M11 advanced to `5918/7700` with
raw rollout SR `0.7617` and mandatory SR `0.7550`; plate-slide M3 advanced to
`648/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs are still waiting for M11's formal summary.
**Live poll (2026-10-03 07:48 UTC):** PnP M11 advanced to `5928/7700` with
raw rollout SR `0.7620` and mandatory SR `0.7553`; plate-slide M3 advanced to
`658/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy, and the queue is still waiting for M11 completion.
**Live poll (2026-10-03 07:50 UTC):** PnP M11 advanced to `5938/7700` with
raw rollout SR `0.7622` and mandatory SR `0.7555`; plate-slide M3 advanced to
`668/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy; downstream jobs have not started.
**Live poll (2026-10-03 08:12 UTC):** PnP M11 advanced to `6108/7700` with
raw rollout SR `0.7650` and mandatory SR `0.7585`; plate-slide M3 advanced to
`828/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
healthy.  The rescue queue is still waiting for M11's `training_summary`, so
M12--M15 and the heterogeneous jobs have not started yet.
**Live poll (2026-10-03 08:14 UTC):** PnP M11 advanced to `6128/7700` with
raw rollout SR `0.7656` and mandatory SR `0.7591`; plate-slide M3 advanced to
`838/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU trainers are
still making progress; the rescue queue continues waiting for M11's summary.
**Diagnostic (2026-10-03 08:17 UTC):** A CPU evaluation of the in-progress
plate-slide M3 checkpoint `model_800.pt` on 64 held-out episodes completed
the first two macros in all episodes but completed the third in none: strict
prefix counts `[64,64,0]`, raw/mandatory full SR `0/64`.  This localizes the
current failure to the newly appended third suffix rather than the first
macro or the first handoff; it is an intermediate checkpoint and not a final
composition result.
**Diagnostic (2026-10-03 08:19 UTC):** A CPU evaluation of the in-progress
PnP M11 checkpoint `model_6100.pt` on 64 held-out episodes completed 62/64
full chains (raw and mandatory SR `0.96875`).  Prefix counts were
`[64,64,63,63,62,62,62,62,62,62,62]`, indicating a small progressive loss
across later macros rather than a collapse at the first macro.  This is an
intermediate diagnostic, not the final 256-episode endpoint.
**Live poll (2026-10-03 08:22 UTC):** PnP M11 advanced to `6188/7700` with
raw rollout SR `0.7667` and mandatory SR `0.7602`; plate-slide M3 advanced to
`898/2100` with rollout raw/mandatory SR `0.0000`.  Both processes remain
healthy, while the rescue queue still waits for M11's training summary.
**Debug queue (2026-10-03 08:27 UTC):** Because the M3 intermediate and an
older full M3 run both showed prefix `[all,all,0]`, an isolated third-suffix
control was queued as `plate_slide_v2_suffix2_isolated_M1_stage16_seed42_base700_env128`.
It uses `source_macro_index=2`, the same v2 target/reward, stage width 16 and
700 iterations, with the object initialized at the predecessor target and the
arm reset only for this standalone learnability diagnostic.  It waits for the
main M3 summary and GPU availability; it is not a replacement for the
continuous no-reset composition row.
**Live poll (2026-10-03 08:28 UTC):** PnP M11 advanced to `6228/7700` with
raw rollout SR `0.7676` and mandatory SR `0.7611`; plate-slide M3 advanced to
`938/2100` with rollout raw/mandatory SR `0.0000`.  Both main trainers remain
healthy.  The M11-dependent queues and the isolated suffix control are still
waiting on their prerequisites; no later composition job has started.
**Live poll (2026-10-03 08:29 UTC):** PnP M11 advanced to `6238/7700` with
raw rollout SR `0.7677` and mandatory SR `0.7613`; plate-slide M3 advanced to
`948/2100` with rollout raw/mandatory SR `0.0000`.  No M11 or M12 summary is
present yet, and the downstream queue remains in its prerequisite-wait state.
**Live poll (2026-10-03 08:30 UTC):** PnP M11 advanced to `6248/7700` with
raw rollout SR `0.7679` and mandatory SR `0.7615`; plate-slide M3 advanced to
`958/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU trainers are
actively progressing; no M11 summary or downstream M12 start is present.
**Live poll (2026-10-03 08:32 UTC):** PnP M11 advanced to `6258/7700` with
raw rollout SR `0.7682` and mandatory SR `0.7618`; plate-slide M3 advanced to
`968/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU trainers remain
active; M11 still has no summary and M12 has not started.
**Live poll (2026-10-03 08:34 UTC):** PnP M11 advanced to `6268/7700` with
raw rollout SR `0.7684` and mandatory SR `0.7620`; plate-slide M3 advanced to
`978/2100` with rollout raw/mandatory SR `0.0000`.  Both main trainers remain
healthy; the M11 summary and all downstream starts are still pending.
**Live poll (2026-10-03 08:35 UTC):** PnP M11 advanced to `6278/7700` with
raw rollout SR `0.7687` and mandatory SR `0.7623`; plate-slide M3 advanced to
`988/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active and no M11 summary is present yet.
**Live poll (2026-10-03 08:37 UTC):** PnP M11 advanced to `6288/7700` with
raw rollout SR `0.7689` and mandatory SR `0.7625`; plate-slide M3 advanced to
`998/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU jobs remain
healthy; the downstream queue is still waiting for M11 completion.
**Live poll (2026-10-03 08:39 UTC):** PnP M11 advanced to `6298/7700` with
raw rollout SR `0.7692` and mandatory SR `0.7628`; plate-slide M3 advanced to
`1008/2100` with rollout raw/mandatory SR `0.0000`.  M11 is now past 80% of
its budget but has no final summary yet; downstream jobs remain queued.
**Live poll (2026-10-03 08:40 UTC):** PnP M11 advanced to `6318/7700` with
raw rollout SR `0.7697` and mandatory SR `0.7633`; plate-slide M3 advanced to
`1018/2100` with rollout raw/mandatory SR `0.0000`.  M11 remains active and
the rescue queue is still waiting for its summary.
**Live poll (2026-10-03 08:42 UTC):** PnP M11 advanced to `6328/7700` with
raw rollout SR `0.7698` and mandatory SR `0.7635`; plate-slide M3 advanced to
`1038/2100` with rollout raw/mandatory SR `0.0000`.  M11 is still training;
no final summary or downstream M12 start exists yet.
**Live poll (2026-10-03 08:44 UTC):** PnP M11 advanced to `6338/7700` with
raw rollout SR `0.7700` and mandatory SR `0.7636`; plate-slide M3 advanced to
`1048/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
the M11-dependent queue is still waiting for a formal summary.
**Live poll (2026-10-03 08:46 UTC):** PnP M11 advanced to `6358/7700` with
raw rollout SR `0.7704` and mandatory SR `0.7641`; plate-slide M3 advanced to
`1058/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active, and the M11 summary is still pending.
**Live poll (2026-10-03 08:47 UTC):** PnP M11 advanced to `6368/7700` with
raw rollout SR `0.7707` and mandatory SR `0.7643`; plate-slide M3 advanced to
`1068/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU jobs remain
active; no M11 summary or downstream start is present.
**Live poll (2026-10-03 08:49 UTC):** PnP M11 advanced to `6378/7700` with
raw rollout SR `0.7709` and mandatory SR `0.7645`; plate-slide M3 advanced to
`1078/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active and the downstream queue is still waiting for M11 completion.
**Live poll (2026-10-03 08:51 UTC):** PnP M11 advanced to `6388/7700` with
raw rollout SR `0.7710` and mandatory SR `0.7647`; plate-slide M3 advanced to
`1088/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain healthy;
the M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 08:52 UTC):** PnP M11 advanced to `6398/7700` with
raw rollout SR `0.7712` and mandatory SR `0.7649`; plate-slide M3 advanced to
`1098/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; the queue is still waiting for M11's summary.
**Live poll (2026-10-03 08:53 UTC):** PnP M11 advanced to `6408/7700` with
raw rollout SR `0.7714` and mandatory SR `0.7651`; plate-slide M3 advanced to
`1108/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU trainers remain
active; no M11 summary or downstream M12 start is present.
**Live poll (2026-10-03 08:55 UTC):** PnP M11 advanced to `6418/7700` with
raw rollout SR `0.7716` and mandatory SR `0.7653`; plate-slide M3 advanced to
`1118/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; the M11-dependent queue is still waiting for completion.
**Live poll (2026-10-03 08:57 UTC):** PnP M11 advanced to `6428/7700` with
raw rollout SR `0.7719` and mandatory SR `0.7656`; plate-slide M3 advanced to
`1128/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
the M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 08:57 UTC, later sample):** PnP M11 advanced to
`6438/7700` with raw rollout SR `0.7722` and mandatory SR `0.7659`; plate-slide
M3 advanced to `1138/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs
remain active; no M11 summary is present yet.
**Live poll (2026-10-03 08:59 UTC):** PnP M11 advanced to `6448/7700` with
raw rollout SR `0.7724` and mandatory SR `0.7661`; plate-slide M3 advanced to
`1148/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; the M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:00 UTC):** PnP M11 advanced to `6458/7700` with
raw rollout SR `0.7726` and mandatory SR `0.7663`; plate-slide M3 advanced to
`1158/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU jobs remain
active; the M11-dependent queue is still waiting for a summary.
**Live poll (2026-10-03 09:02 UTC):** PnP M11 advanced to `6478/7700` with
raw rollout SR `0.7729` and mandatory SR `0.7667`; plate-slide M3 advanced to
`1168/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 has not yet produced its summary.
**Live poll (2026-10-03 09:04 UTC):** PnP M11 advanced to `6488/7700` with
raw rollout SR `0.7731` and mandatory SR `0.7668`; plate-slide M3 advanced to
`1188/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
the M11 summary and all downstream starts are still pending.
**Live poll (2026-10-03 09:06 UTC):** PnP M11 advanced to `6498/7700` with
raw rollout SR `0.7733` and mandatory SR `0.7670`; plate-slide M3 advanced to
`1198/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; no M11 summary or downstream start is present.
**Live poll (2026-10-03 09:08 UTC):** PnP M11 advanced to `6508/7700` with
raw rollout SR `0.7734` and mandatory SR `0.7672`; plate-slide M3 advanced to
`1208/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 has not yet produced its summary.
**Live poll (2026-10-03 09:09 UTC):** PnP M11 advanced to `6528/7700` with
raw rollout SR `0.7737` and mandatory SR `0.7675`; plate-slide M3 advanced to
`1218/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU jobs remain
active; the M11 summary is still pending.
**Live poll (2026-10-03 09:11 UTC):** PnP M11 advanced to `6538/7700` with
raw rollout SR `0.7739` and mandatory SR `0.7677`; plate-slide M3 advanced to
`1228/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; the downstream queue is still waiting for M11 completion.
**Live poll (2026-10-03 09:13 UTC):** PnP M11 advanced to `6548/7700` with
raw rollout SR `0.7741` and mandatory SR `0.7679`; plate-slide M3 advanced to
`1238/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:15 UTC):** PnP M11 advanced to `6558/7700` with
raw rollout SR `0.7743` and mandatory SR `0.7681`; plate-slide M3 advanced to
`1258/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; the M11-dependent queue is still waiting for completion.
**Live poll (2026-10-03 09:16 UTC):** PnP M11 advanced to `6578/7700` with
raw rollout SR `0.7746` and mandatory SR `0.7684`; plate-slide M3 advanced to
`1268/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:18 UTC):** PnP M11 advanced to `6588/7700` with
raw rollout SR `0.7748` and mandatory SR `0.7686`; plate-slide M3 advanced to
`1278/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; the downstream queue is still waiting for M11 completion.
**Live poll (2026-10-03 09:20 UTC):** PnP M11 advanced to `6598/7700` with
raw rollout SR `0.7749` and mandatory SR `0.7687`; plate-slide M3 advanced to
`1298/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
the M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:22 UTC):** PnP M11 advanced to `6618/7700` with
raw rollout SR `0.7753` and mandatory SR `0.7692`; plate-slide M3 advanced to
`1308/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 summary and all downstream starts are still pending.
**Live poll (2026-10-03 09:24 UTC):** PnP M11 advanced to `6628/7700` with
raw rollout SR `0.7756` and mandatory SR `0.7694`; plate-slide M3 advanced to
`1318/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU trainers remain
active; no M11 summary or downstream start exists yet.
**Live poll (2026-10-03 09:25 UTC):** PnP M11 advanced to `6638/7700` with
raw rollout SR `0.7757` and mandatory SR `0.7696`; plate-slide M3 advanced to
`1328/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
the M11 summary is still pending.
**Live poll (2026-10-03 09:27 UTC):** PnP M11 advanced to `6648/7700` with
raw rollout SR `0.7759` and mandatory SR `0.7698`; plate-slide M3 advanced to
`1338/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:29 UTC):** PnP M11 advanced to `6668/7700` with
raw rollout SR `0.7763` and mandatory SR `0.7702`; plate-slide M3 advanced to
`1348/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU trainers remain
active; the M11-dependent queue is still waiting for completion.
**Live poll (2026-10-03 09:30 UTC):** PnP M11 advanced to `6678/7700` with
raw rollout SR `0.7765` and mandatory SR `0.7704`; plate-slide M3 advanced to
`1368/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:32 UTC):** PnP M11 advanced to `6688/7700` with
raw rollout SR `0.7767` and mandatory SR `0.7706`; plate-slide M3 advanced to
`1378/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:34 UTC):** PnP M11 advanced to `6698/7700` with
raw rollout SR `0.7769` and mandatory SR `0.7707`; plate-slide M3 advanced to
`1388/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
the M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:36 UTC):** PnP M11 advanced to `6718/7700` with
raw rollout SR `0.7772` and mandatory SR `0.7711`; plate-slide M3 advanced to
`1398/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU jobs remain
active; the M11 summary is still pending.
**Live poll (2026-10-03 09:37 UTC):** PnP M11 advanced to `6728/7700` with
raw rollout SR `0.7775` and mandatory SR `0.7713`; plate-slide M3 advanced to
`1408/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:39 UTC):** PnP M11 advanced to `6738/7700` with
raw rollout SR `0.7777` and mandatory SR `0.7716`; plate-slide M3 advanced to
`1428/2100` with rollout raw/mandatory SR `0.0000`.  Both jobs remain active;
the M11 summary is still pending.
**Live poll (2026-10-03 09:41 UTC):** PnP M11 advanced to `6748/7700` with
raw rollout SR `0.7779` and mandatory SR `0.7717`; plate-slide M3 advanced to
`1438/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; M11 summary and downstream starts are still pending.
**Live poll (2026-10-03 09:43 UTC):** PnP M11 advanced to `6768/7700` with
raw rollout SR `0.7782` and mandatory SR `0.7721`; plate-slide M3 advanced to
`1458/2100` with rollout raw/mandatory SR `0.0000`.  Both GPU jobs remain
active; the M11 summary is still pending.
**Live poll (2026-10-03 09:45 UTC):** PnP M11 advanced to `6778/7700` with
raw rollout SR `0.7784` and mandatory SR `0.7723`; plate-slide M3 advanced to
`1468/2100` with rollout raw/mandatory SR `0.0000`.  Both trainers remain
active; no M11 summary or downstream start is present.
**Live poll (2026-10-03 09:50 UTC):** PnP M11 advanced to `6818/7700` with
raw rollout SR `0.7791` and mandatory SR `0.7730`; plate-slide M3 advanced to
`1518/2100` with rollout raw/mandatory SR `0.0000`. Both trainers remain
active; the rescue queue is still waiting for the M11 training summary.
**Checkpoint validation (2026-10-03):** The intermediate PnP M11
`model_1250.pt` that reached raw SR `251/256 = 0.9805` on seed `1042` was
re-evaluated deterministically for 256 episodes on independent seed `2042`.
It obtained raw/mandatory SR `241/256 = 0.9414`; per-macro rates were
`[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9805,0.9414,0.9609]`. Thus the
checkpoint is a strong intermediate result but not a stable 0.98 M11 result.
**Live poll (2026-10-03 09:56 UTC):** PnP M11 advanced to `6858/7700` with
raw rollout SR `0.7799` and mandatory SR `0.7739`; plate-slide M3 advanced to
`1558/2100` with rollout raw/mandatory SR `0.0000`. Both trainers remain
active and the downstream rescue queue is still waiting for the M11 summary.
**Live poll (2026-10-03 09:58 UTC):** PnP M11 advanced to `6878/7700` with
raw rollout SR `0.7803` and mandatory SR `0.7743`; plate-slide M3 advanced to
`1568/2100` with rollout raw/mandatory SR `0.0000`. Both GPU trainers are
still alive; the rescue queue remains blocked only on M11 completion.
**Live poll (2026-10-03 11:09 UTC):** PnP M11 advanced to `7388/7700` with
raw rollout SR `0.7872` and mandatory SR `0.7815`. Plate-slide M3 completed its
`2100/2100` training iterations with rollout SR `0.0000`; its held-out
256-episode evaluation has been launched and is still pending. The M11 rescue
queue remains waiting for the M11 training summary before starting M12--M15.
**Plate-slide M3 final evaluation (2026-10-03 11:11 UTC):** The completed
plate-slide M3 checkpoint was evaluated deterministically for 256 episodes
(seed `1042`) and obtained raw/mandatory SR `0.0/0.0`. This confirms that the
current M3 slide protocol does not yet provide a usable long-composition
result; the isolated suffix diagnostic remains important before interpreting
this as a composition boundary.
**Live poll (2026-10-03 11:11 UTC):** PnP M11 advanced to `7408/7700` with
raw rollout SR `0.7874` and mandatory SR `0.7816`. The suffix-isolation queue
is now unblocked by the M3 summary but is waiting for GPU 3 while M11 is still
running; no duplicate suffix run was launched.
**Live poll (2026-10-03 11:30 UTC):** PnP M11 advanced to `7538/7700` with
raw rollout SR `0.7859` and mandatory SR `0.7802`; the process remains alive
and no final `training_summary.json` exists yet. The next PnP jobs (M12--M15)
have therefore not started. The alternative plate-slide sweep has started
M4 (`128/2800`, rollout raw/mandatory SR `0.0000/0.0000`) while its earlier
M3 run remains recorded as a diagnostic failure rather than a clean boundary.
**Live poll (2026-10-03 11:32 UTC):** PnP M11 reached `7558/7700` with
rollout raw/mandatory SR `0.7856/0.7800`; its final summary is still absent.
Plate-slide M4 reached `148/2800` and remains in the early-training phase with
rollout raw/mandatory SR `0.0000/0.0000`. The rescue queue continues to wait
for M11 and has not launched M12.
**PnP M11 final training summary (2026-10-03 11:53 UTC):** The scratch run
completed all `7700` iterations (`167777` training episodes). Its aggregate
raw/mandatory success rates were `0.780643/0.775106`; macro-wise raw rates
decreased from `0.9797` for macro 1 to `0.8334` for macro 11, while prefix
rates decreased from `0.9797` to `0.7806`. Thus the earlier isolated 0.98
checkpoint is not representative of the converged long-run result.
**PnP M11 held-out evaluation (2026-10-03 11:55 UTC):** The final checkpoint
(`model_7699.pt`) was evaluated with the deterministic rule gate for 256
requested episodes (257 completed by the vectorized evaluator). It obtained
raw/mandatory SR `0.871595` (`224/257`), with the failure concentrated at
macro 6 and inherited by the remaining prefix. The rescue queue has now
queued M12; GPU 3 is temporarily occupied by the already-authorized isolated
plate-slide suffix diagnostic, so M12 will start when that diagnostic frees
the device.
**Live poll (2026-10-03 11:57 UTC):** The isolated plate-slide suffix diagnostic
has reached `38/700` and the full plate-slide M4 run has reached `308/2800`;
both remain active. The rescue queue still reports GPU 3 occupied by the
suffix diagnostic and has not duplicated or preempted either run; M12 remains
queued.
**Live poll (2026-10-03 11:58 UTC):** The suffix diagnostic reached `48/700`
and plate-slide M4 remains at `308/2800`; both processes are alive. M12 is
still queued behind the suffix diagnostic on GPU 3.
**Live poll (2026-10-03 12:03 UTC):** The isolated suffix run reached
`108/700`, while plate-slide M4 reached `348/2800`. Both are still actively
training; the PnP M12 queue remains waiting for GPU 3.
**Live poll (2026-10-03 12:06 UTC):** The isolated suffix run reached
`148/700`; plate-slide M4 reached `368/2800`. M12 has not started because
the suffix diagnostic still owns GPU 3.
**Live poll (2026-10-03 12:08 UTC):** The isolated suffix run reached
`168/700`, and plate-slide M4 reached `378/2800`; M12 remains queued behind
the active suffix diagnostic on GPU 3.
**Live poll (2026-10-03 12:09 UTC):** The suffix diagnostic reached `188/700`
and plate-slide M4 reached `388/2800`; the M12 queue continues to wait for
GPU 3 and no PnP M12 process exists yet.
**Live poll (2026-10-03 12:11 UTC):** The suffix diagnostic reached `198/700`.
Plate-slide M4 remains at `388/2800` during its current checkpoint interval;
M12 is still queued behind the GPU-3 suffix run.
**Live poll (2026-10-03 12:12 UTC):** The suffix diagnostic reached `218/700`
and plate-slide M4 reached `398/2800`; both trainers remain alive. M12 still
has no training process because GPU 3 is occupied by the suffix run.
**Live poll (2026-10-03 12:14 UTC):** The suffix diagnostic reached `238/700`
and plate-slide M4 reached `408/2800`; M12 remains queued while GPU 3 is
occupied by the suffix run.
**Live poll (2026-10-03 12:15 UTC):** The suffix diagnostic reached `258/700`
and plate-slide M4 reached `418/2800`; the GPU-3 wait for M12 is unchanged.
**Live poll (2026-10-03 12:16 UTC):** The suffix diagnostic reached `268/700`
and plate-slide M4 reached `428/2800`; M12 remains queued behind the active
GPU-3 suffix diagnostic.
**Live poll (2026-10-03 12:18 UTC):** The suffix diagnostic reached `288/700`
and plate-slide M4 reached `438/2800`; GPU 3 remains occupied and M12 has not
started.
**Live poll (2026-10-03 12:19 UTC):** The suffix diagnostic reached `308/700`
and plate-slide M4 reached `448/2800`; M12 remains queued behind the active
GPU-3 suffix diagnostic.
**Live poll (2026-10-03 12:21 UTC):** The suffix diagnostic reached `328/700`
and plate-slide M4 reached `458/2800`; GPU 3 remains occupied, so M12 has not
started.
**Live poll (2026-10-03 12:22 UTC):** The suffix diagnostic reached `338/700`
and plate-slide M4 reached `468/2800`; the M12 queue is still waiting on GPU 3.
**Live poll (2026-10-03 12:24 UTC):** The suffix diagnostic reached `358/700`
and plate-slide M4 reached `478/2800`; M12 remains queued behind the suffix
diagnostic.
**Live poll (2026-10-03 12:25 UTC):** The suffix diagnostic reached `378/700`
and plate-slide M4 reached `488/2800`; GPU 3 remains occupied and M12 has not
started.
**Live poll (2026-10-03 12:26 UTC):** The suffix diagnostic reached `398/700`
and plate-slide M4 reached `498/2800`; the rescue queue is still waiting for
GPU 3 before starting M12.
**Live poll (2026-10-03 12:28 UTC):** The suffix diagnostic reached `408/700`
and plate-slide M4 reached `508/2800`; GPU 3 remains occupied and M12 has not
started.
**Live poll (2026-10-03 12:29 UTC):** The suffix diagnostic reached `428/700`
and plate-slide M4 reached `518/2800`; M12 remains queued behind the active
suffix run.
**Live poll (2026-10-03 12:31 UTC):** The suffix diagnostic reached `448/700`
and plate-slide M4 reached `528/2800`; the suffix process remains alive and
M12 is still waiting for GPU 3.
**Live poll (2026-10-03 12:32 UTC):** The suffix diagnostic reached `468/700`
and plate-slide M4 reached `538/2800`; M12 remains queued behind the suffix
diagnostic on GPU 3.
**Live poll (2026-10-03 12:34 UTC):** The suffix diagnostic reached `478/700`
and plate-slide M4 reached `548/2800`; GPU 3 is still occupied and M12 has
not started.
**Live poll (2026-10-03 12:35 UTC):** The suffix diagnostic reached `498/700`
and plate-slide M4 reached `558/2800`; M12 remains queued behind the active
GPU-3 suffix run.
**Live poll (2026-10-03 12:37 UTC):** The suffix diagnostic reached `518/700`
and plate-slide M4 reached `568/2800`; GPU 3 remains occupied, so M12 has not
started.
**Live poll (2026-10-03 12:38 UTC):** The suffix diagnostic reached `528/700`
and plate-slide M4 reached `578/2800`; M12 remains queued behind suffix on GPU 3.
**Live poll (2026-10-03 12:40 UTC):** The suffix diagnostic reached `548/700`
and plate-slide M4 reached `588/2800`; GPU 3 remains occupied and M12 has not
started.
**Live poll (2026-10-03 12:41 UTC):** The suffix diagnostic reached `568/700`
and plate-slide M4 reached `598/2800`; M12 remains queued behind suffix on GPU 3.
**Live poll (2026-10-03 12:43 UTC):** The suffix diagnostic reached `578/700`
and plate-slide M4 reached `608/2800`; GPU 3 remains occupied and M12 has not
started.
**Live poll (2026-10-03 12:44 UTC):** The suffix diagnostic reached `598/700`
and plate-slide M4 reached `618/2800`; M12 remains queued behind the suffix
diagnostic on GPU 3.
**Live poll (2026-10-03 12:46 UTC):** The suffix diagnostic reached `618/700`
and plate-slide M4 reached `628/2800`; M12 remains queued while suffix uses
GPU 3.
**Live poll (2026-10-03 12:47 UTC):** The suffix diagnostic reached `628/700`
and plate-slide M4 reached `638/2800`; GPU 3 is still occupied and M12 has not
started.
**Live poll (2026-10-03 12:49 UTC):** The suffix diagnostic reached `648/700`
and plate-slide M4 reached `648/2800`; M12 remains queued behind suffix on
GPU 3.
**Live poll (2026-10-03 12:50 UTC):** The suffix diagnostic reached `668/700`
and plate-slide M4 reached `658/2800`; M12 remains queued while suffix uses
GPU 3.
**Plate-slide suffix diagnostic completed (2026-10-03 12:54 UTC):** The
isolated source-macro-index 2 run completed all `700` iterations and produced
its training summary (training raw/mandatory SR `0.0/0.0`). This is a diagnostic
for the previously failing third suffix, not a replacement composition result;
its held-out evaluation is queued separately.
**PnP M12 started (2026-10-03 12:55 UTC):** After the suffix trainer exited,
the rescue queue launched scratch PnP M12 on GPU 3 with the same v10 geometry,
stage-onehot, and `700*M` budget protocol. The M12 training log is now live.
**Live poll (2026-10-03 12:58 UTC):** PnP M12 reached `8/8400` iterations
(early rollout raw/mandatory `0.0000/0.0000`), while plate-slide M4 reached
`718/2800`. Both trainers are alive; the early M12 zero is not an endpoint
measurement.
**Live poll (2026-10-03 12:59 UTC):** PnP M12 reached `18/8400` and remains in
the initialization phase with rollout raw/mandatory `0.0000/0.0000`; plate-slide
M4 reached `728/2800`. Both jobs remain active.
**Live poll (2026-10-03 13:01 UTC):** PnP M12 reached `28/8400`, still in the
initialization phase with rollout raw/mandatory `0.0000/0.0000`; plate-slide M4
reached `738/2800`. Both processes remain alive.
**Live poll (2026-10-03 13:03 UTC):** PnP M12 reached `38/8400` with rollout
raw `0.0000`; plate-slide M4 reached `748/2800`. Both jobs remain active and
M12 is still in its early initialization phase.
**Live poll (2026-10-03 13:04 UTC):** PnP M12 reached `48/8400` with rollout
raw/mandatory `0.0000/0.0000`; plate-slide M4 reached `758/2800`. Both remain
active, and M12 is still far too early for an endpoint interpretation.
**Live poll (2026-10-03 13:06 UTC):** PnP M12 reached `58/8400` with rollout
raw `0.0000`; plate-slide M4 reached `768/2800`. Both processes remain alive,
and M12 is still in early training.
**Live poll (2026-10-03 13:08 UTC):** PnP M12 reached `68/8400` with rollout
raw/mandatory `0.0000/0.0000`; plate-slide M4 reached `788/2800`. M12 remains
in its early initialization phase.
**Live poll (2026-10-03 13:09 UTC):** PnP M12 reached `78/8400` with rollout
raw `0.0000`; plate-slide M4 reached `798/2800`. M12 remains queued behind the
active GPU-3 run.
**Live poll (2026-10-03 13:11 UTC):** PnP M12 reached `88/8400` with rollout
raw `0.0000`; plate-slide M4 reached `808/2800`. Both jobs remain active and
M12 is still in early training.
**Live poll (2026-10-03 13:13 UTC):** PnP M12 reached `98/8400` with rollout
raw `0.0000`; plate-slide M4 reached `818/2800`. Both processes remain active;
M12 is still in early training.
**Live poll (2026-10-03 13:14 UTC):** PnP M12 reached `108/8400` with rollout
raw `0.0000`; plate-slide M4 reached `828/2800`. Both trainers remain active
and M12 is still in early training.
**Live poll (2026-10-03 13:16 UTC):** PnP M12 reached `118/8400` with rollout
raw `0.0000`; plate-slide M4 reached `848/2800`. Both processes remain active
and M12 is still in the initialization phase.
**Live poll (2026-10-03 13:18 UTC):** PnP M12 reached `128/8400` with rollout
raw `0.0000`; plate-slide M4 reached `858/2800`. Both remain active and M12 is
still in early training.
**Live poll (2026-10-03 13:19 UTC):** PnP M12 reached `138/8400` with rollout
raw `0.0000`; plate-slide M4 reached `868/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:21 UTC):** PnP M12 reached `148/8400` with rollout
raw `0.0000`; plate-slide M4 reached `878/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:22 UTC):** PnP M12 reached `158/8400` with rollout
raw `0.0000`; plate-slide M4 reached `888/2800`. Both remain active and M12
has not yet left the initialization phase.
**Live poll (2026-10-03 13:24 UTC):** PnP M12 reached `168/8400` with rollout
raw `0.0000`; plate-slide M4 reached `898/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:26 UTC):** PnP M12 reached `178/8400` with rollout
raw `0.0000`; plate-slide M4 reached `908/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:27 UTC):** PnP M12 reached `188/8400` with rollout
raw `0.0000`; plate-slide M4 reached `918/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:29 UTC):** PnP M12 reached `198/8400` with rollout
raw `0.0000`; plate-slide M4 reached `928/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:30 UTC):** PnP M12 reached `208/8400` with rollout
raw `0.0000`; plate-slide M4 reached `938/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:32 UTC):** PnP M12 reached `218/8400` with rollout
raw `0.0000`; plate-slide M4 reached `958/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:34 UTC):** PnP M12 reached `228/8400` with rollout
raw `0.0000`; plate-slide M4 reached `968/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:35 UTC):** PnP M12 reached `238/8400` with rollout
raw `0.0000`; plate-slide M4 reached `978/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:37 UTC):** PnP M12 reached `248/8400` with rollout
raw `0.0000`; plate-slide M4 reached `988/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:39 UTC):** PnP M12 reached `258/8400` with rollout
raw `0.0000`; plate-slide M4 reached `998/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:40 UTC):** PnP M12 reached `268/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1008/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:42 UTC):** PnP M12 reached `278/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1018/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:44 UTC):** PnP M12 reached `288/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1028/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:45 UTC):** PnP M12 reached `298/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1038/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:48 UTC):** PnP M12 reached `318/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1058/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:49 UTC):** PnP M12 reached `328/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1068/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:51 UTC):** PnP M12 reached `338/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1078/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:53 UTC):** PnP M12 reached `348/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1088/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:54 UTC):** PnP M12 reached `358/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1098/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:56 UTC):** PnP M12 reached `368/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1108/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 13:58 UTC):** PnP M12 reached `378/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1118/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 14:00 UTC):** PnP M12 reached `388/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1138/2800`. Both remain active and M12
is still in early training.
**Live poll (2026-10-03 14:03 UTC):** PnP M12 reached `408/8400` with rollout
raw `0.0000`; plate-slide M4 reached `1168/2800`. Both remain active and M12
is still in early training.

**Live poll (2026-10-03 14:20 UTC):** PnP M12 reached `518/8400` with rollout
raw `0.0002`; plate-slide v2 M4 reached `1288/2800` with rollout raw `0.0000`.
Both trainers remain active.  The new v3 M1--M5 queue is alive and waiting for
the existing GPU1 worker.

**Alternative-task protocol audit (2026-10-03):** The v2 plate-slide chain is
not yet a clean boundary study: isolated training reaches 100% on target
`(0.08,0.88)`, while the isolated suffix targeting `(-0.08,0.88)` fails, and
the formal M3 fails exactly at macro 3.  This is a reachability/trajectory
confound rather than evidence that composition itself fails.  A v3 protocol
was therefore added with central, distinct targets
`(0,0.85)->(0.08,0.88)->(0.08,0.80)->(0.08,0.90)->(0,0.80)`; it retains the
same plate, three atomic stages, observation/action interface, no-reset
semantics, and `700*M` scratch budget.  A full M1--M5 v3 queue was launched on
GPU 1 after a one-iteration CLI smoke test passed; the queue is currently
waiting for the known Qwen worker on that GPU.

**Live poll (2026-10-03 14:22 UTC):** PnP M12 reached `528/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1308/2800` with rollout raw
`0.0000`.  The v3 reachable-task queue remains alive at its GPU1 wait guard;
the known worker is still using about 33 GB on that device.

**Live poll (2026-10-03 14:29 UTC):** PnP M12 reached `568/8400` with rollout
raw `0.0001`; old plate-slide v2 M4 reached `1348/2800` with rollout raw
`0.0000`.  Both trainers remain alive with no terminal summary.  The v3
queue remains in its safe GPU1 wait loop while the persistent worker rotates.

**Live poll (2026-10-03 14:26 UTC):** PnP M12 reached `548/8400` with rollout
raw `0.0001`; old plate-slide v2 M4 reached `1338/2800` with rollout raw
`0.0000`.  Neither has produced a terminal summary yet.  The v3 queue is
still alive and waiting on the rotating GPU1 worker.

**Macro-task choice rationale:** Plate-slide is the cleanest second homogeneous
primitive for a no-reset study: the same puck remains present after each
move, every subtask has the same approach/contact/transport-to-target
structure, and only the destination changes.  One-shot state-changing tasks
such as button-press or drawer-open would either become undefined after the
first macro or require an implicit reset, so they are reserved for the
heterogeneous study rather than used as the homogeneous replacement.

**Live poll (2026-10-03 14:31 UTC):** PnP M12 reached `578/8400` with rollout
raw `0.0001`; old plate-slide v2 M4 reached `1368/2800` with rollout raw
`0.0000`.  No M12--M15 summary exists yet, and v3 M1 is still waiting for
GPU1 rather than running under a reduced budget.

**Live poll (2026-10-03 14:51 UTC):** PnP M12 reached `698/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1508/2800` with rollout raw
`0.0000`; v3 M1 reached `148/700` with rollout raw SR `0.9348`.  The v3
single-macro sanity run remains healthy and is not yet at its endpoint.

**Live poll (2026-10-03 14:38 UTC):** PnP M12 reached `628/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1418/2800` with rollout raw
`0.0000`; the new v3 M1 reached `28/700` with rollout raw `0.0011`.  All
three trainers remain alive.

**Live poll (2026-10-03 14:42 UTC):** PnP M12 reached `648/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1438/2800` with rollout raw
`0.0000`; v3 M1 reached `68/700` and rollout raw SR increased to `0.7608`.
The v3 trajectory remains trainable under the full protocol.

**Live poll (2026-10-03 14:43 UTC):** PnP M12 reached `658/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1458/2800` with rollout raw
`0.0000`; v3 M1 reached `78/700` with rollout raw SR `0.8074`.  Its
`model_75.pt` checkpoint is present and training continues toward `model_699.pt`.

**Live poll (2026-10-03 14:47 UTC):** PnP M12 reached `678/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1478/2800` with rollout raw
`0.0000`; v3 M1 reached `118/700` with rollout raw SR `0.9029`.  The new
protocol continues to improve before its final held-out evaluation.

**Live poll (2026-10-03 14:53 UTC):** PnP M12 reached `708/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1518/2800` with rollout raw
`0.0000`; v3 M1 reached `158/700` with rollout raw SR `0.9417`.  M1 remains
healthy and has not yet reached its endpoint.

**Live poll (2026-10-03 14:54 UTC):** PnP M12 reached `718/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1528/2800` with rollout raw
`0.0000`; v3 M1 reached `178/700` with rollout raw SR `0.9522`.  The v3
M1 curve remains healthy and no endpoint evaluation has started yet.

**Live poll (2026-10-03 14:49 UTC):** PnP M12 reached `688/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1488/2800` with rollout raw
`0.0000`; v3 M1 reached `128/700` with rollout raw SR `0.9160`.  No endpoint
summary exists yet, so M2 has not been released.

**Live poll (2026-10-03 14:45 UTC):** PnP M12 reached `668/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1468/2800` with rollout raw
`0.0000`; v3 M1 reached `98/700` with rollout raw SR `0.8656`.  The v3 run
has reached `model_100.pt` and remains on track for the full endpoint.

**Live poll (2026-10-03 14:40 UTC):** PnP M12 reached `638/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1428/2800` with rollout raw
`0.0000`; v3 M1 reached `48/700` and its rollout raw SR rose to `0.5836`.
The v3 M1 run is therefore learning normally; this is still an intermediate
rollout value, not the held-out endpoint.

**Live poll (2026-10-03 14:36 UTC):** PnP M12 reached `618/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1408/2800` with rollout raw
`0.0000`.  GPU1 is now available and the reachability-screened v3 M1 has
started at `18/700` with rollout raw `0.0020`, using the full 128-environment
and 700-iteration setting.

**Live poll (2026-10-03 14:34 UTC):** PnP M12 reached `598/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1388/2800` with rollout raw
`0.0000`.  Both processes remain alive, no new summary or evaluation file has
appeared, and v3 M1 is still waiting for GPU1.

**Live poll (2026-10-03 14:32 UTC):** PnP M12 reached `588/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1378/2800` with rollout raw
`0.0000`.  GPU1 is still occupied by the known worker (`~33 GB`), so the v3
queue remains safely waiting.

**Live poll (2026-10-03 14:56 UTC):** PnP M12 reached `728/8400` with rollout
raw `0.0002`; old plate-slide v2 M4 reached `1548/2800` with rollout raw
`0.0000`; v3 M1 reached `198/700` with rollout raw SR `0.9596`.  Checkpoints
through `model_175.pt` are present and the full-budget run remains healthy.

**Live poll (2026-10-03 14:59 UTC):** PnP M12 reached `748/8400` with rollout
raw `0.0004`; old plate-slide v2 M4 reached `1558/2800` with rollout raw
`0.0000`; v3 M1 reached `218/700` with rollout raw SR `0.9650`.  No endpoint
summary or held-out evaluation has appeared yet.

**Live poll (2026-10-03 16:00 UTC):** v3 plate-slide M1 completed the full 700
iterations and its CPU held-out evaluation is `256/256` (`SR=1.0`, strict
prefix `1.0`).  The v3 queue has advanced to M2, but M2 is waiting for the
unrelated Qwen worker occupying GPU1.  PnP M12 and old v2 M4 remain active in
their respective queues.

**Live poll (2026-10-03 16:03 UTC):** PnP M12 reached `1168/8400` with rollout
raw `0.0445`; old plate-slide v2 M4 reached `1998/2800` with rollout raw
`0.0000`.  The clean v3 M2 queue remains alive and waiting on GPU1; no process
was restarted or duplicated.

**Live poll (2026-10-03 16:08 UTC):** PnP M12 reached `1198/8400` with rollout
raw `0.0535`.  The clean v3 M2 queue is still waiting on the unrelated GPU1
Qwen worker.  The obsolete v2 plate-slide lane was stopped after its target-
reachability failure was established; it is not used for the composition claim.

**Live poll (2026-10-03 16:10 UTC):** PnP M12 reached `1208/8400` with rollout
raw `0.0560`; it remains the only active composition trainer on its GPU.  No
M12 terminal summary exists yet, and the clean v3 M2 queue continues waiting
on the external GPU1 worker; no heterogeneous run has started yet.

**Live poll (2026-10-03 16:10 UTC, later poll):** PnP M12 reached `1218/8400`
with rollout raw `0.0594` and remains healthy.  No M12 summary exists yet; v3
M2 is still waiting on GPU1, and the heterogeneous lane has not started.

**Live poll (2026-10-03 16:12 UTC):** PnP M12 advanced to `1228/8400` with
rollout raw `0.0624`; its process is alive and training normally.  No terminal
summary exists yet; GPU1 remains occupied by the external Qwen worker, so v3
M2 and the downstream heterogeneous queue remain pending.

**Live poll (2026-10-03 16:14 UTC):** PnP M12 reached `1238/8400` with rollout
raw `0.0658`; the trainer remains alive.  M12 has not reached its declared
`8400`-iteration endpoint, so M13–M15 are not started yet.

**Live poll (2026-10-03 16:15 UTC):** PnP M12 advanced to `1248/8400` with
rollout raw `0.0691`; no training or held-out summary exists yet.  The v3
alternative-task queue remains alive at its GPU1 wait guard.

**Live poll (2026-10-03 16:17 UTC):** PnP M12 reached `1258/8400` with rollout
raw `0.0728`; the trainer remains alive and no terminal summary exists.  M13–M15,
the v3 M2–M5 continuation, and the heterogeneous lane remain queued in order.

**Live poll (2026-10-03 16:18 UTC):** PnP M12 advanced to `1268/8400` with
rollout raw `0.0763`; the process remains healthy.  No endpoint summary exists,
so downstream training has not started.

**Live poll (2026-10-03 16:20 UTC):** PnP M12 reached `1278/8400` with rollout
raw `0.0801`; the process remains alive and no terminal summary has appeared.
The downstream queues remain pending rather than being duplicated.

**Live poll (2026-10-03 16:22 UTC):** PnP M12 advanced to `1298/8400` with
rollout raw `0.0868`; the trainer is still alive and the M12 summary directory
is empty.  M13–M15 and later macro/heterogeneous runs remain queued.

**Live poll (2026-10-03 16:24 UTC):** PnP M12 reached `1308/8400` with rollout
raw `0.0899`; the process remains healthy.  No terminal summary or downstream
job has appeared yet.

**Live poll (2026-10-03 16:26 UTC):** PnP M12 reached `1318/8400` with rollout
raw `0.0934`; the process remains alive.  There is still no M12 summary, so no
downstream job has been launched.

**Live poll (2026-10-03 16:28 UTC):** PnP M12 reached `1328/8400` with rollout
raw `0.0964`; training remains healthy and no terminal summary has appeared.

**Live poll (2026-10-03 16:29 UTC):** PnP M12 advanced to `1338/8400` with
rollout raw `0.0999`; its process remains alive and no summary exists yet.

**Live poll (2026-10-03 16:31 UTC):** PnP M12 reached `1348/8400` with rollout
raw `0.1037`; the trainer remains healthy and downstream jobs are still queued.

**Live poll (2026-10-03 16:34 UTC):** PnP M12 reached `1368/8400` with rollout
raw `0.1106`; GPU3 remains active and the trainer is alive.  No terminal summary
has appeared yet.

**Live poll (2026-10-03 16:36 UTC):** PnP M12 advanced to `1388/8400` with
rollout raw `0.1184`; the trainer remains healthy and no terminal summary exists.

**Live poll (2026-10-03 16:37 UTC):** PnP M12 reached `1398/8400` with rollout
raw `0.1224`; the process remains alive and the downstream queue is unchanged.

**Live poll (2026-10-03 16:39 UTC):** PnP M12 advanced to `1408/8400` with
rollout raw `0.1268`; the trainer remains healthy and no endpoint summary exists.

**Live poll (2026-10-03 16:41 UTC):** PnP M12 reached `1418/8400` with rollout
raw `0.1309`; the process remains alive and downstream runs are still queued.

**Live poll (2026-10-03 16:42 UTC):** PnP M12 advanced to `1428/8400` with
rollout raw `0.1351`; training remains healthy and no terminal summary exists.

**Live poll (2026-10-03 16:44 UTC):** PnP M12 reached `1438/8400` with rollout
raw `0.1394`; the process remains alive and downstream jobs remain queued.

**Live poll (2026-10-03 16:47 UTC):** PnP M12 reached `1458/8400` with rollout
raw `0.1480`; the trainer remains alive and no terminal summary exists yet.

**Live poll (2026-10-03 16:49 UTC):** PnP M12 reached `1478/8400` with rollout
raw `0.1564`; the trainer remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 16:51 UTC):** PnP M12 advanced to `1488/8400` with
rollout raw `0.1606`; the process remains alive and no terminal summary exists.

**Live poll (2026-10-03 16:53 UTC):** PnP M12 reached `1498/8400` with rollout
raw `0.1647`; the trainer remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 16:54 UTC):** PnP M12 advanced to `1508/8400` with
rollout raw `0.1688`; the process remains alive and no terminal summary exists.

**Live poll (2026-10-03 16:56 UTC):** PnP M12 reached `1518/8400` with rollout
raw `0.1727`; the trainer remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 16:58 UTC):** PnP M12 advanced to `1528/8400` with
rollout raw `0.1769`; the process remains alive and no terminal summary exists.

**Live poll (2026-10-03 16:59 UTC):** PnP M12 reached `1538/8400` with rollout
raw `0.1814`; training remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 17:02 UTC):** PnP M12 reached `1558/8400` with rollout
raw `0.1899`; the process remains alive and no terminal summary exists.

**Live poll (2026-10-03 17:05 UTC):** PnP M12 advanced to `1578/8400` with
rollout raw `0.1986`; the trainer remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 17:06 UTC):** PnP M12 reached `1588/8400` with rollout
raw `0.2030`; process `1372142` is still alive.  No M12 training/evaluation
summary exists yet, and the v3 alternative queue is still waiting on GPU1.

**Live poll (2026-10-03 17:08 UTC):** PnP M12 advanced to `1598/8400` with
rollout raw `0.2070`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 17:11 UTC):** PnP M12 reached `1618/8400` with rollout
raw `0.2155`; training remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 17:14 UTC):** PnP M12 advanced to `1638/8400` with
rollout raw `0.2239`; process `1372142` remains alive (temporarily sleeping
between updates) and no terminal summary exists.

**Live poll (2026-10-03 17:16 UTC):** PnP M12 reached `1648/8400` with rollout
raw `0.2279`; the process remains healthy and no terminal summary exists.

**Live poll (2026-10-03 17:18 UTC):** PnP M12 advanced to `1658/8400` with
rollout raw `0.2318`; the trainer remains alive and M13–M15 remain queued.

**Live poll (2026-10-03 17:20 UTC):** PnP M12 reached `1678/8400` with rollout
raw `0.2397`; the process remains healthy and no terminal summary exists.

**Live poll (2026-10-03 17:22 UTC):** PnP M12 advanced to `1688/8400` with
rollout raw `0.2433`; the process remains alive and M13–M15 remain queued.

**Live poll (2026-10-03 17:25 UTC):** PnP M12 reached `1708/8400` with rollout
raw `0.2509`; the trainer remains healthy and no terminal summary exists.

**Live poll (2026-10-03 17:28 UTC):** PnP M12 advanced to `1728/8400` with
rollout raw `0.2592`; the process remains alive and M13–M15 remain queued.

**Live poll (2026-10-03 17:30 UTC):** PnP M12 reached `1738/8400` with rollout
raw `0.2629`; the trainer remains healthy and no terminal summary exists.

**Live poll (2026-10-03 17:33 UTC):** PnP M12 advanced to `1758/8400` with
rollout raw `0.2707`; the process remains alive and M13–M15 remain queued.

**Live poll (2026-10-03 17:34 UTC):** PnP M12 reached `1768/8400` with rollout
raw `0.2742`; the trainer remains healthy and no terminal summary exists.

**Live poll (2026-10-03 17:38 UTC):** PnP M12 reached `1788/8400` with rollout
raw `0.2819`; the process remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 17:41 UTC):** PnP M12 advanced to `1808/8400` with
rollout raw `0.2896`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 17:43 UTC):** PnP M12 reached `1828/8400` with rollout
raw `0.2973`; the process remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 17:45 UTC):** PnP M12 advanced to `1838/8400` with
rollout raw `0.3007`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 17:48 UTC):** PnP M12 reached `1858/8400` with rollout
raw `0.3077`; the process remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 17:52 UTC):** PnP M12 advanced to `1888/8400` with
rollout raw `0.3188`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 17:54 UTC):** PnP M12 reached `1898/8400` with rollout
raw `0.3225`; the process remains healthy and M13–M15 remain queued.

**Live poll (2026-10-03 17:55 UTC):** PnP M12 advanced to `1908/8400` with
rollout raw `0.3261`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 17:50 UTC):** PnP M12 advanced to `1878/8400` with
rollout raw `0.3150`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 17:59 UTC):** PnP M12 reached `1938/8400` with rollout
raw `0.3364`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:01 UTC):** PnP M12 advanced to `1948/8400` with
rollout raw `0.3400`; the trainer remains alive. The alternative-task M2 queue
is still waiting for external GPU-1 worker `1426068` to release the device.

**Live poll (2026-10-03 18:03 UTC):** PnP M12 advanced to `1958/8400` with
rollout raw `0.3432`; no terminal summary exists yet. The alternative-task M2
queue remains blocked only by external GPU-1 worker `1426068`.

**Live poll (2026-10-03 18:05 UTC):** PnP M12 advanced to `1968/8400` with
rollout raw `0.3466`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:06 UTC):** PnP M12 advanced to `1978/8400` with
rollout raw `0.3501`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:08 UTC):** PnP M12 advanced to `1988/8400` with
rollout raw `0.3531`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:09 UTC):** PnP M12 advanced to `1998/8400` with
rollout raw `0.3565`; the trainer remains alive. M13 has not started.

**Live poll (2026-10-03 18:10 UTC):** PnP M12 advanced to `2008/8400` with
rollout raw `0.3595`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:12 UTC):** PnP M12 advanced to `2018/8400` with
rollout raw `0.3625`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:13 UTC):** PnP M12 advanced to `2028/8400` with
rollout raw `0.3655`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:15 UTC):** PnP M12 advanced to `2038/8400` with
rollout raw `0.3685`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:16 UTC):** PnP M12 advanced to `2048/8400` with
rollout raw `0.3715`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:18 UTC):** PnP M12 advanced to `2058/8400` with
rollout raw `0.3747`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:19 UTC):** PnP M12 advanced to `2068/8400` with
rollout raw `0.3777`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:21 UTC):** PnP M12 advanced to `2078/8400` with
rollout raw `0.3806`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:22 UTC):** PnP M12 advanced to `2088/8400` with
rollout raw `0.3834`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:24 UTC):** PnP M12 advanced to `2098/8400` with
rollout raw `0.3862`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:25 UTC):** PnP M12 advanced to `2108/8400` with
rollout raw `0.3889`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:26 UTC):** PnP M12 advanced to `2118/8400` with
rollout raw `0.3916`; the trainer remains alive and no terminal summary exists.

**Live poll (2026-10-03 18:28 UTC):** PnP M12 advanced to `2128/8400` with
rollout raw `0.3942` (mandatory rollout `0.3782`); the trainer remains alive
and the GPU process is active.

**Live poll (2026-10-03 18:30 UTC):** PnP M12 advanced to `2138/8400` with
rollout raw `0.3969` (mandatory rollout `0.3810`); the trainer remains alive.

**Live poll (2026-10-03 18:31 UTC):** PnP M12 advanced to `2148/8400` with
rollout raw `0.3998` (mandatory rollout `0.3839`); the trainer remains alive.

**Live poll (2026-10-03 18:33 UTC):** PnP M12 advanced to `2158/8400` with
rollout raw `0.4025` (mandatory rollout `0.3867`); the trainer remains alive
and no terminal summary exists.

**Live poll (2026-10-03 18:34 UTC):** PnP M12 advanced to `2168/8400` with
rollout raw `0.4054` (mandatory rollout `0.3896`); the trainer remains alive.

**Live poll (2026-10-03 18:36 UTC):** PnP M12 advanced to `2178/8400` with
rollout raw `0.4081` (mandatory rollout `0.3924`); the trainer remains alive.

**Live poll (2026-10-03 18:37 UTC):** PnP M12 advanced to `2188/8400` with
rollout raw `0.4109` (mandatory rollout `0.3951`); the trainer remains alive.

**Live poll (2026-10-03 18:39 UTC):** PnP M12 advanced to `2198/8400` with
rollout raw `0.4135` (mandatory rollout `0.3978`); the trainer remains alive.

**Live poll (2026-10-03 18:40 UTC):** PnP M12 advanced to `2208/8400` with
rollout raw `0.4165` (mandatory rollout `0.4008`); the trainer remains alive.

**Live poll (2026-10-03 18:42 UTC):** PnP M12 advanced to `2218/8400` with
rollout raw `0.4191` (mandatory rollout `0.4034`); the trainer remains alive.

**Live poll (2026-10-03 18:43 UTC):** PnP M12 advanced to `2228/8400` with
rollout raw `0.4217` (mandatory rollout `0.4061`); the trainer remains alive.

**Live poll (2026-10-03 18:45 UTC):** PnP M12 advanced to `2238/8400` with
rollout raw `0.4243` (mandatory rollout `0.4087`); the trainer remains alive.

**Live poll (2026-10-03 18:46 UTC):** PnP M12 advanced to `2248/8400` with
rollout raw `0.4268` (mandatory rollout `0.4112`); the trainer remains alive.

**Live poll (2026-10-03 18:48 UTC):** PnP M12 advanced to `2258/8400` with
rollout raw `0.4292` (mandatory rollout `0.4136`); the trainer remains alive.

**Live poll (2026-10-03 18:50 UTC):** PnP M12 advanced to `2268/8400` with
rollout raw `0.4315` (mandatory rollout `0.4159`); the trainer remains alive.

**Live poll (2026-10-03 18:51 UTC):** PnP M12 advanced to `2278/8400` with
rollout raw `0.4339` (mandatory rollout `0.4183`); the trainer remains alive.

**Live poll (2026-10-03 18:53 UTC):** PnP M12 advanced to `2288/8400` with
rollout raw `0.4361` (mandatory rollout `0.4204`); the trainer remains alive.

**Live poll (2026-10-03 18:54 UTC):** PnP M12 advanced to `2298/8400` with
rollout raw `0.4384` (mandatory rollout `0.4228`); the trainer remains alive.

**Live poll (2026-10-03 18:56 UTC):** PnP M12 advanced to `2308/8400` with
rollout raw `0.4409` (mandatory rollout `0.4253`); the trainer remains alive.

**Live poll (2026-10-03 18:57 UTC):** PnP M12 advanced to `2318/8400` with
rollout raw `0.4431` (mandatory rollout `0.4274`); the trainer remains alive.

**Live poll (2026-10-03 18:59 UTC):** PnP M12 advanced to `2328/8400` with
rollout raw `0.4453` (mandatory rollout `0.4296`); the trainer remains alive.

**Live poll (2026-10-03 19:00 UTC):** PnP M12 advanced to `2338/8400` with
rollout raw `0.4476` (mandatory rollout `0.4319`); the trainer remains alive.

**Live poll (2026-10-03 19:02 UTC):** PnP M12 advanced to `2348/8400` with
rollout raw `0.4499` (mandatory rollout `0.4342`); the trainer remains alive.

**Live poll (2026-10-03 19:03 UTC):** PnP M12 advanced to `2358/8400` with
rollout raw `0.4520` (mandatory rollout `0.4363`); the trainer remains alive.

**Live poll (2026-10-03 19:05 UTC):** PnP M12 advanced to `2368/8400` with
rollout raw `0.4540` (mandatory rollout `0.4383`); the trainer remains alive.

**Live poll (2026-10-03 19:06 UTC):** PnP M12 advanced to `2378/8400` with
rollout raw `0.4564` (mandatory rollout `0.4407`); the trainer remains alive.

**Live poll (2026-10-03 19:08 UTC):** PnP M12 advanced to `2388/8400` with
rollout raw `0.4586` (mandatory rollout `0.4429`); the trainer remains alive.

**Live poll (2026-10-03 19:09 UTC):** PnP M12 advanced to `2398/8400` with
rollout raw `0.4605` (mandatory rollout `0.4449`); the trainer remains alive.

**Live poll (2026-10-03 19:11 UTC):** PnP M12 advanced to `2408/8400` with
rollout raw `0.4627` (mandatory rollout `0.4470`); the trainer remains alive.

**Live poll (2026-10-03 19:12 UTC):** PnP M12 advanced to `2418/8400` with
rollout raw `0.4647` (mandatory rollout `0.4489`); the trainer remains alive.

**Live poll (2026-10-03 19:14 UTC):** PnP M12 advanced to `2428/8400` with
rollout raw `0.4666` (mandatory rollout `0.4507`); the trainer remains alive.

**Live poll (2026-10-03 19:15 UTC):** PnP M12 advanced to `2438/8400` with
rollout raw `0.4686` (mandatory rollout `0.4527`); the trainer remains alive.

**Live poll (2026-10-03 19:17 UTC):** PnP M12 advanced to `2448/8400` with
rollout raw `0.4708` (mandatory rollout `0.4548`); the trainer remains alive.

**Live poll (2026-10-03 19:18 UTC):** PnP M12 advanced to `2458/8400` with
rollout raw `0.4727` (mandatory rollout `0.4565`); the trainer remains alive.

**Live poll (2026-10-03 19:20 UTC):** PnP M12 advanced to `2468/8400` with
rollout raw `0.4744` (mandatory rollout `0.4582`); the trainer remains alive.

**Live poll (2026-10-03 19:21 UTC):** PnP M12 advanced to `2478/8400` with
rollout raw `0.4763` (mandatory rollout `0.4600`); the trainer remains alive.

**Live poll (2026-10-03 19:23 UTC):** PnP M12 advanced to `2488/8400` with
rollout raw `0.4781` (mandatory rollout `0.4617`); the trainer remains alive.

**Live poll (2026-10-03 19:24 UTC):** PnP M12 advanced to `2498/8400` with
rollout raw `0.4796` (mandatory rollout `0.4632`); the trainer remains alive.

**Live poll (2026-10-03 19:26 UTC):** PnP M12 advanced to `2508/8400` with
rollout raw `0.4812` (mandatory rollout `0.4648`); the trainer remains alive.

**Live poll (2026-10-03 19:27 UTC):** PnP M12 advanced to `2518/8400` with
rollout raw `0.4831` (mandatory rollout `0.4665`); the trainer remains alive.

**Live poll (2026-10-03 19:29 UTC):** PnP M12 advanced to `2528/8400` with
rollout raw `0.4847` (mandatory rollout `0.4681`); the trainer remains alive.

**Live poll (2026-10-03 19:31 UTC):** PnP M12 advanced to `2538/8400` with
rollout raw `0.4865` (mandatory rollout `0.4699`); the trainer remains alive.

**Live poll (2026-10-03 19:32 UTC):** PnP M12 advanced to `2548/8400` with
rollout raw `0.4883` (mandatory rollout `0.4716`); the trainer remains alive.

**Live poll (2026-10-03 19:34 UTC):** PnP M12 advanced to `2558/8400` with
rollout raw `0.4901` (mandatory rollout `0.4735`); the trainer remains alive.

**Live poll (2026-10-03 19:35 UTC):** PnP M12 advanced to `2568/8400` with
rollout raw `0.4919` (mandatory rollout `0.4753`); the trainer remains alive.

**Live poll (2026-10-03 19:37 UTC):** PnP M12 advanced to `2578/8400` with
rollout raw `0.4938` (mandatory rollout `0.4772`); the trainer remains alive.

**Live poll (2026-10-03 19:38 UTC):** PnP M12 advanced to `2588/8400` with
rollout raw `0.4955` (mandatory rollout `0.4789`); the trainer remains alive.

**Live poll (2026-10-03 19:40 UTC):** PnP M12 advanced to `2598/8400` with
rollout raw `0.4971` (mandatory rollout `0.4805`); the trainer remains alive.

**Live poll (2026-10-03 19:41 UTC):** PnP M12 advanced to `2608/8400` with
rollout raw `0.4989` (mandatory rollout `0.4824`); the trainer remains alive.

**Live poll (2026-10-03 19:43 UTC):** PnP M12 advanced to `2618/8400` with
rollout raw `0.5006` (mandatory rollout `0.4841`); the trainer remains alive.

**Live poll (2026-10-03 19:44 UTC):** PnP M12 advanced to `2628/8400` with
rollout raw `0.5021` (mandatory rollout `0.4856`); the trainer remains alive.

**Live poll (2026-10-03 19:46 UTC):** PnP M12 advanced to `2638/8400` with
rollout raw `0.5039` (mandatory rollout `0.4873`); the trainer remains alive.

**Live poll (2026-10-03 19:47 UTC):** PnP M12 advanced to `2648/8400` with
rollout raw `0.5054` (mandatory rollout `0.4889`); the trainer remains alive.

**Live poll (2026-10-03 19:49 UTC):** PnP M12 advanced to `2658/8400` with
rollout raw `0.5072` (mandatory rollout `0.4906`); the trainer remains alive.

**Live poll (2026-10-03 19:50 UTC):** PnP M12 advanced to `2668/8400` with
rollout raw `0.5087` (mandatory rollout `0.4921`); the trainer remains alive.

**Live poll (2026-10-03 19:52 UTC):** PnP M12 advanced to `2678/8400` with
rollout raw `0.5102` (mandatory rollout `0.4937`); the trainer remains alive.

**Live poll (2026-10-03 19:53 UTC):** PnP M12 advanced to `2688/8400` with
rollout raw `0.5117` (mandatory rollout `0.4952`); the trainer remains alive.

**Live poll (2026-10-03 19:55 UTC):** PnP M12 advanced to `2698/8400` with
rollout raw `0.5131` (mandatory rollout `0.4966`); the trainer remains alive.

**Live poll (2026-10-03 19:56 UTC):** PnP M12 advanced to `2708/8400` with
rollout raw `0.5143` (mandatory rollout `0.4978`); the trainer remains alive.

**Live poll (2026-10-03 19:58 UTC):** PnP M12 advanced to `2718/8400` with
rollout raw `0.5155` (mandatory rollout `0.4989`); the trainer remains alive.

**Live poll (2026-10-03 19:59 UTC):** PnP M12 advanced to `2728/8400` with
rollout raw `0.5169` (mandatory rollout `0.5004`); the trainer remains alive.

**Live poll (2026-10-03 20:01 UTC):** PnP M12 advanced to `2738/8400` with
rollout raw `0.5182` (mandatory rollout `0.5017`); the trainer remains alive.

**Live poll (2026-10-03 20:03 UTC):** PnP M12 advanced to `2748/8400` with
rollout raw `0.5194` (mandatory rollout `0.5029`); the trainer remains alive.

**Live poll (2026-10-03 20:04 UTC):** PnP M12 advanced to `2758/8400` with
rollout raw `0.5207` (mandatory rollout `0.5042`); the trainer remains alive.

**Live poll (2026-10-03 20:06 UTC):** PnP M12 advanced to `2768/8400` with
rollout raw `0.5220` (mandatory rollout `0.5055`); the trainer remains alive.

**Live poll (2026-10-03 20:07 UTC):** PnP M12 advanced to `2778/8400` with
rollout raw `0.5235` (mandatory rollout `0.5070`); the trainer remains alive.

**Live poll (2026-10-03 20:09 UTC):** PnP M12 advanced to `2788/8400` with
rollout raw `0.5248` (mandatory rollout `0.5083`); the trainer remains alive.

**Live poll (2026-10-03 20:10 UTC):** PnP M12 advanced to `2798/8400` with
rollout raw `0.5261` (mandatory rollout `0.5095`); the trainer remains alive.

**Live poll (2026-10-03 20:12 UTC):** PnP M12 advanced to `2808/8400` with
rollout raw `0.5276` (mandatory rollout `0.5110`); the trainer remains alive.

**Live poll (2026-10-03 20:13 UTC):** PnP M12 advanced to `2818/8400` with
rollout raw `0.5289` (mandatory rollout `0.5124`); the trainer remains alive.

**Live poll (2026-10-03 20:15 UTC):** PnP M12 advanced to `2828/8400` with
rollout raw `0.5304` (mandatory rollout `0.5138`); the trainer remains alive.

**Live poll (2026-10-03 20:18 UTC):** PnP M12 advanced to `2838/8400` with
rollout raw `0.5318` (mandatory rollout `0.5151`); the trainer remains alive.

**Live poll (2026-10-03 20:19 UTC):** PnP M12 advanced to `2848/8400` with
rollout raw `0.5330` (mandatory rollout `0.5162`); the trainer remains alive
at 99.4% CPU utilization.

**Live poll (2026-10-03 20:19 UTC, later poll):** PnP M12 advanced to
`2858/8400` with rollout raw `0.5343` (mandatory rollout `0.5176`); the
trainer remains alive at 99.4% CPU utilization.

**Live poll (2026-10-03 20:21 UTC):** PnP M12 advanced to `2868/8400` with
rollout raw `0.5356` (mandatory rollout `0.5189`); the log and checkpoint
timestamps are advancing and the trainer remains alive.

**Live poll (2026-10-03 20:25 UTC):** PnP M12 advanced to `2888/8400` with
rollout raw `0.5382` (mandatory rollout `0.5215`); checkpoint `model_2875.pt`
was written and the trainer remains alive.

**Queue adjustment (2026-10-03 20:27 UTC):** The v3 continuous plate-slide
lane was moved off the GPU-1 Qwen wait and relaunched on GPU3 with an explicit
guard to start only after the PnP M15 training summary exists. This avoids
competing with PnP M12--M15 while preserving the no-reset, full-budget M1--M5
alternative-task sweep.

**Live poll (2026-10-03 20:28 UTC):** PnP M12 advanced to `2908/8400` with
rollout raw `0.5411` (mandatory rollout `0.5243`); its trainer remains alive.
The replacement plate-slide queue is alive and waiting for the PnP M15 summary,
and has not started training prematurely.

**Live poll (2026-10-03 20:31 UTC):** PnP M12 advanced to `2928/8400` with
rollout raw `0.5436` (mandatory rollout `0.5267`); the log timestamp advanced
and the trainer remains alive at 99.4% CPU utilization.

**Live poll (2026-10-03 20:33 UTC):** PnP M12 advanced to `2938/8400` with
rollout raw `0.5449` (mandatory rollout `0.5281`); the trainer remains alive.

**Live poll (2026-10-03 20:34 UTC):** PnP M12 advanced to `2948/8400` with
rollout raw `0.5462` (mandatory rollout `0.5293`); the trainer remains alive.

**Live poll (2026-10-03 20:36 UTC):** PnP M12 advanced to `2958/8400` with
rollout raw `0.5472` (mandatory rollout `0.5303`); the trainer remains alive
at 99.4% CPU utilization.

**Live poll (2026-10-03 20:37 UTC):** PnP M12 advanced to `2968/8400` with
rollout raw `0.5484` (mandatory rollout `0.5315`); the trainer remains alive.

**Live poll (2026-10-03 20:39 UTC):** PnP M12 advanced to `2978/8400` with
rollout raw `0.5495` (mandatory rollout `0.5326`); the trainer remains alive
at 99.4% CPU utilization.

**Live poll (2026-10-03 20:40 UTC):** PnP M12 advanced to `2988/8400` with
rollout raw `0.5508` (mandatory rollout `0.5339`); the trainer remains alive.

**Live poll (2026-10-03 20:42 UTC):** PnP M12 advanced to `2998/8400` with
rollout raw `0.5520` (mandatory rollout `0.5352`); the trainer remains alive
at 99.4% CPU utilization.

**Live poll (2026-10-03 20:45 UTC):** PnP M12 advanced to `3018/8400` with
rollout raw `0.5545` (mandatory rollout `0.5377`); the trainer remains alive.

**Live poll (2026-10-03 20:47 UTC):** PnP M12 advanced to `3038/8400` with
rollout raw `0.5569` (mandatory rollout `0.5402`); the trainer remains alive
at 99.4% CPU utilization.

**Live poll (2026-10-03 20:50 UTC):** PnP M12 advanced to `3048/8400` with
rollout raw `0.5580` (mandatory rollout `0.5414`); the trainer remains alive.

**Live poll (2026-10-03 20:52 UTC):** PnP M12 advanced to `3068/8400` with
rollout raw `0.5606` (mandatory rollout `0.5440`); the trainer remains alive
at 99.4% CPU utilization.

**Live poll (2026-10-03 20:55 UTC):** PnP M12 advanced to `3088/8400` with
rollout raw `0.5627` (mandatory rollout `0.5461`); the trainer remains alive.

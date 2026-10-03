# Frame-VLM and GateVLM protocol

The release separates the two VLM roles.

## Frame-VLM reward compilation

The compiler receives the task description, the simulator-state interface, an
ordered frame packet, the current reward program, and training diagnostics. It
returns a validated `RewardProgram` with one stage record per active step. The
75 records in `task_registry/reward_machines/` are the frozen programs used by
the main-table recipe.

The compiler is run before PPO and the resulting JSON is frozen. A repair round
may replace the program during development, but a training run uses one fixed
program from start to finish.

## GateVLM transition verification

GateVLM is called only after the native adapter reports a persistent candidate
for the current stage. It receives the fixed-camera reference frame and the
ordered candidate frames, together with the task and stage contract. It emits
exactly one of `accept`, `reject`, or `abstain`.

The environment continues its fixed-rate rollout while a request is pending.
When a response arrives, the adapter checks episode ID, stage ID, and request
step before applying it. A stale response is discarded; an abstention blocks
the transition. GateVLM never supplies a dense scalar reward and never replaces
the policy's state observation.

## Local worker

The model worker is a JSON-lines process. One input line is one compile or gate
request and one output line is the corresponding program or decision. Connect
it through `SubprocessJSONLBackend`. Record the model identifier and prompt
version beside the frozen response cache. Model weights and credentials are
not bundled in this release.

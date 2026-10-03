#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROBOSTEP_PYTHON:-python}"
TASK="${ROBOSTEP_RECOVERY_TASK:-PickCube-v1}"
if [[ "$TASK" == "PickCube-v1" ]]; then
  CHECKPOINT="${ROBOSTEP_RECOVERY_CHECKPOINT:-$ROOT/artifacts/checkpoints/recovery/pickcube_checkpoint_best.pt}"
else
  CHECKPOINT="${ROBOSTEP_RECOVERY_CHECKPOINT:-$ROOT/artifacts/checkpoints/recovery/picksingleycb_checkpoint_best.pt}"
fi
MODEL="${ROBOSTEP_QWEN_MODEL:-$ROOT/models/Qwen3.5-9B}"
OUTPUT="${ROBOSTEP_OUTPUT_ROOT:-$ROOT/runs/recovery}/$TASK"
export PYTHONPATH="$ROOT/main_method/maniskill:$ROOT/main_method:${PYTHONPATH:-}"

exec "$PYTHON_BIN" "$ROOT/experiments/recovery/run_vlm_backward_recovery.py" \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --method bidirectional \
  --intervention regression \
  --severity "${ROBOSTEP_RECOVERY_SEVERITY:-0.05}" \
  --episodes "${ROBOSTEP_RECOVERY_EPISODES:-50}" \
  --device "${ROBOSTEP_DEVICE:-cuda:0}" \
  --qwen-model "$MODEL" \
  --qwen-worker "$ROOT/experiments/recovery/qwen3_5_worker_v5_rollback.py" \
  --output "$OUTPUT"

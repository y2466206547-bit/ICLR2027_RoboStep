#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python examples/train_task.py \
  --benchmark Meta-World \
  --task "${TASK_ID:-reach-v3}" \
  --seed "${SEED:-42}" \
  --total-timesteps "${TOTAL_TIMESTEPS:-4096}" \
  --output "${OUTPUT:-runs/metaworld_${TASK_ID:-reach-v3}_seed42.json}"

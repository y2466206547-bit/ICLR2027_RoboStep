#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python examples/train_task.py \
  --benchmark SoftGym \
  --task "${TASK_ID:-PassWater}" \
  --seed "${SEED:-42}" \
  --total-timesteps "${TOTAL_TIMESTEPS:-4096}" \
  --output "${OUTPUT:-runs/softgym_${TASK_ID:-PassWater}_seed42.json}"

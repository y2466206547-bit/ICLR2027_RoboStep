#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python examples/train_task.py \
  --benchmark ManiSkill \
  --task PickCube-v1 \
  --seed "${SEED:-42}" \
  --total-timesteps "${TOTAL_TIMESTEPS:-4096}" \
  --output "${OUTPUT:-runs/maniskill_pickcube_seed42.json}"

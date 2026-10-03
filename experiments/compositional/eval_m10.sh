#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROBOSTEP_PYTHON:-python}"
CHECKPOINT="${1:?usage: eval_m10.sh CHECKPOINT [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$ROOT/runs/compositional/m10_eval}"
export PYTHONPATH="$ROOT/main_method/metaworld:$ROOT/main_method:$ROOT/main_method/stage_policy:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m stage_reward.eval_bidirectional_pick_place "$CHECKPOINT" \
  --gate rule --episodes 256 --num-envs 64 --seed 1042 --output "$OUTPUT_DIR"

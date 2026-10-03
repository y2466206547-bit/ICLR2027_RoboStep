#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROBOSTEP_PYTHON:-python}"
OUTPUT_ROOT="${ROBOSTEP_OUTPUT_ROOT:-$ROOT/runs/compositional}"
export PYTHONPATH="$ROOT/main_method/metaworld:$ROOT/main_method:$ROOT/main_method/stage_policy:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m stage_reward.train_bidirectional_pick_place \
  --macro-count 10 \
  --source-macro-index 0 \
  --geometry-protocol tabletop_unique_chain_v8 \
  --gate rule \
  --num-envs "${ROBOSTEP_NUM_ENVS:-128}" \
  --base-iterations 700 \
  --steps-per-env 64 \
  --max-episode-length 250 \
  --seed 42 \
  --train-device "${ROBOSTEP_DEVICE:-cuda:0}" \
  --stage-id-size 31 \
  --raw-stage-onehot \
  --init-noise-std 0.45 \
  --entropy-coef 0.003 \
  --learning-rate 0.0005 \
  --save-interval 25 \
  --output-root "$OUTPUT_ROOT" \
  --run-name "unique10_v11_release_M10_stage31_seed42_base700"

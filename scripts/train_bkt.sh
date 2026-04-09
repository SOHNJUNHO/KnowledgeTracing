#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/train_bkt.sh <data_path> [extra args...]"
  exit 1
fi

DATA_PATH="$1"
shift

uv run python -m knowledge_tracing.main \
  --data_path "$DATA_PATH" \
  --batch_size 32 \
  --max_epochs 50 \
  --lr 1e-3 \
  --n_embd 64 \
  --hidden_dim 128 \
  --lambda_consistency 0.25 \
  --lambda_guess 0.25 \
  --lambda_slip 0.25 \
  "$@"

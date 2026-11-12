#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: bash scripts/run_alternative_models.sh DATA_DIR [OUT_DIR]"; exit 2; fi
DATA="$1"
OUT="runs/alternatives"
if [ "$#" -ge 2 ]; then OUT="$2"; fi
for METHOD in hmm timebin dtw; do
  python -m phaseforge.cli run-alt --data "$DATA" --config configs/kaggle_group.json --method "$METHOD" --output "$OUT/$METHOD"
done
echo "Saved research-only alternative model scores to: $OUT"

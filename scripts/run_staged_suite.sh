#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: bash scripts/run_staged_suite.sh DATA_DIR [OUT_DIR]"; exit 2; fi
DATA="$1"
OUT="runs/staged"
if [ "$#" -ge 2 ]; then OUT="$2"; fi
mkdir -p "$OUT"
python -m phaseforge.cli audit-kaggle --data "$DATA" | tee "$OUT/source_audit.json"
python -m phaseforge.cli run --data "$DATA" --config configs/kaggle_group.json --output "$OUT/group"
python -m phaseforge.cli run --data "$DATA" --config configs/kaggle_temporal.json --output "$OUT/temporal"
python -m phaseforge.cli fuse --runs "$OUT/group" "$OUT/temporal" --output "$OUT/fusion"
python -m phaseforge.cli proxy --data "$DATA" --config configs/kaggle_group.json --output "$OUT/proxy"
python -m phaseforge.cli gmm-study --data "$DATA" --config configs/kaggle_group.json --output "$OUT/behavior" --seeds 20
python -m phaseforge.cli sweep --data "$DATA" --config configs/kaggle_normalization_sweep.json --output "$OUT/normalization"
echo "Completed staged suite: $OUT"

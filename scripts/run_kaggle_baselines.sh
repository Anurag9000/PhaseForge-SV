#!/usr/bin/env bash
set -euo pipefail
DATA="${1:?Usage: scripts/run_kaggle_baselines.sh /path/to/kaggle [run_dir]}"
OUT="${2:-runs/kaggle}"
python -m phaseforge.cli audit-kaggle --data "$DATA"
python -m phaseforge.cli run --data "$DATA" --config configs/kaggle_group.json --output "$OUT/group"
python -m phaseforge.cli run --data "$DATA" --config configs/kaggle_temporal.json --output "$OUT/temporal"
python -m phaseforge.cli finalize --data "$DATA" --runs "$OUT/group" "$OUT/temporal" --output "$OUT/fusion"
echo "Submission with audited Kaggle-only provenance: $OUT/fusion/submission.csv"

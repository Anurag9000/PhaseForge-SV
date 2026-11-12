#!/usr/bin/env bash
set -euo pipefail
DATA="${1:?Usage: bash scripts/reproduce_historical_best.sh /path/to/kaggle [output-dir]}"
OUT="${2:-runs/historical_best_reproduction}"
python -m phaseforge.cli audit-kaggle --data "$DATA" --deep
python -m phaseforge.cli run --data "$DATA" --config configs/historical_461_group.json --output "$OUT/group"
python -m phaseforge.cli run --data "$DATA" --config configs/historical_461_temporal.json --output "$OUT/temporal"
python -m phaseforge.cli finalize --data "$DATA" --runs "$OUT/group" "$OUT/temporal" --output "$OUT/fusion"
python -m phaseforge.cli report --runs "$OUT/group" "$OUT/temporal" --output "$OUT/report"
echo "Reproduction artifacts saved under $OUT"
echo "Reproduction is NOT certified to reach the historical notebook 4.61% EER until measured."

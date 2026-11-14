#!/usr/bin/env bash
set -euo pipefail
DATA="${1:?Usage: bash scripts/run_missing_suite.sh /path/to/kaggle [output_dir] [max_jobs]}"
OUT="${2:-runs/missing_methods}"
LIMIT="${3:-127}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m phaseforge.cli audit-kaggle --data "$DATA"
"$PYTHON" -m phaseforge.cli suite-new --data "$DATA" --config configs/missing_research_suite.json --output "$OUT" --track kaggle --max-jobs "$LIMIT"
echo "Ledger: $OUT/new_suite_ledger.json"
echo "Development leaderboard: $OUT/new_suite_summary.json"

#!/usr/bin/env bash
set -euo pipefail
CORPUS="${1:?Usage: scripts/run_google_research.sh /extracted/google/corpus [run_dir]}"
OUT="${2:-runs/google}"
mkdir -p "$OUT"
python -m phaseforge.cli audit-google --data "$CORPUS" --output "$OUT/benchmark"
python -m phaseforge.cli split-google --manifest "$OUT/benchmark/google_manifest.csv" --output "$OUT/benchmark" --seed 2026
python -m phaseforge.cli run --data "$CORPUS" --config configs/google_research_stop.json --output "$OUT/stop_ubm" --track google --benchmark "$OUT/benchmark" --ubm-words stop
python -m phaseforge.cli run --data "$CORPUS" --config configs/google_research_stop.json --output "$OUT/all_word_ubm" --track google --benchmark "$OUT/benchmark" --ubm-words all
echo "Full Google research results are separate from Kaggle outputs: $OUT"

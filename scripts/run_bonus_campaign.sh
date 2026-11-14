#!/usr/bin/env bash
set -euo pipefail
DATA="${1:?Usage: bash scripts/run_bonus_campaign.sh /absolute/kaggle [out] [smoke|full]}"
OUT="${2:-runs/bonus}"
LEVEL="${3:-smoke}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
DATA="$(cd "$DATA" && pwd)"
PY="${PYTHON:-python3}"
if [[ "$LEVEL" != "smoke" && "$LEVEL" != "full" ]]; then echo "smoke or full expected" >&2; exit 2; fi
"$PY" -m phaseforge.cli audit-kaggle --data "$DATA"
if [[ "$LEVEL" == "smoke" ]]; then
  "$PY" -m phaseforge.bonus_benchmark --config configs/bonus_smoke.json --output "$OUT/competitors_smoke" --backends native torchgmm --device cpu --strict
  "$PY" -m phaseforge.bonus_identical_scoring --config configs/bonus_smoke.json --output "$OUT/identical_smoke" --backends torchgmm --device cpu
  "$PY" -m phaseforge.bonus_end_to_end ablation --data "$DATA" --config configs/bonus_e2e_smoke.json --output "$OUT/e2e_smoke" --max-jobs 2 --strict
  "$PY" -m phaseforge.bonus_frontend_benchmark --data "$DATA" --output "$OUT/frontend_smoke" --mode group --files 16 --batches 1 4 8 --repeats 2 --device cpu
  "$PY" -m phaseforge.bonus_report --benchmark "$OUT/competitors_smoke/bonus_results.json" --ablations "$OUT/e2e_smoke/bonus_e2e_summary.json" --output "$OUT/BONUS_SMOKE_REPORT.md"
  echo "Measured smoke report: $OUT/BONUS_SMOKE_REPORT.md"
  exit 0
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "Full campaign requires CUDA"; print("GPU:", torch.cuda.get_device_name(0))'
BACKENDS=(native torchgmm)
PARITY=(torchgmm)
CUVS_ARGS=()
BENCHDIR="$OUT/gpu_comparison"
if [[ -n "${CUVS_BINARY:-}" ]]; then
  [[ -f "$CUVS_BINARY" ]] || { echo "No compiled C++ cuVS adapter: $CUVS_BINARY" >&2; exit 2; }
  BACKENDS+=(cuvs)
  PARITY+=(cuvs)
  CUVS_ARGS=(--cuvs-binary "$CUVS_BINARY")
  BENCHDIR="$OUT/gpu_comparison_cuvs"
fi
"$PY" -m phaseforge.bonus_frontend_benchmark --data "$DATA" --output "$OUT/frontend_gpu" --mode group --files 256 --batches 1 4 8 16 32 --repeats 5 --device cuda
"$PY" -m phaseforge.bonus_benchmark --config configs/bonus_gpu_benchmark.json --output "$BENCHDIR" --kaggle-data "$DATA" --backends "${BACKENDS[@]}" --device cuda "${CUVS_ARGS[@]}" --strict
"$PY" -m phaseforge.bonus_identical_scoring --config configs/bonus_gpu_benchmark.json --output "$OUT/identical_gpu" --kaggle-data "$DATA" --backends "${PARITY[@]}" --device cuda "${CUVS_ARGS[@]}"
"$PY" -m phaseforge.bonus_end_to_end ablation --data "$DATA" --config configs/bonus_e2e_suite.json --output "$OUT/e2e_full" --max-jobs 35 --strict
CHECKPOINT="$OUT/e2e_full/jobs/group_reference/model_seed_2026.pt"
[[ -f "$CHECKPOINT" ]] || { echo "No frozen Kaggle checkpoint: $CHECKPOINT" >&2; exit 2; }
"$PY" -m phaseforge.bonus_end_to_end compare --data "$DATA" --checkpoint "$CHECKPOINT" --output "$OUT/frozen_scoring_gpu" --repeats 7 --speaker-chunk 16 --frame-chunk 1024 --device cuda
"$PY" -m phaseforge.bonus_report --benchmark "$BENCHDIR/bonus_results.json" --scoring "$OUT/frozen_scoring_gpu/scoring_comparison.json" --ablations "$OUT/e2e_full/bonus_e2e_summary.json" --output "$OUT/BONUS_REPRODUCIBLE_REPORT.md"
"$PY" -m phaseforge.bonus_plots --benchmark "$BENCHDIR/bonus_results.json" --scoring "$OUT/frozen_scoring_gpu/scoring_comparison.json" --e2e "$OUT/e2e_full/bonus_e2e_summary.json" --output "$OUT/figures"
echo "Full measured report: $OUT/BONUS_REPRODUCIBLE_REPORT.md"

#!/usr/bin/env bash
set -euo pipefail
CORPUS="${1:?Usage: scripts/run_google_research.sh /path/to/extracted/google [output-dir]}"
OUT="${2:-runs/google}"
CONFIG="${PHASEFORGE_GOOGLE_CONFIG:-configs/google_research_stop.json}"
ENROL="${GOOGLE_ENROL:-1}"
TEST="${GOOGLE_TEST:-1}"
mkdir -p "$OUT"
python -m phaseforge.cli audit-google --data "$CORPUS" --output "$OUT/benchmark"
python -m phaseforge.cli split-google --manifest "$OUT/benchmark/google_manifest.csv" --output "$OUT/benchmark" --seed 2026 --target-enrol "$ENROL" --target-test "$TEST"
python - "$CONFIG" "$OUT/research_config.json" "$ENROL" <<'PY'
import json, sys
src, dest, enrol = sys.argv[1:]
config = json.load(open(src))
config["enrolment_size"] = int(enrol)
json.dump(config, open(dest, "w"), indent=2)
PY
for pool in stop all; do
  python -m phaseforge.cli run --data "$CORPUS" --config "$OUT/research_config.json" --output "$OUT/development/$pool" --track google --benchmark "$OUT/benchmark" --ubm-words "$pool" --dev-only
done
BEST="$(python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]) / "development"
eers={pool: json.loads((root/pool/"result.json").read_text())["metrics"]["eer"] for pool in ("stop","all")}
print(min(eers, key=eers.get))
PY
)"
echo "Selected UBM word pool using development EER: $BEST"
python -m phaseforge.cli run --data "$CORPUS" --config "$OUT/research_config.json" --output "$OUT/locked_final" --track google --benchmark "$OUT/benchmark" --ubm-words "$BEST"
python - "$OUT" "$BEST" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); pool=sys.argv[2]
metrics={k: json.loads((root/"development"/k/"result.json").read_text())["metrics"]["eer"] for k in ("stop","all")}
final=json.loads((root/"locked_final"/"result.json").read_text())
report={"protocol":"Google independent research only","selected_word_pool":pool,"selection_development_eer":metrics,"locked_test":final.get("locked_google_test_metrics"),"source_track":"google_research"}
(root/"research_comparison.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report, indent=2))
PY
echo "The selected model and its measured locked Google test result are in $OUT/locked_final"

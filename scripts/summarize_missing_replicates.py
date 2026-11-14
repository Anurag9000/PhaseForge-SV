"""Aggregate matched-seed PhaseForge-SV experiments without mixing metrics.

python3 scripts/summarize_missing_replicates.py \
  --ledger runs/new_missing_3seed/new_suite_ledger.json \
  --output runs/new_missing_3seed/aggregate

Speaker-verification EER and keyword-classification accuracy are ALWAYS
separate leaderboards; no unsupported across-protocol comparison is made.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def aggregate_ledger(ledger: dict, expected_seeds: list[int] | None = None) -> dict:
    if not ledger:
        raise ValueError("Empty new suite ledger")
    grouped = defaultdict(list)
    for entry in ledger.values():
        if not isinstance(entry, dict):
            raise ValueError("Malformed ledger entry")
        name = str(entry["id"])
        if "__s" in name and name.rsplit("__s", 1)[1].isdigit():
            base, seed_text = name.rsplit("__s", 1)
            seed = int(seed_text)
        else:
            base = name
            seed = entry.get("config", {}).get("seed")
        grouped[base].append((seed, entry))
    verification, keyword, failures = [], [], []
    for name, entries in sorted(grouped.items()):
        complete = [(seed, e) for seed, e in entries
                    if e.get("status") == "complete"]
        failed = [(seed, e) for seed, e in entries
                  if e.get("status") == "failed"]
        skipped = sum(e.get("status") == "inapplicable" for _, e in entries)
        if complete:
            kinds = {e["kind"] for _, e in complete}
            if len(kinds) != 1:
                raise ValueError(f"Inconsistent job kind among seeds: {name}")
            kind = next(iter(kinds))
            if kind == "keyword_google":
                key, dest, better = "development_accuracy", keyword, "higher"
            else:
                key, dest, better = "development_eer", verification, "lower"
            values = [float(e[key]) for _, e in complete]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite results: {name}")
            std = statistics.stdev(values) if len(values) >= 2 else None
            summary = {
                "base_method": name, "kind": kind,
                "metric": key, "better": better,
                "complete_seeds": [seed for seed, _ in complete],
                "completed": len(complete), "failed": len(failed),
                "skipped": skipped,
                "expected_seeds": expected_seeds,
                "all_expected_seeds_present": (
                    expected_seeds is not None
                    and {seed for seed, _ in complete} == set(expected_seeds)
                ),
                "mean": statistics.mean(values), "std": std,
                "minimum": min(values), "maximum": max(values),
                "per_seed": [
                    {"seed": seed, "value": float(e[key]),
                     "directory": e.get("directory")}
                    for seed, e in complete
                ],
            }
            dest.append(summary)
        if failed:
            failures.append({
                "base_method": name, "failures": [
                    {"seed": seed, "error": e.get("error", "unknown")}
                    for seed, e in failed
                ]
            })
    verification.sort(key=lambda x: (x["mean"], x["std"] or 0))
    keyword.sort(key=lambda x: (-x["mean"], x["std"] or 0))
    return {
        "source": "new_phase_research",
        "speaker_verification": verification,
        "keyword_classification": keyword,
        "failures": failures,
        "important": (
            "Development-only: models selected here have not been scored on "
            "a locked Google test. Single-seed std=null is NOT zero uncertainty. "
            "Keyword accuracy must never be compared with verification EER."
        ),
    }


def save_aggregate(report: dict, destination: str | Path):
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "replicate_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for key, filename in (
        ("speaker_verification", "speaker_verification_replicates.csv"),
        ("keyword_classification", "keyword_replicates.csv"),
    ):
        rows = report[key]
        fields = [
            "base_method", "kind", "metric", "completed", "failed",
            "mean", "std", "minimum", "maximum", "all_expected_seeds_present",
        ]
        with (dest / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()
    ledger = json.loads(Path(args.ledger).read_text())
    report = aggregate_ledger(ledger, args.seeds)
    save_aggregate(report, args.output)
    print(
        f"Aggregated {len(report['speaker_verification'])} speaker-verification "
        f"methods and {len(report['keyword_classification'])} keyword methods; "
        f"{len(report['failures'])} failed method families"
    )


if __name__ == "__main__":
    main()

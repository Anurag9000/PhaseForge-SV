"""Reproducible finite experiment search, resumption, score comparison and ledgers."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .data import read_csv, write_csv
from .engine import RunConfig, run
from .metrics import report
from .scoring import fuse_streams


def _deep_set(config: dict, dotted: str, value) -> None:
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def expand_grid(
    space: dict, max_runs: int = 100, strategy: str = "random", seed: int = 2026,
) -> list[dict]:
    """Return at most max_runs configurations WITHOUT materializing enormous grids.

    strategy=random samples the full Cartesian space reproducibly (better coverage
    under a cap); strategy=grid runs lexicographically, useful for exhaustive
    small grids. Every pair (parameter, value) is recorded in results.
    """
    keys = sorted(space)
    values = [space[key] for key in keys]
    if any(not isinstance(v, list) or not v for v in values):
        raise ValueError("Grid entries must be nonempty lists")
    if strategy not in ("random", "grid"):
        raise ValueError("Unknown grid strategy")
    total = math.prod(len(v) for v in values)
    limit = min(int(max_runs), total)
    if limit < 1:
        raise ValueError("max_runs must be positive")
    if strategy == "grid" or limit == total:
        variants = itertools.islice(itertools.product(*values), limit)
        return [dict(zip(keys, row)) for row in variants]
    rng = random.Random(seed)
    samples = set()
    while len(samples) < limit:
        samples.add(tuple(rng.randrange(len(options)) for options in values))
    indices = sorted(samples)
    return [dict(zip(keys, [values[j][index] for j, index in enumerate(row)]))
            for row in indices]


def sweep(
    data_path: str, sweep_config: dict, output: str, track: str = "kaggle",
    benchmark: str | None = None,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = sweep_config["base"]
    combinations = expand_grid(
        sweep_config["grid"], max_runs=int(sweep_config.get("max_runs", 100)),
        strategy=sweep_config.get("strategy", "random"),
        seed=int(sweep_config.get("seed", 2026)),
    )
    limit = len(combinations)
    records = []
    for i, combo in enumerate(combinations[:limit]):
        cfg = copy.deepcopy(baseline)
        for name, value in combo.items():
            _deep_set(cfg, name, value)
        cfg["evaluate_eval"] = False  # no evaluation trials during selection
        spec = json.dumps(cfg, sort_keys=True)
        digest = hashlib.sha256(spec.encode()).hexdigest()[:12]
        run_dir = output / f"{i:04d}_{digest}"
        result_file = run_dir / "result.json"
        try:
            if result_file.is_file():
                result = json.loads(result_file.read_text())
            else:
                result = run(data_path, RunConfig.from_dict(cfg), run_dir,
                             track=track, benchmark=benchmark)
            row = {"run": i, "id": digest, "eer": result["metrics"]["eer"],
                   "auc": result["metrics"]["auc"],
                   "seconds": result["elapsed_seconds"], "parameters": combo,
                   "status": "complete", "run_dir": str(run_dir)}
        except Exception as exc:
            row = {"run": i, "id": digest, "status": "failed",
                   "error": repr(exc), "parameters": combo, "run_dir": str(run_dir)}
            (run_dir).mkdir(parents=True, exist_ok=True)
            (run_dir / "failure.json").write_text(json.dumps(row, indent=2))
        records.append(row)
        (output / "sweep_results.json").write_text(json.dumps(records, indent=2))
        print(f"{i+1}/{min(limit,len(combinations))}: {row}", flush=True)
    successful = sorted(
        (r for r in records if r["status"] == "complete"), key=lambda x: x["eer"]
    )
    leaderboard = {"tested": len(records), "completed": len(successful),
                   "failed": len(records) - len(successful),
                   "best": successful[0] if successful else None,
                   "ranked": successful}
    (output / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    return leaderboard


def fuse_saved_runs(
    run_directories: list[str], output: str, weights: list[float] | None = None,
    evaluate_eval: bool = True,
) -> dict:
    """Feature/architecture fusion: scales fitted on development scores only."""
    if len(run_directories) < 2:
        raise ValueError("Pass at least two independently trained experiment runs")
    datasets = []
    for directory in run_directories:
        path = Path(directory)
        dev = read_csv(path / "development_scores.csv")
        evaluation = read_csv(path / "evaluation_scores.csv") if evaluate_eval else []
        datasets.append((dev, evaluation))
    keys = [row["trial_id"] for row in datasets[0][0]]
    for dev, ev in datasets:
        if [row["trial_id"] for row in dev] != keys:
            raise ValueError("Development trial order mismatch")
    scales = [
        float(torch.tensor([float(x["score"]) for x in d]).std(unbiased=False))
        for d, _ in datasets
    ]
    merged_dev = fuse_streams([
        torch.tensor([float(x["score"]) for x in d], dtype=torch.float64)
        for d, _ in datasets
    ], scales, weights)
    labels = torch.tensor([int(x["label"]) for x in datasets[0][0]])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = datasets[0][0]
    write_csv(output / "development_scores.csv",
              [{**{k: row[k] for k in ("trial_id", "claimed_id", "test_file", "label")},
                "score": f"{float(score):.12f}"}
               for row, score in zip(original, merged_dev)],
              ["trial_id", "claimed_id", "test_file", "label", "score"])
    if evaluate_eval:
        order = [row["trial_id"] for row in datasets[0][1]]
        if any([x["trial_id"] for x in ev] != order for _, ev in datasets):
            raise ValueError("Evaluation trial ID mismatch")
        merged_eval = fuse_streams([
            torch.tensor([float(x["score"]) for x in ev], dtype=torch.float64)
            for _, ev in datasets
        ], scales, weights)
        original_eval = datasets[0][1]
        write_csv(output / "evaluation_scores.csv",
                  [{"trial_id": row["trial_id"], "claimed_id": row["claimed_id"],
                    "test_file": row["test_file"], "score": f"{float(sc):.12f}"}
                   for row, sc in zip(original_eval, merged_eval)],
                  ["trial_id", "claimed_id", "test_file", "score"])
        # No submission by default: caller must independently establish source track
        # compliance for every constituent run before exporting for Kaggle.
    result = {"runs": run_directories, "weights": weights, "dev_scales": scales,
              "metrics": report(merged_dev, labels)}
    (output / "fusion.json").write_text(json.dumps(result, indent=2))
    return result

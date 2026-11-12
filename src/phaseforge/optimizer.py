"""Stage-wise adaptive model improvement with recorded acceptance/rejection.

Each proposed change modifies ONE experimental dimension. Every trial is
executed on the same validation protocol and seeds before accepting a change.
No hidden Kaggle test labels are used or requested.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .engine import RunConfig, run
from .experiments import _deep_set
from .studies import run_proxy



def aggregate_validation(values: list[float], objective: str = "mean") -> float:
    """Compare equally weighted, genuinely speaker-heldout proxy splits.

    A mean-plus-standard-deviation objective is conservative when one
    candidate has unstable results across different speaker populations.
    """
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Expected nonempty finite validation EERs")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    if objective == "mean":
        return mean
    if objective == "worst":
        return max(values)
    if objective == "mean_plus_std":
        return mean + math.sqrt(variance)
    raise ValueError(f"Unknown robustness objective: {objective}")



def adaptive_optimize(
    data: str | Path, search_spec: dict, output: str | Path,
) -> dict:
    """Finite, resumable greedy coordinate search preserving the experiment setting.

    Default selection is a Kaggle-background speaker-excluded pseudo-verification
    test. Distinct from development trial model selection, and never transfers
    externally trained Google corpus parameters into the Kaggle model.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = copy.deepcopy(search_spec["initial"])
    stages = search_spec["stages"]
    rounds = int(search_spec.get("rounds", 2))
    tolerance = float(search_spec.get("minimum_improvement", 1e-5))
    protocol = search_spec.get("validation", "proxy")
    rotations = int(search_spec.get("proxy_rotations", 1))
    split_seeds = tuple(int(x) for x in search_spec.get("proxy_split_seeds", [2026]))
    objective = str(search_spec.get("selection_objective", "mean"))
    if not split_seeds or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("Require distinct, nonempty proxy split seeds")
    aggregate_validation([.1, .2], objective)
    if protocol not in ("proxy", "dev"):
        raise ValueError("validation must be proxy or dev")
    if rounds < 1:
        raise ValueError("Need at least one optimization round")

    ledger_path = output / "optimization_ledger.json"
    records = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    cache: dict[str, dict] = {record["key"]: record for record in records
                               if record.get("state") == "complete"}

    def assess(config: dict, stage: str, change: dict) -> dict:
        config = copy.deepcopy(config)
        config["evaluate_eval"] = False
        serial = json.dumps(config, sort_keys=True)
        experiment_identity = json.dumps({
            "data_root": str(Path(data).resolve()),
            "protocol": protocol, "rotations": rotations,
            "split_seeds": split_seeds, "objective": objective,
            "config": config,
        }, sort_keys=True)
        key = hashlib.sha256(experiment_identity.encode()).hexdigest()[:16]
        if key in cache:
            return cache[key]
        location = output / "runs" / key
        try:
            if protocol == "proxy":
                split_values = []
                for speaker_seed in split_seeds:
                    result = run_proxy(
                        data, RunConfig.from_dict(config),
                        location / f"split_{speaker_seed}",
                        seed=speaker_seed, rotations=rotations
                    )
                    split_values.append(float(result["mean_eer"]))
            else:
                result = run(data, RunConfig.from_dict(config), location, track="kaggle")
                split_values = [float(result["metrics"]["eer"])]
            value = aggregate_validation(split_values, objective)
            row = {
                "key": key, "stage": stage, "change": change,
                "value": value, "state": "complete",
                "split_eers": split_values,
                "mean_eer": sum(split_values) / len(split_values),
                "config": config, "run_dir": str(location),
            }
            cache[key] = row
        except Exception as error:
            row = {
                "key": key, "stage": stage, "change": change,
                "state": "failed", "error": repr(error),
                "config": config, "run_dir": str(location),
            }
        records.append(row)
        ledger_path.write_text(json.dumps(records, indent=2))
        return row

    incumbent = assess(start, "initial", {})
    if incumbent["state"] != "complete":
        raise RuntimeError("Initial system failed; inspect optimization_ledger.json")
    history = [{"stage": "initial", "value": incumbent["value"],
                "config": incumbent["config"]}]
    current = copy.deepcopy(start)
    for iteration in range(rounds):
        changed = False
        for stage, parameters in stages.items():
            candidate = None
            for dotted_path, choices in parameters.items():
                for value in choices:
                    proposed = copy.deepcopy(current)
                    _deep_set(proposed, dotted_path, value)
                    evaluated = assess(proposed, f"{iteration}:{stage}",
                                       {dotted_path: value})
                    if evaluated["state"] != "complete":
                        continue
                    if candidate is None or evaluated["value"] < candidate["value"]:
                        candidate = evaluated
            if candidate and candidate["value"] + tolerance < incumbent["value"]:
                current = candidate["config"]
                incumbent = candidate
                changed = True
                history.append({
                    "stage": stage, "iteration": iteration,
                    "value": incumbent["value"], "config": incumbent["config"],
                    "key": incumbent["key"],
                })
        if not changed:
            break
    summary = {
        "protocol": protocol, "rounds_requested": rounds,
        "proxy_split_seeds": split_seeds,
        "proxy_rotations": rotations,
        "selection_objective": objective,
        "best_validation_eer": incumbent["value"],
        "best_mean_validation_eer": incumbent["mean_eer"],
        "best_split_eers": incumbent["split_eers"],
        "best_config": incumbent["config"],
        "history": history,
        "trials_completed": len([r for r in records if r["state"] == "complete"]),
        "trials_failed": len([r for r in records if r["state"] == "failed"]),
        "note": "The final Kaggle model must be separately retrained using only Kaggle background.",
    }
    (output / "optimization_summary.json").write_text(json.dumps(summary, indent=2))
    (output / "best_config.json").write_text(json.dumps(incumbent["config"], indent=2))
    return summary

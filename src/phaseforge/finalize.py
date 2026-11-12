"""Source-verified Kaggle submission creation from completed GMM-UBM runs.

This deliberately refuses research-only HMM/DTW runs and requires manifests to
match both the selected training runs and the user's supplied competition data.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .data import KaggleData, read_csv, write_csv
from .metrics import report
from .scoring import fuse_streams


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_scores(path: Path, manifest: list[dict], labeled: bool) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing scored trials: {path}")
    rows = read_csv(path)
    if len(rows) != len(manifest):
        raise ValueError(f"Trial count mismatch: {path}")
    scores: list[float] = []
    for saved, reference in zip(rows, manifest):
        for field in ("trial_id", "claimed_id", "test_file"):
            if saved.get(field) != reference[field]:
                raise ValueError(f"Incorrect {field} or trial order in {path}")
        if labeled and saved.get("label") != reference["label"]:
            raise ValueError(f"Development label mismatch in {path}")
        try:
            score = float(saved["score"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid score in {path}") from exc
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite score in {path}")
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float64)


def finalize_kaggle(
    data_root: str | Path, runs: list[str | Path], output: str | Path,
    weights: list[float] | None = None,
) -> dict:
    """Verify training provenance, trial IDs and scores before exporting a submission.

    Multi-run score fusion uses only development-score standard deviations;
    scores from experimental research-only methods are never eligible.
    """
    if not runs:
        raise ValueError("Select at least one completed Kaggle GMM-UBM run")
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs) or any(w < 0 or not math.isfinite(w) for w in weights):
        raise ValueError("Invalid nonnegative run weights")
    if sum(weights) <= 0:
        raise ValueError("At least one weight must be positive")
    with KaggleData(data_root) as data:
        dev_scores, eval_scores, audit = [], [], []
        for location in runs:
            directory = Path(location).resolve()
            result_file = directory / "result.json"
            if not result_file.is_file():
                raise FileNotFoundError(f"No completed training manifest: {result_file}")
            result = json.loads(result_file.read_text())
            if result.get("track") != "kaggle" or result.get("research_only") is not False:
                raise PermissionError(f"Non-Kaggle or research-only model: {directory}")
            if Path(result.get("source", "")).resolve() != data.root:
                raise PermissionError(f"Model training source differs from supplied Kaggle corpus: {directory}")
            config = result.get("config", {})
            score_config = config.get("score", {})
            if score_config.get("norm", "none") not in ("none", "z", "t", "s"):
                raise PermissionError("Non-per-trial scoring is not eligible")
            if not result.get("seed_diagnostics"):
                raise ValueError("No completed GMM training diagnostics in result")
            dev = directory / "development_scores.csv"
            ev = directory / "evaluation_scores.csv"
            dev_scores.append(_load_scores(dev, data.dev_trials, True))
            eval_scores.append(_load_scores(ev, data.eval_trials, False))
            audit.append({
                "run_dir": str(directory),
                "source": str(data.root),
                "result_sha256": _sha256(result_file),
                "dev_sha256": _sha256(dev),
                "eval_sha256": _sha256(ev),
                "feature": config.get("feature", {}),
                "gmm": config.get("em", {}),
                "MAP": config.get("map", {}),
                "score": score_config,
                "seeds": config.get("seeds", []),
            })
        if len(runs) == 1:
            final_dev, final_eval = dev_scores[0], eval_scores[0]
            scales = [1.0]
        else:
            scales = [float(score.std(unbiased=False)) for score in dev_scores]
            if min(scales) <= 1e-12:
                raise ValueError("A development stream has constant scores")
            final_dev = fuse_streams(dev_scores, scales, weights)
            final_eval = fuse_streams(eval_scores, scales, weights)
        if not torch.isfinite(final_eval).all():
            raise FloatingPointError("Refusing nonfinite evaluation predictions")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "submission.csv",
                  [{"trial_id": row["trial_id"], "score": f"{float(s):.12f}"}
                   for row, s in zip(data.eval_trials, final_eval)],
                  ["trial_id", "score"])
        write_csv(output / "development_scores.csv",
                  [
                      {"trial_id": row["trial_id"], "claimed_id": row["claimed_id"],
                       "test_file": row["test_file"], "label": row["label"],
                       "score": f"{float(s):.12f}"}
                      for row, s in zip(data.dev_trials, final_dev)
                  ],
                  ["trial_id", "claimed_id", "test_file", "label", "score"])
        labels = torch.tensor([int(r["label"]) for r in data.dev_trials], dtype=torch.bool)
        outcome = {
            "source_track": "kaggle_only",
            "eligible_gmm_ubm_only": True,
            "runs": audit,
            "fusion_weights": weights,
            "development_scaling": scales,
            "dev_metrics": report(final_dev, labels),
            "evaluation_trials": len(data.eval_trials),
            "submission_sha256": _sha256(output / "submission.csv"),
            "evaluation_labels": "unavailable",
        }
        (output / "provenance.json").write_text(json.dumps(outcome, indent=2))
        return outcome

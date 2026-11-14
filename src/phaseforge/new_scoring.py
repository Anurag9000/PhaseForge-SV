"""Research score calibration, quality-aware fusion, and adaptive cohort normalization.

Calibration fits ONLY on labeled development trials. Speaker groups and
test recordings are held out as folds to report out-of-fold diagnostics.
The resulting scores are NOT claimed to be unmodified GMM-UBM LLRs.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import read_csv, write_csv
from .metrics import roc, report


def min_dcf(scores, labels, prior: float = .01, miss_cost: float = 1.,
            fa_cost: float = 1.) -> dict:
    if not 0 < prior < 1:
        raise ValueError("Prior must be in (0,1)")
    curve = roc(scores, labels)
    cost = miss_cost * prior * curve["frr"] + fa_cost * (1-prior) * curve["far"]
    index = int(cost.argmin())
    reference = min(miss_cost * prior, fa_cost * (1-prior))
    return {
        "min_dcf": float(cost[index] / reference),
        "raw_dcf": float(cost[index]),
        "threshold": float(curve["thresholds"][index]),
        "prior": prior,
    }


def cllr(scores, labels) -> float:
    """Cross-entropy Cllr for LOG likelihood ratios; not meaningful for raw cosines."""
    x = torch.as_tensor(scores, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.bool)
    if not y.any() or not (~y).any():
        raise ValueError("Need both genuine and impostor scores")
    return float((F.softplus(-x[y]).mean() + F.softplus(x[~y]).mean()) /
                 (2 * math.log(2)))


def adaptive_snorm(raw: torch.Tensor, z_cohort: torch.Tensor,
                   t_cohort: torch.Tensor, top_k: int = 20,
                   floor: float = .03):
    """Per-trial top-k score-relative Z/T cohort statistics.

    Inputs have shapes [trials], [trials,z_cohort], [trials,t_cohort].
    Cohort SPEAKER MODELS and cohort recordings must all originate from
    the designated training background population.
    """
    if raw.ndim != 1 or z_cohort.ndim != 2 or t_cohort.ndim != 2:
        raise ValueError("Adaptive S-norm requires 1D raw and 2D cohort score tables")
    if len(raw) != len(z_cohort) or len(raw) != len(t_cohort):
        raise ValueError("Trial count mismatch")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    z = z_cohort.topk(min(top_k, z_cohort.shape[1]), dim=1).values
    t = t_cohort.topk(min(top_k, t_cohort.shape[1]), dim=1).values
    z_std = z.std(1, unbiased=False).clamp_min(floor)
    t_std = t.std(1, unbiased=False).clamp_min(floor)
    return ((raw - z.mean(1)) / z_std + (raw - t.mean(1)) / t_std) / 2


def length_quality(lengths: torch.Tensor, saturation: float = 100.) -> torch.Tensor:
    """Reliability proxy from duration, bounded in [0,1)."""
    return lengths.float().clamp_min(0) / (lengths.float().clamp_min(0) + saturation)


def quality_fusion(scores: torch.Tensor, quality: torch.Tensor,
                   uncertainty: torch.Tensor, alpha: float = .2,
                   beta: float = .1):
    """Quality-adjusted score, selected on development only."""
    if scores.shape != quality.shape or scores.shape != uncertainty.shape:
        raise ValueError("Quality tables must match scores")
    return scores * (1 + alpha * quality) - beta * uncertainty


class LogisticCalibration:
    """Regularized weighted logistic regression of raw verification scores."""
    def __init__(self, epochs: int = 250, lr: float = .05, prior: float = .5):
        if epochs < 1 or not 0 < prior < 1:
            raise ValueError("Invalid calibrator")
        self.epochs, self.lr, self.prior = epochs, lr, prior

    def fit(self, X: torch.Tensor, labels: torch.Tensor):
        X = X.double()
        if X.ndim == 1:
            X = X[:, None]
        y = labels.double()
        if len(X) != len(y) or not y.any() or not (1 - y).any():
            raise ValueError("Calibration needs both genuine and impostor labels")
        self.mean = X.mean(0)
        self.std = X.std(0, unbiased=False).clamp_min(1e-5)
        Z = (X - self.mean) / self.std
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True,
                        device=X.device)
        b = torch.zeros((), dtype=torch.float64, requires_grad=True, device=X.device)
        optimizer = torch.optim.Adam([w, b], lr=self.lr)
        positive_weight = self.prior / y.sum().item()
        negative_weight = (1 - self.prior) / (len(y) - y.sum().item())
        weights = torch.where(y > .5, positive_weight, negative_weight)
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            z = Z @ w + b
            loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") *
                    weights).sum() + 1e-3 * w.square().sum()
            loss.backward()
            optimizer.step()
        self.w, self.b = w.detach(), b.detach()
        return self

    @torch.no_grad()
    def __call__(self, X: torch.Tensor):
        X = X.double()
        if X.ndim == 1:
            X = X[:, None]
        log_odds = ((X - self.mean) / self.std) @ self.w + self.b
        return log_odds - math.log(self.prior / (1 - self.prior))


def speaker_grouped_calibration(
    scores: torch.Tensor, trials: list[dict], folds: int = 5,
    group_key: str = "test_file", seed: int = 2026,
):
    """No recording's labels are used to fit its own calibration transform."""
    if folds < 2:
        raise ValueError("Need at least two speaker/recording folds")
    groups = defaultdict(list)
    for index, trial in enumerate(trials):
        groups[trial[group_key]].append(index)
    keys = sorted(groups)
    if len(keys) < folds:
        raise ValueError("Too few distinct recording groups")
    random.Random(seed).shuffle(keys)
    labels = torch.tensor([int(x["label"]) for x in trials], dtype=torch.float64)
    scores = scores.double()
    result = torch.full_like(scores, float("nan"))
    for fold in range(folds):
        hold_keys = keys[fold::folds]
        hold = [i for name in hold_keys for i in groups[name]]
        retained = [i for name in keys if name not in set(hold_keys)
                    for i in groups[name]]
        fitted = LogisticCalibration().fit(scores[retained], labels[retained])
        result[hold] = fitted(scores[hold])
    if not torch.isfinite(result).all():
        raise FloatingPointError("Invalid out-of-fold calibration")
    return result


def calibrate_saved_predictions(
    development: str | Path, output: str | Path,
    evaluation: str | Path | None = None, folds: int = 5,
) -> dict:
    rows = read_csv(development)
    if not rows or not all(k in rows[0] for k in
                           ("trial_id", "test_file", "claimed_id", "score", "label")):
        raise ValueError("Need development trial scores and labels")
    raw = torch.tensor([float(x["score"]) for x in rows], dtype=torch.float64)
    labels = torch.tensor([int(x["label"]) for x in rows])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    oof = speaker_grouped_calibration(raw, rows, folds=folds)
    final = LogisticCalibration().fit(raw, labels)
    write_csv(out / "calibrated_development_oof.csv", [
        {**{k: r[k] for k in ("trial_id", "claimed_id", "test_file", "label")},
         "score": f"{float(x):.12f}"} for r, x in zip(rows, oof)
    ], ["trial_id", "claimed_id", "test_file", "label", "score"])
    if evaluation:
        evaluation_rows = read_csv(evaluation)
        values = final(torch.tensor([float(r["score"]) for r in evaluation_rows]))
        write_csv(out / "calibrated_evaluation.csv", [
            {"trial_id": row["trial_id"], "score": f"{float(value):.12f}"}
            for row, value in zip(evaluation_rows, values)
        ], ["trial_id", "score"])
    result = {
        "research_only": True, "calibration": "out-of-fold grouped logistic",
        "raw": {**report(raw, labels), **min_dcf(raw, labels)},
        "calibrated_oof": {
            **report(oof, labels), **min_dcf(oof, labels),
            "cllr": cllr(oof, labels),
        }, "folds": folds,
    }
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    return result

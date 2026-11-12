"""EER, ROC/DET-style operating points, AUC and grouped bootstrap; pure PyTorch."""
from __future__ import annotations

import random
from collections import defaultdict

import torch


def roc(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    s = torch.as_tensor(scores, dtype=torch.float64).flatten()
    y = torch.as_tensor(labels, dtype=torch.bool).flatten()
    if len(s) != len(y) or len(s) == 0 or not torch.isfinite(s).all():
        raise ValueError("Scores and labels must be finite equal-length arrays")
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        raise ValueError("Need genuine and impostor trials")
    ordered = torch.argsort(s, descending=True, stable=True)
    sorted_scores, sorted_labels = s[ordered], y[ordered]
    change = torch.cat([sorted_scores[:-1] != sorted_scores[1:],
                        torch.tensor([True], device=s.device)])
    ends = torch.where(change)[0]
    positives = sorted_labels.to(torch.float64).cumsum(0)[ends]
    negatives = (~sorted_labels).to(torch.float64).cumsum(0)[ends]
    fpr = torch.cat([s.new_zeros(1), negatives / nneg])
    tpr = torch.cat([s.new_zeros(1), positives / npos])
    frr = 1 - tpr
    thresholds = torch.cat([s.new_full((1,), float("inf")), sorted_scores[ends]])
    return {"far": fpr, "frr": frr, "tpr": tpr, "thresholds": thresholds}


def eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    curve = roc(scores, labels)
    far, frr = curve["far"], curve["frr"]
    dif = far - frr
    index = int(torch.where(dif >= 0)[0][0])
    if index == 0:
        return float(far[0])
    left, right = index - 1, index
    d0, d1 = float(dif[left]), float(dif[right])
    frac = -d0 / (d1 - d0) if d1 != d0 else 0.0
    return float(far[left] + frac * (far[right] - far[left]))


def report(scores: torch.Tensor, labels: torch.Tensor, target_far: float = .01) -> dict:
    curve = roc(scores, labels)
    fpr, tpr = curve["far"], curve["tpr"]
    auc = float(torch.trapz(tpr, fpr))
    eligible = torch.where(fpr <= target_far)[0]
    index = int(eligible[-1])
    # Score > theta is the assignment's decision rule. Move theta just below
    # the selected ROC score so ties at that value count as accepted.
    selected = curve["thresholds"][index]
    threshold = float(torch.nextafter(selected, selected.new_full((), -float("inf"))))
    s = torch.as_tensor(scores).flatten()
    y = torch.as_tensor(labels).bool().flatten()
    accepted = s > threshold
    return {
        "eer": eer(scores, labels),
        "auc": auc,
        "target_far": target_far,
        "operating_threshold": threshold,
        "far": float(accepted[~y].double().mean()),
        "frr": float((~accepted[y]).double().mean()),
        "balanced_accuracy": float(
            (accepted[y].double().mean() + (~accepted[~y]).double().mean()) / 2
        ),
        "genuine": int(y.sum()), "impostor": int((~y).sum()),
    }


def grouped_bootstrap(
    trials: list[dict], scores: torch.Tensor, iterations: int = 1000,
    group_key: str = "claimed_id", seed: int = 2026,
) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, trial in enumerate(trials):
        groups[trial[group_key]].append(i)
    rng = random.Random(seed)
    keys = sorted(groups)
    values = []
    labels = torch.tensor([int(t["label"]) for t in trials])
    for _ in range(iterations):
        indices = [index for _ in keys for index in groups[rng.choice(keys)]]
        y = labels[indices]
        if y.any() and (~y.bool()).any():
            values.append(eer(scores[indices], y))
    if not values:
        raise ValueError("No valid resampled groups")
    values.sort()
    return {
        "mean": sum(values) / len(values),
        "ci95": [values[int(.025 * (len(values) - 1))],
                 values[int(.975 * (len(values) - 1))]],
        "resamples": len(values), "group_key": group_key,
    }

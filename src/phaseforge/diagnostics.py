"""Speaker-level verification diagnostics and paired model comparisons (torch only)."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from .data import read_csv, write_csv
from .metrics import eer, report, roc


def per_speaker_diagnostics(
    trials: list[dict], scores: torch.Tensor, threshold: float,
) -> tuple[list[dict], dict]:
    if len(trials) != len(scores):
        raise ValueError("Score count mismatch")
    groups = defaultdict(list)
    for i, row in enumerate(trials):
        groups[row["claimed_id"]].append(i)
    details = []
    for claimed, indices in sorted(groups.items()):
        positives = [float(scores[i]) for i in indices if int(trials[i]["label"]) == 1]
        negatives = [float(scores[i]) for i in indices if int(trials[i]["label"]) == 0]
        fn = sum(score <= threshold for score in positives)
        fp = sum(score > threshold for score in negatives)
        details.append({
            "speaker": claimed, "genuine_trials": len(positives),
            "impostor_trials": len(negatives),
            "false_rejections": fn, "false_acceptances": fp,
            "genuine_mean": sum(positives) / len(positives) if positives else None,
            "impostor_mean": sum(negatives) / len(negatives) if negatives else None,
            "within_speaker_margin": min(positives) - max(negatives)
            if positives and negatives else None,
            "perfect_within_speaker_ranking": int(
                min(positives) > max(negatives)
            ) if positives and negatives else None,
        })
    return details, {
        "speakers": len(details),
        "speakers_with_any_error": sum(bool(r["false_rejections"] or
                                           r["false_acceptances"]) for r in details),
        "perfect_within_speaker_ranking": sum(
            r["perfect_within_speaker_ranking"] or 0 for r in details
        ),
    }


def impostor_condition_report(
    trials: list[dict], scores: torch.Tensor,
) -> dict:
    """Separate wrong enrolled claims from fully unenrolled impostor recordings.

    A recording with any genuine development claim is treated as an enrolled
    target's recording. Unenrolled speaker recordings have no positive claim.
    """
    genuine_files = {x["test_file"] for x in trials if int(x["label"]) == 1}
    positive = [i for i, x in enumerate(trials) if int(x["label"]) == 1]
    enrolled_wrong = [i for i, x in enumerate(trials)
                      if int(x["label"]) == 0 and x["test_file"] in genuine_files]
    unknown_wrong = [i for i, x in enumerate(trials)
                     if int(x["label"]) == 0 and x["test_file"] not in genuine_files]
    labels = torch.tensor([int(x["label"]) for x in trials])
    output = {}
    for name, negatives in (
        ("enrolled_impostor", enrolled_wrong),
        ("unknown_impostor", unknown_wrong),
    ):
        if negatives and positive:
            index = positive + negatives
            output[name] = {
                "genuine": len(positive), "impostor": len(negatives),
                **report(scores[index], labels[index])
            }
    return output


def paired_group_bootstrap(
    trials: list[dict], model_a: torch.Tensor, model_b: torch.Tensor,
    group_key: str = "test_file", repetitions: int = 1000, seed: int = 2026,
) -> dict:
    """Resample shared recordings or shared claimed-speaker groups *together*."""
    if len(trials) != len(model_a) or len(trials) != len(model_b):
        raise ValueError("Paired trial predictions must have identical lengths")
    groups = defaultdict(list)
    for i, row in enumerate(trials):
        groups[row[group_key]].append(i)
    keys = sorted(groups)
    rng = random.Random(seed)
    labels = torch.tensor([int(row["label"]) for row in trials])
    improvements = []
    for _ in range(repetitions):
        sample = [i for _ in keys for i in groups[rng.choice(keys)]]
        y = labels[sample]
        if not y.any() or not (~y.bool()).any():
            continue
        improvements.append(
            eer(model_a[sample], y) - eer(model_b[sample], y)
        )
    if not improvements:
        raise ValueError("No balanced bootstrap sample")
    improvements.sort()
    return {
        "grouping": group_key, "repetitions": len(improvements),
        "delta_eer_a_minus_b": eer(model_a, labels) - eer(model_b, labels),
        "median": improvements[len(improvements) // 2],
        "ci95": [
            improvements[int(.025 * (len(improvements) - 1))],
            improvements[int(.975 * (len(improvements) - 1))],
        ],
        "improvement_frequency": sum(x > 0 for x in improvements) / len(improvements),
    }


def analyze_score_csv(
    csv_path: str | Path, output: str | Path, target_far: float = .01,
    compare_csv: str | Path | None = None,
) -> dict:
    rows = read_csv(csv_path)
    required = {"trial_id", "claimed_id", "test_file", "label", "score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Expected labelled development score CSV")
    scores = torch.tensor([float(x["score"]) for x in rows], dtype=torch.float64)
    labels = torch.tensor([int(x["label"]) for x in rows])
    metrics = report(scores, labels, target_far)
    per_target, speaker_summary = per_speaker_diagnostics(
        rows, scores, metrics["operating_threshold"]
    )
    result = {
        "overall": metrics,
        "speaker_summary": speaker_summary,
        "impostor_conditions": impostor_condition_report(rows, scores),
    }
    if compare_csv is not None:
        other = read_csv(compare_csv)
        if [r["trial_id"] for r in other] != [r["trial_id"] for r in rows]:
            raise ValueError("Paired model trial IDs are not identical")
        second = torch.tensor([float(x["score"]) for x in other], dtype=torch.float64)
        result["paired_comparison"] = {
            "recordings": paired_group_bootstrap(rows, scores, second, "test_file"),
            "claimed_speakers": paired_group_bootstrap(rows, scores, second, "claimed_id"),
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "speaker_diagnostics.csv", per_target, list(per_target[0]))
    (output / "analysis.json").write_text(json.dumps(result, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curve = roc(scores, labels)
        plt.figure(figsize=(6, 5))
        plt.plot(curve["far"].cpu(), curve["tpr"].cpu(),
                 label=f"AUC={metrics['auc']:.4f}")
        plt.xlabel("False accept rate")
        plt.ylabel("True accept rate")
        plt.legend()
        plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(output / "roc.png", dpi=140)
        plt.close()
    except ImportError:
        pass
    return result

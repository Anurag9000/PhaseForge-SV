"""Verified per-trial log-likelihood-ratio scoring, cohort norms and exploratory score fusion."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .gmm import MAPConfig, TorchGMM


@dataclass(frozen=True)
class ScoreConfig:
    smoothing: int = 1  # odd median window, 1=no smoothing
    top_components: int | None = None
    norm: str = "none"  # none / z / t / s
    cohort_speakers: int = 32
    cohort_test_files: int = 50

    def validate(self) -> None:
        if self.smoothing % 2 != 1 or self.smoothing < 1:
            raise ValueError("Median smoothing window must be positive and odd")
        if self.norm not in ("none", "z", "t", "s"):
            raise ValueError("Unsupported score normalization")


@torch.no_grad()
def likelihood_ratio(
    background: TorchGMM, speaker: TorchGMM,
    frames: torch.Tensor, smoothing: int = 1,
    top_components: int | None = None,
) -> torch.Tensor:
    if smoothing < 1 or smoothing % 2 != 1:
        raise ValueError("Odd positive smoothing required")
    numerator = speaker.log_components(frames)
    denominator = background.log_components(frames)
    if top_components is not None and top_components < denominator.shape[-1]:
        if top_components <= 0:
            raise ValueError("top_components must be positive")
        indices = denominator.topk(top_components, dim=1).indices
        numerator = numerator.gather(1, indices)
        denominator = denominator.gather(1, indices)
    llr = torch.logsumexp(numerator, -1) - torch.logsumexp(denominator, -1)
    if smoothing > 1 and len(llr) > 1:
        half = smoothing // 2
        padded = F.pad(llr[None, None], (half, half), mode="replicate")
        llr = padded.unfold(-1, smoothing, 1).median(-1).values.flatten()
    return llr.mean()


def enrolled_models(
    ubm: TorchGMM, features: dict[str, torch.Tensor],
    enrolment: list[dict], map_config: MAPConfig, k: int = 3,
) -> dict[str, TorchGMM]:
    if not 1 <= k <= 3:
        raise ValueError("Use 1/2/3 enrolment utterances")
    grouped = defaultdict(list)
    for row in enrolment:
        grouped[row["speaker_id"]].append(row)
    adapted = {}
    for speaker, rows in grouped.items():
        selection = sorted(rows, key=lambda row: int(row["enrol_index"]))[:k]
        frames = torch.cat([features[row["file_id"]] for row in selection])
        adapted[speaker] = ubm.adapt(frames, map_config)
    return adapted


@torch.no_grad()
def score_trials(
    ubm: TorchGMM, features: dict[str, torch.Tensor], trials: list[dict],
    speakers: dict[str, TorchGMM], config: ScoreConfig = ScoreConfig(),
) -> torch.Tensor:
    config.validate()
    cache = {}
    output = []
    for row in trials:
        claimed, test = row["claimed_id"], row["test_file"]
        if claimed not in speakers:
            raise ValueError(f"Unenrolled claimed speaker: {claimed}")
        key = (claimed, test)
        if key not in cache:
            cache[key] = likelihood_ratio(
                ubm, speakers[claimed], features[test],
                smoothing=config.smoothing, top_components=config.top_components
            )
        output.append(cache[key])
    return torch.stack(output).to(dtype=torch.float64)


@torch.no_grad()
def cohort_normalize(
    ubm: TorchGMM, feature: dict[str, torch.Tensor],
    trials: list[dict], scores: torch.Tensor,
    targets: dict[str, TorchGMM], cohort: dict[str, TorchGMM],
    background_test_files: list[str], config: ScoreConfig,
) -> torch.Tensor:
    """Z uses target vs background tests, T uses test vs background cohort speaker models.

    Fit cohorts on background enrolment only and choose background test IDs not from
    those cohort speakers. No target trial recording is used to fit any model/statistic.
    """
    config.validate()
    if config.norm == "none":
        return scores
    if not background_test_files or not cohort:
        raise ValueError("Score normalization requires background-only cohort")
    z_stats, t_stats = {}, {}
    if config.norm in ("z", "s"):
        for claimed, model in targets.items():
            baseline = torch.stack([
                likelihood_ratio(ubm, model, feature[file], config.smoothing,
                                 config.top_components)
                for file in background_test_files
            ]).double()
            z_stats[claimed] = (baseline.mean(), baseline.std(unbiased=False).clamp_min(.03))
    if config.norm in ("t", "s"):
        for test in sorted({r["test_file"] for r in trials}):
            cohort_scores = torch.stack([
                likelihood_ratio(ubm, model, feature[test], config.smoothing,
                                 config.top_components)
                for model in cohort.values()
            ]).double()
            t_stats[test] = (
                cohort_scores.mean(), cohort_scores.std(unbiased=False).clamp_min(.03)
            )
    norm_scores = []
    for row, score in zip(trials, scores):
        if config.norm in ("z", "s"):
            mu, sd = z_stats[row["claimed_id"]]
            z = (score - mu) / sd
        if config.norm in ("t", "s"):
            mu, sd = t_stats[row["test_file"]]
            t = (score - mu) / sd
        norm_scores.append(z if config.norm == "z" else t if config.norm == "t" else (z + t) / 2)
    return torch.stack(norm_scores)


def fuse_streams(
    scores: list[torch.Tensor], development_scales: list[float] | None = None,
    weights: list[float] | None = None,
) -> torch.Tensor:
    if not scores:
        raise ValueError("No scores to fuse")
    if any(len(x) != len(scores[0]) for x in scores):
        raise ValueError("Mismatched trial lists for fusion")
    weights = weights or [1 / len(scores)] * len(scores)
    development_scales = development_scales or [1.0] * len(scores)
    if len(weights) != len(scores) or len(development_scales) != len(scores):
        raise ValueError("Fusion sizes mismatch")
    total = sum(weights)
    if total <= 0 or any(scale <= 0 for scale in development_scales):
        raise ValueError("Invalid stream scales/weights")
    return sum(w * score / scale for w, score, scale in zip(
        weights, scores, development_scales
    )) / total


def exploratory_four_claim_adjustment(
    trials: list[dict], raw_scores: torch.Tensor, alpha: float,
) -> torch.Tensor:
    """NOT the required Kaggle per-trial LLR; research-only group-dependent score."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(trials):
        grouped[row["test_file"]].append(i)
    adjusted = raw_scores.clone()
    for indices in grouped.values():
        if len(indices) != 4:
            raise ValueError("Four-claim adjustment requires exactly 4 claims/test recording")
        peak = raw_scores[indices].max()
        adjusted[indices] = raw_scores[indices] - alpha * (peak - raw_scores[indices])
    return adjusted

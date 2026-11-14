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
    norm_stage: str = "per_seed"  # per_seed | after_ensemble
    cohort_seed: int = 2026
    cohort_speakers: int = 32
    cohort_test_files: int = 50
    cohort_top_k: int = 20

    def validate(self) -> None:
        if self.smoothing % 2 != 1 or self.smoothing < 1:
            raise ValueError("Median smoothing window must be positive and odd")
        if self.norm not in ("none", "z", "t", "s", "adaptive_s"):
            raise ValueError("Unsupported score normalization")
        if self.norm_stage not in ("per_seed", "after_ensemble"):
            raise ValueError("Unknown score-normalization stage")
        if self.cohort_top_k < 1:
            raise ValueError("cohort_top_k must be positive")


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
        if len(rows) < k:
            raise ValueError(
                f"Speaker {speaker!r} has {len(rows)} enrolment recordings; "
                f"the experiment requested {k}. Refusing to silently change protocol."
            )
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
    if config.norm in ("z", "s", "adaptive_s"):
        for claimed, model in targets.items():
            baseline = torch.stack([
                likelihood_ratio(ubm, model, feature[file], config.smoothing,
                                 config.top_components)
                for file in background_test_files
            ]).double()
            if config.norm == "adaptive_s":
                baseline = baseline.topk(
                    min(config.cohort_top_k, len(baseline))
                ).values
            z_stats[claimed] = (baseline.mean(), baseline.std(unbiased=False).clamp_min(.03))
    if config.norm in ("t", "s", "adaptive_s"):
        for test in sorted({r["test_file"] for r in trials}):
            cohort_scores = torch.stack([
                likelihood_ratio(ubm, model, feature[test], config.smoothing,
                                 config.top_components)
                for model in cohort.values()
            ]).double()
            if config.norm == "adaptive_s":
                cohort_scores = cohort_scores.topk(
                    min(config.cohort_top_k, len(cohort_scores))
                ).values
            t_stats[test] = (
                cohort_scores.mean(), cohort_scores.std(unbiased=False).clamp_min(.03)
            )
    norm_scores = []
    for row, score in zip(trials, scores):
        if config.norm in ("z", "s", "adaptive_s"):
            mu, sd = z_stats[row["claimed_id"]]
            z = (score - mu) / sd
        if config.norm in ("t", "s", "adaptive_s"):
            mu, sd = t_stats[row["test_file"]]
            t = (score - mu) / sd
        norm_scores.append(z if config.norm == "z" else t if config.norm == "t" else (z + t) / 2)
    return torch.stack(norm_scores)



@torch.no_grad()
def cohort_score_tables(
    ubm: TorchGMM, feature: dict[str, torch.Tensor],
    trials: list[dict], targets: dict[str, TorchGMM],
    cohort: dict[str, TorchGMM], background_test_files: list[str],
    config: ScoreConfig,
) -> dict:
    """Raw cohort LLR *matrices*, retaining per-seed samples for later averaging.

    This is needed to reproduce the historical experiment: average the five
    independent models' scores first, THEN estimate Z/T means and deviations.
    Normalizing each seed before averaging is mathematically different.
    """
    config.validate()
    if not cohort or not background_test_files:
        raise ValueError("Cohort model and background score population required")
    target_names = sorted(targets)
    test_names = sorted({row["test_file"] for row in trials})
    cohort_names = sorted(cohort)
    background_files = sorted(background_test_files)
    z = None
    t = None
    if config.norm in ("z", "s", "adaptive_s"):
        z = torch.stack([
            torch.stack([
                likelihood_ratio(
                    ubm, targets[name], feature[file],
                    config.smoothing, config.top_components
                ) for file in background_files
            ]) for name in target_names
        ]).double()
    if config.norm in ("t", "s", "adaptive_s"):
        t = torch.stack([
            torch.stack([
                likelihood_ratio(
                    ubm, cohort[speaker], feature[file],
                    config.smoothing, config.top_components
                ) for speaker in cohort_names
            ]) for file in test_names
        ]).double()
    return {
        "target_names": target_names,
        "test_names": test_names,
        "cohort_names": cohort_names,
        "background_files": background_files,
        "z": z,
        "t": t,
    }


@torch.no_grad()
def ensemble_cohort_normalize(
    trials: list[dict], mean_raw_scores: torch.Tensor,
    seed_tables: list[dict], config: ScoreConfig,
) -> torch.Tensor:
    """Normalize ENSEMBLE-MEAN scores using ENSEMBLE-MEAN cohort matrices.

    Historical 2026 notebook used sample standard deviations (unbiased=True).
    This is an explicit reproducibility mode; regular per-seed normalization
    remains available for separate ablations.
    """
    config.validate()
    if config.norm == "none":
        return mean_raw_scores
    if not seed_tables:
        raise ValueError("No per-seed cohort tables to average")
    names = ("target_names", "test_names", "cohort_names", "background_files")
    baseline = seed_tables[0]
    if any(any(table[key] != baseline[key] for key in names) for table in seed_tables):
        raise ValueError("Every seed must use the same background-only cohort population")
    norm = {}
    if config.norm in ("z", "s", "adaptive_s"):
        scores = torch.stack([table["z"].cpu() for table in seed_tables]).mean(0)
        if config.norm == "adaptive_s":
            scores = scores.topk(min(config.cohort_top_k, scores.shape[1]), dim=1).values
        norms = scores.std(1, unbiased=scores.shape[1] > 1).clamp_min(.03)
        norm["z"] = {
            name: (scores[i].mean(), norms[i])
            for i, name in enumerate(baseline["target_names"])
        }
    if config.norm in ("t", "s", "adaptive_s"):
        scores = torch.stack([table["t"].cpu() for table in seed_tables]).mean(0)
        if config.norm == "adaptive_s":
            scores = scores.topk(min(config.cohort_top_k, scores.shape[1]), dim=1).values
        norms = scores.std(1, unbiased=scores.shape[1] > 1).clamp_min(.03)
        norm["t"] = {
            name: (scores[i].mean(), norms[i])
            for i, name in enumerate(baseline["test_names"])
        }
    corrected = []
    for row, score in zip(trials, mean_raw_scores.cpu().double()):
        if config.norm in ("z", "s", "adaptive_s"):
            mu, sd = norm["z"][row["claimed_id"]]
            zscore = (score - mu) / sd
        if config.norm in ("t", "s", "adaptive_s"):
            mu, sd = norm["t"][row["test_file"]]
            tscore = (score - mu) / sd
        corrected.append(
            zscore if config.norm == "z" else
            tscore if config.norm == "t" else (zscore + tscore) / 2
        )
    return torch.stack(corrected)

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

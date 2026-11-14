"""MathWorks MFCC/GMM-UBM research replication on ORIGINAL GOOGLE ONLY.

This is never importable as a Kaggle GMM feature path. It implements the
reference's 40ms/10ms MFCC, global background feature standardization,
local cepstral centering, 32-mixture UBM, all-moment MAP (r=16), and
3-frame median-smoothed LLR. Its speech-region detection is an explicitly
energy-based approximation of MATLAB detectSpeech, not byte-exact MATLAB.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .data import GoogleResearchData, write_csv
from .engine import load_research_manifests, select_device, _background_frames
from .gmm import EMConfig, MAPConfig, TorchGMM
from .google_reference_features import (
    ReferenceFeature, extract_google_reference,
)
from .metrics import report
from .scoring import ScoreConfig, enrolled_models, score_trials


@dataclass(frozen=True)
class MathWorksResearch:
    reference: ReferenceFeature = field(default_factory=ReferenceFeature)
    em: EMConfig = field(default_factory=lambda: EMConfig(
        mixtures=32, covariance="diag", init="random", iterations=20,
        reg=.02
    ))
    map: MAPConfig = field(default_factory=lambda: MAPConfig(
        relevance=16, adapt_variances=True, adapt_weights=True
    ))
    score: ScoreConfig = field(default_factory=lambda: ScoreConfig(
        norm="none", smoothing=3
    ))
    device: str = "auto"
    seed: int = 2026
    max_background_files: int | None = None
    frame_cap: int = 32
    evaluate_locked_test: bool = False

    @classmethod
    def from_dict(cls, data):
        return cls(
            reference=ReferenceFeature(**data.get("reference", {})),
            em=EMConfig(**data.get("em", {})),
            map=MAPConfig(**data.get("map", {})),
            score=ScoreConfig(**data.get("score", {})),
            **{k: v for k, v in data.items() if k not in ("reference","em","map","score")},
        )


def run_google_mathworks(
    root: str | Path, benchmark: str | Path, config: MathWorksResearch,
    output: str | Path,
):
    """Train/score ONLY from explicit original Google speaker-disjoint manifests."""
    config.reference.validate()
    config.em.validate()
    config.map.validate()
    config.score.validate()
    benchmark = Path(benchmark)
    data = GoogleResearchData(root, benchmark / "google_manifest.csv")
    manifests = load_research_manifests(benchmark, "stop")
    protocol = json.loads((benchmark / "google_benchmark.json").read_text())
    k = int(protocol["enrolment_takes_per_speaker"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    bg = manifests["background"]
    if config.max_background_files is not None:
        selected = set(random.Random(config.seed).sample(
            sorted({r["file_id"] for r in bg}),
            min(len(bg), config.max_background_files),
        ))
        bg = [r for r in bg if r["file_id"] in selected]
    bg_ids = [r["file_id"] for r in bg]
    if not bg_ids:
        raise ValueError("MathWorks research UBM population empty")
    requested = set(bg_ids)
    requested.update(r["file_id"] for r in manifests["dev_enrol"])
    requested.update(r["test_file"] for r in manifests["dev_trials"])
    if config.evaluate_locked_test:
        requested.update(r["file_id"] for r in manifests["eval_enrol"])
        requested.update(r["test_file"] for r in manifests["eval_trials"])
    started = time.perf_counter()
    raw = extract_google_reference(
        data, sorted(requested), config.reference, device
    )
    pooled_background = torch.cat([raw[file] for file in bg_ids]).to(device)
    mean = pooled_background.mean(0)
    std = pooled_background.std(0, unbiased=False).clamp_min(.06)
    # MathWorks reference order: global feature normalization followed by
    # local scalar cepstral mean subtraction per utterance.
    features = {}
    for file, x in raw.items():
        z = (x.to(device) - mean) / std
        features[file] = z - z.mean()
    X = _background_frames(features, bg_ids, config.frame_cap, device)
    torch.manual_seed(config.seed)
    ubm = TorchGMM(config.em).fit(X)
    scores_by_partition = {}
    metrics = {}
    for partition in ("dev", "eval"):
        if partition == "eval" and not config.evaluate_locked_test:
            continue
        enrol = manifests[f"{partition}_enrol"]
        trials = manifests[f"{partition}_trials"]
        # The core GMM function has a 1-3 take limit matching Google protocol.
        targets = enrolled_models(ubm, features, enrol, config.map, k)
        scores = score_trials(ubm, features, trials, targets, config.score).cpu()
        labeled = partition == "dev" or config.evaluate_locked_test
        columns = ["trial_id", "claimed_id", "test_file"] + (
            ["label"] if labeled else []
        ) + ["score"]
        write_csv(output / f"{partition}_scores.csv", [
            {**{key: r[key] for key in columns if key != "score"},
             "score": f"{float(value):.12f}"}
            for r, value in zip(trials, scores)
        ], columns)
        if labeled:
            labels = torch.tensor([int(r["label"]) for r in trials])
            metrics[partition] = report(scores, labels)
        scores_by_partition[partition] = len(scores)
    torch.save({
        "original_google_only": True,
        "reference_frontend": asdict(config.reference),
        "config": asdict(config),
        "global_mean": mean.cpu(), "global_std": std.cpu(),
        "ubm": ubm.snapshot(),
    }, output / "checkpoint.pt")
    result = {
        "source_track": "original_google_mfcc_reference",
        "research_only": True,
        "eligible_for_mandatory_kaggle_submission": False,
        "reference_note": (
            "MFCC magnitude features permitted only for original-Google research. "
            "Speech trimming is energy-based, not exact MATLAB detectSpeech."
        ),
        "config": asdict(config),
        "metrics": metrics["dev"],
        "locked_google_test_metrics": metrics.get("eval"),
        "trial_counts": scores_by_partition,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result

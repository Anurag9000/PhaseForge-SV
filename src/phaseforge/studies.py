"""Controlled GMM initialization/covariance study and speaker-disjoint proxy.

All proxy experiment recordings originate from Kaggle background, and target
and impostor identities are excluded from proxy UBM and PCA fitting.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .data import KaggleData, write_csv
from .engine import RunConfig, _background_frames, extract_features, run, select_device
from .gmm import EMConfig, TorchGMM
from .transforms import fit_background_transform


def proxy_manifests(
    data: KaggleData,
    seed: int = 2026,
    rotation: int = 0,
    maximum_targets: int = 92,
    maximum_impostors: int = 100,
) -> dict:
    """Background-speaker pseudo-verification with three enrolment files per target.

    Targets need >=4 takes; their fourth is held out. All utterances from target
    and unknown impostor speakers are excluded from the proxy UBM. Rotating the
    heldout take is possible for four-fold robustness but folds share speakers.
    """
    if not 0 <= rotation < 4:
        raise ValueError("Four rotations (0-3) supported")
    grouped = defaultdict(list)
    for row in data.background:
        grouped[row["speaker_id"]].append(row["file_id"])
    rng = random.Random(seed)
    eligible = sorted(s for s, x in grouped.items() if len(x) >= 4)
    if len(eligible) < 4:
        raise ValueError("Insufficient four-take speakers for a proxy benchmark")
    targets = rng.sample(eligible, min(maximum_targets, len(eligible)))
    others = sorted(set(grouped) - set(targets))
    impostors = rng.sample(others, min(maximum_impostors, len(others)))
    excluded = set(targets) | set(impostors)
    fit_background = [r for r in data.background if r["speaker_id"] not in excluded]
    if not fit_background:
        raise ValueError("Background proxy requires remaining UBM speakers")
    enrol, trials = [], []
    for speaker in targets:
        recordings = sorted(grouped[speaker])[:4]
        test = recordings[rotation]
        retained = [file for i, file in enumerate(recordings) if i != rotation]
        enrol.extend({"speaker_id": speaker, "file_id": f, "enrol_index": i}
                     for i, f in enumerate(retained))
        trials.append({"claimed_id": speaker, "test_file": test, "label": 1})
        for wrong in rng.sample([s for s in targets if s != speaker], min(3, len(targets) - 1)):
            trials.append({"claimed_id": wrong, "test_file": test, "label": 0})
    for speaker in impostors:
        test = sorted(grouped[speaker])[0]
        for wrong in rng.sample(targets, min(4, len(targets))):
            trials.append({"claimed_id": wrong, "test_file": test, "label": 0})
    rng.shuffle(trials)
    for i, item in enumerate(trials):
        item["trial_id"] = f"proxy_{rotation}_{i:06d}"
    return {
        "background": fit_background,
        "dev_enrol": enrol,
        "dev_trials": trials,
        "eval_enrol": [],
        "eval_trials": [],
    }


def run_proxy(
    data_root: str | Path, config: RunConfig, output: str | Path,
    seed: int = 2026, rotations: int = 4,
) -> dict:
    if not 1 <= rotations <= 4:
        raise ValueError("rotations must be 1..4")
    with KaggleData(data_root) as data:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        outcomes = []
        for rotation in range(rotations):
            manifests = proxy_manifests(data, seed, rotation)
            c = replace(config, evaluate_eval=False)
            target = output / f"rotation_{rotation}"
            write_csv(target / "proxy_enrol.csv", manifests["dev_enrol"],
                      ["speaker_id", "file_id", "enrol_index"])
            write_csv(target / "proxy_trials.csv", manifests["dev_trials"],
                      ["trial_id", "claimed_id", "test_file", "label"])
            outcome = run(data_root, c, target, track="kaggle",
                          manifests_override=manifests)
            outcomes.append({
                "rotation": rotation, "eer": outcome["metrics"]["eer"],
                "auc": outcome["metrics"]["auc"],
                "targets": len({x["speaker_id"] for x in manifests["dev_enrol"]}),
                "ubm_speakers": len({x["speaker_id"] for x in manifests["background"]}),
                "trials": len(manifests["dev_trials"]),
            })
        result = {
            "track": "kaggle_background_proxy",
            "research_only": True,
            "mean_eer": sum(x["eer"] for x in outcomes) / len(outcomes),
            "rotations": outcomes,
            "note": "Rotations share speaker identities; not independent datasets.",
        }
        (output / "proxy_summary.json").write_text(json.dumps(result, indent=2))
        return result


def gmm_behavior_study(
    root: str | Path, config: RunConfig, output: str | Path,
    seeds_per_initializer: int = 20, holdout_fraction: float = .20,
) -> dict:
    """Full assignment behavior study on speaker-disjoint background only.

    Train/validation speaker partition is applied *before* fitting PCA/ICA.
    Outputs 20 random and 20 kmeans fits, covariance/AIC/BIC/heldout LL study
    and an intentional overparameterized degeneracy/remedy comparison.
    """
    if seeds_per_initializer < 20:
        raise ValueError("Assignment requires at least 20 seeds per initializer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with KaggleData(root) as data:
        grouped = defaultdict(list)
        for row in data.background:
            grouped[row["speaker_id"]].append(row["file_id"])
        speakers = sorted(grouped)
        rng = random.Random(6102)
        rng.shuffle(speakers)
        cut = max(1, round(len(speakers) * holdout_fraction))
        holdout = set(speakers[:cut])
        fit_ids = [r["file_id"] for r in data.background if r["speaker_id"] not in holdout]
        held_ids = [r["file_id"] for r in data.background if r["speaker_id"] in holdout]
        data.assert_background_only(fit_ids + held_ids)
        device = select_device(config.device)
        raw = extract_features(data, fit_ids + held_ids, config.feature, output / "cache",
                               device)
        transformed, _ = fit_background_transform(raw, fit_ids, "pca", min(
            12, config.dimensions
        ), seed=2026)
        X = _background_frames(transformed, fit_ids, 12, device)[:5000]
        H = _background_frames(transformed, held_ids, 10, device)[:1400]
        if len(X) < 50 or len(H) < 20:
            raise ValueError("Need larger background speaker populations for diagnostic")
        rows = []
        for method in ("random", "kmeans"):
            for seed in range(2400, 2400 + seeds_per_initializer):
                c = replace(config.em, mixtures=8, covariance="diag", init=method,
                            iterations=35, seed=seed)
                start = time.perf_counter()
                model = TorchGMM(c).fit(X)
                rows.append({
                    "kind": "initialization", "init": method, "seed": seed,
                    "mixtures": 8, "covariance": "diag",
                    "training_ll": model.history[-1],
                    "heldout_ll": model.mean_log_prob(H),
                    "seconds": time.perf_counter() - start,
                    "iterations": len(model.history),
                    "dead_max": max(model.dead_components),
                })
        covariances = []
        for cov in ("spherical", "diag", "tied", "full"):
            for mixtures in (4, 8, 16):
                c = replace(config.em, mixtures=mixtures, covariance=cov,
                            init="kmeans", iterations=25, seed=2026)
                start = time.perf_counter()
                model = TorchGMM(c).fit(X)
                covariances.append({
                    "kind": "covariance", "mixtures": mixtures, "covariance": cov,
                    "training_ll": model.history[-1],
                    "heldout_ll": model.mean_log_prob(H),
                    "seconds": time.perf_counter() - start,
                    "iterations": len(model.history),
                    **model.information_criteria(X),
                })
        tiny = X[:40, :8]
        pathological = TorchGMM(EMConfig(
            mixtures=32, covariance="diag", reg=1e-10,
            init="random", iterations=25, seed=124, min_occupancy=.001
        )).fit(tiny)
        remedied = TorchGMM(EMConfig(
            mixtures=8, covariance="diag", reg=.05, init="kmeans",
            iterations=25, seed=124
        )).fit(tiny)
        init_winners = {}
        for method in ("random", "kmeans"):
            matches = [row for row in rows if row["init"] == method]
            winner = max(matches, key=lambda x: x["training_ll"])
            init_winners[method] = {
                "seed": winner["seed"],
                "training_ll": winner["training_ll"],
                "heldout_ll": winner["heldout_ll"],
                "seed_mean_training_ll": sum(x["training_ll"] for x in matches) / len(matches),
                "seed_mean_heldout_ll": sum(x["heldout_ll"] for x in matches) / len(matches),
            }
        result = {
            "source_track": "kaggle_background_only",
            "seed": 6102, "fit_background_speakers": len(speakers) - len(holdout),
            "heldout_background_speakers": len(holdout),
            "fit_frames": len(X), "heldout_frames": len(H),
            "initialization_fits": rows, "initialization_best_of_20": init_winners,
            "covariance_study": covariances,
            "degeneracy": {
                "pathological_min_variance": float(pathological.cov.min()),
                "pathological_min_weight": float(pathological.weights.min()),
                "remedied_min_variance": float(remedied.cov.min()),
                "remedied_min_weight": float(remedied.weights.min()),
                "interpretation": "Compare unstable tiny fitting population and regularized small K.",
            },
        }
        (output / "gmm_behavior.json").write_text(json.dumps(result, indent=2))
        return result

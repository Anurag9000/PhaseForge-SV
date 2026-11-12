"""Executable research alternatives: true HMM-UBM, time-binned GMM, phase-DTW.

Results are saved separately, and no submission.csv is generated because the
competition's mandatory primary algorithm is the conventional GMM-UBM.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from .data import GoogleResearchData, KaggleData
from .engine import (
    RunConfig, extract_features, load_research_manifests, select_device, _save_scores
)
from .hmm import HMMConfig, LeftRightHMMUBM
from .metrics import grouped_bootstrap, report
from .temporal import PhaseDTW, TimeBinnedUBM
from .transforms import fit_background_transform
from .scratch import ScratchGMMComparison
from .gmm import EMConfig, TorchGMM


def _adapted_templates(
    features: dict[str, torch.Tensor], enrol: list[dict], k: int,
) -> dict[str, list[torch.Tensor]]:
    grouped = defaultdict(list)
    for row in enrol:
        grouped[row["speaker_id"]].append(row)
    return {
        speaker: [features[row["file_id"]]
                  for row in sorted(rows, key=lambda x: int(x["enrol_index"]))[:k]]
        for speaker, rows in grouped.items()
    }


def _models_for_claims(alternative, enrol: dict[str, list[torch.Tensor]], method: str,
                       config: RunConfig) -> dict:
    if method == "dtw":
        return enrol
    return {
        speaker: alternative.adapt(frames, config.map)
        for speaker, frames in enrol.items()
    }


@torch.no_grad()
def _trial_scores(
    alternative, trials: list[dict], features: dict[str, torch.Tensor],
    adapted: dict, method: str,
) -> torch.Tensor:
    scores, cache = [], {}
    for row in trials:
        key = row["claimed_id"], row["test_file"]
        if key not in cache:
            x = features[key[1]]
            if method == "dtw":
                value = alternative.score(x, adapted[key[0]])
            else:
                value = alternative.score(x, adapted[key[0]])
            cache[key] = value
        scores.append(cache[key])
    return torch.stack(scores).double().cpu()


def run_alternative(
    root: str | Path, config: RunConfig, output: str | Path,
    method: str = "hmm", track: str = "kaggle", benchmark: str | None = None,
    ubm_pool: str = "stop", hmm: HMMConfig | None = None,
    time_states: int = 4, mixtures_per_state: int = 8,
) -> dict:
    if method not in ("hmm", "timebin", "dtw", "scratch"):
        raise ValueError("Alternative method must be hmm, timebin, dtw or scratch")
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    if track == "kaggle":
        data = KaggleData(root)
        bg, dev_enrol, dev_trials = data.background, data.dev_enrol, data.dev_trials
        eval_enrol, eval_trials = data.eval_enrol, data.eval_trials
    elif track == "google":
        if not benchmark:
            raise ValueError("Google research requires benchmark")
        data = GoogleResearchData(root, Path(benchmark) / "google_manifest.csv")
        manifests = load_research_manifests(benchmark, ubm_pool)
        bg = manifests["background"]
        dev_enrol, dev_trials = manifests["dev_enrol"], manifests["dev_trials"]
        eval_enrol, eval_trials = manifests["eval_enrol"], manifests["eval_trials"]
    else:
        raise ValueError(track)
    bg_ids = sorted(set(r["file_id"] for r in bg))
    if config.max_background_files is not None:
        bg_ids = random.Random(2026).sample(bg_ids, min(len(bg_ids), config.max_background_files))
    if isinstance(data, KaggleData):
        data.assert_background_only(bg_ids)
    requested = set(bg_ids)
    requested.update(r["file_id"] for r in dev_enrol)
    requested.update(r["test_file"] for r in dev_trials)
    if config.evaluate_eval:
        requested.update(r["file_id"] for r in eval_enrol)
        requested.update(r["test_file"] for r in eval_trials)
    start = time.perf_counter()
    feature = extract_features(
        data, sorted(requested), config.feature,
        output.parent / ".phase_cache", device
    )
    transformed, transform = fit_background_transform(
        feature, bg_ids, config.transform, config.dimensions,
        config.max_fit_transform_frames
    )
    feature = {key: tensor.to(device) for key, tensor in transformed.items()}
    recordings = [feature[ident] for ident in bg_ids]
    if method == "hmm":
        alternative = LeftRightHMMUBM(hmm or HMMConfig()).fit(recordings)
        history = alternative.history
    elif method == "timebin":
        alternative = TimeBinnedUBM(time_states, mixtures_per_state).fit(
            recordings, config.em
        )
        history = [m.history for m in alternative.models]
    elif method == "scratch":
        training_frames = torch.cat([
            recording[torch.linspace(
                0, len(recording) - 1, min(len(recording), config.max_frames_per_background_file),
                device=recording.device,
            ).long()] for recording in recordings
        ], dim=0)
        bg_ubm = TorchGMM(config.em).fit(training_frames)
        alternative = ScratchGMMComparison(
            bg_ubm, mixtures=min(8, config.em.mixtures),
            covariance_floor=max(.05, config.em.reg),
            max_iterations=config.em.iterations,
            seed=config.em.seed
        )
        history = bg_ubm.history
    else:
        alternative = PhaseDTW()
        history = []
    enroll_dev = _adapted_templates(feature, dev_enrol, config.enrolment_size)
    adapted_dev = _models_for_claims(alternative, enroll_dev, method, config)
    dev_scores = _trial_scores(alternative, dev_trials, feature, adapted_dev, method)
    labels = torch.tensor([int(r["label"]) for r in dev_trials])
    _save_scores(output / "development_scores.csv", dev_trials, dev_scores, labeled=True)
    if config.evaluate_eval:
        enroll_eval = _adapted_templates(feature, eval_enrol, config.enrolment_size)
        adapted_eval = _models_for_claims(alternative, enroll_eval, method, config)
        eval_scores = _trial_scores(alternative, eval_trials, feature, adapted_eval, method)
        _save_scores(
            output / "evaluation_scores.csv", eval_trials, eval_scores,
            labeled=(track == "google"),
        )
    test_metrics = None
    if track == "google" and config.evaluate_eval:
        if not all("label" in trial for trial in eval_trials):
            raise ValueError("Missing labels in locked Google research test")
        test_labels = torch.tensor([int(row["label"]) for row in eval_trials])
        test_metrics = report(eval_scores, test_labels, config.target_far)
        test_metrics["bootstrap_by_speaker"] = grouped_bootstrap(
            eval_trials, eval_scores, group_key="claimed_id"
        )
        (output / "locked_google_test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2)
        )
    result = {
        "method": method, "track": track, "research_only": True,
        "eligible_for_mandatory_kaggle_baseline": False,
        "config": config.to_dict(), "hmm_config": asdict(hmm or HMMConfig())
        if method == "hmm" else None,
        "metrics": report(dev_scores, labels),
        "locked_google_test_metrics": test_metrics,
        "bootstrap_by_speaker": grouped_bootstrap(dev_trials, dev_scores, group_key="claimed_id"),
        "training_history": history,
        "elapsed_seconds": time.perf_counter() - start,
        "background_recordings": len(bg_ids),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    if isinstance(data, KaggleData):
        data.close()
    return result

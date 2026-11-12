"""Robustness tests on frozen, Kaggle-trained models and development trials only.

This reuses a trained UBM and designated enrolment MAP models. Audio perturbations
are applied exclusively to *test* recordings. No external noise data, no Google
training, no fitting on dev/eval tests. Mandatory Kaggle scoring is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .data import KaggleData, write_csv
from .engine import RunConfig, _cohort, extract_features, select_device
from .features import PhaseExtractor
from .gmm import TorchGMM
from .learned import PhaseConvEncoder, PhaseEncoderConfig
from .metrics import report
from .scoring import cohort_normalize, enrolled_models, score_trials


def perturb_waveform(
    audio: torch.Tensor, kind: str, parameter: float = 0,
    seed: int = 2026,
) -> torch.Tensor:
    """Shape-preserving gain, shift or Gaussian noise at a specified SNR dB."""
    y = audio.flatten().float()
    if kind == "clean":
        return y.clone()
    if kind == "gain":
        if parameter <= 0:
            raise ValueError("Gain must be positive")
        return y * parameter
    if kind == "shift":
        shift = int(parameter)
        if shift < 0 or shift >= len(y):
            raise ValueError("Shift must be within recording length")
        result = torch.zeros_like(y)
        if shift == 0:
            result.copy_(y)
        else:
            result[shift:] = y[:-shift]
        return result
    if kind == "noise":
        # SNR=20log10(rms_signal/rms_noise). Zero signal stays exactly silent.
        power = y.square().mean()
        if float(power) < 1e-15:
            return y.clone()
        generator = torch.Generator(device=y.device).manual_seed(seed)
        noise = torch.randn(y.shape, generator=generator, device=y.device)
        noise = noise / noise.square().mean().sqrt().clamp_min(1e-12)
        scale = power.sqrt() / (10.0 ** (parameter / 20.0))
        return y + noise * scale
    raise ValueError(f"Unknown perturbation: {kind}")


def default_conditions() -> list[dict]:
    return [
        {"name": "clean", "kind": "clean", "value": 0},
        {"name": "gain_half", "kind": "gain", "value": 0.5},
        {"name": "gain_double", "kind": "gain", "value": 2},
        {"name": "noise_20dB", "kind": "noise", "value": 20},
        {"name": "noise_10dB", "kind": "noise", "value": 10},
        {"name": "noise_5dB", "kind": "noise", "value": 5},
        {"name": "shift_80samples", "kind": "shift", "value": 80},
        {"name": "shift_160samples", "kind": "shift", "value": 160},
    ]


def _transformer(artifact: dict, device: torch.device):
    transform = artifact["transformation"]
    if isinstance(transform, dict) and "encoder_state" in transform:
        config = PhaseEncoderConfig(**transform["config"])
        model = PhaseConvEncoder(
            transform["original_phase_dimensions"], config
        ).to(device)
        model.load_state_dict(transform["encoder_state"])
        model.eval()
        return lambda x: model(x[None]).squeeze(0)
    if transform is None:
        return lambda x: x
    # PCA/ICA tensors are saved as objects by the engine, moved explicitly here.
    if hasattr(transform, "mean"):
        transform.mean = transform.mean.to(device)
    if hasattr(transform, "components"):
        transform.components = transform.components.to(device)
        transform.values = transform.values.to(device)
    if hasattr(transform, "pca"):
        transform.pca.mean = transform.pca.mean.to(device)
        transform.pca.components = transform.pca.components.to(device)
        transform.pca.values = transform.pca.values.to(device)
        transform.W = transform.W.to(device)
    return transform


@torch.no_grad()
def run_robustness(
    data_root: str | Path, trained_run: str | Path, output: str | Path,
    seed: int | None = None, conditions: list[dict] | None = None,
) -> dict:
    """Freeze one checkpoint and probe the SAME dev trial structure under corruption.

    A single specified seed is compared against its own unperturbed score,
    NOT against a multi-seed ensemble's EER.
    """
    trained_run, output = Path(trained_run), Path(output)
    result_file = trained_run / "result.json"
    result = json.loads(result_file.read_text())
    if result.get("track") != "kaggle" or result.get("research_only") is not False:
        raise PermissionError("Robustness needs a Kaggle-only trained UBM checkpoint")
    config = RunConfig.from_dict(result["config"])
    if seed is None:
        seed = config.seeds[0]
    if seed not in config.seeds:
        raise ValueError("Requested seed not present in trained run")
    checkpoint = trained_run / f"model_seed_{seed}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    # Only load model artifacts produced by yourself; torch.load pickle is not
    # safe for untrusted checkpoints.
    artifact = torch.load(checkpoint, map_location=device, weights_only=False)
    if artifact.get("source_track") != "kaggle":
        raise PermissionError("External model parameters are prohibited")
    if artifact["feature"] != asdict(config.feature):
        raise ValueError("Checkpoint feature configuration differs from training record")
    with KaggleData(data_root) as data:
        if Path(result.get("source", "")).resolve() != data.root:
            raise PermissionError("Dataset root differs from recorded training source")
        ubm = TorchGMM.from_snapshot(artifact["gmm"], device=str(device))
        bg_ids = list(dict.fromkeys(r["file_id"] for r in data.background))
        bg = data.background
        if config.max_background_files is not None:
            subset = random.Random(2026).sample(
                bg_ids, min(len(bg_ids), config.max_background_files)
            )
            bg_ids = sorted(subset)
            keep = set(bg_ids)
            bg = [r for r in bg if r["file_id"] in keep]
        data.assert_background_only(bg_ids)
        tests = sorted({r["test_file"] for r in data.dev_trials})
        ids = list(dict.fromkeys(
            bg_ids + [r["file_id"] for r in data.dev_enrol] + tests
        ))
        raw = extract_features(
            data, ids, config.feature, output.parent / ".phase_cache", device
        )
        transform = _transformer(artifact, device)
        clean_features = {k: transform(v.to(device)) for k, v in raw.items()}
        speakers = enrolled_models(
            ubm, clean_features, data.dev_enrol, config.map, config.enrolment_size
        )
        cohorts, background_tests = _cohort(
            bg, clean_features, ubm, config.map, config.score, seed
        )
        extractor = PhaseExtractor(config.feature)
        baseline_labels = torch.tensor([int(r["label"]) for r in data.dev_trials])
        summaries = []
        baseline_scores = None
        for spec in conditions or default_conditions():
            kind, value = str(spec["kind"]), float(spec.get("value", 0))
            name = str(spec["name"])
            altered = dict(clean_features)
            if kind != "clean":
                for ident in tests:
                    # Stable per-file seed, never randomized by Python hash salt.
                    digest = hashlib.sha256(
                        f"{seed}:{ident}:{name}".encode()
                    ).digest()
                    item_seed = int.from_bytes(digest[:4], "little")
                    waveform = data.load(ident).to(device)
                    modified = perturb_waveform(waveform, kind, value, item_seed)
                    altered[ident] = transform(extractor(modified))
            raw_score = score_trials(
                ubm, altered, data.dev_trials, speakers, config.score
            )
            if config.score.norm != "none":
                scores = cohort_normalize(
                    ubm, altered, data.dev_trials, raw_score, speakers,
                    cohorts, background_tests, config.score
                ).cpu()
            else:
                scores = raw_score.cpu()
            if baseline_scores is None:
                baseline_scores = scores
            detail = report(scores, baseline_labels, config.target_far)
            detail.update({
                "condition": name,
                "kind": kind,
                "value": value,
                "score_mean_absolute_shift": float(
                    (scores - baseline_scores).abs().mean()
                ),
            })
            summaries.append(detail)
            write_csv(
                output / f"development_{name}.csv",
                [
                    {"trial_id": r["trial_id"], "claimed_id": r["claimed_id"],
                     "test_file": r["test_file"], "label": r["label"],
                     "score": f"{float(score):.12f}"}
                    for r, score in zip(data.dev_trials, scores)
                ],
                ["trial_id", "claimed_id", "test_file", "label", "score"]
            )
        output_report = {
            "source_track": "kaggle_background_only",
            "method": "frozen_model_perturbed_dev_trials",
            "checkpoint_seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "conditions": summaries,
            "reference_original_ensemble_dev_eer": result["metrics"]["eer"],
            "warning": "Results are one frozen seed; do not compare against multi-seed original without accounting for this.",
        }
        (output / "robustness.json").write_text(json.dumps(output_report, indent=2))
        return output_report

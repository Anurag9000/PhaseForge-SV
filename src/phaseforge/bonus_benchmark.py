"""Reproducible bonus: native PyTorch GMM versus CSOgroup TorchGMM and NVIDIA cuVS.

This is a BENCHMARK/RESEARCH tool; external packages are never imported
by the mandatory Kaggle training pipeline. cuVS GMM uses the verified
C++ interface, via the separately compiled bonus_cuvs_gmm binary.

Two distinct experiments:
  * Fit+initialization+EM, followed by held-out log likelihood and MAP EER.
  * Held-out score_samples using the trained model.
Implementations do not share identical EM initialization/convergence
semantics. Compare them on identical data/feature/K/covariance/seed and
record the differences; never assert parameter equivalence across fits.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .engine import extract_features
from .features import FeatureConfig
from .gmm import EMConfig, MAPConfig, TorchGMM
from .metrics import report
from .scoring import ScoreConfig, enrolled_models, score_trials


@dataclass(frozen=True)
class BenchCase:
    name: str
    source: str = "synthetic"  # synthetic | kaggle
    frames: int = 3000
    dimensions: int = 16
    mixtures: int = 8
    covariance: str = "diag"
    iterations: int = 15
    reg: float = .02
    seed: int = 2026
    repeats: int = 2
    eval_frames: int = 500
    regime: str = "separated"  # separated/overlap/near_singular/imbalanced/offset/outliers
    dtype: str = "float32"
    init: str = "kmeans"
    frame_chunk: int = 1024
    feature_mode: str = "group"
    max_background_files: int = 240
    max_dev_trials: int = 0  # 0=skip real verification, >0=score full dev protocol

    def validate(self):
        if (self.source not in ("synthetic", "kaggle") or
            self.regime not in (
                "separated", "overlap", "near_singular", "imbalanced", "offset", "outliers"
            ) or self.covariance not in ("diag", "spherical", "tied", "full") or
            self.dtype not in ("float32", "float64") or
            self.init not in ("random", "kmeans")):
            raise ValueError("Unsupported benchmark case")
        if (self.frames <= self.mixtures or self.eval_frames < 2 or
            self.dimensions < 1 or self.mixtures < 1 or self.iterations < 1 or
            self.repeats < 1 or self.frame_chunk < 1 or self.reg <= 0):
            raise ValueError("Invalid benchmark size or numerical configuration")
        if self.max_dev_trials < 0:
            raise ValueError("max_dev_trials cannot be negative")


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_logp(model: TorchGMM, data: torch.Tensor, chunk: int):
    return torch.cat([model.log_prob(x).detach() for x in data.split(chunk)])


def _rss_highwater_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024. if sys.platform.startswith("linux") else rss / (1024.**2)


def _cuda_snapshot(device: torch.device):
    if device.type != "cuda":
        return None
    return {
        "allocated_peak_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "reserved_peak_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def machine_info():
    info = {
        "python": sys.version, "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "packages": {},
    }
    if torch.cuda.is_available():
        info["gpus"] = [
            {"name": torch.cuda.get_device_name(i),
             "capability": torch.cuda.get_device_capability(i),
             "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())
        ]
    for package in ("torchgmm", "lightning", "cuvs-cu12", "cuvs-cu13"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    try:
        cp = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True, timeout=5)
        info["git_commit"] = cp.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


def synthetic_data(case: BenchCase):
    """Seeded known-mixture data; true generating parameters are recorded."""
    rng = torch.Generator(device="cpu").manual_seed(case.seed)
    dtype = torch.float64 if case.dtype == "float64" else torch.float32
    separation = .7 if case.regime in ("overlap", "near_singular") else 4.
    means = torch.randn(case.mixtures, case.dimensions, generator=rng).to(dtype) * separation
    probs = torch.ones(case.mixtures, dtype=dtype)
    if case.regime == "imbalanced":
        probs = torch.linspace(1, 15, case.mixtures, dtype=dtype) ** -2
    probs /= probs.sum()
    assign = torch.multinomial(probs, case.frames + case.eval_frames,
                                replacement=True, generator=rng)
    noise = torch.randn(case.frames + case.eval_frames, case.dimensions,
                        generator=rng).to(dtype)
    if case.regime == "near_singular" and case.dimensions > 1:
        # Perfectly correlated directions up to a small explicit perturbation.
        noise[:, 1:] = noise[:, :1] + 1e-4 * noise[:, 1:]
    data = means[assign] + noise * (.5 if case.regime == "separated" else 1)
    if case.regime == "offset":
        data += 10000.
    if case.regime == "outliers":
        count = max(1, len(data) // 100)
        data[-count:] += 75 * torch.randn(count, case.dimensions, generator=rng).to(dtype)
    train = data[:case.frames].contiguous()
    held = data[case.frames:].contiguous()
    return train, held, {
        "case_seed": case.seed, "true_mixture_weights": probs.tolist(),
        "source": "synthetic", "regime": case.regime,
        "true_means_sha256": hashlib.sha256(
            array("d" if case.dtype == "float64" else "f", means.cpu().reshape(-1).tolist()).tobytes()
        ).hexdigest(),
    }, None


def kaggle_data(case: BenchCase, root: Path, cache: Path):
    """Train on background.csv only; held-out likelihood uses OTHER background speakers.

    Optional dev speaker trials receive MAP adaptation but never affect the UBM.
    """
    from .data import KaggleData
    data = KaggleData(root)
    bg = list(data.background)
    speakers = sorted({r["speaker_id"] for r in bg})
    if len(speakers) < 4:
        raise ValueError("Need >=4 distinct competition background speakers")
    # Deterministically hold out speakers from the fit to measure density generalization.
    chosen = set(speakers[::5])
    train_rows = [r for r in bg if r["speaker_id"] not in chosen]
    hold_rows = [r for r in bg if r["speaker_id"] in chosen]
    if case.max_background_files:
        train_rows = train_rows[:case.max_background_files]
        hold_rows = hold_rows[:max(5, case.max_background_files // 5)]
    ids = {r["file_id"] for r in train_rows + hold_rows}
    trials = list(data.dev_trials[:case.max_dev_trials]) if case.max_dev_trials else []
    if trials:
        ids.update(r["file_id"] for r in data.dev_enrol)
        ids.update(r["test_file"] for r in trials)
    cfg = FeatureConfig(mode=case.feature_mode)
    features = extract_features(data, sorted(ids), cfg, cache, torch.device("cpu"))
    train = torch.cat([features[r["file_id"]] for r in train_rows])
    held = torch.cat([features[r["file_id"]] for r in hold_rows])
    if case.dimensions < train.shape[1]:
        # Feature truncation is identical for all backends; avoid fitting PCA with
        # the held-out data, and record this as a truncation (not PCA).
        train, held = train[:, :case.dimensions], held[:, :case.dimensions]
        features = {key: val[:, :case.dimensions] for key, val in features.items()}
    elif case.dimensions != train.shape[1]:
        raise ValueError("Requested dimension exceeds extracted phase features")
    if len(train) < case.mixtures:
        raise ValueError("Too few background phase frames for requested GMM")
    train = train[:case.frames].contiguous()
    held = held[:case.eval_frames].contiguous()
    dev = (features, list(data.dev_enrol), trials) if trials else None
    provenance = {
        "source": "kaggle_background_only", "dataset": str(root.resolve()),
        "fit_background_speakers": len({r["speaker_id"] for r in train_rows}),
        "heldout_background_speakers": len({r["speaker_id"] for r in hold_rows}),
        "dev_label_uses": "evaluation only",
        "feature_mode": case.feature_mode,
        "feature_dimensions": train.shape[1],
    }
    data.close()
    return train, held, provenance, dev


def _to_model(
    weights: torch.Tensor, means: torch.Tensor, cov: torch.Tensor,
    case: BenchCase, device: torch.device,
) -> TorchGMM:
    config = EMConfig(
        mixtures=case.mixtures, covariance=case.covariance,
        init=case.init, iterations=case.iterations, reg=case.reg,
        chunk=case.frame_chunk, seed=case.seed,
    )
    model = TorchGMM(config)
    model.weights = weights.to(device).detach()
    model.mu = means.to(device).detach()
    model.cov = cov.to(device).detach()
    if not (torch.isfinite(model.weights).all()
            and torch.isfinite(model.mu).all()
            and torch.isfinite(model.cov).all()):
        raise FloatingPointError("Backend produced non-finite UBM parameters")
    if not torch.isclose(model.weights.sum(), model.weights.new_tensor(1.), atol=1e-3):
        raise FloatingPointError("Backend mixture weights are not normalized")
    return model


def _dev_metrics(model, dev, device, case):
    if dev is None:
        return None
    features, enrol, trials = dev
    F = {k: v.to(device) for k, v in features.items()
         if k in {r["file_id"] for r in enrol} | {r["test_file"] for r in trials}}
    adapted = enrolled_models(model, F, enrol, MAPConfig(relevance=8.), k=3)
    scores = score_trials(model, F, trials, adapted, ScoreConfig(norm="none")).cpu()
    labels = torch.tensor([int(t["label"]) for t in trials])
    if not labels.any() or not (~labels.bool()).any():
        raise ValueError("Selected development subset lacks both genuine/impostor trials")
    return report(scores, labels)


def _export_float32(path, tensor):
    with Path(path).open("wb") as writer:
        array("f", tensor.detach().float().cpu().reshape(-1).tolist()).tofile(writer)


def _read_float32(path):
    with Path(path).open("rb") as reader:
        values = array("f")
        values.fromfile(reader, Path(path).stat().st_size // 4)
    return torch.tensor(values, dtype=torch.float32)


def _run_native(case, train, held, device, dev):
    X = train.to(device)
    V = held.to(device)
    torch.manual_seed(case.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_highwater_mib()
    synchronize(device)
    start = time.perf_counter()
    cfg = EMConfig(
        mixtures=case.mixtures, covariance=case.covariance,
        init=case.init, iterations=case.iterations, tolerance=1e-10,
        reg=case.reg, chunk=case.frame_chunk, seed=case.seed,
        min_occupancy=2,
    )
    model = TorchGMM(cfg).fit(X)
    synchronize(device)
    fit_seconds = time.perf_counter() - start
    _ = _mean_logp(model, V[:min(20, len(V))], case.frame_chunk)
    synchronize(device)
    t0 = time.perf_counter()
    values = _mean_logp(model, V, case.frame_chunk)
    synchronize(device)
    score_seconds = time.perf_counter() - t0
    sanity = {
        "sum_weights": float(model.weights.sum()),
        "min_covariance_value": float(model.cov.min()),
        "nonfinite_heldout_logprobs": int((~torch.isfinite(values)).sum()),
        "iterations_reported": len(model.history),
    }
    metrics = _dev_metrics(model, dev, device, case)
    return {
        "backend": "native", "status": "complete", "fit_seconds": fit_seconds,
        "score_seconds": score_seconds, "score_frames": len(V),
        "log_likelihood_mean": float(values.double().mean()),
        "log_likelihood_std": float(values.double().std(unbiased=False)),
        "rss_highwater_mib": _rss_highwater_mib(),
        "rss_highwater_before_mib": rss_before,
        "gpu_memory": _cuda_snapshot(device),
        "sanity": sanity, "dev_verification": metrics,
        "model": model,
    }


def _run_torchgmm(case, train, held, device, dev):
    from torchgmm.bayes import GaussianMixture as ThirdPartyGaussianMixture
    # API verified in CSOgroup/torchgmm estimator.py. The estimator's
    # score_samples returns NEGATIVE log likelihood; the model returns log p.
    accelerator = "gpu" if device.type == "cuda" else "cpu"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_highwater_mib()
    torch.manual_seed(case.seed)
    trainer = {
        "accelerator": accelerator, "devices": 1,
        "max_epochs": case.iterations,
        "enable_progress_bar": False, "logger": False,
        "enable_checkpointing": False,
        "precision": "64-true" if case.dtype == "float64" else "32-true",
    }
    estimator = ThirdPartyGaussianMixture(
        num_components=case.mixtures,
        covariance_type=case.covariance,
        init_strategy=case.init,
        covariance_regularization=case.reg,
        convergence_tolerance=1e-10,
        batch_size=min(case.frame_chunk, len(train)),
        trainer_params=trainer,
    )
    # Lightning manages device moves; inputs remain CPU for its DataLoader.
    X = train.cpu()
    synchronize(device)
    start = time.perf_counter()
    estimator.fit(X)
    synchronize(device)
    fit_seconds = time.perf_counter() - start
    module = estimator.model_
    model_device = module.means.device
    held_device = held.to(model_device)
    with torch.no_grad():
        _ = module(held_device[:min(20, len(held_device))])[1]
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        values = torch.cat([
            module(block)[1].detach()
            for block in held_device.split(case.frame_chunk)
        ])
    synchronize(device)
    score_seconds = time.perf_counter() - t0
    model = _to_model(
        module.component_probs, module.means,
        module.covariances, case, device,
    )
    metrics = _dev_metrics(model, dev, device, case)
    return {
        "backend": "torchgmm", "status": "complete",
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds, "score_frames": len(held),
        "log_likelihood_mean": float(values.double().mean()),
        "log_likelihood_std": float(values.double().std(unbiased=False)),
        "rss_highwater_mib": _rss_highwater_mib(),
        "rss_highwater_before_mib": rss_before,
        "gpu_memory": _cuda_snapshot(device),
        "sanity": {
            "sum_weights": float(module.component_probs.sum()),
            "min_covariance_value": float(module.covariances.min()),
            "nonfinite_heldout_logprobs": int((~torch.isfinite(values)).sum()),
            "iterations_reported": estimator.num_iter_,
            "converged": estimator.converged_,
            "nll_train": float(estimator.nll_),
        },
        "dev_verification": metrics,
        "model": model,
    }


def _run_cuvs(case, train, held, device, dev, binary, directory):
    if case.dtype != "float32":
        raise ValueError("The bundled cuVS binary uses float32; use a float32 case")
    if device.type != "cuda":
        raise ValueError("cuVS requires a CUDA device")
    if binary is None or not Path(binary).is_file():
        raise FileNotFoundError("cuVS native C++ binary not installed/compiled")
    directory.mkdir(parents=True, exist_ok=True)
    tx, vx, model_path = (
        directory / "train.f32", directory / "heldout.f32", directory / "fitted.f32"
    )
    _export_float32(tx, train)
    _export_float32(vx, held)
    args = [
        str(binary), "--train", str(tx), "--eval", str(vx),
        "--model", str(model_path),
        "--n", str(len(train)), "--d", str(train.shape[1]),
        "--eval-n", str(len(held)), "--k", str(case.mixtures),
        "--covariance", case.covariance, "--iterations", str(case.iterations),
        "--reg", str(case.reg), "--seed", str(case.seed),
        "--init", case.init,
    ]
    p = subprocess.run(args, capture_output=True, text=True, timeout=7200)
    if p.returncode:
        raise RuntimeError(
            f"cuVS C++ harness exit {p.returncode}: {p.stderr[-1500:]}"
        )
    result = json.loads(p.stdout.strip().splitlines()[-1])
    if not model_path.is_file():
        raise FileNotFoundError("cuVS did not export fitted model")
    K, D = case.mixtures, train.shape[1]
    cov_n = {"diag": K * D, "spherical": K, "tied": D * D,
             "full": K * D * D}[case.covariance]
    raw = _read_float32(model_path)
    if raw.numel() != K + K * D + cov_n:
        raise ValueError("cuVS fitted parameters have unexpected length")
    mean = raw[K:K+K*D].reshape(K, D)
    covariance = raw[K+K*D:]
    if case.covariance == "diag":
        covariance = covariance.reshape(K, D)
    elif case.covariance == "full":
        covariance = covariance.reshape(K, D, D)
    elif case.covariance == "tied":
        covariance = covariance.reshape(D, D)
    model = _to_model(raw[:K], mean, covariance, case, device)
    result.update({
        "backend": "cuvs", "status": "complete",
        "dev_verification": _dev_metrics(model, dev, device, case),
        "sanity": {
            "sum_weights": float(model.weights.sum()),
            "min_covariance_value": float(model.cov.min()),
            "iterations_reported": result.get("iterations_reported"),
            "nonfinite_heldout_logprobs": result.get("nonfinite_heldout_logprobs"),
        }
    })
    return {**result, "model": model}


def _serializable(item: dict[str, Any]):
    return {key: value for key, value in item.items() if key != "model"}


def run_bonus(
    cases: list[BenchCase], output: Path, backends: list[str],
    device: torch.device, kaggle_root: Path | None = None,
    cuvs_binary: Path | None = None, strict: bool = False,
):
    output.mkdir(parents=True, exist_ok=True)
    if not cases:
        raise ValueError("No benchmark configurations")
    rows = []
    info = machine_info()
    for case in cases:
        case.validate()
        directory = output / case.name
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if case.source == "synthetic":
                train, held, provenance, dev = synthetic_data(case)
            else:
                if kaggle_root is None:
                    raise ValueError("Kaggle case requires --kaggle-data")
                train, held, provenance, dev = kaggle_data(
                    case, kaggle_root, output / "phase_cache"
                )
            source_hash = hashlib.sha256(
                array("d" if case.dtype == "float64" else "f", train.cpu().reshape(-1).tolist()).tobytes()
                + array("d" if case.dtype == "float64" else "f", held.cpu().reshape(-1).tolist()).tobytes()
            ).hexdigest()
        except Exception as exc:
            if strict:
                raise
            rows.append({"case": case.name, "backend": "data",
                         "status": "failed", "error": repr(exc)})
            continue
        for repeat in range(case.repeats):
            repeated = BenchCase(**{**asdict(case), "seed": case.seed + repeat})
            for name in backends:
                try:
                    if name == "native":
                        result = _run_native(repeated, train, held, device, dev)
                    elif name == "torchgmm":
                        result = _run_torchgmm(repeated, train, held, device, dev)
                    elif name == "cuvs":
                        result = _run_cuvs(
                            repeated, train, held, device, dev,
                            cuvs_binary, directory / f"cuvs_seed_{repeated.seed}"
                        )
                    else:
                        raise ValueError(f"Unknown benchmark backend: {name}")
                    entry = _serializable(result)
                except (ImportError, FileNotFoundError) as exc:
                    entry = {"backend": name, "status": "skipped",
                             "reason": repr(exc)}
                    if strict:
                        raise
                except Exception as exc:
                    entry = {"backend": name, "status": "failed",
                             "error": repr(exc),
                             "traceback": traceback.format_exc(limit=8)}
                    if strict:
                        raise
                entry.update({
                    "case": case.name, "case_settings": asdict(repeated),
                    "repeat": repeat, "data_sha256": source_hash,
                    "source_provenance": provenance,
                    "device": str(device),
                    "fit_comparison_warning": (
                        "Init and stop rules differ between implementations; "
                        "matched data/K/cov/seed/regularization, not same EM iterates."
                    ),
                })
                rows.append(entry)
                (output / "bonus_results.json").write_text(json.dumps({
                    "environment": info, "runs": rows
                }, indent=2))
                print(
                    f"BONUS {case.name} repetition={repeat} backend={name}: "
                    f"{entry['status']} fit={entry.get('fit_seconds')}",
                    flush=True,
                )
    with (output / "bonus_summary.csv").open("w", newline="") as fh:
        fields = [
            "case", "repeat", "backend", "status", "fit_seconds", "score_seconds",
            "score_frames", "log_likelihood_mean", "log_likelihood_std",
            "rss_highwater_mib", "data_sha256", "device",
        ]
        writer = csv.DictWriter(fh, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backends", nargs="+", default=["native", "torchgmm", "cuvs"],
                        choices=["native", "torchgmm", "cuvs"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--kaggle-data", default=None)
    parser.add_argument("--cuvs-binary", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but this machine has no CUDA-enabled PyTorch")
    cases = [BenchCase(**o) for o in cfg["cases"]]
    rows = run_bonus(
        cases, Path(args.output), args.backends, device,
        Path(args.kaggle_data) if args.kaggle_data else None,
        Path(args.cuvs_binary) if args.cuvs_binary else None,
        args.strict
    )
    if not any(r.get("status") == "complete" for r in rows):
        raise RuntimeError("No successful benchmark comparisons; see bonus_results.json")


if __name__ == "__main__":
    main()

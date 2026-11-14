"""Strictly identical-parameter GMM inference: native, TorchGMM and cuVS.

All comparators receive the SAME fitted weights, means and variances, not
independently trained mixture models. Reports max/mean absolute log-p
differences, throughput, and numerical robustness. cuVS uses its C++ API
through the optional compiled adapter; no invented Python GMM interface.
"""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import subprocess
import time

import torch

from .bonus_benchmark import (
    BenchCase, _export_float32, _read_float32, kaggle_data, synthetic_data,
    synchronize,
)
from .gmm import EMConfig, TorchGMM


@torch.no_grad()
def compare_identical(
    case: BenchCase, device: torch.device,
    output: str | Path, backends: list[str],
    cuvs_binary: str | Path | None = None,
    kaggle_root: str | Path | None = None,
) -> dict:
    """Fit ONE native GMM and time different scoring kernels on SAME model."""
    if case.covariance not in ("diag", "spherical"):
        raise ValueError("Frozen-parameter bonus supports diag/spherical only")
    if case.dtype != "float32":
        raise ValueError("Frozen cuVS parity uses float32 inputs")
    case.validate()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if case.source == "synthetic":
        train, held, provenance, _ = synthetic_data(case)
    else:
        if kaggle_root is None:
            raise ValueError("Missing Kaggle source for real frozen-GMM parity")
        train, held, provenance, _ = kaggle_data(case, Path(kaggle_root),
                                               out / "phase_cache")
    X, V = train.to(device), held.to(device)
    native = TorchGMM(EMConfig(
        mixtures=case.mixtures, covariance=case.covariance,
        init=case.init, iterations=case.iterations, tolerance=1e-10,
        reg=case.reg, seed=case.seed, chunk=case.frame_chunk
    )).fit(X)
    for item in (native.weights, native.mu, native.cov):
        if not torch.isfinite(item).all():
            raise FloatingPointError("Unstable reference model")
    def timed(func):
        _ = func()
        synchronize(device)
        begin = time.perf_counter()
        result = func().detach().cpu().double()
        synchronize(device)
        return result, time.perf_counter() - begin
    truth, baseline_s = timed(lambda: native.log_prob(V))
    rows = {
        "native": {
            "status":"complete", "score_seconds":baseline_s,
            "max_abs_logp_diff":0.0, "mean_abs_logp_diff":0.0,
            "frames":len(V), "device":str(device)
        }
    }
    if "torchgmm" in backends:
        try:
            from torchgmm.bayes.gmm.model import (
                GaussianMixtureModel, GaussianMixtureModelConfig,
            )
            module = GaussianMixtureModel(
                GaussianMixtureModelConfig(
                    num_components=case.mixtures,
                    num_features=V.shape[1],
                    covariance_type=case.covariance,
                )
            ).to(device)
            with torch.no_grad():
                module.component_probs.copy_(native.weights)
                module.means.copy_(native.mu)
                module.precisions_cholesky.copy_(native.cov.rsqrt())
            scores, seconds = timed(lambda: module(V)[1])
            diff = (scores - truth).abs()
            rows["torchgmm"] = {
                "status":"complete", "score_seconds":seconds,
                "max_abs_logp_diff":float(diff.max()),
                "mean_abs_logp_diff":float(diff.mean()),
                "frames":len(V), "device":str(device),
            }
        except (ImportError, ModuleNotFoundError) as err:
            rows["torchgmm"] = {"status":"skipped","reason":repr(err)}
    if "cuvs" in backends:
        binary = Path(cuvs_binary) if cuvs_binary else None
        if binary is None or not binary.is_file():
            rows["cuvs"] = {
                "status":"skipped",
                "reason":"cuVS C++ binary not built/available",
            }
        elif device.type != "cuda":
            rows["cuvs"] = {"status":"skipped","reason":"CUDA device required"}
        else:
            train_bin, eval_bin, model_bin, score_bin = [
                out / file for file in (
                    "native_train.f32", "native_eval.f32",
                    "native_ubm.f32", "cuvs_scores.f32"
                )
            ]
            _export_float32(train_bin, train)
            _export_float32(eval_bin, held)
            with model_bin.open("wb") as stream:
                for tensor in (native.weights, native.mu, native.cov):
                    array("f", tensor.cpu().reshape(-1).tolist()).tofile(stream)
            command = [
                str(binary), "--score-only",
                "--train", str(train_bin), "--eval", str(eval_bin),
                "--model", str(model_bin), "--scores-out", str(score_bin),
                "--n", str(len(train)), "--eval-n", str(len(held)),
                "--d", str(V.shape[1]), "--k", str(case.mixtures),
                "--covariance", case.covariance, "--iterations", str(case.iterations),
                "--reg", str(case.reg), "--init", case.init,
                "--seed", str(case.seed),
            ]
            p = subprocess.run(command, text=True, capture_output=True, timeout=3600)
            if p.returncode:
                raise RuntimeError(
                    f"cuVS frozen scoring failed ({p.returncode}): {p.stderr[-1500:]}"
                )
            result = json.loads(p.stdout.strip().splitlines()[-1])
            scores = _read_float32(score_bin).double()
            if len(scores) != len(truth):
                raise ValueError("cuVS scores do not match frame count")
            diff = (scores - truth).abs()
            rows["cuvs"] = {
                "status":"complete",
                "score_seconds":result["score_seconds"],
                "max_abs_logp_diff":float(diff.max()),
                "mean_abs_logp_diff":float(diff.mean()),
                "frames":len(V), "device":str(device),
            }
    result = {
        "case":case.name,
        "experiment":"identical_parameters_frozen_loglikelihood",
        "source":provenance,
        "covariance":case.covariance,
        "mixtures":case.mixtures, "dimensions":V.shape[1],
        "frames":len(V), "results":rows,
        "warning": (
            "Fitting happens once using native PyTorch. All backend scores "
            "are evaluated with that exact UBM. FFT or classification scores "
            "are NOT involved. cuVS is compiled C++, not Python bindings."
        ),
    }
    (out / "identical_parameter_scoring.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bonus_smoke.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backends", nargs="+", choices=["torchgmm","cuvs"],
                        default=["torchgmm","cuvs"])
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--cuvs-binary")
    parser.add_argument("--kaggle-data")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    device = torch.device(
        "cuda" if (args.device=="auto" and torch.cuda.is_available()) else
        "cpu" if args.device=="auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen cuVS scoring")
    results = []
    for case_json in config["cases"]:
        case = BenchCase(**case_json)
        if case.covariance not in ("diag","spherical"):
            continue
        results.append(compare_identical(
            case, device, Path(args.output) / case.name, args.backends,
            args.cuvs_binary, args.kaggle_data,
        ))
    if not results:
        raise RuntimeError("No applicable frozen-parameter cases")
    print(json.dumps([{"case":r["case"],"results":r["results"]} for r in results],
                     indent=2)[:8000])


if __name__ == "__main__":
    main()

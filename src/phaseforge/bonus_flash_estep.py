"""OPTIONAL IBM Flash-GMM GPU E-step comparator, not a full EM training benchmark.

External source stays external: clone https://github.com/IBM/Flash-GMM
and pass --flash-source /path/to/Flash-GMM/flash_gmm.py.
The Triton Flash-GMM kernel supports per-component SPHERICAL covariance.
Its published results are E-step only, NOT GMM-UBM end-to-end speedups.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from .gmm import EMConfig, TorchGMM


def native_estep(model: TorchGMM, X: torch.Tensor, chunk_size: int = 4096):
    K, D = model.mu.shape
    n = X.new_zeros(K)
    first = X.new_zeros(K, D)
    scattered = X.new_zeros(K)
    logs = []
    for block in X.split(chunk_size):
        c = model.log_components(block)
        logz = torch.logsumexp(c, -1)
        resp = torch.softmax(c, dim=-1)
        logs.append(logz)
        count = resp.sum(0)
        weighted = resp.T @ block
        n += count
        first += weighted
        scattered += (
            (resp.T @ block.square()).sum(-1)
            - 2 * (model.mu * weighted).sum(-1)
            + count * model.mu.square().sum(-1)
        )
    return torch.cat(logs), n, first, scattered


def load_external_flash(source: str | Path):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError("Explicit IBM Flash-GMM source file required")
    spec = importlib.util.spec_from_file_location("ibm_flash_gmm_bench", source)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import external Flash-GMM")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "flash_gmm_estep", None)):
        raise ImportError("External Flash-GMM lacks flash_gmm_estep")
    return module.flash_gmm_estep


@torch.no_grad()
def benchmark_case(
    n: int, k: int, d: int, output: str | Path,
    source: str | Path | None = None,
    repeats: int = 3, seed: int = 2026, chunk_size: int = 4096,
    allow_missing_flash: bool = False,
) -> dict:
    if n <= k or k < 1 or d < 1 or repeats < 1:
        raise ValueError("Invalid Flash-GMM benchmark dimensions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_missing_flash:
        raise RuntimeError("IBM Flash-GMM comparator requires an NVIDIA CUDA GPU")
    torch.manual_seed(seed)
    X = torch.randn(n, d, device=device)
    cfg = EMConfig(
        mixtures=k, covariance="spherical", init="random",
        iterations=2, reg=.02, seed=seed, chunk=chunk_size,
    )
    model = TorchGMM(cfg)
    model._initialize(X)
    log_sigma = model.cov.clamp_min(1e-6).log().contiguous()
    log_pi = model.weights.clamp_min(1e-12).log().contiguous()
    flash = None
    failure = None
    if source and device.type == "cuda":
        try:
            flash = load_external_flash(source)
        except (ImportError, FileNotFoundError) as error:
            failure = repr(error)
            if not allow_missing_flash:
                raise
    else:
        failure = "cuVS-independent Flash-GMM comparison needs CUDA and --flash-source"
    methods = {"native_chunked":lambda:native_estep(model,X,chunk_size)}
    if flash is not None:
        methods["ibm_flash"] = lambda:flash(X, model.mu, log_sigma, log_pi)
    trials = {}
    outputs = {}
    for name, func in methods.items():
        _ = func()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        peak = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            current = func()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            times.append(elapsed)
            peak.append(torch.cuda.max_memory_allocated() if device.type=="cuda" else None)
        outputs[name] = tuple(t.detach().cpu().double() for t in current)
        trials[name] = {
            "status":"complete",
            "median_seconds":statistics.median(times),
            "std_seconds":statistics.stdev(times) if repeats > 1 else None,
            "timings_seconds":times,
            "peak_torch_cuda_allocated_bytes":max(
                (x for x in peak if x is not None), default=None
            ),
        }
    if "ibm_flash" in outputs:
        reference = outputs["native_chunked"]
        predicted = outputs["ibm_flash"]
        trials["ibm_flash"]["max_abs_errors"] = {
            key:float((a-b).abs().max())
            for key,a,b in zip(
                ("log_likelihood_per_frame", "occupancy",
                 "first_moment", "scatter_about_old_mean"),
                reference,predicted,
            )
        }
        trials["ibm_flash"]["all_finite"] = all(
            torch.isfinite(v).all() for v in predicted
        )
    else:
        trials["ibm_flash"] = {"status":"skipped","reason":failure}
    result = {
        "case":{"n":n,"k":k,"d":d,"seed":seed,
                "repeats":repeats,"chunk":chunk_size},
        "device":str(device),
        "scope":"one E-step only, not GMM fit, MAP or speaker EER",
        "covariance":"spherical",
        "same_initial_weights_means_variances":True,
        "numerical_inputs_are_identical":True,
        "measurements":trials,
        "warning":(
            "Flash-GMM uses a Triton fused/atomic E-step; its BF16/TF32 "
            "accumulation can deviate from PyTorch's reductions. Do not "
            "attribute this speedup to whole EM fitting."
        ),
    }
    path=Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path/"flash_estep_comparison.json").write_text(json.dumps(result,indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flash-source")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-missing-flash", action="store_true")
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text())
    results=[]
    for case in config["cases"]:
        label=case["name"]
        params={key:case[key] for key in ("n","k","d")}
        result=benchmark_case(
            **params, output=Path(args.output)/label,
            source=args.flash_source,repeats=args.repeats,
            seed=case.get("seed",2026),
            allow_missing_flash=args.allow_missing_flash,
        )
        results.append({"name":label,"result":result})
        print(label,result["measurements"],flush=True)
    (Path(args.output)/"flash_summary.json").write_text(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()

"""Measured CPU/GPU phase-only extraction batch-size benchmark.

Compares the SAME original waveform tensors on one device. Separates
preloaded computation-only timing from archive decoding and end-to-end
loading, so data caching is not misreported as an algorithmic speedup.

Usage:
python -m phaseforge.bonus_frontend_benchmark \
  --data /path/to/kaggle --output runs/bonus/frontend \
  --mode group --files 128 --batches 1 4 8 16 32 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch

from .bonus_frontend import extract_grouped_features
from .data import KaggleData
from .features import FeatureConfig


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_frontend(
    data_root: str | Path, output: str | Path,
    mode: str = "group", max_files: int = 128,
    batches: tuple[int, ...] = (1, 4, 8, 16, 32),
    repeats: int = 3, device_name: str = "auto",
    seed: int = 2026, bands: int = 24,
) -> dict:
    if repeats < 1 or max_files < 2 or not batches or any(x < 1 for x in batches):
        raise ValueError("Invalid frontend benchmark budget")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA frontend benchmark requires GPU")
    cfg = FeatureConfig(mode=mode, bands=bands)
    cfg.validate()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with KaggleData(data_root) as data:
        source = sorted({r["file_id"] for r in data.background})
        ids = random.Random(seed).sample(source, min(max_files, len(source)))
        ids.sort()
        started = time.perf_counter()
        raw = {file: data.load(file).to(device) for file in ids}
        _sync(device)
        read_seconds = time.perf_counter() - started

        class MemoryData:
            def load(self, file_id):
                return raw[file_id]

        preload = MemoryData()
        measured = {}
        for batch_size in batches:
            # Warm-up waveforms never influence model training or selection.
            _ = extract_grouped_features(
                preload, ids[:min(5,len(ids))], cfg, device, batch_size
            )
            _sync(device)
            timings = []
            gpu_peaks = []
            produced = None
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                produced = extract_grouped_features(
                    preload, ids, cfg, device, batch_size
                )
                _sync(device)
                timings.append(time.perf_counter() - t0)
                gpu_peaks.append(
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else None
                )
            if batch_size == batches[0]:
                reference = produced
                if batch_size != 1:
                    # Always compare to the literal reference singleton frontend.
                    reference = extract_grouped_features(
                        preload, ids, cfg, device, batch_size=1
                    )
            diffs = [
                float((reference[file] - produced[file]).abs().max())
                for file in ids
            ]
            if any(reference[file].shape != produced[file].shape for file in ids):
                raise AssertionError("Batched frontend changed number of frames")
            measured[str(batch_size)] = {
                "median_seconds": statistics.median(timings),
                "std_seconds": (
                    statistics.stdev(timings) if len(timings) > 1 else None
                ),
                "seconds": timings,
                "peak_torch_cuda_allocated_bytes": max(
                    (p for p in gpu_peaks if p is not None), default=None
                ),
                "max_abs_phase_feature_error": max(diffs),
                "exact_parity_within_2e_4": max(diffs) <= 2e-4,
                "utterances":len(ids),
                "total_phase_frames":sum(len(produced[file]) for file in ids),
            }
            if not measured[str(batch_size)]["exact_parity_within_2e_4"]:
                raise AssertionError(
                    f"Phase batch_size={batch_size} differs from reference: "
                    f"{max(diffs)}"
                )
        baseline = measured["1"]["median_seconds"]
        for m in measured.values():
            m["speedup_over_single"] = baseline / max(m["median_seconds"], 1e-12)
        result = {
            "track":"kaggle_background_only",
            "phase_input_only":True,
            "mode":mode,"bands":bands,
            "device":str(device),
            "n_files":len(ids),
            "audio_read_once_seconds":read_seconds,
            "excludes_archive_io_from_algorithm_speedups":True,
            "file_manifest_sha256":hashlib.sha256(
                "\n".join(ids).encode()
            ).hexdigest(),
            "measurements":measured,
        }
    (out/"frontend_benchmark.json").write_text(json.dumps(result,indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["group","temporal","cosphase"], default="group")
    parser.add_argument("--files", type=int, default=128)
    parser.add_argument("--batches", type=int, nargs="+",
                        default=[1,4,8,16,32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = parser.parse_args(argv)
    result = benchmark_frontend(
        args.data,args.output,args.mode,args.files,tuple(args.batches),
        args.repeats,args.device,
    )
    print(json.dumps(result,indent=2)[:10000])


if __name__ == "__main__":
    main()

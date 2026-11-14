"""Bonus: reproducible phase-only full-pipeline and exact LLR optimization ablations.

Two experiments:
- compare: use ONE pre-fitted Kaggle UBM checkpoint, identical phase frames and
  MAP enrolment, then compare ordinary and batched scoring speed and EER.
- ablation: re-train from background and measure full end-to-end elapsed time
  and development EER for predeclared feature/EM/MAP/scoring changes.

The hidden evaluation audio is never accessed in this bonus selection suite.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from .data import KaggleData, read_csv
from .engine import RunConfig, extract_features, run, select_device
from .features import FeatureConfig
from .gmm import EMConfig, MAPConfig, TorchGMM
from .metrics import grouped_bootstrap, report
from .scoring import ScoreConfig, enrolled_models, score_trials


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compare_scoring(
    data_root: str | Path, checkpoint: str | Path, output: str | Path,
    device_name: str = "auto", repeats: int = 5,
    speaker_chunk: int = 16, frame_chunk: int = 1024,
) -> dict:
    """Use the same frozen UBM, phase features and MAP speaker models."""
    if repeats < 1:
        raise ValueError("Need >=1 repeat")
    device = select_device(device_name)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # This checkpoint is a trusted artifact produced by this repository.
    artifact = torch.load(checkpoint, map_location=device, weights_only=False)
    if artifact.get("source_track") != "kaggle":
        raise PermissionError("Only Kaggle-background-trained GMM checkpoints permitted")
    ubm = TorchGMM.from_snapshot(artifact["gmm"], device=str(device))
    if ubm.config.covariance != "diag":
        raise ValueError("Batched LLR speed comparison requires diagonal GMM")
    cfg = ScoreConfig(
        norm="none", smoothing=1,
        top_components=None, implementation="reference",
    )
    with KaggleData(data_root) as data:
        trials = list(data.dev_trials)
        enrol = list(data.dev_enrol)
        ids = sorted(
            {r["file_id"] for r in enrol}
            | {r["test_file"] for r in trials}
        )
        t0 = time.perf_counter()
        raw = extract_features(
            data, ids, FeatureConfig(**artifact["feature"]),
            out / "phase_cache", device,
        )
        _sync(device)
        preprocessing_seconds = time.perf_counter() - t0
        transform = artifact["transformation"]
        transformed = {
            file: (transform(features.to(device)) if transform is not None
                   else features.to(device))
            for file, features in raw.items()
        }
    t0 = time.perf_counter()
    adapted = enrolled_models(
        ubm, transformed, enrol, MAPConfig(**artifact["map"]), k=3
    )
    _sync(device)
    adaptation_seconds = time.perf_counter() - t0
    modes = {
        "reference": replace(cfg, implementation="reference"),
        "batched": replace(
            cfg, implementation="batched", speaker_chunk=speaker_chunk,
            frame_chunk=frame_chunk,
        ),
    }
    measurements = {}
    score_vectors = {}
    labels = torch.tensor([int(t["label"]) for t in trials])
    for mode, score_config in modes.items():
        # Warm up both backends before timing. Synchronize CUDA around each run.
        _ = score_trials(ubm, transformed, trials, adapted, score_config)
        _sync(device)
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            values = score_trials(ubm, transformed, trials, adapted, score_config)
            _sync(device)
            samples.append({
                "seconds": time.perf_counter() - start,
                "peak_torch_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else None
                ),
            })
        score_vectors[mode] = values.cpu()
        measurements[mode] = {
            "samples": samples,
            "median_score_seconds": statistics.median(
                sample["seconds"] for sample in samples
            ),
            "trials": len(trials),
            "development": report(values.cpu(), labels),
        }
    diff = (score_vectors["reference"] - score_vectors["batched"]).abs()
    comparison = {
        "track": "kaggle_background_only",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest(),
        "n_dev_trials": len(trials),
        "feature_extraction_seconds": preprocessing_seconds,
        "map_adaptation_seconds": adaptation_seconds,
        "scoring": measurements,
        "max_absolute_score_difference": float(diff.max()),
        "mean_absolute_score_difference": float(diff.mean()),
        "eer_difference": abs(
            measurements["reference"]["development"]["eer"]
            - measurements["batched"]["development"]["eer"]
        ),
        "measured_speedup": (
            measurements["reference"]["median_score_seconds"] /
            max(1e-12, measurements["batched"]["median_score_seconds"])
        ),
        "exact_parity_within_2e_4": bool(torch.allclose(
            score_vectors["reference"], score_vectors["batched"],
            rtol=1e-5, atol=2e-4,
        )),
        "note": (
            "Comparison uses identical frozen GMM and per-speaker MAP models. "
            "Raw per-trial LLRs only; not ensemble or S-norm timing."
        ),
    }
    (out / "scoring_comparison.json").write_text(
        json.dumps(comparison, indent=2)
    )
    for mode, values in score_vectors.items():
        with (out / f"{mode}_development_scores.csv").open("w") as fp:
            fp.write("trial_id,score\n")
            for row, score in zip(trials, values):
                fp.write(f"{row['trial_id']},{float(score):.12f}\n")
    if not comparison["exact_parity_within_2e_4"]:
        raise AssertionError("Batched inference differs from reference beyond tolerance")
    return comparison


def _apply_overrides(config: dict, changes: dict):
    new = json.loads(json.dumps(config))
    for path, value in changes.items():
        parts = path.split(".")
        cursor = new
        for name in parts[:-1]:
            cursor = cursor.setdefault(name, {})
        cursor[parts[-1]] = value
    return new


def run_ablation(
    data_root: str | Path, manifest: dict,
    output: str | Path, limit: int | None = None,
    strict: bool = False,
) -> dict:
    """Full Kaggle background -> phase extraction -> MAP -> scoring experiments."""
    jobs = manifest.get("jobs", [])
    if not jobs:
        raise ValueError("Empty bonus end-to-end ablation")
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate bonus job IDs")
    if limit is not None:
        jobs = jobs[:limit]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(manifest["base_config"]).read_text())
    ledger_path = out / "bonus_ablation_ledger.json"
    previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    records = {}
    for job in jobs:
        if (not job["id"].replace("_", "").replace("-", "").isalnum()
                or "/" in job["id"]):
            raise ValueError("Unsafe bonus job name")
        config = _apply_overrides(base, job.get("overrides", {}))
        config["evaluate_eval"] = False
        config["save_model"] = True
        cfg = RunConfig.from_dict(config)
        cfg.validate()
        # Exact implementation/config/data version, not just a label.
        digest = hashlib.sha256(json.dumps(
            {"config": config, "source": str(Path(data_root).resolve())},
            sort_keys=True,
        ).encode()).hexdigest()
        result_dir = out / "jobs" / job["id"]
        prev = previous.get(job["id"], {})
        if (prev.get("digest") == digest and prev.get("status") == "complete"
                and (result_dir / "result.json").is_file()):
            records[job["id"]] = prev
            continue
        result_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run(data_root, cfg, result_dir, "kaggle")
            entry = {
                "id": job["id"], "status": "complete",
                "digest": digest, "EER": result["metrics"]["eer"],
                "AUC": result["metrics"]["auc"],
                "elapsed_seconds": result["elapsed_seconds"],
                "GPU_peak_allocated_bytes": (
                    result["hardware"]["peak_cuda_allocated_bytes"]
                ),
                "config": config, "directory": str(result_dir),
            }
        except Exception as exc:
            entry = {
                "id": job["id"], "status": "failed",
                "digest": digest, "error": repr(exc),
                "config": config, "directory": str(result_dir),
            }
        records[job["id"]] = entry
        previous[job["id"]] = entry
        ledger_path.write_text(json.dumps(previous, indent=2))
        print(
            f"BONUS E2E {job['id']}: {entry['status']} EER={entry.get('EER')}",
            flush=True,
        )
        if strict and entry["status"] == "failed":
            raise RuntimeError(f"Bonus experiment {job['id']} failed: {entry['error']}")
    ranking = sorted(
        [entry for entry in records.values() if entry["status"] == "complete"],
        key=lambda item: item["EER"]
    )
    result = {
        "complete": len(ranking),
        "failed": sum(e["status"] == "failed" for e in records.values()),
        "development_ranked": ranking,
        "no_hidden_eval_access": True,
        "selection_warning": (
            "Rank on development only. Matching seed/cache/hardware must be "
            "audited before attributing end-to-end runtime speedups."
        ),
    }
    (out / "bonus_e2e_summary.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--data", required=True)
    cmp.add_argument("--checkpoint", required=True)
    cmp.add_argument("--output", required=True)
    cmp.add_argument("--device", default="auto")
    cmp.add_argument("--repeats", type=int, default=5)
    cmp.add_argument("--speaker-chunk", type=int, default=16)
    cmp.add_argument("--frame-chunk", type=int, default=1024)
    abl = sub.add_parser("ablation")
    abl.add_argument("--data", required=True)
    abl.add_argument("--config", required=True)
    abl.add_argument("--output", required=True)
    abl.add_argument("--max-jobs", type=int, default=None)
    abl.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "compare":
        result = compare_scoring(
            args.data, args.checkpoint, args.output, args.device,
            args.repeats, args.speaker_chunk, args.frame_chunk,
        )
    else:
        manifest = json.loads(Path(args.config).read_text())
        result = run_ablation(
            args.data, manifest, args.output, args.max_jobs, args.strict
        )
    print(json.dumps(result, indent=2)[:8000])


if __name__ == "__main__":
    main()

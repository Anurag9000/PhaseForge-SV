"""Resumable, provenance-recorded heterogeneous Kaggle-only experiment suite.

Runs multiple phase feature/GMM/HMM/DTW/CNN families sequentially with the
SAME allowed competition corpus. Development is the only official labeled
population; proxy validation is explicitly separate and excludes its
speakers from UBM fitting. Never loads the original Google archive.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from .alternatives import run_alternative
from .data import KaggleData
from .engine import RunConfig, run
from .experiments import _deep_set
from .studies import run_proxy


def _sha256_config(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True,
            check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def execute_suite(
    data_root: str | Path, spec: dict, output: str | Path,
    max_jobs: int | None = None,
) -> dict:
    """Execute/resume finite experiment jobs without touching evaluation WAVs.

    Each job gets a separate filesystem directory and a configuration digest.
    A changed configuration, code revision or corpus manifest is NOT treated as
    a previously completed run, even when the human-readable job ID is reused.
    """
    if spec.get("track", "kaggle") != "kaggle":
        raise ValueError("This heterogeneous suite is Kaggle-only")
    jobs = list(spec["jobs"])
    names = [j["id"] for j in jobs]
    if not jobs or any(not name or not isinstance(name, str) for name in names):
        raise ValueError("Require named experimental jobs")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate job identifiers")
    if any("/" in name or ".." in name or "\\" in name for name in names):
        raise ValueError("Job names may not contain path separators")
    if max_jobs is not None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        jobs = jobs[:max_jobs]
    root = Path(data_root).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with KaggleData(root) as data:
        manifest = hashlib.sha256()
        for filename in ("background.csv", "dev_enrol.csv", "dev_trials.csv",
                         "eval_enrol.csv", "eval_trials.csv", "sample_submission.csv"):
            manifest.update((root / filename).read_bytes())
        manifest_sha = manifest.hexdigest()
        data_summary = data.summary()
    revision = _revision()
    ledger_file = output / "suite_ledger.json"
    previous = json.loads(ledger_file.read_text()) if ledger_file.is_file() else {}
    records = []
    started = time.perf_counter()

    for index, job in enumerate(jobs):
        kind = job.get("kind", "gmm")
        if kind not in ("gmm", "alternative"):
            raise ValueError(f"Unknown job kind {kind}")
        config_file = Path(job["config"])
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        config_data = json.loads(config_file.read_text())
        for key, value in job.get("overrides", {}).items():
            _deep_set(config_data, key, value)
        # Never load evaluation files during development model-family searches.
        config_data["evaluate_eval"] = False
        config_data["save_model"] = False
        cfg = RunConfig.from_dict(config_data)
        cfg.validate()
        if kind == "alternative" and job.get("method", "dtw") not in (
            "hmm", "timebin", "dtw", "scratch"
        ):
            raise ValueError("Unknown alternative method")
        identity = {
            "job": job,
            "config": config_data,
            "dataset_root": str(root),
            "manifest_sha256": manifest_sha,
            "git_revision": revision,
        }
        digest = _sha256_config(identity)
        directory = output / "jobs" / f"{index:03d}_{job['id']}"
        expected_result = directory / "result.json"
        old = previous.get(job["id"])
        if (old and old.get("identity") == digest and old.get("status") == "complete"
                and expected_result.is_file()):
            records.append(old)
            continue
        began = time.perf_counter()
        record = {
            "id": job["id"], "kind": kind,
            "method": job.get("method") if kind == "alternative" else "gmm_ubm",
            "identity": digest, "config": config_data,
            "git_revision": revision, "source_manifest_sha256": manifest_sha,
            "source_track": "kaggle_background_only",
            "directory": str(directory),
        }
        try:
            if kind == "gmm":
                result = run(root, cfg, directory, track="kaggle")
            else:
                result = run_alternative(
                    root, cfg, directory, method=job["method"], track="kaggle"
                )
            record["development_eer"] = result["metrics"]["eer"]
            record["development_auc"] = result["metrics"]["auc"]
            if kind == "gmm" and job.get("proxy", False):
                proxy = run_proxy(
                    root, replace(cfg, evaluate_eval=False),
                    directory / "proxy", seed=int(job.get("proxy_seed", 2026)),
                    rotations=int(job.get("proxy_rotations", 1)),
                )
                record["proxy_mean_eer"] = proxy["mean_eer"]
                record["proxy_rotations"] = proxy["rotations"]
            record["status"] = "complete"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "failure.json").write_text(json.dumps(record, indent=2))
        record["seconds"] = time.perf_counter() - began
        records.append(record)
        previous[job["id"]] = record
        ledger_file.write_text(json.dumps(previous, indent=2))
        print(
            f"Suite {index+1}/{len(jobs)} {job['id']}: "
            f"{record['status']} {record.get('development_eer')}",
            flush=True
        )

    completed = [r for r in records if r["status"] == "complete"]
    eligible = [r for r in completed if r["kind"] == "gmm"]
    development_ranked = sorted(eligible, key=lambda x: x["development_eer"])
    proxy_ranked = sorted(
        (r for r in eligible if "proxy_mean_eer" in r),
        key=lambda x: x["proxy_mean_eer"],
    )
    # Never rank a proxy EER against a development EER as if they came
    # from the same population. Different validation protocols require
    # separate leaderboards and paired comparisons.
    summary = {
        "track": "kaggle",
        "dataset": str(root),
        "dataset_summary": data_summary,
        "manifest_sha256": manifest_sha,
        "git_revision": revision,
        "attempted": len(records), "completed": len(completed),
        "failed": len(records) - len(completed),
        "selection": (
            "Separate proxy and official development leaderboards. "
            "Never compare their EER numbers as the same metric."
        ),
        "proxy_validated_ranking": [
            {"id": r["id"], "dev_eer": r["development_eer"],
             "proxy_eer": r["proxy_mean_eer"], "config": r["config"]}
            for r in proxy_ranked
        ],
        "development_ranking": [
            {"id": r["id"], "dev_eer": r["development_eer"],
             "has_proxy": "proxy_mean_eer" in r, "config": r["config"]}
            for r in development_ranked
        ],
        "other_research_methods": [
            {"id": r["id"], "dev_eer": r["development_eer"]}
            for r in completed if r["kind"] != "gmm"
        ],
        "seconds": time.perf_counter() - started,
        "warning": (
            "This is development/proxy selection only. Rerun selected Kaggle-only "
            "GMM configs with evaluation enabled, then finalize with provenance checks."
        ),
    }
    (output / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

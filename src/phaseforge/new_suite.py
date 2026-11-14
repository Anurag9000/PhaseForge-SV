"""Independent new research-suite executor for previously absent model families.

Does not modify the original 36-job GMM suite. Supports phase-only CNN,
Transformer/ECAPA/GE2E, GMM-supervector, i-vector, PLDA, PBM, score calibration,
MathWorks-inspired phase-GMM ablations, and original Google keyword tasks.
On Kaggle, NO neural or classical research run creates a submission.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

from .data import KaggleData
from .engine import RunConfig, run
from .new_neural_runner import NeuralRun, run_neural
from .new_classical_runner import ClassicalRun, run_classical
from .new_keyword import KeywordRun, run_keyword
from .new_scoring import calibrate_saved_predictions
from .google_mathworks_reference import MathWorksResearch, run_google_mathworks


ALLOWED = (
    "gmm_phase", "neural_sv", "classical_sv", "keyword_google",
    "mathworks_reference", "calibrate",
)


def _sha256(obj: dict):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _revision():
    p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, timeout=5, check=False)
    return p.stdout.strip() or "unknown"


def _implementation_fingerprint() -> str:
    """Hash executable Python implementation, not docs/tests or Git metadata.

    A documentation-only commit should not retrain a 127-job GPU suite;
    any implementation edit must invalidate the previously completed job.
    """
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    files = sorted(directory.glob("*.py"))
    if not files:
        raise RuntimeError("PhaseForge implementation modules not found")
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _deep_set(root: dict, key: str, value):
    path = key.split(".")
    node = root
    for name in path[:-1]:
        node = node.setdefault(name, {})
    node[path[-1]] = value


def execute_new_suite(
    data_root: str | Path, manifest: dict, output: str | Path,
    track: str = "kaggle", benchmark: str | Path | None = None,
    max_jobs: int | None = None, strict: bool = False,
):
    """Stop at max_jobs but retain per-job state; reruns resume by digest."""
    if track not in ("kaggle", "google"):
        raise ValueError("Unknown dataset track")
    if track == "google" and not benchmark:
        raise ValueError("Original-Google research requires locked speaker benchmark")
    if manifest.get("suite_family") != "new_phase_research":
        raise ValueError("Not the new missing-methods suite registry")
    jobs = manifest.get("jobs", [])
    if not jobs:
        raise ValueError("Empty research suite")
    names = [j["id"] for j in jobs]
    if len(set(names)) != len(names) or any(
        not n or "/" in n or "\\" in n or ".." in n for n in names
    ):
        raise ValueError("Unsafe/duplicate suite job IDs")
    if max_jobs is not None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be >=1")
        jobs = jobs[:max_jobs]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(data_root).resolve()
    if track == "kaggle":
        with KaggleData(source) as ds:
            signature = ds.summary()
            files = ["background.csv", "dev_enrol.csv", "dev_trials.csv",
                     "eval_enrol.csv", "eval_trials.csv", "sample_submission.csv"]
            digest = hashlib.sha256()
            for name in files:
                digest.update((source / name).read_bytes())
            signature["manifest_digest"] = digest.hexdigest()
            # Hash ZIP central-directory identity without reading/decoding
            # the complete multi-gigabyte archive before every resumed job.
            zip_digest = hashlib.sha256()
            with zipfile.ZipFile(source / "audio.zip") as archive:
                for info in sorted(archive.infolist(), key=lambda x: x.filename):
                    zip_digest.update(
                        f"{info.filename}|{info.file_size}|{info.compress_size}|"
                        f"{info.CRC:08x}\n".encode()
                    )
            signature["audio_zip_directory_sha256"] = zip_digest.hexdigest()
    else:
        benchmark = Path(benchmark).resolve()
        locked_files = (
            "google_benchmark.json", "google_manifest.csv",
            "google_speaker_splits.csv",
            "google_dev_enrol.csv", "google_dev_trials.csv",
            "google_test_enrol.csv", "google_test_trials.csv",
            "google_ubm_stop.csv", "google_ubm_all_words.csv",
        )
        signature = {
            "benchmark": str(benchmark),
            "file_sha256": {
                name: hashlib.sha256((benchmark / name).read_bytes()).hexdigest()
                for name in locked_files
            },
        }
    revision = _revision()
    implementation_sha = _implementation_fingerprint()
    state_file = output / "new_suite_ledger.json"
    ledger = json.loads(state_file.read_text()) if state_file.exists() else {}
    entries = []
    for i, job in enumerate(jobs):
        kind = job.get("kind")
        if kind not in ALLOWED:
            raise ValueError(f"Unsupported research job type {kind}")
        if kind in ("keyword_google", "mathworks_reference") and track != "google":
            # The new suite may contain a Google-only job. This is an explicit
            # skip, not a silent misuse of the competition dataset.
            record = {"id": job["id"], "status": "inapplicable",
                      "reason": f"{kind} requires original Google research"}
            entries.append(record)
            continue
        if job.get("tracks") and track not in job["tracks"]:
            entries.append({"id": job["id"], "status": "inapplicable",
                            "reason": f"{track} not in allowed tracks"})
            continue
        template = Path(job["config"])
        if not template.is_file():
            raise FileNotFoundError(f"New suite config not found: {template}")
        config = json.loads(template.read_text())
        for dotted, value in job.get("overrides", {}).items():
            _deep_set(config, dotted, value)
        if kind == "gmm_phase":
            config["evaluate_eval"] = False
            config["save_model"] = False
            if track == "google":
                # A Google research benchmark may use 1+1 original stop
                # takes, unlike Kaggle's mandatory three-enrolment format.
                # Its manifest—not dev metrics—defines the correct count.
                protocol = json.loads(
                    (Path(benchmark) / "google_benchmark.json").read_text()
                )
                config["enrolment_size"] = int(
                    protocol["enrolment_takes_per_speaker"]
                )
        elif kind in ("neural_sv", "classical_sv", "keyword_google",
                      "mathworks_reference"):
            config["evaluate_locked_test"] = False
            if kind == "neural_sv" and track == "google":
                # Original stop recordings are not guaranteed to have three
                # distinct takes per background speaker. GE2E requires >=2.
                config["takes_per_speaker"] = min(
                    2, int(config.get("takes_per_speaker", 3))
                )
        identity = {
            "job": job, "config": config, "data": str(source), "track": track,
            "source_signature": signature,
            "implementation_sha256": implementation_sha,
        }
        if kind == "calibrate":
            # A changed source model must invalidate the calibrated output
            # even when the calibration hyperparameters stayed the same.
            parent = next(
                (entry for entry in entries if entry["id"] == job.get("source_job")),
                None,
            )
            identity["source_job_key"] = parent.get("key") if parent else None
            identity["source_job_status"] = parent.get("status") if parent else None
        key = _sha256(identity)
        path = output / "jobs" / f"{i:03d}_{job['id']}"
        previous = ledger.get(job["id"])
        if (previous and previous.get("key") == key
                and previous.get("status") == "complete"
                and (path / "result.json").is_file()):
            entries.append(previous)
            continue
        path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        item = {
            "id": job["id"], "kind": kind, "key": key, "track": track,
            "config": config, "directory": str(path),
            "revision": revision, "source_signature": signature,
            "implementation_sha256": implementation_sha,
        }
        try:
            if kind == "gmm_phase":
                result = run(source, RunConfig.from_dict(config), path, track,
                             benchmark=benchmark)
            elif kind == "neural_sv":
                result = run_neural(source, NeuralRun.from_dict(config), path,
                                    track=track, benchmark=benchmark)
            elif kind == "classical_sv":
                result = run_classical(source, ClassicalRun.from_dict(config), path,
                                       track=track, benchmark=benchmark)
            elif kind == "keyword_google":
                result = run_keyword(source, benchmark, KeywordRun.from_dict(config), path)
            elif kind == "mathworks_reference":
                result = run_google_mathworks(
                    source, benchmark, MathWorksResearch.from_dict(config), path
                )
            else:
                previous_id = job.get("source_job")
                if not previous_id:
                    raise ValueError("Calibration must specify source_job")
                parent = next((entry for entry in entries
                               if entry["id"] == previous_id and
                               entry.get("status") == "complete"), None)
                if parent is None:
                    raise ValueError(f"Calibration source job not completed: {previous_id}")
                source_csv = Path(parent["directory"]) / "development_scores.csv"
                result = calibrate_saved_predictions(
                    source_csv, path, folds=int(config.get("folds", 5))
                )
                (path / "result.json").write_text(json.dumps(result, indent=2))
            if kind == "keyword_google":
                item["development_accuracy"] = result["metrics"]["development"]["accuracy"]
                item["development_macro_f1"] = result["metrics"]["development"]["macro_f1"]
            else:
                metric = result.get("metrics", result.get("calibrated_oof", {}))
                item["development_eer"] = metric["eer"]
                item["development_auc"] = metric["auc"]
            item["status"] = "complete"
        except Exception as error:
            item["status"], item["error"] = "failed", repr(error)
            (path / "failure.json").write_text(json.dumps(item, indent=2))
        item["seconds"] = time.perf_counter() - start
        ledger[job["id"]] = item
        entries.append(item)
        state_file.write_text(json.dumps(ledger, indent=2))
        print(f"NEW {i+1}/{len(jobs)} {job['id']}: {item['status']} "
              f"{item.get('development_eer')}", flush=True)
        if strict and item["status"] == "failed":
            raise RuntimeError(f"Suite stopped on {job['id']}: {item['error']}")
    completed = [r for r in entries if r["status"] == "complete"]
    sv = [r for r in completed if "development_eer" in r]
    keywords = [r for r in completed if "development_accuracy" in r]
    summary = {
        "track": track, "suite_family": "new_phase_research",
        "source_signature": signature, "git_revision": revision,
        "implementation_sha256": implementation_sha,
        "registered_jobs": len(manifest["jobs"]), "attempted": len(entries),
        "completed": len(completed),
        "failed": sum(x["status"] == "failed" for x in entries),
        "skipped_other_track": sum(x["status"] == "inapplicable" for x in entries),
        "speaker_verification_ranked": sorted(
            [{"id": r["id"], "kind": r["kind"],
              "development_eer": r["development_eer"],
              "development_auc": r["development_auc"]}
             for r in sv], key=lambda x: x["development_eer"]
        ),
        "keyword_accuracy_ranked": sorted(
            [{"id": r["id"], "accuracy": r["development_accuracy"]}
             for r in keywords], key=lambda x: -x["accuracy"]
        ),
        "note": (
            "All results are development-only and RESEARCH-ONLY. A neural, PLDA, "
            "keyword or calibrated score is not the mandatory Kaggle GMM-UBM LLR. "
            "Google locked test remains untouched until a separate final execution."
        ),
    }
    (output / "new_suite_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

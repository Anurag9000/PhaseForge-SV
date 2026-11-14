import csv
import io
import json
import math
import struct
import wave
import zipfile
from pathlib import Path

import pytest

from phaseforge.data import KaggleData, google_benchmark, inventory_google
from phaseforge.engine import RunConfig, run


def _wav(seed: int, nframes: int = 2048) -> bytes:
    buffer = io.BytesIO()
    samples = [
        int(3000 * math.sin(2 * math.pi * (180 + seed * 13) * i / 16000))
        for i in range(nframes)
    ]
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<" + "h" * nframes, *samples))
    return buffer.getvalue()


def _csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def tiny_kaggle(tmp_path):
    bg = [{"speaker_id": f"b{i}", "file_id": f"b{i}_{j}"}
          for i in range(4) for j in range(3)]
    dev = [{"speaker_id": f"d{i}", "file_id": f"d{i}_{j}", "enrol_index": j}
           for i in range(2) for j in range(3)]
    ev = [{"speaker_id": f"e{i}", "file_id": f"e{i}_{j}", "enrol_index": j}
          for i in range(2) for j in range(3)]
    dtrials = [
        {"trial_id": "d1", "claimed_id": "d0", "test_file": "dt0", "label": 1},
        {"trial_id": "d2", "claimed_id": "d1", "test_file": "dt0", "label": 0},
        {"trial_id": "d3", "claimed_id": "d1", "test_file": "dt1", "label": 1},
        {"trial_id": "d4", "claimed_id": "d0", "test_file": "dt1", "label": 0},
    ]
    etrials = [
        {"trial_id": "t1", "claimed_id": "e0", "test_file": "et0"},
        {"trial_id": "t2", "claimed_id": "e1", "test_file": "et0"},
    ]
    _csv(tmp_path / "background.csv", ["file_id", "speaker_id"], bg)
    _csv(tmp_path / "dev_enrol.csv", ["speaker_id", "file_id", "enrol_index"], dev)
    _csv(tmp_path / "eval_enrol.csv", ["speaker_id", "file_id", "enrol_index"], ev)
    _csv(tmp_path / "dev_trials.csv", ["trial_id", "claimed_id", "test_file", "label"],
         dtrials)
    _csv(tmp_path / "eval_trials.csv", ["trial_id", "claimed_id", "test_file"], etrials)
    _csv(tmp_path / "sample_submission.csv", ["trial_id", "score"],
         [{"trial_id": x["trial_id"], "score": 0} for x in etrials])
    ids = [x["file_id"] for x in bg + dev + ev] + ["dt0", "dt1", "et0"]
    with zipfile.ZipFile(tmp_path / "audio.zip", "w") as archive:
        for i, ident in enumerate(ids):
            archive.writestr(f"audio/{ident}.wav", _wav(i))
    return tmp_path


def test_kaggle_audit_rejects_manifest_file_overlap(tiny_kaggle):
    trials = list(csv.DictReader((tiny_kaggle / "dev_trials.csv").open()))
    trials[0]["test_file"] = "b0_0"
    _csv(
        tiny_kaggle / "dev_trials.csv",
        ["trial_id", "claimed_id", "test_file", "label"], trials
    )
    with pytest.raises(ValueError, match="audio file-ID overlap"):
        KaggleData(tiny_kaggle)


def test_kaggle_audit_rejects_unenrolled_claim(tiny_kaggle):
    trials = list(csv.DictReader((tiny_kaggle / "dev_trials.csv").open()))
    trials[0]["claimed_id"] = "not_enrolled"
    _csv(
        tiny_kaggle / "dev_trials.csv",
        ["trial_id", "claimed_id", "test_file", "label"], trials
    )
    with pytest.raises(ValueError, match="speaker not enrolled"):
        KaggleData(tiny_kaggle)


def test_kaggle_allowlist_and_source_isolation(tiny_kaggle):
    with KaggleData(tiny_kaggle) as d:
        assert d.summary()["background_utterances"] == 12
        assert d.load("dt0").numel() > 100
        d.assert_background_only(["b0_0"])
        with pytest.raises(ValueError):
            d.assert_background_only(["d0_0"])
        with pytest.raises(PermissionError):
            d.load("google_stop_0")


def test_tiny_end_to_end_kaggle_only(tiny_kaggle):
    config = RunConfig.from_dict({
        "feature": {"mode": "group", "bands": 8, "normalization": "utterance"},
        "em": {"mixtures": 2, "iterations": 4, "chunk": 32, "reg": .05},
        "map": {"relevance": 4},
        "score": {"norm": "none"},
        "transform": "pca", "dimensions": 4,
        "seeds": [12], "evaluate_eval": True, "save_model": False,
        "max_frames_per_background_file": 8, "device": "cpu",
    })
    out = tiny_kaggle / "runs" / "smoke"
    result = run(tiny_kaggle, config, out)
    assert result["track"] == "kaggle"
    assert (out / "submission.csv").exists()
    rows = (out / "submission.csv").read_text().splitlines()
    assert rows[0] == "trial_id,score"
    assert len(rows) == 3
    assert (out / "result.json").exists()


def test_google_benchmark_stays_research_only(tmp_path):
    root = tmp_path / "google"
    folder = root / "stop"
    folder.mkdir(parents=True)
    for speaker in range(31):
        for take in range(5):
            (folder / f"s{speaker:03d}_nohash_{take}.wav").write_bytes(
                _wav(speaker * 7 + take, nframes=256)
            )
    dest = tmp_path / "research"
    report = inventory_google(root, dest)
    assert report["unique_speakers"] == 31
    split = google_benchmark(dest / "google_manifest.csv", dest, seed=2026)
    assert split["research_only"]
    assert (dest / "google_test_trials.csv").exists()



def test_zip_subfolder_layout_and_integrity_audit(tiny_kaggle):
    archive = tiny_kaggle / "audio.zip"
    with zipfile.ZipFile(archive) as source:
        recordings = [(Path(item.filename).name, source.read(item))
                      for item in source.infolist() if item.filename.endswith(".wav")]
    with zipfile.ZipFile(archive, "w") as destination:
        for name, pcm in recordings:
            destination.writestr("repackaged/nested/" + name, pcm)
    with KaggleData(tiny_kaggle) as data:
        wave = data.load("dt0")
        assert wave.numel() > 100
        audit = data.audit_audio(check_crc=True)
        assert audit["audited_files"] == len(data.allowed_ids)
        assert audit["zip_crc_verified"]
        assert audit["min_frames"] >= 2048



def test_ensemble_first_full_kaggle_pipeline(tiny_kaggle):
    """Two seeds and shared background cohorts must produce a valid Kaggle CSV."""
    config = RunConfig.from_dict({
        "feature": {"mode": "group", "bands": 8, "normalization": "utterance",
                    "masked_mean": False},
        "em": {"mixtures": 2, "iterations": 4, "chunk": 32, "reg": .05,
               "init_subsample": 40, "kmeans_steps": 3,
               "dead_policy": "hard_examples",
               "init_variance_source": "initialization",
               "legacy_pre_mstep_likelihood": True},
        "map": {"relevance": .5},
        "score": {"norm": "s", "norm_stage": "after_ensemble",
                  "cohort_seed": 2026, "cohort_speakers": 2,
                  "cohort_test_files": 2, "smoothing": 1},
        "transform": "pca", "dimensions": 4,
        "seeds": [12, 13], "evaluate_eval": True, "save_model": False,
        "max_frames_per_background_file": 8, "device": "cpu",
    })
    output = tiny_kaggle / "runs" / "after_ensemble"
    result = run(tiny_kaggle, config, output)
    assert result["track"] == "kaggle"
    assert result["config"]["score"]["norm_stage"] == "after_ensemble"
    assert (output / "submission.csv").exists()
    assert len((output / "submission.csv").read_text().splitlines()) == 3
    assert result["metrics"]["eer"] >= 0



def test_resume_heterogeneous_kaggle_suite(tiny_kaggle):
    """Repeated identical jobs read the successful ledger, not rerun audio training."""
    from phaseforge.suite import execute_suite
    import json

    settings = {
        "feature": {"mode": "group", "bands": 8},
        "em": {"mixtures": 2, "iterations": 3, "reg": 0.05, "chunk": 32},
        "map": {"relevance": 4},
        "score": {"norm": "none"},
        "transform": "pca", "dimensions": 4, "seeds": [2026],
        "save_model": False, "evaluate_eval": False,
        "max_frames_per_background_file": 8, "device": "cpu",
    }
    config_file = tiny_kaggle / "single.json"
    config_file.write_text(json.dumps(settings))
    spec = {
        "track": "kaggle",
        "jobs": [{"id": "single", "kind": "gmm", "config": str(config_file)}]
    }
    output = tiny_kaggle / "runs" / "suite"
    first = execute_suite(tiny_kaggle, spec, output)
    assert first["completed"] == 1
    assert first["failed"] == 0
    assert len(first["development_ranking"]) == 1
    path = output / "jobs" / "000_single" / "result.json"
    timestamp = path.stat().st_mtime_ns
    second = execute_suite(tiny_kaggle, spec, output)
    assert second["completed"] == 1
    assert path.stat().st_mtime_ns == timestamp
    assert not (output / "jobs" / "000_single" / "submission.csv").exists()


def test_research_corpus_refused_by_kaggle_suite():
    from phaseforge.suite import execute_suite

    with pytest.raises(ValueError, match="Kaggle-only"):
        execute_suite("irrelevant", {"track":"google","jobs":[]}, "irrelevant")



def test_new_neural_real_wav_pipeline_is_background_only(tiny_kaggle):
    """Train, enrol and score an actual PCM-16 Kaggle-shaped synthetic dataset."""
    from phaseforge.new_neural_runner import NeuralRun, run_neural
    cfg = NeuralRun.from_dict({
        "feature": {"mode":"group","bands":8},
        "architecture":{"name":"bcnn","channels":16,"embedding":8,"depth":1},
        "objective":"ge2e",
        "epochs":1,"steps_per_epoch":1,
        "speakers_per_batch":2,"takes_per_speaker":2,
        "max_frames":24,"device":"cpu",
        "evaluate_locked_test":False,
    })
    out = tiny_kaggle / "neural_research"
    result = run_neural(tiny_kaggle, cfg, out)
    assert result["source_track"] == "kaggle"
    assert result["research_only"]
    assert (out / "checkpoint.pt").exists()
    assert (out / "development_scores.csv").exists()
    assert not (out / "evaluation_scores.csv").exists()
    assert not (out / "submission.csv").exists()


@pytest.mark.parametrize("method", [
    "supervector_cosine", "supervector_svm",
    "ivector_cosine", "ivector_plda", "ivector_lda_plda",
])
def test_new_classical_real_wav_pipeline_methods(tiny_kaggle, method):
    from phaseforge.new_classical_runner import ClassicalRun, run_classical
    cfg = ClassicalRun.from_dict({
        "method":method,
        "feature":{"mode":"group","bands":8},
        "em":{"mixtures":2,"covariance":"diag","iterations":4,"reg":.05},
        "map":{"relevance":4},
        "transform":"pca","dimensions":4,
        "tv_dimension":2,"tv_iterations":2,
        "lda_dimension":2,"frame_cap":4,
        "device":"cpu","evaluate_locked_test":False,
    })
    out = tiny_kaggle / f"classical_{method}"
    result = run_classical(tiny_kaggle, cfg, out)
    assert result["method"] == method
    assert result["research_only"]
    assert (out / "checkpoint.pt").exists()
    assert (out / "development_scores.csv").exists()
    assert not (out / "submission.csv").exists()



def test_new_heterogeneous_suite_executes_and_resumes(tiny_kaggle, monkeypatch):
    """Resume across documentation-only revisions and remain dev-only."""
    from phaseforge import new_suite
    from phaseforge.new_suite import execute_new_suite
    import json
    root = tiny_kaggle
    config = root / "tiny_neural.json"
    config.write_text(json.dumps({
        "feature": {"mode":"group","bands":8},
        "architecture":{"name":"xvector","channels":16,
                        "embedding":8,"depth":1},
        "objective":"aam", "epochs":1,"steps_per_epoch":1,
        "speakers_per_batch":2,"takes_per_speaker":2,
        "max_frames":16,"device":"cpu",
    }))
    registry = {"suite_family":"new_phase_research","jobs":[
        {"id":"first_neural","kind":"neural_sv","config":str(config)},
        {"id":"google_only","kind":"keyword_google","config":str(config),
         "tracks":["google"]},
    ]}
    destination = root / "missing_methods_results"
    first = execute_new_suite(root, registry, destination, "kaggle")
    assert first["completed"] == 1
    assert first["failed"] == 0
    assert first["skipped_other_track"] == 1
    checkpoint = destination / "jobs/000_first_neural/result.json"
    assert checkpoint.is_file()
    assert not (checkpoint.parent / "submission.csv").exists()
    timestamp = checkpoint.stat().st_mtime_ns
    # The Git HEAD can change for documentation/tests without changing
    # executable PhaseForge code. It must NOT force GPU model retraining.
    monkeypatch.setattr(new_suite, "_revision", lambda: "docs-only-new-commit")
    second = execute_new_suite(root, registry, destination, "kaggle")
    assert second["completed"] == 1
    assert checkpoint.stat().st_mtime_ns == timestamp


def test_mathworks_phase_preprocessing_synthetic_audio():
    import torch
    from phaseforge.features import FeatureConfig, PhaseExtractor
    waveform = torch.zeros(16000)
    waveform[4000:9000] = .3 * torch.sin(
        torch.arange(5000) * (2 * 3.14159 * 300 / 16000)
    )
    config = FeatureConfig(
        mode="group", win=640, hop=160, n_fft=1024, bands=24,
        peak_normalize=True, trim_longest_region=True,
        normalization="none",
    )
    features = PhaseExtractor(config)(waveform)
    assert features.shape[-1] == 48
    assert torch.isfinite(features).all()

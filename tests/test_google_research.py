"""True locked Google research test metrics and strict enrolment protocol smoke test."""
from __future__ import annotations

import csv
import io
import math
import struct
import wave

import pytest

from phaseforge.data import google_benchmark, inventory_google, read_csv
from phaseforge.engine import RunConfig, run
from phaseforge.gmm import EMConfig, MAPConfig, TorchGMM
from phaseforge.scoring import enrolled_models
import torch


def _pcm16(frequency: int, samples: int = 1024) -> bytes:
    buffer = io.BytesIO()
    pcm = [
        round(12000 * math.sin(2 * math.pi * frequency * i / 16000))
        for i in range(samples)
    ]
    with wave.open(buffer, "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(16000)
        fp.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    return buffer.getvalue()


def test_speaker_disjoint_google_one_plus_one_produces_test_eer(tmp_path):
    corpus = tmp_path / "original_google_speech_commands"
    (corpus / "stop").mkdir(parents=True)
    for speaker in range(31):
        for take in range(2):
            (corpus / "stop" / f"p{speaker:03d}_nohash_{take}.wav").write_bytes(
                _pcm16(220 + 11 * speaker + 2 * take)
            )
    for speaker in range(5):
        (corpus / "stop" / f"u{speaker:03d}_nohash_0.wav").write_bytes(
            _pcm16(1200 + 15 * speaker)
        )
    benchmark = tmp_path / "speaker_split"
    inventory_google(corpus, benchmark)
    summary = google_benchmark(
        benchmark / "google_manifest.csv", benchmark,
        seed=2026, target_enrol=1, target_test=1, minimum_targets=30,
    )
    assert summary["research_only"] is True
    assert summary["enrolment_takes_per_speaker"] == 1
    splits = read_csv(benchmark / "google_speaker_splits.csv")
    lookup = {x["speaker_id"]: x["split"] for x in splits}
    for partition in ("dev", "test"):
        enrol = read_csv(benchmark / f"google_{partition}_enrol.csv")
        trials = read_csv(benchmark / f"google_{partition}_trials.csv")
        for target in enrol:
            assert lookup[target["speaker_id"]] == partition
        assert all(x["label"] in ("0", "1") for x in trials)
        assert len([x for x in trials if x["label"] == "1"]) > 0
    cfg = RunConfig.from_dict({
        "feature": {"mode": "group", "bands": 8, "vad_db": None},
        "em": {"mixtures": 2, "iterations": 3, "reg": 0.05, "chunk": 32},
        "map": {"relevance": 4.},
        "score": {"norm": "none"},
        "transform": "none",
        "seeds": [11], "enrolment_size": 1,
        "max_frames_per_background_file": 4,
        "evaluate_eval": True,
        "save_model": False,
        "device": "cpu",
    })
    output = tmp_path / "experiment"
    result = run(corpus, cfg, output, track="google", benchmark=benchmark)
    assert result["track"] == "google" and result["research_only"]
    assert result["locked_google_test_metrics"]["genuine"] > 0
    assert 0 <= result["locked_google_test_metrics"]["eer"] <= 1
    saved = read_csv(output / "evaluation_scores.csv")
    assert "label" in saved[0] and "score" in saved[0]
    assert (output / "locked_google_test_metrics.json").is_file()
    assert not (output / "submission.csv").exists()


def test_enrolled_models_fail_on_fewer_takes():
    x = torch.randn(50, 4, generator=torch.Generator().manual_seed(3))
    bg = TorchGMM(EMConfig(mixtures=2, iterations=3)).fit(x)
    feats = {"e": x[:5]}
    with pytest.raises(ValueError, match="requested 3"):
        enrolled_models(
            bg, feats,
            [{"speaker_id": "s", "file_id": "e", "enrol_index": 0}],
            MAPConfig(), k=3
        )

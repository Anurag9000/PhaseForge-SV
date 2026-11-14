"""Google-only MFCC, logmel and passphrase research tests; never Kaggle training."""
import io
import math
import struct
import wave

import pytest
import torch

from phaseforge.data import GoogleResearchData, inventory_google, google_benchmark
from phaseforge.google_reference_features import (
    ReferenceFeature, GoogleOnlyReferenceExtractor, extract_google_reference,
)
from phaseforge.google_mathworks_reference import MathWorksResearch, run_google_mathworks
from phaseforge.new_keyword import KeywordRun, run_keyword
from phaseforge.new_classical_runner import ClassicalRun, run_classical


def _wav(freq: int):
    buffer = io.BytesIO()
    samples = [int(6000 * math.sin(2 * math.pi * freq * t / 16000))
               for t in range(2048)]
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return buffer.getvalue()


@pytest.fixture
def original_google_research(tmp_path):
    root = tmp_path / "google_original"
    for word in ("stop", "go"):
        (root / word).mkdir(parents=True)
    for person in range(31):
        for take in (0, 1):
            (root / "stop" / f"spk{person:03d}_nohash_{take}.wav").write_bytes(
                _wav(300 + person * 7 + take)
            )
        (root / "go" / f"spk{person:03d}_nohash_0.wav").write_bytes(
            _wav(700 + person * 5)
        )
    benchmark = tmp_path / "original_google_benchmark"
    inventory_google(root, benchmark)
    google_benchmark(
        benchmark / "google_manifest.csv", benchmark, seed=2026,
        target_enrol=1, target_test=1, minimum_targets=30,
    )
    return root, benchmark


def test_google_only_reference_prohibits_kaggle_data_type():
    with pytest.raises(PermissionError, match="Google research ONLY"):
        extract_google_reference(
            object(), ["f"], ReferenceFeature(), torch.device("cpu")
        )


@pytest.mark.parametrize("kind", ["mfcc", "logmel"])
def test_reference_feature_frames_finite(kind):
    t = torch.arange(2000) / 16000
    y = (torch.sin(2 * math.pi * 300 * t) * .4).float()
    cfg = ReferenceFeature(kind=kind, bands=8, coefficients=4)
    z = GoogleOnlyReferenceExtractor(cfg)(y)
    assert z.ndim == 2
    assert z.shape[-1] == (4 if kind=="mfcc" else 8)
    assert torch.isfinite(z).all()


def test_mathworks_original_google_reference_pipeline(original_google_research, tmp_path):
    root, benchmark = original_google_research
    cfg = MathWorksResearch.from_dict({
        "reference":{"kind":"mfcc","bands":8,"coefficients":4,
                     "win":640,"hop":160,"n_fft":1024},
        "em":{"mixtures":2,"covariance":"diag","iterations":3,"reg":.05},
        "map":{"relevance":4,"adapt_variances":True,"adapt_weights":True},
        "score":{"norm":"none","smoothing":3},
        "frame_cap":4,"max_background_files":12,
        "device":"cpu",
        "evaluate_locked_test":False,
    })
    out = tmp_path / "original_google_mfcc"
    result = run_google_mathworks(root, benchmark, cfg, out)
    assert result["source_track"] == "original_google_mfcc_reference"
    assert result["research_only"] and result["metrics"]["genuine"] > 0
    assert (out / "dev_scores.csv").exists()
    assert not (out / "eval_scores.csv").exists()
    assert not (out / "submission.csv").exists()


def test_original_google_keyword_logmel_braided(original_google_research, tmp_path):
    root, benchmark = original_google_research
    cfg = KeywordRun.from_dict({
        "frontend":"logmel", "words":["stop","go"],
        "reference":{"kind":"logmel","bands":8,"coefficients":4},
        "architecture":{"name":"bcnn_2d","channels":8,"embedding":8,"depth":1},
        "epochs":1,"steps_per_epoch":1,"batch_size":4,
        "max_frames":32,"max_files_per_split":20,"device":"cpu",
    })
    out = tmp_path / "original_keyword"
    result = run_keyword(root, benchmark, cfg, out)
    assert result["source_track"] == "original_google_keyword_research"
    assert result["metrics"]["development"]["count"] > 0
    assert 0 <= result["metrics"]["development"]["accuracy"] <= 1
    assert result["frontend"] == "logmel"
    assert not (out / "locked_test_keywords.csv").exists()


def test_google_word_specific_background_ubm(original_google_research, tmp_path):
    root, benchmark = original_google_research
    cfg = ClassicalRun.from_dict({
        "method":"passphrase_ubm",
        "feature":{"mode":"group","bands":8},
        "em":{"mixtures":2,"iterations":3,"reg":.05},
        "map":{"relevance":4},
        "transform":"pca","dimensions":4,
        "frame_cap":4,"max_background_files":25,
        "device":"cpu",
    })
    out = tmp_path / "research_passphrase"
    result = run_classical(root, cfg, out, "google", benchmark)
    assert result["research_only"]
    assert result["method"] == "passphrase_ubm"
    assert (out / "development_scores.csv").exists()

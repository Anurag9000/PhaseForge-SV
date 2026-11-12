import math

import pytest
import torch

from phaseforge.features import FeatureConfig, PhaseExtractor, mel_bands


@pytest.fixture
def signal():
    t = torch.arange(16000) / 16000
    return .4 * torch.sin(2 * math.pi * 224 * t) + .2 * torch.sin(2 * math.pi * 501 * t)


@pytest.mark.parametrize("mode", [
    "group", "temporal", "ifd", "cosphase", "direct", "group_cepstrum",
    "modified_group_delay", "lp_residual", "all_pole", "multi",
])
def test_phase_modes_are_finite(signal, mode):
    cfg = FeatureConfig(mode=mode, lpc_order=8, bands=12, normalization="partial",
                        mean_strength=.5, std_strength=.5)
    features = PhaseExtractor(cfg)(signal)
    assert features.ndim == 2
    assert 10 <= len(features) <= 101
    assert features.shape[1] >= 12
    assert torch.isfinite(features).all()


def test_masked_pooling_and_silent_input():
    x = torch.zeros(16000)
    for mode in ("group", "temporal", "lp_residual", "all_pole"):
        out = PhaseExtractor(FeatureConfig(mode=mode))(x)
        assert len(out) >= 1
        assert torch.isfinite(out).all()


def test_partial_mean_and_scale_are_independent(signal):
    f0 = PhaseExtractor(FeatureConfig(
        normalization="none", vad_db=None
    ))(signal)
    f1 = PhaseExtractor(FeatureConfig(
        normalization="partial", mean_strength=0, std_strength=0, vad_db=None
    ))(signal)
    assert torch.allclose(f0, f1, atol=1e-6)
    centered = PhaseExtractor(FeatureConfig(
        normalization="utterance", vad_db=None
    ))(signal)
    assert float(centered.mean(0).abs().max()) < 1e-4


def test_phase_band_weights_normalized():
    W = mel_bands(257, 24, 16000, torch.device("cpu"))
    assert W.shape == (257, 24)
    assert torch.allclose(W.sum(0), torch.ones(24), atol=1e-5)


def test_fft_window_validation():
    with pytest.raises(ValueError):
        FeatureConfig(win=640, n_fft=512).validate()

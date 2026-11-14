"""Batched GPU-friendly phase-STFT must match the reference exactly."""
from dataclasses import replace

import pytest
import torch

from phaseforge.bonus_frontend import (
    batched_phase_features, can_batch, extract_grouped_features,
)
from phaseforge.features import FeatureConfig, PhaseExtractor


@pytest.mark.parametrize("mode", ["group", "temporal", "cosphase", "direct"])
@pytest.mark.parametrize("normalization", ["utterance", "partial", "none"])
@pytest.mark.parametrize("vad_db", [None, -30.0])
def test_batched_phase_matches_original_utterance_extraction(
    mode, normalization, vad_db,
):
    generator = torch.Generator().manual_seed(17)
    signals = [
        torch.randn(10000, generator=generator) * (.001 if i == 0 else .3)
        for i in range(4)
    ]
    cfg = FeatureConfig(
        mode=mode, bands=8, win=400, hop=160, n_fft=512,
        vad_db=vad_db, normalization=normalization,
        mean_strength=.5, std_strength=.7,
        masked_mean=True, reliability=1e-4, delta=True,
    )
    expected = [PhaseExtractor(cfg)(v) for v in signals]
    actual = batched_phase_features(signals, cfg)
    assert len(actual) == len(expected)
    for old, new in zip(expected, actual):
        assert old.shape == new.shape
        assert torch.allclose(old, new, atol=1e-4, rtol=2e-4)


def test_batched_variable_length_falls_back_without_padding_bias():
    signals = [torch.randn(2000), torch.randn(2200)]
    cfg = FeatureConfig(mode="group", bands=8)
    results = batched_phase_features(signals, cfg)
    for waveform, result in zip(signals, results):
        assert torch.equal(result, PhaseExtractor(cfg)(waveform))


def test_batched_unsupported_phase_mode_uses_original():
    config = FeatureConfig(mode="lp_residual", bands=8)
    assert not can_batch(config)
    signal = torch.randn(2048)
    assert torch.equal(
        batched_phase_features([signal], config)[0],
        PhaseExtractor(config)(signal),
    )


def test_batched_frontend_maintains_order_and_per_file_identity():
    gen = torch.Generator().manual_seed(18)
    waves = {
        "b":torch.randn(1600, generator=gen),
        "a":torch.randn(2000, generator=gen),
        "e":torch.randn(1600, generator=gen),
        "c":torch.randn(1800, generator=gen),
        "f":torch.randn(1600, generator=gen),
        "d":torch.randn(1800, generator=gen),
    }

    class Data:
        def load(self, key):
            return waves[key]

    cfg = FeatureConfig(mode="temporal", bands=8)
    order = ["d", "a", "c", "b", "f", "e"]
    actual = extract_grouped_features(
        Data(), order, cfg, torch.device("cpu"), batch_size=3,
    )
    expected = {file:PhaseExtractor(cfg)(waves[file]) for file in order}
    assert set(actual) == set(expected)
    for file in order:
        assert torch.allclose(actual[file], expected[file], atol=1e-4, rtol=2e-4)


def test_nonpositive_batch_size_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        extract_grouped_features(
            object(), [], FeatureConfig(),
            torch.device("cpu"), batch_size=0,
        )

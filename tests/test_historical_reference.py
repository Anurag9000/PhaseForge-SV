"""Compare current phase features directly with original 2026 notebook equations.

This reference is deliberately independent of PhaseExtractor.mel_bands and
tests waveform-to-feature numerical compatibility, not just config equality.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from phaseforge.features import FeatureConfig, PhaseExtractor


def _legacy_band_matrix(nfreq: int, nbands: int, sr: int) -> torch.Tensor:
    freqs = torch.linspace(0, sr / 2, nfreq)
    mel = lambda x: 2595. * torch.log10(1 + x / 700.)
    unmel = lambda x: 700. * (
        torch.pow(torch.tensor(10.), x / 2595.) - 1
    )
    centers = unmel(torch.linspace(mel(freqs[0]), mel(freqs[-1]), nbands + 2))
    bands = []
    for k in range(nbands):
        lo, middle, hi = centers[k:k + 3]
        tri = torch.minimum(
            (freqs - lo) / (middle - lo + 1e-8),
            (hi - freqs) / (hi - middle + 1e-8),
        ).clamp_min(0)
        bands.append(tri / (tri.sum() + 1e-8))
    return torch.stack(bands, 1)


def _original_notebook_phase(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Original notebook's unmodified group/temporal/cosphase equations."""
    win, hop, fft, bands, sr = 400, 160, 512, 24, 16000
    window = torch.hann_window(win)
    X = torch.stft(
        x, n_fft=fft, hop_length=hop, win_length=win,
        window=window, center=False, return_complex=True
    )
    magnitude = X.abs()
    peak = magnitude.amax().clamp_min(1e-8)
    unit = X / magnitude.clamp_min(1e-7 * peak)
    valid = (magnitude > peak * 1e-4).float()
    W = _legacy_band_matrix(fft // 2 + 1, bands, sr)
    if mode == "group":
        cross = unit[1:] * unit[:-1].conj()
        circular = cross / cross.abs().clamp_min(1e-8)
        mask = valid[1:] * valid[:-1]
        circular = F.pad(circular, (0, 0, 1, 0))
        mask = F.pad(mask, (0, 0, 1, 0))
    elif mode == "temporal":
        shift = (2 * math.pi * hop / fft) * torch.arange(fft // 2 + 1)
        cross = unit[:, 1:] * unit[:, :-1].conj() * torch.exp(
            -1j * shift[:, None]
        )
        circular = cross / cross.abs().clamp_min(1e-8)
        mask = valid[:, 1:] * valid[:, :-1]
        circular = F.pad(circular, (1, 0))
        mask = F.pad(mask, (1, 0))
    elif mode == "cosphase":
        circular, mask = unit, valid
    else:
        raise ValueError(mode)
    features = torch.cat(
        [(circular.real * mask).T @ W, (circular.imag * mask).T @ W], dim=-1
    )
    frames = x.unfold(0, fft, hop)
    rms = frames.square().mean(1).sqrt()
    active = rms > rms.max().clamp_min(1e-8) * 10 ** (-30 / 20)
    if int(active.sum()) < 10:
        top = rms.topk(min(10, len(rms))).indices
        active = torch.zeros(len(rms), dtype=torch.bool)
        active[top] = True
    features = features[active]
    avg = features.mean(0, keepdim=True)
    std = features.std(0, unbiased=False, keepdim=True).clamp_min(.06)
    return (features - avg) / std


@pytest.mark.parametrize("mode", ["group", "temporal", "cosphase"])
def test_historical_phase_streams_match_original_notebook(mode):
    gen = torch.Generator().manual_seed(43)
    t = torch.arange(16000) / 16000
    samples = (
        .34 * torch.sin(2 * math.pi * 261.7 * t)
        + .19 * torch.sin(2 * math.pi * 519.2 * t)
        + .02 * torch.randn(16000, generator=gen)
    )
    config = FeatureConfig(
        mode=mode, win=400, hop=160, n_fft=512, bands=24,
        vad_db=-30, masked_mean=False, normalization="utterance",
        delta=False, reliability=1e-4,
    )
    actual = PhaseExtractor(config)(samples)
    expected = _original_notebook_phase(samples, mode)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4)

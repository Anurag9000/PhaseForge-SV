import pytest
import torch

from phaseforge.robustness import perturb_waveform, default_conditions


def test_noise_perturbation_reproducible():
    x = torch.linspace(-1, 1, 1000)
    a = perturb_waveform(x, "noise", 10, seed=222)
    b = perturb_waveform(x, "noise", 10, seed=222)
    assert torch.equal(a, b)
    noise = a - x
    measured_snr = 20 * torch.log10(
        x.square().mean().sqrt() / noise.square().mean().sqrt()
    )
    assert float(measured_snr) == pytest.approx(10, abs=.01)


def test_gain_and_shift_preserve_shape():
    x = torch.arange(30).float()
    assert torch.equal(perturb_waveform(x, "gain", .5), x / 2)
    shifted = perturb_waveform(x, "shift", 4)
    assert len(shifted) == len(x)
    assert torch.equal(shifted[:4], torch.zeros(4))
    assert torch.equal(shifted[4:], x[:-4])


def test_silence_and_invalid_perturbations():
    x = torch.zeros(400)
    assert torch.equal(perturb_waveform(x, "noise", 5), x)
    assert len(default_conditions()) >= 6
    with pytest.raises(ValueError):
        perturb_waveform(x, "gain", 0)
    with pytest.raises(ValueError):
        perturb_waveform(x, "shift", 500)
    with pytest.raises(ValueError):
        perturb_waveform(x, "unknown", 0)

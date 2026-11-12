"""Data-limited independently trained GMM must not inherit UBM parameters."""
import pytest
import torch

from phaseforge.gmm import EMConfig, MAPConfig, TorchGMM
from phaseforge.scratch import ScratchGMMComparison


def test_scratch_trains_new_gmm_and_scores_per_frame_llr():
    generator = torch.Generator().manual_seed(2026)
    bg_frames = torch.cat((
        torch.randn(100, 4, generator=generator) - 2.,
        torch.randn(100, 4, generator=generator) + 2.
    ))
    ubm = TorchGMM(EMConfig(mixtures=4, iterations=5)).fit(bg_frames)
    enrollment = [
        torch.randn(15, 4, generator=generator) + 1.5
        for _ in range(3)
    ]
    method = ScratchGMMComparison(ubm, mixtures=2, max_iterations=6)
    speaker = method.adapt(enrollment, MAPConfig())
    assert speaker is not ubm
    assert not torch.allclose(speaker.mu, ubm.mu[:2])
    test = torch.randn(12, 4, generator=generator)
    expected = (speaker.log_prob(test) - ubm.log_prob(test)).mean()
    assert torch.allclose(method.score(test, speaker), expected, atol=1e-6)


def test_scratch_rejects_empty_enrollment():
    ubm = TorchGMM(EMConfig(mixtures=2))
    with pytest.raises(ValueError):
        ScratchGMMComparison(ubm).adapt([])

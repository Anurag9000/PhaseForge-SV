import torch

from phaseforge.gmm import MAPConfig
from phaseforge.hmm import HMMConfig, LeftRightHMMUBM


def test_baum_welch_hmm_posterior_and_llr_are_finite():
    generator = torch.Generator().manual_seed(194)
    recordings = []
    for _ in range(8):
        segments = [
            torch.randn(8, 2, generator=generator) * .3 + center
            for center in (-1.5, 0, 1.5)
        ]
        recordings.append(torch.cat(segments))
    hmm = LeftRightHMMUBM(HMMConfig(
        states=3, mixtures_per_state=2, iterations=3,
        init_em_iterations=5, covariance_floor=.05
    )).fit(recordings)
    value, gamma, self_count, next_count = hmm.forward_backward(recordings[0])
    assert torch.isfinite(value)
    assert gamma.shape == (24, 3)
    assert torch.allclose(gamma.sum(-1), torch.ones(24), atol=2e-4)
    assert torch.all(self_count >= 0) and torch.all(next_count >= 0)
    adapted = hmm.adapt(recordings[:3], MAPConfig(relevance=8))
    llr = hmm.score(recordings[4], adapted)
    assert torch.isfinite(llr)
    assert len(adapted) == 3
    assert torch.allclose(hmm.self_probability[-1], torch.tensor(1.0))

"""Verify optional one-pass EM stats/LL remains a true GMM EM update."""
from dataclasses import replace

import pytest
import torch

from phaseforge.gmm import EMConfig, TorchGMM


@pytest.mark.parametrize("covariance", ["diag", "spherical", "tied", "full"])
def test_fused_em_matches_reference_fixed_iterations(covariance):
    g = torch.Generator().manual_seed(117)
    X = torch.cat([
        torch.randn(80, 4, generator=g) - 1.,
        torch.randn(80, 4, generator=g) + 1.,
    ])
    base = EMConfig(
        mixtures=3, covariance=covariance, init="random",
        iterations=3, tolerance=1e-15, chunk=37, reg=.05,
        seed=17,
    )
    control = TorchGMM(base).fit(X)
    fused = TorchGMM(replace(base, em_impl="fused")).fit(X)
    assert len(control.history) == len(fused.history)
    assert torch.allclose(control.mu, fused.mu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(control.weights, fused.weights, atol=1e-4, rtol=1e-4)
    assert torch.allclose(control.cov, fused.cov, atol=1e-4, rtol=1e-4)
    assert abs(control.mean_log_prob(X) - fused.mean_log_prob(X)) < 1e-4


@pytest.mark.parametrize("covariance", ["diag", "spherical", "tied", "full"])
def test_fused_e_step_statistics_match_reference(covariance):
    X = torch.randn(90, 4, generator=torch.Generator().manual_seed(8))
    m = TorchGMM(EMConfig(mixtures=3, covariance=covariance, iterations=2,
                         init="random", seed=14, reg=.02, chunk=17))
    m._initialize(X)
    control = m.sufficient_statistics(X, with_full_second=covariance == "full")
    optimized = m._fused_stats_and_ll(X, with_full_second=covariance == "full")
    for a, b in zip(control, optimized[:4]):
        if a is None:
            assert b is None
        else:
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert abs(optimized[4] - m.mean_log_prob(X)) < 1e-5


def test_em_impl_rejects_unknown_backend():
    with pytest.raises(ValueError, match="EM implementation"):
        EMConfig(em_impl="not_a_thing").validate()

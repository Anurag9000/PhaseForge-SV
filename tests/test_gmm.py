import pytest
import torch

from phaseforge.gmm import EMConfig, MAPConfig, TorchGMM


@pytest.fixture
def mixture():
    gen = torch.Generator().manual_seed(41)
    a = torch.randn(200, 3, generator=gen) * .35 - 2
    b = torch.randn(200, 3, generator=gen) * .65 + 2
    return torch.cat([a, b])


@pytest.mark.parametrize("covariance", ["diag", "spherical", "tied", "full"])
def test_gaussian_em_all_covariances(mixture, covariance):
    model = TorchGMM(EMConfig(
        mixtures=2, covariance=covariance, iterations=14, reg=.05,
        chunk=64, seed=1
    )).fit(mixture)
    assert model.mu.shape == (2, 3)
    assert len(model.history) >= 1
    assert torch.isfinite(model.log_prob(mixture)).all()
    assert torch.allclose(model.weights.sum(), torch.tensor(1.), atol=1e-5)
    assert model.information_criteria(mixture)["aic"] > 0


@pytest.mark.parametrize("init", ["random", "kmeans"])
def test_initializations(mixture, init):
    fitted = TorchGMM(EMConfig(
        mixtures=4, init=init, iterations=5, seed=42
    )).fit(mixture)
    assert all(torch.isfinite(torch.tensor(fitted.history)))


def test_map_means_only_and_zero_occupancy(mixture):
    model = TorchGMM(EMConfig(mixtures=2, iterations=8)).fit(mixture)
    enroll = mixture[:20]
    adapted = model.adapt(enroll, MAPConfig(relevance=8))
    assert torch.allclose(adapted.cov, model.cov)
    assert torch.allclose(adapted.weights, model.weights)
    heavily_regularized = model.adapt(enroll, MAPConfig(relevance=1e8))
    assert torch.allclose(heavily_regularized.mu, model.mu, atol=1e-4)


def test_map_variance_and_weight_extensions(mixture):
    model = TorchGMM(EMConfig(mixtures=2, iterations=8)).fit(mixture)
    adapted = model.adapt(mixture[:60], MAPConfig(
        relevance=4, adapt_variances=True, adapt_weights=True
    ))
    assert bool((adapted.cov > 0).all())
    assert torch.allclose(adapted.weights.sum(), torch.tensor(1.), atol=1e-5)
    assert torch.isfinite(adapted.log_prob(mixture[:10])).all()


def test_parameter_count():
    K, D = 4, 3
    model = TorchGMM(EMConfig(mixtures=K, covariance="diag"))
    model.mu = torch.zeros(K, D)
    assert model.parameter_count() == (K - 1) + 2 * K * D

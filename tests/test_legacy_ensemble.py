"""Regression tests for the five-seed historical notebook scoring protocol."""
from __future__ import annotations

import pytest
import torch

from phaseforge.gmm import EMConfig, TorchGMM
from phaseforge.scoring import ScoreConfig, ensemble_cohort_normalize


def _table(z, t, cohort=("c1", "c2")):
    return {
        "target_names": ["a", "b"],
        "test_names": ["f1", "f2"],
        "cohort_names": list(cohort),
        "background_files": ["bg1", "bg2"],
        "z": torch.tensor(z, dtype=torch.float64),
        "t": torch.tensor(t, dtype=torch.float64),
    }


def test_ensemble_cohort_uses_mean_samples_then_unbiased_statistics():
    rows = [
        {"claimed_id": "a", "test_file": "f1"},
        {"claimed_id": "b", "test_file": "f2"},
    ]
    raw = torch.tensor([3., 5.], dtype=torch.float64)
    one = _table([[0., 4.], [1., 7.]], [[-1., 3.], [2., 8.]])
    two = _table([[2., 6.], [3., 9.]], [[1., 5.], [4., 10.]])
    config = ScoreConfig(norm="s", norm_stage="after_ensemble")
    result = ensemble_cohort_normalize(rows, raw, [one, two], config)
    z = (one["z"] + two["z"]) / 2
    t = (one["t"] + two["t"]) / 2
    expected = torch.stack([
        ((raw[0] - z[0].mean()) / z[0].std(unbiased=True)
         + (raw[0] - t[0].mean()) / t[0].std(unbiased=True)) / 2,
        ((raw[1] - z[1].mean()) / z[1].std(unbiased=True)
         + (raw[1] - t[1].mean()) / t[1].std(unbiased=True)) / 2,
    ])
    assert torch.allclose(result, expected, atol=1e-10)


def test_changing_ensemble_cohort_population_is_rejected():
    rows = [{"claimed_id": "a", "test_file": "f1"}]
    one = _table([[1., 2.], [2., 3.]], [[1., 2.], [2., 3.]])
    two = _table([[1., 2.], [2., 3.]], [[1., 2.], [2., 3.]],
                 cohort=("c2", "c3"))
    with pytest.raises(ValueError, match="same background-only cohort"):
        ensemble_cohort_normalize(
            rows, torch.tensor([2.]), [one, two],
            ScoreConfig(norm="s", norm_stage="after_ensemble")
        )


def test_legacy_kmeans_initialization_and_dead_component_config():
    cfg = EMConfig(
        mixtures=2, init_subsample=60, kmeans_steps=10,
        init_variance_source="initialization",
        dead_policy="hard_examples", dead_weight_fraction=.1,
        legacy_pre_mstep_likelihood=True, iterations=6,
    )
    data = torch.cat([
        torch.randn(60, 3, generator=torch.Generator().manual_seed(4)) - 2.,
        torch.randn(60, 3, generator=torch.Generator().manual_seed(5)) + 2.,
    ])
    model = TorchGMM(cfg).fit(data)
    assert len(model.history) >= 2
    assert torch.isfinite(model.log_prob(data)).all()
    assert abs(float(model.weights.sum()) - 1) < 1e-6

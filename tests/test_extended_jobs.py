"""Extended runnable job tests; synthetic data only, no Kaggle/Google network."""
import json

import pytest
import torch

from phaseforge.engine import RunConfig
from phaseforge.alternatives import run_alternative
from phaseforge.hmm import HMMConfig
from phaseforge.optimizer import adaptive_optimize
from phaseforge.experiments import expand_grid
from phaseforge.studies import proxy_manifests


def test_random_grid_never_materializes_the_full_cartesian_product():
    grid = {f"p{i}": list(range(10)) for i in range(12)}
    sampled = expand_grid(grid, max_runs=17, strategy="random", seed=2026)
    assert len(sampled) == 17
    assert len({tuple(x[k] for k in sorted(x)) for x in sampled}) == 17


def test_invalid_proxy_has_explicit_error():
    class FakeData:
        background = [{"speaker_id": "b0", "file_id": "b0"}]
    with pytest.raises(ValueError):
        proxy_manifests(FakeData())


def test_basic_hmm_alternative_training_import_and_scoring(tmp_path):
    from phaseforge.hmm import LeftRightHMMUBM
    generator = torch.Generator().manual_seed(3)
    recordings = []
    for _ in range(4):
        frames = [
            torch.randn(7, 3, generator=generator) + bias
            for bias in (-1.0, 0.5, 2.0)
        ]
        recordings.append(torch.cat(frames))
    m = LeftRightHMMUBM(HMMConfig(
        states=3, mixtures_per_state=2, iterations=2, init_em_iterations=3
    )).fit(recordings)
    val = m.score(recordings[0], m.adapt(recordings[1:], __import__(
        "phaseforge.gmm", fromlist=["MAPConfig"]
    ).MAPConfig(relevance=8)))
    assert torch.isfinite(val)


def test_adaptive_optimizer_validation_argument():
    from phaseforge.optimizer import adaptive_optimize
    with pytest.raises(ValueError):
        adaptive_optimize("unused", {
            "initial": {}, "stages": {}, "rounds": 1,
            "validation": "hidden-kaggle-test",
        }, "/tmp/phaseforge_test_invalid_protocol")

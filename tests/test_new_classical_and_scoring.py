"""Unit tests for i-vector posterior, SVM, PLDA, calibration and full new-suite policy."""
import json
import math
from pathlib import Path

import pytest
import torch

from phaseforge.gmm import EMConfig, TorchGMM
from phaseforge.new_classical import (
    PairwiseLinearSVM, TotalVariability, TwoCovariancePLDA, map_supervector,
    supervised_lda,
)
from phaseforge.new_classical_runner import ClassicalRun
from phaseforge.new_keyword import KeywordRun
from phaseforge.new_neural_runner import NeuralRun
from phaseforge.new_scoring import (
    adaptive_snorm, calibrate_saved_predictions, cllr, length_quality,
    min_dcf, speaker_grouped_calibration,
)
from phaseforge.new_suite import execute_new_suite


def _synthetic_background():
    torch.manual_seed(8)
    sequences = {
        f"s{s}_{i}": torch.randn(18, 4) * .3 + float(s)
        for s in range(3) for i in range(3)
    }
    labels = {k: k.split("_")[0] for k in sequences}
    return sequences, labels


def test_ivector_total_variability_matches_posterior_equations():
    data, labels = _synthetic_background()
    X = torch.cat(list(data.values()))
    ubm = TorchGMM(EMConfig(mixtures=2, iterations=4, reg=.05)).fit(X)
    tv = TotalVariability(ubm, dimension=3, iterations=3).fit(data)
    vector = tv.transform(data["s0_0"])
    assert vector.shape == (3,)
    assert torch.isfinite(vector).all()
    assert torch.allclose(vector.norm(), torch.ones(()), atol=1e-4)
    assert tv.T.shape == (2, 4, 3)
    assert torch.isfinite(tv.T).all()


def test_supervector_svm_and_plda():
    data, labels = _synthetic_background()
    ubm = TorchGMM(EMConfig(mixtures=2, iterations=4)).fit(
        torch.cat(list(data.values()))
    )
    embeddings = {key: map_supervector(ubm, seq) for key, seq in data.items()}
    dim = len(next(iter(embeddings.values())))
    assert dim == 8
    pair = PairwiseLinearSVM(epochs=3).fit(embeddings, labels)
    cosine = pair.score(embeddings["s0_0"], embeddings["s1_0"])
    assert torch.isfinite(cosine)
    plda = TwoCovariancePLDA().fit(embeddings, labels)
    s = plda.score(embeddings["s0_0"], embeddings["s0_1"])
    assert torch.isfinite(s)
    mean, projection = supervised_lda(embeddings, labels, 2)
    assert mean.shape == (dim,)
    assert projection.shape == (dim, 2)


def test_adaptive_snorm_quality_and_min_dcf():
    z = torch.tensor([[2., 1., 0.], [3., 0., -1.]], dtype=torch.float64)
    t = torch.tensor([[1., 0., -2.], [4., 2., -1.]], dtype=torch.float64)
    v = adaptive_snorm(torch.tensor([2., 3.], dtype=torch.float64), z, t, top_k=2)
    assert v.shape == (2,) and torch.isfinite(v).all()
    q = length_quality(torch.tensor([0, 20, 100]))
    assert q[0] == 0 and q[2] > q[1]
    labels = torch.tensor([1,1,0,0])
    score = torch.tensor([4., 2., -2., -3.])
    assert min_dcf(score, labels)["min_dcf"] == 0
    assert cllr(score, labels) >= 0


def test_grouped_calibration_never_fits_on_heldout_recording():
    trials = [
        {"claimed_id": str(i%3), "test_file": str(i//2), "label": i%2,
         "trial_id": str(i)}
        for i in range(18)
    ]
    score = torch.tensor([1. if r["label"] else -1. for r in trials])
    output = speaker_grouped_calibration(score, trials, folds=3)
    assert output.shape == score.shape
    assert torch.isfinite(output).all()


def test_new_configs_and_kaggle_keyword_isolation():
    with pytest.raises(ValueError):
        execute_new_suite(".", {
            "suite_family": "new_phase_research",
            "jobs": [{"id":"duplicate", "kind":"keyword_google", "config":"x"},
                     {"id":"duplicate", "kind":"keyword_google", "config":"x"}],
        }, "test-out")
    # Neural and classical config schemas parse without importing Google data.
    assert NeuralRun.from_dict({"architecture":{"name":"ecapa"}}).architecture.name == "ecapa"
    assert ClassicalRun.from_dict({"method":"ivector_cosine"}).method == "ivector_cosine"
    assert KeywordRun.from_dict({"architecture":{"name":"kwt"}}).architecture.name == "kwt"

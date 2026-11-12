import math

import torch

from phaseforge.gmm import EMConfig, MAPConfig, TorchGMM
from phaseforge.metrics import eer, grouped_bootstrap, report, roc
from phaseforge.scoring import (
    ScoreConfig, enrolled_models, exploratory_four_claim_adjustment,
    likelihood_ratio, score_trials,
)
from phaseforge.temporal import PhaseDTW, TimeBinnedUBM, dtw_cost


def test_perfect_and_tied_eer():
    scores = torch.tensor([3., 2., -2., -3.])
    labels = torch.tensor([1, 1, 0, 0])
    assert eer(scores, labels) == 0
    assert math.isclose(eer(torch.ones(4), labels), .5, abs_tol=1e-8)
    assert report(scores, labels)["auc"] == 1.0
    assert torch.isfinite(roc(scores, labels)["far"]).all()


def test_llr_is_utterance_average():
    bg = torch.cat([torch.randn(100, 2) - 1, torch.randn(100, 2) + 1])
    base = TorchGMM(EMConfig(mixtures=2, iterations=8, seed=8)).fit(bg)
    speaker = base.adapt(bg[:30], MAPConfig(relevance=2))
    frames = bg[:15]
    expected = (speaker.log_prob(frames) - base.log_prob(frames)).mean()
    measured = likelihood_ratio(base, speaker, frames)
    assert torch.allclose(measured, expected, atol=1e-5)


def test_enrolment_three_and_trial_scoring():
    x = torch.randn(200, 2)
    bg = TorchGMM(EMConfig(mixtures=2, iterations=6)).fit(x)
    feat = {"e1": x[:10], "e2": x[10:20], "e3": x[20:30], "test": x[30:40]}
    enrol = [
        {"speaker_id": "spk", "file_id": f"e{i}", "enrol_index": i - 1}
        for i in (1, 2, 3)
    ]
    speaker = enrolled_models(bg, feat, enrol, MAPConfig(relevance=8), k=3)
    trials = [{"claimed_id": "spk", "test_file": "test"}]
    z = score_trials(bg, feat, trials, speaker)
    assert z.shape == (1,)
    assert torch.isfinite(z).all()


def test_group_dependent_extension_explicitly_not_baseline():
    trials = [{"test_file": "file", "claimed_id": f"spk{i}"} for i in range(4)]
    scores = torch.tensor([2., 1., -1., 0.])
    shifted = exploratory_four_claim_adjustment(trials, scores, alpha=.5)
    assert shifted[0] == scores[0]
    assert shifted[1] < scores[1]


def test_group_bootstrap_with_positives_and_negatives():
    trials = [
        {"claimed_id": f"s{i // 2}", "test_file": f"f{i // 2}", "label": i % 2}
        for i in range(10)
    ]
    scores = torch.tensor([float(x["label"]) for x in trials])
    result = grouped_bootstrap(trials, scores, iterations=30, seed=9)
    assert result["resamples"] == 30
    assert result["ci95"] == [0.0, 0.0]


def test_temporal_gmm_and_dtw():
    a = torch.sin(torch.linspace(0, 9, 24))[:, None]
    b = torch.cos(torch.linspace(0, 9, 24))[:, None]
    assert float(dtw_cost(a, a)) < float(dtw_cost(a, b))
    model = PhaseDTW()
    assert model.score(a, [a, b]) > model.score(b, [a])
    tgmm = TimeBinnedUBM(states=3, mixture_per_state=2)
    tgmm.fit([a, b, a + .1, b + .1], EMConfig(
        mixtures=2, iterations=4, reg=.05
    ))
    adapted = tgmm.adapt([a, a + .05, a + .1], MAPConfig(relevance=4))
    assert torch.isfinite(tgmm.score(a, adapted))

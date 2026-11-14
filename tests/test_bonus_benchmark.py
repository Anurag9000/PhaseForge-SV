"""Numerical and trial-protocol checks for the end-to-end GPU bonus."""
from dataclasses import replace
import json

import pytest
import torch

from phaseforge.bonus_benchmark import BenchCase, run_bonus, synthetic_data
from phaseforge.bonus_scoring import accelerated_score_trials
from phaseforge.gmm import EMConfig, MAPConfig, TorchGMM
from phaseforge.scoring import ScoreConfig, score_trials


@pytest.fixture()
def speaker_setup():
    g = torch.Generator().manual_seed(42)
    training = torch.cat((
        torch.randn(100, 6, generator=g)-1.8,
        torch.randn(100, 6, generator=g)+1.8,
    ))
    bg = TorchGMM(EMConfig(mixtures=3, iterations=5, reg=.05, chunk=40)).fit(training)
    features = {}
    claimed = {}
    for i in range(5):
        name = f"speaker_{i}"
        sample = torch.randn(30, 6, generator=g) + i * .3
        claimed[name] = bg.adapt(sample, MAPConfig(relevance=4))
    for i in range(7):
        features[f"test_{i}"] = torch.randn(21+i, 6, generator=g)
    trials = [
        {"trial_id":f"t{i}_{j}", "claimed_id":f"speaker_{j}",
         "test_file":f"test_{i}", "label": str(int(i == j))}
        for i in range(7) for j in range(5)
    ]
    # Include duplicates and shuffled ID ordering; never reorder trial scores.
    trials += [dict(trials[3], trial_id="repeated3")]
    return bg, features, claimed, trials


@pytest.mark.parametrize("speaker_chunk", [1,2,9])
@pytest.mark.parametrize("frame_chunk", [1,7,100])
def test_batched_llr_matches_reference_for_all_chunks(
    speaker_setup, speaker_chunk, frame_chunk
):
    bg, features, models, trials = speaker_setup
    control = score_trials(bg, features, trials, models, ScoreConfig(norm="none"))
    config = ScoreConfig(norm="none", implementation="batched",
                         speaker_chunk=speaker_chunk, frame_chunk=frame_chunk)
    faster = score_trials(bg, features, trials, models, config)
    assert faster.shape == control.shape
    assert torch.allclose(faster, control, atol=2e-4, rtol=1e-5)


def test_unsupported_smoothing_falls_back_to_reference(speaker_setup):
    bg, features, models, trials = speaker_setup
    ordinary = score_trials(bg, features, trials, models, ScoreConfig(smoothing=3))
    adapted = score_trials(bg, features, trials, models,
                           ScoreConfig(smoothing=3, implementation="batched"))
    assert torch.equal(ordinary, adapted)
    with pytest.raises(ValueError, match="diagonal"):
        accelerated_score_trials(
            bg, features, trials, models,
            ScoreConfig(smoothing=3), strict_acceleration=True
        )


def test_fast_mode_does_not_use_labels(speaker_setup):
    bg, features, models, trials = speaker_setup
    config = ScoreConfig(implementation="batched")
    original = score_trials(bg, features, trials, models, config)
    unlabeled = [{k:v for k,v in row.items() if k!="label"} for row in trials]
    assert torch.equal(original, score_trials(bg, features, unlabeled, models, config))


def test_synthetic_bonus_fit_and_research_artifacts(tmp_path):
    case = BenchCase(name="tiny", frames=160, eval_frames=40,
                     dimensions=4, mixtures=2, iterations=3,
                     repeats=1, reg=.05)
    tr, ev, provenance, dev = synthetic_data(case)
    assert tr.shape == (160, 4)
    assert ev.shape == (40, 4)
    assert dev is None
    rows = run_bonus([case], tmp_path, ["native"], torch.device("cpu"), strict=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "complete"
    assert row["fit_seconds"] > 0
    assert row["score_seconds"] >= 0
    assert row["sanity"]["nonfinite_heldout_logprobs"] == 0
    assert row["sanity"]["sum_weights"] == pytest.approx(1, abs=1e-5)
    assert (tmp_path/"bonus_results.json").exists()
    assert (tmp_path/"bonus_summary.csv").exists()
    document = json.loads((tmp_path/"bonus_results.json").read_text())
    assert document["runs"][0]["backend"] == "native"
    assert document["runs"][0]["source_provenance"]["source"] == "synthetic"


def test_missing_external_backend_must_never_claim_success(tmp_path):
    case = BenchCase(name="nobinary", frames=100, eval_frames=20,
                     dimensions=3, mixtures=2, iterations=2, repeats=1)
    # CI has no GPU: explicit skipped/failed status, no invented speedup.
    rows = run_bonus([case], tmp_path, ["cuvs"], torch.device("cpu"))
    assert rows[0]["status"] != "complete"

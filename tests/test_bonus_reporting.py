"""Bonus reporting is evidence-grounded: no invented GPU or EER results."""
import json
from pathlib import Path

import pytest

from phaseforge.bonus_report import aggregate_bench, render_report


def _fixture():
    return {
        "environment":{
            "git_commit":"synthetic-fixture",
            "pytorch":"test", "torch_cuda":None, "gpus":None,
            "packages":{"torchgmm":None}
        },
        "runs":[
            {"case":"tiny","backend":"native","repeat":0,
             "status":"complete", "fit_seconds":.004, "score_seconds":.001,
             "log_likelihood_mean":-3., "gpu_memory":None,
             "sanity":{"nonfinite_heldout_logprobs":0},
             "dev_verification":None},
            {"case":"tiny","backend":"native","repeat":1,
             "status":"complete", "fit_seconds":.006, "score_seconds":.002,
             "log_likelihood_mean":-2.9, "gpu_memory":None,
             "sanity":{"nonfinite_heldout_logprobs":0},
             "dev_verification":None},
            {"case":"tiny","backend":"torchgmm","repeat":0,
             "status":"complete", "fit_seconds":.03, "score_seconds":.0007,
             "log_likelihood_mean":-2.8, "gpu_memory":None,
             "sanity":{"nonfinite_heldout_logprobs":0},
             "dev_verification":None},
            {"case":"tiny","backend":"cuvs","repeat":0,
             "status":"skipped","reason":"No compiled CUDA binary"}
        ],
    }


def test_benchmark_report_does_not_invent_missing_gpu_measurements():
    report = render_report(_fixture())
    assert "tiny" in report
    assert "native" in report
    assert "torchgmm" in report
    assert "cuvs" in report
    assert "N/A" in report
    assert "No hidden evaluation labels" in report


def test_aggregate_keeps_seed_variance_and_failure_counts_separate():
    values = aggregate_bench(_fixture())
    native = next(x for x in values if x["backend"] == "native")
    cuvs = next(x for x in values if x["backend"] == "cuvs")
    assert native["complete"] == 2
    assert native["fit_seconds"] == pytest.approx(.005)
    assert native["fit_seconds_std"] > 0
    assert cuvs["complete"] == 0
    assert cuvs["skipped"] == 1
    assert cuvs["fit_seconds"] is None
    assert cuvs["development_eer_median"] is None


def test_report_sections_can_include_measured_trial_parity_and_ablations():
    scoring = {
        "checkpoint_sha256":"toy-checkpoint",
        "n_dev_trials":8,
        "max_absolute_score_difference":.00001,
        "exact_parity_within_2e_4":True,
        "scoring":{
            "reference":{"median_score_seconds":.02},
            "batched":{"median_score_seconds":.01}
        },
        "measured_speedup":2.,
        "eer_difference":0.,
    }
    e2e = {"development_ranked":[
        {"id":"fast","EER":.1,"AUC":.93,"elapsed_seconds":10.}
    ]}
    text=render_report(_fixture(),scoring,e2e)
    assert "toy-checkpoint" in text
    assert "fast" in text
    assert "2" in text
    assert "End-to-end" in text

"""Identical-parameter likelihood benchmark remains independent of fitter."""
import json

import pytest
import torch

from phaseforge.bonus_benchmark import BenchCase
from phaseforge.bonus_identical_scoring import compare_identical


def test_native_frozen_parameter_scoring_produces_measured_record(tmp_path):
    case = BenchCase(
        name="frozen_smoke", source="synthetic",
        frames=150, dimensions=4, mixtures=2, iterations=3,
        eval_frames=40, reg=.05, repeats=1,
    )
    result = compare_identical(case, torch.device("cpu"), tmp_path, [])
    assert result["experiment"] == "identical_parameters_frozen_loglikelihood"
    assert result["results"]["native"]["status"] == "complete"
    assert result["results"]["native"]["max_abs_logp_diff"] == 0
    assert result["frames"] == 40
    assert (tmp_path/"identical_parameter_scoring.json").is_file()


def test_cuvs_frozen_parameters_explicitly_skipped_without_binary(tmp_path):
    case = BenchCase(
        name="frozen_missing_cuvs", frames=150, dimensions=4,
        mixtures=2, iterations=3, eval_frames=40,
    )
    result = compare_identical(case, torch.device("cpu"), tmp_path, ["cuvs"])
    assert result["results"]["cuvs"]["status"] == "skipped"


def test_full_covariance_frozen_parity_refuses_unsupported_path(tmp_path):
    case = BenchCase(
        name="frozen_full", frames=150, dimensions=4,
        mixtures=2, iterations=3, eval_frames=40, covariance="full",
    )
    with pytest.raises(ValueError, match="diag/spherical"):
        compare_identical(case, torch.device("cpu"), tmp_path, [])

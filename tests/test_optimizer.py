import pytest
from phaseforge.optimizer import aggregate_validation


def test_aggregate_proxy_robustness_objectives():
    values = [0.1, 0.2]
    assert aggregate_validation(values, "mean") == pytest.approx(.15)
    assert aggregate_validation(values, "worst") == pytest.approx(.20)
    assert aggregate_validation(values, "mean_plus_std") == pytest.approx(.20)
    assert aggregate_validation([.1], "mean_plus_std") == pytest.approx(.1)


def test_empty_nonfinite_and_bad_objectives_are_rejected():
    with pytest.raises(ValueError):
        aggregate_validation([], "mean")
    with pytest.raises(ValueError):
        aggregate_validation([float("nan")], "mean")
    with pytest.raises(ValueError):
        aggregate_validation([.1], "invalid")

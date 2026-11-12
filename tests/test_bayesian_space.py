"""Bayesian search controllers never touch evaluation labels or model inference."""
import pytest

from phaseforge.bayes_search import suggest_value


class DummyTrial:
    def suggest_categorical(self, key, values):
        return values[0]

    def suggest_float(self, key, low, high, **kwargs):
        return low

    def suggest_int(self, key, low, high, **kwargs):
        return low


def test_bayesian_search_parameter_definitions():
    t = DummyTrial()
    assert suggest_value(t, "feature.mode", ["group","temporal"]) == "group"
    assert suggest_value(t, "map.relevance", {
        "type":"float", "low":.5, "high":32, "log":True
    }) == .5
    assert suggest_value(t, "em.mixtures", {
        "type":"int", "low":8, "high":128
    }) == 8
    with pytest.raises(ValueError):
        suggest_value(t,"unknown",{"type":"mystery"})

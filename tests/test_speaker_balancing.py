import pytest
import torch

from phaseforge.engine import _background_frames, RunConfig


def test_background_speaker_balancing_equalizes_per_speaker_budget():
    features = {
        "a1": torch.ones(20, 2) * 1.0,
        "a2": torch.ones(20, 2) * 1.0,
        "a3": torch.ones(20, 2) * 1.0,
        "b1": torch.ones(20, 2) * 2.0,
    }
    speakers = {"a1": "a", "a2": "a", "a3": "a", "b1": "b"}
    ids = list(features)
    raw = _background_frames(features, ids, 10, torch.device("cpu"))
    equal = _background_frames(
        features, ids, 10, torch.device("cpu"),
        speaker_by_file=speakers, strategy="speaker_balanced"
    )
    assert len(raw) == 40
    assert len(equal) == 40
    assert int((raw[:, 0] == 1).sum()) == 30
    assert int((raw[:, 0] == 2).sum()) == 10
    assert int((equal[:, 0] == 1).sum()) == 20
    assert int((equal[:, 0] == 2).sum()) == 20


def test_invalid_balancing_requires_speaker_metadata():
    with pytest.raises(ValueError):
        _background_frames(
            {"f": torch.randn(10, 3)}, ["f"], 5, torch.device("cpu"),
            strategy="speaker_balanced"
        )


def test_config_roundtrip_and_validation():
    config = RunConfig.from_dict({"background_sampling": "speaker_balanced"})
    assert config.background_sampling == "speaker_balanced"
    assert config.to_dict()["background_sampling"] == "speaker_balanced"
    with pytest.raises(ValueError):
        RunConfig.from_dict({"background_sampling": "unknown"}).validate()

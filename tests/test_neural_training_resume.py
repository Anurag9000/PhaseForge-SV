"""Exact interrupted-vs-uninterrupted neural training regression.

Tests that the training checkpoint resumes optimization and RNG/EMA states,
not merely reuses a final encoder snapshot.
"""
import torch
import pytest

from phaseforge.new_neural_runner import NeuralRun, train_phase_network


@pytest.mark.parametrize("objective", ["aam", "byol"])
def test_epoch_resume_matches_uninterrupted_training(tmp_path, objective):
    generator = torch.Generator().manual_seed(29)
    features = {
        f"s{s}_{take}": torch.randn(24, 8, generator=generator) * .2 + s * .1
        for s in range(4) for take in range(3)
    }
    rows = [
        {"speaker_id":f"s{s}", "file_id":f"s{s}_{take}"}
        for s in range(4) for take in range(3)
    ]
    config = NeuralRun.from_dict({
        "feature":{"mode":"group","bands":8},
        "architecture":{"name":"xvector","channels":16,
                        "embedding":8,"depth":1},
        "objective":objective,
        "epochs":3,"steps_per_epoch":2,
        "speakers_per_batch":3,"takes_per_speaker":2,
        "max_frames":16,"device":"cpu","seed":2026,
    })
    full, _, uninterrupted = train_phase_network(
        features, rows, config, torch.device("cpu"),
        source_tag="synthetic-resume-test",
    )
    path = tmp_path / "train_checkpoint.pt"
    partial, _, partial_history = train_phase_network(
        features, rows, config, torch.device("cpu"),
        checkpoint_file=path, max_epochs_this_invocation=1,
        source_tag="synthetic-resume-test",
    )
    assert partial_history["completed_epochs"] == 1
    assert path.is_file()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["next_epoch"] == 1
    assert saved["optimizer"]
    assert saved["torch_rng"].numel() > 0
    if objective == "byol":
        assert saved["teacher"] is not None
    resumed, _, completed = train_phase_network(
        features, rows, config, torch.device("cpu"),
        checkpoint_file=path, source_tag="synthetic-resume-test",
    )
    assert completed["completed_epochs"] == 3
    assert completed["loss_history"] == uninterrupted["loss_history"]
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    assert completed["objective_state"] is not None


def test_changed_neural_configuration_restarts_checkpoint(tmp_path):
    generator = torch.Generator().manual_seed(9)
    features = {f"s{s}_{k}":torch.randn(16, 8, generator=generator)
                for s in range(3) for k in range(2)}
    rows = [{"speaker_id":f"s{s}", "file_id":f"s{s}_{k}"}
            for s in range(3) for k in range(2)]
    common = {
        "architecture":{"name":"xvector","channels":16,"embedding":8,"depth":1},
        "epochs":2,"steps_per_epoch":1,
        "speakers_per_batch":2,"takes_per_speaker":2,
        "max_frames":16, "seed":13, "device":"cpu",
    }
    path = tmp_path / "resume.pt"
    old = NeuralRun.from_dict(common)
    train_phase_network(
        features, rows, old, torch.device("cpu"),
        checkpoint_file=path, max_epochs_this_invocation=1
    )
    first_identity = torch.load(path, weights_only=True)["identity"]
    new = NeuralRun.from_dict({**common, "learning_rate":.002})
    _, _, artifact = train_phase_network(
        features, rows, new, torch.device("cpu"),
        checkpoint_file=path, max_epochs_this_invocation=1
    )
    second_identity = torch.load(path, weights_only=True)["identity"]
    assert first_identity != second_identity
    assert artifact["completed_epochs"] == 1

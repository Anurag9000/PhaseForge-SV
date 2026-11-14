"""Numerical smoke tests of ALL newly added phase-speaker backbones and losses."""
import pytest
import torch

from phaseforge.new_architectures import Architecture, MODEL_NAMES, PhaseSpeakerEncoder
from phaseforge.new_neural_runner import (
    NeuralRun, train_phase_network, verify_embeddings,
)
from phaseforge.new_objectives import (
    AAMSoftmax, GE2ELoss, PairVerifier, batch_hard_triplet,
    pairwise_bce, prototypical_loss, supervised_contrastive,
)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_new_backbone_embeddings_have_gradients(name):
    torch.manual_seed(13)
    config = Architecture(name=name, channels=16, embedding=8, depth=1, heads=2)
    model = PhaseSpeakerEncoder(8, config)
    frames = torch.randn(4, 32, 8)
    valid = torch.ones(4, 32, dtype=torch.bool)
    valid[0, 20:] = False
    scores = model(frames, valid)
    assert scores.shape == (4, 8)
    assert torch.isfinite(scores).all()
    loss = (scores.square() * torch.linspace(1, 2, 8)).sum()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_ge2e_leave_one_out_and_metric_objectives():
    torch.manual_seed(22)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    embedding = torch.randn(6, 8, requires_grad=True)
    losses = [
        GE2ELoss()(embedding, 3, 2),
        AAMSoftmax(8, 4)(embedding, labels),
        supervised_contrastive(embedding, labels),
        batch_hard_triplet(embedding, labels),
        prototypical_loss(embedding, 3, 2),
        pairwise_bce(PairVerifier(8), embedding, labels),
    ]
    for loss in losses:
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    assert embedding.grad is not None


@pytest.mark.parametrize("objective", [
    "cross_entropy", "aam", "ge2e", "supcon", "triplet",
    "prototype", "pairwise", "byol", "jepa",
])
def test_new_neural_training_objectives_with_background_only(objective):
    torch.manual_seed(3)
    features = {
        f"b{s}_{t}": torch.randn(18, 8) + .1 * s
        for s in range(3) for t in range(2)
    }
    features["dev_test"] = torch.randn(18, 8)
    background = [
        {"speaker_id": f"s{s}", "file_id": f"b{s}_{t}"}
        for s in range(3) for t in range(2)
    ]
    cfg = NeuralRun(
        architecture=Architecture(name="xvector", channels=16, embedding=8, depth=1),
        objective=objective, epochs=1, steps_per_epoch=1,
        speakers_per_batch=2, takes_per_speaker=2,
        max_frames=16, device="cpu", score="pairwise" if objective=="pairwise" else "cosine",
    )
    net, scorer, artifact = train_phase_network(
        features, background, cfg, torch.device("cpu")
    )
    assert artifact["training_utterances"] == 6
    assert len(artifact["loss_history"]) == 1
    assert torch.isfinite(torch.tensor(artifact["loss_history"])).all()
    sample = {
        x: net(torch.randn(1, 16, 8), torch.ones(1,16,dtype=torch.bool))[0].detach()
        for x in ("en0", "en1", "dev")
    }
    out = verify_embeddings(
        sample, [
            {"speaker_id":"s", "file_id":"en0", "enrol_index":0},
            {"speaker_id":"s", "file_id":"en1", "enrol_index":1},
        ], [{"claimed_id":"s", "test_file":"dev"}], 2,
        scorer="pairwise" if scorer is not None else "cosine", pair=scorer
    )
    assert out.shape == (1,) and torch.isfinite(out).all()

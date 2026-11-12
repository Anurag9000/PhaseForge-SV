import torch

from phaseforge.learned import (
    PhaseConvEncoder, PhaseEncoderConfig, learn_phase_encoder
)


def test_supervised_phase_cnn_from_scratch():
    gen = torch.Generator().manual_seed(7)
    feats, bg = {}, []
    for speaker in range(3):
        for take in range(2):
            file_id = f"x{speaker}_{take}"
            feats[file_id] = torch.randn(18, 8, generator=gen) + speaker * .1
            bg.append({"file_id": file_id, "speaker_id": f"s{speaker}"})
    feats["target_test"] = torch.randn(18, 8, generator=gen)
    cfg = PhaseEncoderConfig(hidden=12, embedding=6, epochs=2, batch_size=3)
    out, artifact = learn_phase_encoder(feats, bg, cfg, torch.device("cpu"))
    assert out["target_test"].shape == (18, 6)
    assert torch.isfinite(out["target_test"]).all()
    assert artifact["background_files"] == 6
    assert len(artifact["training_losses"]) == 2


def test_selfsup_phase_cnn():
    feats = {f"u{i}": torch.randn(12, 8) for i in range(4)}
    bg = [{"speaker_id": f"s{i}", "file_id": f"u{i}"} for i in range(4)]
    cfg = PhaseEncoderConfig(task="contrastive", hidden=10, embedding=5,
                             batch_size=2, epochs=1)
    result, artifact = learn_phase_encoder(feats, bg, cfg, torch.device("cpu"))
    assert result["u0"].shape == (12, 5)
    assert artifact["head_state"] is None

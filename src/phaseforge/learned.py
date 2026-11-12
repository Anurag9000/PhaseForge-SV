"""Learned phase-only CNN encoders, trained from scratch on background audio features.

Supervised speaker classes are restricted to Kaggle background speakers. The
alternative self-supervised InfoNCE objective uses two augmented phase views
of the SAME background utterance. No external encoders or magnitude inputs.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PhaseEncoderConfig:
    task: str = "background_speaker"  # background_speaker | contrastive
    hidden: int = 64
    embedding: int = 24
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = .001
    weight_decay: float = .0001
    dropout: float = .1
    max_training_frames: int = 128
    temperature: float = .1
    seed: int = 2026

    def validate(self) -> None:
        if self.task not in ("background_speaker", "contrastive"):
            raise ValueError("Unknown phase encoder objective")
        if self.hidden < 2 or self.embedding < 1 or self.epochs < 1:
            raise ValueError("Invalid encoder dimensions/epochs")
        if self.batch_size < 2 or self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("Invalid optimization hyperparameters")


class PhaseConvEncoder(nn.Module):
    """Frame-to-frame convolutions over phase-derived channels, no magnitude input."""

    def __init__(self, input_dim: int, config: PhaseEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, config.hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(config.hidden, config.hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden, config.embedding, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, frames, phase_dimensions]")
        return self.layers(x.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def pool(encoded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(encoded.dtype)
        return (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)


def _batch(
    features: dict[str, torch.Tensor], ids: list[str],
    max_frames: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cropped = [features[i][:max_frames] for i in ids]
    n = max(len(x) for x in cropped)
    dimension = cropped[0].shape[1]
    x = torch.zeros(len(ids), n, dimension, device=device)
    mask = torch.zeros(len(ids), n, device=device, dtype=torch.bool)
    for i, sequence in enumerate(cropped):
        size = len(sequence)
        x[i, :size] = sequence.to(device)
        mask[i, :size] = True
    return x, mask


def _augment(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # Phase feature masking/jitter only; no magnitude/other recording input.
    retain = (torch.rand_like(x) > .10) & valid[:, :, None]
    jitter = torch.randn_like(x) * .015
    return torch.where(retain, x + jitter, torch.zeros_like(x))


def learn_phase_encoder(
    features: dict[str, torch.Tensor], background_rows: list[dict],
    config: PhaseEncoderConfig, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return embedded features and model record. Speaker classes are BACKGROUND ONLY."""
    config.validate()
    bgrows = list(background_rows)
    file_ids = [x["file_id"] for x in bgrows]
    if not file_ids or any(x not in features for x in file_ids):
        raise ValueError("Training requires available background feature IDs")
    speakers = sorted({row["speaker_id"] for row in bgrows})
    if config.task == "background_speaker" and len(speakers) < 2:
        raise ValueError("Need >=2 background speakers")
    lookup = {speaker: index for index, speaker in enumerate(speakers)}
    targets = {row["file_id"]: lookup[row["speaker_id"]] for row in bgrows}
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    encoder = PhaseConvEncoder(features[file_ids[0]].shape[1], config).to(device)
    head = nn.Linear(config.embedding, max(2, len(speakers))).to(device)
    parameters = list(encoder.parameters())
    if config.task == "background_speaker":
        parameters += list(head.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    rng = random.Random(config.seed)
    for epoch in range(config.epochs):
        order = file_ids.copy()
        rng.shuffle(order)
        losses = []
        encoder.train()
        head.train()
        for pos in range(0, len(order), config.batch_size):
            selected = order[pos:pos + config.batch_size]
            if config.task == "contrastive" and len(selected) < 2:
                continue
            x, valid = _batch(features, selected, config.max_training_frames, device)
            if config.task == "background_speaker":
                encoded = encoder(_augment(x, valid))
                pooled = encoder.pool(encoded, valid)
                labels = torch.tensor([targets[i] for i in selected], device=device)
                loss = F.cross_entropy(head(pooled), labels)
            else:
                first = F.normalize(encoder.pool(encoder(_augment(x, valid)), valid), dim=-1)
                second = F.normalize(encoder.pool(encoder(_augment(x, valid)), valid), dim=-1)
                similarities = (first @ second.T) / config.temperature
                labels = torch.arange(len(selected), device=device)
                loss = (F.cross_entropy(similarities, labels)
                        + F.cross_entropy(similarities.T, labels)) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("Insufficient background samples for contrastive training")
        history.append(sum(losses) / len(losses))
    encoder.eval()
    transformed = {}
    with torch.no_grad():
        for ident, seq in features.items():
            transformed[ident] = encoder(seq[None].to(device)).squeeze(0).cpu()
    artifact = {
        "config": asdict(config),
        "encoder_state": {k: v.cpu() for k, v in encoder.state_dict().items()},
        "head_state": {k: v.cpu() for k, v in head.state_dict().items()}
                      if config.task == "background_speaker" else None,
        "training_losses": history,
        "background_speakers": speakers,
        "background_files": len(file_ids),
        "original_phase_dimensions": features[file_ids[0]].shape[1],
    }
    return transformed, artifact

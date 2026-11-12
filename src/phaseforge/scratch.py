"""Independent speaker GMM from three enrolment takes, measured against a UBM.

This is an intentionally data-limited baseline to quantify how much MAP
regularization contributes. It is NOT used for eligible competition submissions.
The UBM is trained only on designated background speakers.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from .gmm import EMConfig, TorchGMM


class ScratchGMMComparison:
    """Independent per-speaker EM, with background GMM as scoring denominator."""

    def __init__(
        self, ubm: TorchGMM, *,
        mixtures: int = 8, covariance: str = "diag",
        covariance_floor: float = .05, max_iterations: int = 30,
        seed: int = 2026,
    ):
        if mixtures < 1 or covariance_floor <= 0 or max_iterations < 1:
            raise ValueError("Invalid scratch speaker model configuration")
        self.ubm = ubm
        self.config = EMConfig(
            mixtures=mixtures, covariance=covariance,
            reg=covariance_floor, iterations=max_iterations, seed=seed
        )

    def adapt(self, enrollment: list[torch.Tensor], _unused_map_config=None) -> TorchGMM:
        """Fit from SCRATCH: no UBM means, covariances or weights transferred."""
        if len(enrollment) < 1:
            raise ValueError("Need speaker enrolment recordings")
        frames = torch.cat(enrollment, dim=0)
        mixtures = min(self.config.mixtures, len(frames))
        return TorchGMM(replace(self.config, mixtures=mixtures)).fit(frames)

    @torch.no_grad()
    def score(self, recording: torch.Tensor, speaker_model: TorchGMM) -> torch.Tensor:
        return (speaker_model.log_prob(recording) - self.ubm.log_prob(recording)).mean()

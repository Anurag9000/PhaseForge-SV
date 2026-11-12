"""Temporal phase models and phase-template DTW, all implemented in PyTorch.

TimeBinnedUBM is explicitly a *time-binned GMM-UBM* and not a Baum-Welch HMM.
It is useful as a low-data temporal verification ablation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import torch

from .gmm import EMConfig, MAPConfig, TorchGMM


def time_bins(frames: torch.Tensor, states: int) -> list[torch.Tensor]:
    if states < 1 or len(frames) < states:
        raise ValueError("Need at least one frame per temporal state")
    boundaries = torch.linspace(0, len(frames), states + 1).long()
    return [frames[boundaries[i]:boundaries[i + 1]] for i in range(states)]


class TimeBinnedUBM:
    def __init__(self, states: int = 4, mixture_per_state: int = 8, seed: int = 2026):
        self.states, self.mixtures, self.seed = states, mixture_per_state, seed
        self.models: list[TorchGMM] = []

    def fit(self, background_features: list[torch.Tensor], em: EMConfig) -> "TimeBinnedUBM":
        state_frames = [[] for _ in range(self.states)]
        for utterance in background_features:
            if len(utterance) < self.states:
                continue
            for i, segment in enumerate(time_bins(utterance, self.states)):
                state_frames[i].append(segment)
        self.models = [
            TorchGMM(replace(em, mixtures=self.mixtures, seed=self.seed + i)).fit(
                torch.cat(state_frames[i], 0)
            ) for i in range(self.states)
        ]
        return self

    def adapt(self, enrol: list[torch.Tensor], map_cfg: MAPConfig) -> list[TorchGMM]:
        per_state = [[] for _ in range(self.states)]
        for utterance in enrol:
            if len(utterance) >= self.states:
                for i, segment in enumerate(time_bins(utterance, self.states)):
                    per_state[i].append(segment)
        if any(not x for x in per_state):
            raise ValueError("No enrolment features in one or more states")
        return [m.adapt(torch.cat(parts), map_cfg) for m, parts in zip(self.models, per_state)]

    @torch.no_grad()
    def score(self, test: torch.Tensor, adapted: list[TorchGMM]) -> torch.Tensor:
        parts = time_bins(test, self.states)
        contributions = [
            (speaker.log_prob(segment) - bg.log_prob(segment)).sum()
            for segment, bg, speaker in zip(parts, self.models, adapted)
        ]
        return torch.stack(contributions).sum() / len(test)


def dtw_cost(
    X: torch.Tensor, Y: torch.Tensor, band_ratio: float = .4,
    distance: str = "euclidean",
) -> torch.Tensor:
    """Normalized DTW; diagonal-ish band, positive costs, no gradient required."""
    if not len(X) or not len(Y):
        raise ValueError("Empty DTW sequence")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension mismatch")
    if distance == "euclidean":
        dist = torch.cdist(X, Y)
    elif distance == "cosine":
        X = torch.nn.functional.normalize(X, dim=-1)
        Y = torch.nn.functional.normalize(Y, dim=-1)
        dist = (1 - X @ Y.T).clamp_min(0)
    else:
        raise ValueError(distance)
    n, m = dist.shape
    width = max(abs(n - m), int(max(n, m) * band_ratio))
    dp = torch.full((n + 1, m + 1), float("inf"), device=X.device)
    lengths = torch.zeros(n + 1, m + 1, device=X.device)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        jmin = max(1, round(i * m / n) - width)
        jmax = min(m, round(i * m / n) + width)
        for j in range(jmin, jmax + 1):
            previous = torch.stack([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]])
            which = int(previous.argmin())
            prev_length = (lengths[i - 1, j], lengths[i, j - 1],
                           lengths[i - 1, j - 1])[which]
            dp[i, j] = dist[i - 1, j - 1] + previous[which]
            lengths[i, j] = prev_length + 1
    return dp[n, m] / lengths[n, m].clamp_min(1)


class PhaseDTW:
    """Research-only speaker-template matching, not a replacement for mandatory LLR."""

    def __init__(self, distance: str = "cosine", band_ratio: float = .4):
        self.distance, self.band_ratio = distance, band_ratio

    @torch.no_grad()
    def score(self, test: torch.Tensor, enrolment: list[torch.Tensor]) -> torch.Tensor:
        return -torch.stack([
            dtw_cost(test, x, self.band_ratio, self.distance)
            for x in enrolment
        ]).min()

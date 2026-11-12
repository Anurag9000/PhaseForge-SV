"""True left-to-right Gaussian-mixture HMM-UBM, trained using Baum-Welch in torch.

The conventional framewise GMM-UBM remains the competition's mandatory baseline.
This module is an experimental text-dependent *sequence* likelihood model;
it adapts emission distributions but shares word-state transitions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .gmm import EMConfig, MAPConfig, TorchGMM
from .temporal import time_bins


@dataclass(frozen=True)
class HMMConfig:
    states: int = 4
    mixtures_per_state: int = 4
    iterations: int = 10
    init_em_iterations: int = 16
    covariance_floor: float = .02
    transition_prior: float = 2.0
    seed: int = 2026
    convergence: float = 1e-4

    def validate(self) -> None:
        if (self.states < 2 or self.mixtures_per_state < 1 or self.iterations < 1
                or self.covariance_floor <= 0 or self.transition_prior <= 0):
            raise ValueError("Invalid HMM configuration")


class LeftRightHMMUBM:
    def __init__(self, config: HMMConfig):
        config.validate()
        self.config = config
        self.models: list[TorchGMM] = []
        self.self_probability: torch.Tensor | None = None
        self.history: list[float] = []

    def initialize(self, recordings: list[torch.Tensor]) -> None:
        S = self.config.states
        recordings = [x for x in recordings if len(x) >= S]
        if not recordings:
            raise ValueError("Need recordings long enough for every HMM state")
        state_pool = [[] for _ in range(S)]
        for frames in recordings:
            for j, segment in enumerate(time_bins(frames, S)):
                state_pool[j].append(segment)
        self.models = []
        for j in range(S):
            X = torch.cat(state_pool[j])
            model = TorchGMM(EMConfig(
                mixtures=self.config.mixtures_per_state,
                covariance="diag", iterations=self.config.init_em_iterations,
                reg=self.config.covariance_floor, seed=self.config.seed + j,
            )).fit(X)
            self.models.append(model)
        device = recordings[0].device
        self.self_probability = torch.full((S,), .75, device=device)
        self.self_probability[-1] = 1.0

    def _emissions(
        self, recording: torch.Tensor,
        models: list[TorchGMM] | None = None,
    ) -> torch.Tensor:
        models = models if models is not None else self.models
        return torch.stack([model.log_prob(recording) for model in models], -1)

    def forward_backward(
        self, recording: torch.Tensor, models: list[TorchGMM] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns log-likelihood, state occupancies, expected self and next counts.

        Start is state zero; terminal state is forced to S-1; transitions permit
        only staying in place or moving one state forward. No task/word labels.
        """
        if self.self_probability is None or len(recording) < self.config.states:
            raise ValueError("HMM not initialized or recording too short")
        E = self._emissions(recording, models)
        T, S = E.shape
        neg_inf = torch.tensor(-float("inf"), device=E.device, dtype=E.dtype)
        ps = self.self_probability
        log_stay = ps.clamp_min(1e-10).log()
        log_next = (1 - ps[:-1]).clamp_min(1e-10).log()
        alpha = torch.full_like(E, -float("inf"))
        alpha[0, 0] = E[0, 0]
        for t in range(1, T):
            alpha[t, 0] = alpha[t - 1, 0] + log_stay[0] + E[t, 0]
            for j in range(1, S):
                alpha[t, j] = torch.logaddexp(
                    alpha[t - 1, j] + log_stay[j],
                    alpha[t - 1, j - 1] + log_next[j - 1]
                ) + E[t, j]
        value = alpha[-1, -1]
        beta = torch.full_like(E, -float("inf"))
        beta[-1, -1] = 0.0
        for t in range(T - 2, -1, -1):
            for j in range(S):
                same = log_stay[j] + E[t + 1, j] + beta[t + 1, j]
                if j < S - 1:
                    move = log_next[j] + E[t + 1, j + 1] + beta[t + 1, j + 1]
                    beta[t, j] = torch.logaddexp(same, move)
                else:
                    beta[t, j] = same
        gamma = (alpha + beta - value).exp()
        self_count = torch.stack([
            (alpha[:-1, j] + log_stay[j] + E[1:, j] + beta[1:, j] - value).exp().sum()
            for j in range(S)
        ])
        next_count = torch.cat([
            torch.stack([
                (alpha[:-1, j] + log_next[j] + E[1:, j + 1] + beta[1:, j + 1]
                 - value).exp().sum()
                for j in range(S - 1)
            ]),
            torch.zeros(1, device=E.device, dtype=E.dtype),
        ])
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite forward-backward likelihood")
        return value, gamma, self_count, next_count

    @torch.no_grad()
    def fit(self, recordings: list[torch.Tensor]) -> "LeftRightHMMUBM":
        self.initialize(recordings)
        S, K = self.config.states, self.config.mixtures_per_state
        D = recordings[0].shape[-1]
        self.history = []
        for iteration in range(self.config.iterations):
            count = torch.zeros(S, K, device=recordings[0].device)
            first = torch.zeros(S, K, D, device=recordings[0].device)
            second = torch.zeros(S, K, D, device=recordings[0].device)
            stay = torch.zeros(S, device=recordings[0].device)
            advance = torch.zeros(S, device=recordings[0].device)
            total_ll, valid = 0.0, 0
            for utterance in recordings:
                if len(utterance) < S:
                    continue
                likelihood, gamma, cstay, cnext = self.forward_backward(utterance)
                total_ll += float(likelihood)
                valid += 1
                stay += cstay
                advance += cnext
                for state, model in enumerate(self.models):
                    responsibility = model.responsibilities(utterance) * gamma[:, state, None]
                    count[state] += responsibility.sum(0)
                    first[state] += responsibility.T @ utterance
                    second[state] += responsibility.T @ utterance.square()
            self.history.append(total_ll / max(valid, 1))
            for state, model in enumerate(self.models):
                safe = count[state].clamp_min(1e-8)
                inferred_mean = first[state] / safe[:, None]
                inferred_var = (second[state] / safe[:, None]
                                - inferred_mean.square()).clamp_min(
                                    self.config.covariance_floor
                                )
                reliable = count[state] >= 1
                model.mu = torch.where(reliable[:, None], inferred_mean, model.mu)
                model.cov = torch.where(reliable[:, None], inferred_var, model.cov)
                new_weight = torch.where(reliable, count[state], model.weights)
                model.weights = new_weight.clamp_min(1e-9) / new_weight.sum().clamp_min(1e-9)
            prior = self.config.transition_prior
            transition = (stay[:-1] + prior) / (stay[:-1] + advance[:-1] + 2 * prior)
            self.self_probability = torch.cat([
                transition.clamp(.05, .99), torch.ones_like(transition[:1])
            ])
            if len(self.history) > 2:
                previous = self.history[-2]
                if abs(self.history[-1] - previous) < (
                    self.config.convergence * max(1, abs(previous))
                ):
                    break
        return self

    @torch.no_grad()
    def adapt(self, enrolment: list[torch.Tensor], map_cfg: MAPConfig) -> list[TorchGMM]:
        """MAP emission parameters with UBM HMM state occupancy as fractional weights."""
        map_cfg.validate()
        S, K = self.config.states, self.config.mixtures_per_state
        D = self.models[0].mu.shape[-1]
        device = self.models[0].mu.device
        count = torch.zeros(S, K, device=device)
        first = torch.zeros(S, K, D, device=device)
        second = torch.zeros(S, K, D, device=device)
        for recording in enrolment:
            if len(recording) < S:
                continue
            _, gamma, _, _ = self.forward_backward(recording)
            for state, bg in enumerate(self.models):
                posterior = bg.responsibilities(recording) * gamma[:, state, None]
                count[state] += posterior.sum(0)
                first[state] += posterior.T @ recording
                second[state] += posterior.T @ recording.square()
        adapted = []
        for state, original in enumerate(self.models):
            counts = count[state]
            alpha = counts / (counts + map_cfg.relevance)
            sample_mu = first[state] / counts.clamp_min(1e-12)[:, None]
            speaker = TorchGMM(original.config)
            speaker.mu = (alpha[:, None] * sample_mu
                          + (1 - alpha[:, None]) * original.mu)
            speaker.cov = original.cov.clone()
            speaker.weights = original.weights.clone()
            if map_cfg.adapt_variances:
                old_m2 = original.cov + original.mu.square()
                new_m2 = (alpha[:, None] * second[state] /
                          counts.clamp_min(1e-12)[:, None]
                          + (1 - alpha[:, None]) * old_m2)
                speaker.cov = (new_m2 - speaker.mu.square()).clamp_min(
                    map_cfg.variance_floor
                )
            if map_cfg.adapt_weights:
                empirical = counts / counts.sum().clamp_min(1e-12)
                speaker.weights = alpha * empirical + (1 - alpha) * original.weights
                speaker.weights = speaker.weights.clamp_min(1e-9)
                speaker.weights /= speaker.weights.sum()
            adapted.append(speaker)
        return adapted

    @torch.no_grad()
    def log_likelihood(
        self, recording: torch.Tensor, models: list[TorchGMM] | None = None,
    ) -> torch.Tensor:
        score, _, _, _ = self.forward_backward(recording, models)
        return score

    @torch.no_grad()
    def score(self, recording: torch.Tensor, speaker_models: list[TorchGMM]) -> torch.Tensor:
        """Sequence-level LLR divided by speech-frame count, research-only."""
        return (self.log_likelihood(recording, speaker_models)
                - self.log_likelihood(recording)) / len(recording)

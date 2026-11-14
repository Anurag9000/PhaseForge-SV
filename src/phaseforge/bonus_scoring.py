"""Exact batched diagonal-GMM trial LLR scoring for the performance bonus.

All scoring remains phase-derived, per-trial and PyTorch-native. The
reference method is used for unsupported covariance/smoothing/top-k modes.
No group-dependent or hidden-label adjustments are introduced.

The optimized path groups by TEST RECORDING, reuses the UBM frame log
likelihood once, and scores several claimed speakers in a single GEMM.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import torch

from .gmm import TorchGMM
from .scoring import ScoreConfig, score_trials


@torch.no_grad()
def _stacked_diag_log_probs(
    frames: torch.Tensor, speakers: list[TorchGMM],
) -> torch.Tensor:
    """Return [num_frames,num_speakers] log p(X|speaker) without KxNxD tensors."""
    if not speakers:
        return frames.new_empty((len(frames), 0))
    d = frames.shape[1]
    mixture_count = len(speakers[0].weights)
    for model in speakers:
        if (model.config.covariance != "diag" or model.mu.shape != (mixture_count, d)
                or model.mu.device != frames.device):
            raise ValueError("Incompatible batched diag speaker model")
    means = torch.cat([model.mu for model in speakers], dim=0)
    covariance = torch.cat([model.cov for model in speakers], dim=0)
    prior = torch.cat([model.weights for model in speakers], dim=0)
    precision = covariance.reciprocal()
    bias = -.5 * (
        (means.square() * precision + covariance.log()).sum(-1)
        + d * 1.8378770664093453
    ) + prior.clamp_min(1e-12).log()
    likelihoods = frames @ (means * precision).T
    likelihoods -= .5 * (frames.square() @ precision.T)
    likelihoods += bias[None]
    return likelihoods.reshape(len(frames), len(speakers), mixture_count).logsumexp(-1)


@torch.no_grad()
def accelerated_score_trials(
    ubm: TorchGMM, features: dict[str, torch.Tensor], trials: list[dict],
    speakers: dict[str, TorchGMM], config: ScoreConfig = ScoreConfig(),
    speaker_chunk: int = 16, frame_chunk: int = 1024,
    strict_acceleration: bool = False,
) -> torch.Tensor:
    """Return in original CSV order; cache each unique (test_file, claimed_id).

    The output is equivalent (within floating-point matmul roundoff) to the
    reference per-trial LLR, and retains proper genuine/impostor independence.
    Speaker and frame chunk limits bound scoring memory use.
    """
    config.validate()
    if speaker_chunk < 1 or frame_chunk < 1:
        raise ValueError("Chunk limits must be positive")
    compatible = (
        config.smoothing == 1 and config.top_components is None
        and ubm.config.covariance == "diag"
        and all(model.config.covariance == "diag" for model in speakers.values())
    )
    if not compatible:
        if strict_acceleration:
            raise ValueError("Batched scoring supports diagonal, unsmoothed full-mixture LLR only")
        return score_trials(ubm, features, trials, speakers, replace(config, implementation="reference"))
    if not trials:
        return torch.empty((0,), dtype=torch.float64)
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, trial in enumerate(trials):
        speaker, test = trial["claimed_id"], trial["test_file"]
        if speaker not in speakers or test not in features:
            raise ValueError(f"Unknown speaker/test in trial {i}: {speaker} {test}")
        grouped[test][speaker].append(i)
    output = torch.empty(len(trials), dtype=torch.float64)
    for test_file, claims in grouped.items():
        x = features[test_file]
        if x.ndim != 2 or not len(x):
            raise ValueError(f"Empty or invalid feature frames: {test_file}")
        names = list(claims)
        total = torch.zeros((len(names),), dtype=torch.float64, device=x.device)
        for block in x.split(frame_chunk):
            denominator = ubm.log_prob(block)
            for first in range(0, len(names), speaker_chunk):
                chunk_names = names[first:first + speaker_chunk]
                logps = _stacked_diag_log_probs(
                    block, [speakers[name] for name in chunk_names]
                )
                total[first:first + len(chunk_names)] += (
                    logps.double() - denominator.double()[:, None]
                ).sum(0)
        total /= len(x)
        for index, name in enumerate(names):
            for trial_index in claims[name]:
                output[trial_index] = total[index].cpu()
    if not torch.isfinite(output).all():
        raise FloatingPointError("Batched speaker LLR produced nonfinite scores")
    return output

"""Opt-in batched phase frontend for the end-to-end efficiency bonus.

Batches same-length waveforms into ONE PyTorch STFT and matrix projection.
Supports exact original group/temporal/cosphase/direct configurations;
all other modes keep the untouched PhaseExtractor reference path.
No magnitude-based MFCC or trained external model is introduced.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F

from .features import FeatureConfig, PhaseExtractor, mel_bands


SUPPORTED_BATCH_MODES = ("group", "temporal", "cosphase", "direct")


def can_batch(cfg: FeatureConfig) -> bool:
    return cfg.mode in SUPPORTED_BATCH_MODES and not cfg.trim_longest_region


@torch.no_grad()
def batched_phase_features(
    waveforms: list[torch.Tensor], cfg: FeatureConfig,
) -> list[torch.Tensor]:
    """Vectorize waveform-to-phase features for equal-length PCM arrays.

    The intermediate magnitude is used only for the original reference
    reliability mask and speech activity, never a model magnitude feature.
    """
    cfg.validate()
    if not waveforms:
        return []
    if not can_batch(cfg):
        return [PhaseExtractor(cfg)(x) for x in waveforms]
    if len({len(x) for x in waveforms}) != 1:
        return [PhaseExtractor(cfg)(x) for x in waveforms]
    y = torch.stack([x.float().flatten() for x in waveforms])
    if cfg.peak_normalize:
        y = y / y.abs().amax(1, keepdim=True).clamp_min(1e-8)
    if y.shape[1] < cfg.n_fft:
        y = F.pad(y, (0, cfg.n_fft - y.shape[1]))
    frames = y.unfold(1, cfg.n_fft, cfg.hop)
    window = torch.hann_window(cfg.win, device=y.device)
    X = torch.stft(
        y, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
        window=window, center=False, return_complex=True,
    )
    magnitude = X.abs()
    maximum = magnitude.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    unit = X / magnitude.clamp_min(maximum * 1e-7)
    valid = (magnitude > maximum * cfg.reliability).float()
    W = mel_bands(X.shape[1], cfg.bands, cfg.sample_rate, y.device)
    if cfg.mode == "group":
        crossed = unit[:, 1:] * unit[:, :-1].conj()
        crossed = crossed / crossed.abs().clamp_min(1e-8)
        phase = F.pad(crossed, (0, 0, 1, 0))
        mask = F.pad(valid[:, 1:] * valid[:, :-1], (0, 0, 1, 0))
    elif cfg.mode == "temporal":
        advance = 2 * math.pi * cfg.hop / cfg.n_fft * torch.arange(
            X.shape[1], device=y.device
        )
        crossed = (
            unit[:, :, 1:] * unit[:, :, :-1].conj()
            * torch.exp(-1j * advance[None, :, None])
        )
        crossed = crossed / crossed.abs().clamp_min(1e-8)
        phase = F.pad(crossed, (1, 0))
        mask = F.pad(valid[:, :, 1:] * valid[:, :, :-1], (1, 0))
    else:
        phase, mask = unit, valid
    projected_mask = mask.permute(0, 2, 1)
    if cfg.masked_mean:
        denom = (projected_mask @ W).clamp_min(1e-4)
    else:
        denom = torch.ones(
            y.shape[0], phase.shape[2], cfg.bands,
            device=y.device,
        )
    cos = ((phase.real * mask).permute(0, 2, 1) @ W) / denom
    sin = ((phase.imag * mask).permute(0, 2, 1) @ W) / denom
    features = torch.cat((cos, sin), dim=-1).float()

    if cfg.vad_db is not None:
        energies = frames.square().mean(-1).sqrt()
        threshold = energies.max(dim=1, keepdim=True).values.clamp_min(
            1e-8
        ) * (10 ** (cfg.vad_db / 20))
        active = energies >= threshold
        for i in range(len(waveforms)):
            required = min(10, active.shape[1])
            if int(active[i].sum()) < required:
                top = energies[i].topk(required).indices
                active[i].zero_()
                active[i, top] = True
    else:
        active = torch.ones(
            features.shape[:2], dtype=torch.bool, device=y.device
        )
    result = []
    for index in range(len(waveforms)):
        single = features[index, active[index]]
        if not len(single):
            single = features.new_zeros((1, features.shape[-1]))
        if cfg.delta:
            derivative = torch.zeros_like(single)
            if len(single) >= 3:
                derivative[1:-1] = (single[2:] - single[:-2]) / 2
            single = torch.cat((single, derivative), -1)
        if cfg.normalization != "none":
            mean = single.mean(0, keepdim=True)
            std = single.std(0, unbiased=False, keepdim=True).clamp_min(
                cfg.std_floor
            )
            alpha = cfg.mean_strength if cfg.normalization == "partial" else 1.
            beta = cfg.std_strength if cfg.normalization == "partial" else 1.
            single = (single - alpha * mean) / std.pow(beta)
        if not torch.isfinite(single).all():
            raise FloatingPointError("Nonfinite batched phase frontend")
        result.append(single.contiguous())
    return result


@torch.no_grad()
def extract_grouped_features(
    data, file_ids: list[str], cfg: FeatureConfig,
    device: torch.device, batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Bounded wave loader buffering and exact-length STFT batch processing."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ids = list(dict.fromkeys(file_ids))
    if batch_size == 1 or not can_batch(cfg):
        extractor = PhaseExtractor(cfg)
        return {
            file: extractor(data.load(file).to(device)).cpu()
            for file in ids
        }
    result = {}
    groups: dict[int, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    buffered = 0

    def flush(length: int):
        nonlocal buffered
        candidates = groups.pop(length, [])
        buffered -= len(candidates)
        if not candidates:
            return
        names = [name for name, _ in candidates]
        values = batched_phase_features(
            [wave for _, wave in candidates], cfg
        )
        for name, value in zip(names, values):
            result[name] = value.cpu()

    for file in ids:
        waveform = data.load(file).to(device)
        length = len(waveform)
        groups[length].append((file, waveform))
        buffered += 1
        if len(groups[length]) >= batch_size:
            flush(length)
        elif buffered >= batch_size * 2:
            flush(max(groups, key=lambda n: len(groups[n])))
    for length in list(groups):
        flush(length)
    if set(result) != set(ids):
        raise RuntimeError("Batched phase frontend lost or duplicated files")
    return result

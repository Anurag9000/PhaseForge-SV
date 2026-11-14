"""Phase-derived feature extraction. Magnitude is used ONLY for VAD/reliability.

The group-delay and modified variants here are phase-only circular derivatives,
NOT conventional magnitude-normalized modified group delay (which would violate
the competition's strict magnitude-input requirement).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "group"
    sample_rate: int = 16000
    win: int = 400
    hop: int = 160
    n_fft: int = 512
    bands: int = 24
    vad_db: float | None = -30.0
    reliability: float = 1e-4
    masked_mean: bool = True
    normalization: str = "utterance"
    mean_strength: float = 1.0
    std_strength: float = 1.0
    std_floor: float = 0.06
    delta: bool = False
    lpc_order: int = 12
    power: float = 0.7
    multi_modes: tuple[str, ...] = ("group", "temporal")
    multi_windows: tuple[int, ...] = (400, 640)
    peak_normalize: bool = False
    trim_longest_region: bool = False
    speech_region_db: float = -25.0

    def validate(self) -> None:
        if self.win > self.n_fft or self.win < 32 or self.hop < 1 or self.bands < 2:
            raise ValueError("Invalid window, hop, FFT or band count")
        if self.normalization not in ("utterance", "partial", "none"):
            raise ValueError("Unsupported normalization")
        if not 0 <= self.mean_strength <= 1 or not 0 <= self.std_strength <= 1:
            raise ValueError("Normalization strengths must be in [0,1]")
        if self.mode not in (
            "group", "temporal", "ifd", "cosphase", "direct", "group_cepstrum",
            "modified_group_delay", "lp_residual", "all_pole", "multi",
            "unwrap_phase_cepstrum", "ifcc_surrogate", "cnpcc_surrogate",
            "apcc_surrogate", "phase_flux"
        ):
            raise ValueError(f"Unknown phase mode: {self.mode}")


def mel_bands(freqs: int, count: int, sr: int, device: torch.device) -> torch.Tensor:
    f = torch.linspace(0, sr / 2, freqs, device=device)
    to_mel = lambda v: 2595 * torch.log10(1 + v / 700)
    points = torch.linspace(to_mel(f[0]), to_mel(f[-1]), count + 2, device=device)
    points = 700 * (torch.pow(torch.tensor(10., device=device), points / 2595) - 1)
    cols = []
    for i in range(count):
        lo, mid, hi = points[i:i + 3]
        w = torch.minimum((f - lo) / (mid - lo + 1e-9), (hi - f) / (hi - mid + 1e-9))
        cols.append(w.clamp_min(0))
    W = torch.stack(cols, -1)
    return W / W.sum(0, keepdim=True).clamp_min(1e-10)


def _dct(X: torch.Tensor) -> torch.Tensor:
    d = X.shape[-1]
    n = torch.arange(d, device=X.device, dtype=X.dtype)
    k = n[:, None]
    D = torch.cos(math.pi * (n[None, :] + .5) * k / d)
    D[0] *= (1 / d) ** .5
    D[1:] *= (2 / d) ** .5
    return X @ D.T


def _lp_coefficients(frames: torch.Tensor, order: int) -> torch.Tensor:
    """Regularized batched Yule-Walker linear prediction (no external LPC library)."""
    T, width = frames.shape
    if order >= width // 2:
        raise ValueError("LPC order too large")
    win = torch.hann_window(width, device=frames.device, dtype=frames.dtype)
    x = frames * win
    r = torch.stack([
        (x[:, lag:] * x[:, :width - lag]).sum(-1) / width
        for lag in range(order + 1)
    ], -1)
    indices = (torch.arange(order, device=frames.device)[:, None]
               - torch.arange(order, device=frames.device)[None, :]).abs()
    R = r[:, indices]
    eye = torch.eye(order, device=frames.device, dtype=frames.dtype)
    regularizer = (r[:, 0].clamp_min(1e-5) * .01).view(T, 1, 1)
    return torch.linalg.solve(R + regularizer * eye[None], r[:, 1:].unsqueeze(-1)).squeeze(-1)


def _residual(y: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """One stationary LP filter fitted on the utterance, output subsequently phase transformed."""
    result = y.clone()
    for lag in range(1, coeff.numel() + 1):
        result[lag:] -= coeff[lag - 1] * y[:-lag]
    return result




def longest_speech_region(
    waveform: torch.Tensor, sample_rate: int, threshold_db: float = -25.0,
) -> torch.Tensor:
    """Energy-gated longest contiguous speech-like region.

    Unlike MATLAB detectSpeech this is an explicitly named approximation
    using non-learned waveform RMS, not a drop-in replica.
    """
    frame = max(64, int(sample_rate * .025))
    hop = max(32, int(sample_rate * .010))
    if len(waveform) < frame:
        return waveform
    windows = waveform.unfold(0, frame, hop)
    rms = windows.square().mean(-1).sqrt()
    limit = rms.max().clamp_min(1e-8) * 10 ** (threshold_db / 20)
    flags = (rms > limit).tolist()
    longest, start, length = (0, 0), 0, 0
    for i, flag in enumerate(flags + [False]):
        if flag:
            if length == 0:
                start = i
            length += 1
        else:
            if length > longest[1]:
                longest = (start, length)
            length = 0
    if longest[1] < 2:
        return waveform
    begin = longest[0] * hop
    end = min(len(waveform), (longest[0] + longest[1] - 1) * hop + frame)
    return waveform[begin:end]


class PhaseExtractor:
    def __init__(self, config: FeatureConfig):
        config.validate()
        self.config = config

    def _single(self, y: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
        if len(y) < cfg.n_fft:
            y = F.pad(y, (0, cfg.n_fft - len(y)))
        frames = y.unfold(0, cfg.n_fft, cfg.hop)
        if cfg.mode == "lp_residual":
            coeff = _lp_coefficients(frames, cfg.lpc_order).mean(0)
            transformed = _residual(y, coeff)
        else:
            transformed = y
        window = torch.hann_window(cfg.win, device=y.device)
        X = torch.stft(
            transformed, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
            window=window, center=False, return_complex=True,
        )  # frequency x time
        mag = X.abs()
        maximum = mag.amax().clamp_min(1e-8)
        unit = X / mag.clamp_min(maximum * 1e-7)
        valid = (mag > maximum * cfg.reliability).float()
        W = mel_bands(X.shape[0], cfg.bands, cfg.sample_rate, y.device)

        if cfg.mode in (
            "group", "group_cepstrum", "modified_group_delay", "lp_residual",
            "cnpcc_surrogate",
        ):
            cross = unit[1:] * unit[:-1].conj()
            phase = cross / cross.abs().clamp_min(1e-8)
            mask = valid[1:] * valid[:-1]
            phase = F.pad(phase, (0, 0, 1, 0))
            mask = F.pad(mask, (0, 0, 1, 0))
        elif cfg.mode in ("temporal", "ifd", "ifcc_surrogate", "phase_flux"):
            advance = (2 * math.pi * cfg.hop / cfg.n_fft) * torch.arange(
                X.shape[0], device=y.device
            )
            ph = unit[:, 1:] * unit[:, :-1].conj() * torch.exp(-1j * advance[:, None])
            ph = ph / ph.abs().clamp_min(1e-8)
            phase = F.pad(ph, (1, 0))
            mask = F.pad(valid[:, 1:] * valid[:, :-1], (1, 0))
        elif cfg.mode in ("all_pole", "apcc_surrogate"):
            coefficients = _lp_coefficients(frames, cfg.lpc_order)
            k = torch.arange(1, cfg.lpc_order + 1, device=y.device)
            omega = 2 * math.pi / cfg.n_fft * torch.arange(X.shape[0], device=y.device)
            prediction_polynomial = 1 - coefficients.to(torch.complex64) @ torch.exp(
                -1j * k[:, None] * omega[None, :]
            )
            cross = prediction_polynomial[:, :-1] * prediction_polynomial[:, 1:].conj()
            ph = cross / cross.abs().clamp_min(1e-8)
            phase = F.pad(ph.T, (0, 0, 1, 0))
            mask = valid
        elif cfg.mode in ("cosphase", "direct", "unwrap_phase_cepstrum"):
            phase = unit
            mask = valid
        else:
            raise ValueError(cfg.mode)

        weights = mask.T @ W if cfg.masked_mean else torch.ones(
            mask.shape[1], cfg.bands, device=y.device
        )
        denom = weights.clamp_min(1e-4)
        if cfg.mode in (
            "group_cepstrum", "modified_group_delay", "ifd",
            "cnpcc_surrogate", "ifcc_surrogate", "apcc_surrogate",
            "unwrap_phase_cepstrum",
        ):
            angles = torch.angle(phase)
            if cfg.mode == "unwrap_phase_cepstrum":
                # Unwrap along frequency using wrapped adjacent-bin differences.
                wrapped = angles[1:] - angles[:-1]
                delta = torch.atan2(wrapped.sin(), wrapped.cos())
                angles = torch.cat(
                    (angles[:1], angles[:1] + delta.cumsum(0)), dim=0
                )
                angles = angles - angles.mean(0, keepdim=True)
            if cfg.mode in ("modified_group_delay", "cnpcc_surrogate"):
                # CNPCC-like circular-power group phase proxy; NOT a faithful
                # magnitude-normalized published CNPCC implementation.
                angles = angles.clamp(-math.pi / 2, math.pi / 2)
                angles = angles.sign() * angles.abs().pow(cfg.power)
            if cfg.mode == "modified_group_delay":
                # Phase-only robust signed-power group derivative; NOT classic MGD.
                pass
            bands = (angles * mask).T @ W / denom
            features = bands if cfg.mode == "ifd" else _dct(bands)
        else:
            co = (phase.real * mask).T @ W / denom
            si = (phase.imag * mask).T @ W / denom
            features = torch.cat((co, si), -1).float()

        if cfg.vad_db is not None:
            energies = frames.square().mean(-1).sqrt()
            active = energies >= energies.max().clamp_min(1e-8) * (10 ** (cfg.vad_db / 20))
            if int(active.sum()) < min(10, len(active)):
                best = energies.topk(min(10, len(energies))).indices
                active = torch.zeros_like(active)
                active[best] = True
            features = features[active]
        if not len(features):
            features = torch.zeros(1, features.shape[-1], device=y.device)
        return features.float()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        y = waveform.float().flatten()
        if cfg.peak_normalize:
            y = y / y.abs().max().clamp_min(1e-8)
        if cfg.trim_longest_region:
            y = longest_speech_region(y, cfg.sample_rate, cfg.speech_region_db)
        if cfg.mode == "multi":
            base = replace(cfg, mode=cfg.multi_modes[0], win=cfg.multi_windows[0],
                           n_fft=max(512, 2 ** (cfg.multi_windows[0] - 1).bit_length()),
                           vad_db=None, normalization="none", delta=False)
            length = self._single(y, base).shape[0]
            parts = []
            for mode in cfg.multi_modes:
                for win in cfg.multi_windows:
                    child = replace(cfg, mode=mode, win=win,
                                    n_fft=max(512, 2 ** (win - 1).bit_length()),
                                    vad_db=None, normalization="none", delta=False)
                    part = self._single(y, child)
                    if part.shape[0] != length:
                        part = F.interpolate(part.T[None], size=length,
                                             mode="linear", align_corners=False)[0].T
                    parts.append(part)
            features = torch.cat(parts, -1)
            if cfg.vad_db is not None:
                sample_windows = y
                if len(sample_windows) < base.n_fft:
                    sample_windows = F.pad(sample_windows, (0, base.n_fft - len(sample_windows)))
                frames = sample_windows.unfold(0, base.n_fft, cfg.hop)
                energy = frames.square().mean(-1).sqrt()
                use = energy >= energy.max().clamp_min(1e-8) * 10 ** (cfg.vad_db / 20)
                if use.any():
                    features = features[use]
        else:
            features = self._single(y, cfg)
        if cfg.delta:
            derivative = torch.zeros_like(features)
            if len(features) >= 3:
                derivative[1:-1] = (features[2:] - features[:-2]) / 2
            features = torch.cat([features, derivative], -1)
        if cfg.normalization != "none":
            mean = features.mean(0, keepdim=True)
            std = features.std(0, unbiased=False, keepdim=True).clamp_min(cfg.std_floor)
            alpha = cfg.mean_strength if cfg.normalization == "partial" else 1.
            beta = cfg.std_strength if cfg.normalization == "partial" else 1.
            features = (features - alpha * mean) / std.pow(beta)
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite phase-derived features")
        return features.contiguous()

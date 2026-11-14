"""Original Google-only log-mel/MFCC reference extraction.

Magnitude-derived mel and MFCC ARE NOT legal competition model features.
Every entrypoint in this module explicitly accepts GoogleResearchData only.
Used to compare the actual MathWorks MFCC baseline and original GSC keyword
models to separate phase-only alternatives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import GoogleResearchData
from .features import _dct, longest_speech_region, mel_bands


@dataclass(frozen=True)
class ReferenceFeature:
    kind: str = "mfcc"  # mfcc | logmel
    win: int = 640
    hop: int = 160
    n_fft: int = 1024
    bands: int = 40
    coefficients: int = 13
    peak_normalize: bool = True
    trim_longest_region: bool = True
    speech_db: float = -25.

    def validate(self):
        if self.kind not in ("mfcc", "logmel"):
            raise ValueError("Reference must be MFCC or logmel")
        if self.win < 32 or self.n_fft < self.win or self.hop < 1:
            raise ValueError("Invalid reference STFT")
        if not 1 <= self.coefficients <= self.bands:
            raise ValueError("Invalid cepstral coefficient count")


class GoogleOnlyReferenceExtractor:
    def __init__(self, config: ReferenceFeature):
        config.validate()
        self.config = config

    def __call__(self, waveform: torch.Tensor):
        c = self.config
        y = waveform.flatten().float()
        if c.peak_normalize:
            y = y / y.abs().max().clamp_min(1e-8)
        if c.trim_longest_region:
            y = longest_speech_region(y, 16000, c.speech_db)
        if len(y) < c.n_fft:
            y = torch.nn.functional.pad(y, (0, c.n_fft - len(y)))
        X = torch.stft(
            y, n_fft=c.n_fft, hop_length=c.hop, win_length=c.win,
            center=False, window=torch.hann_window(c.win, device=y.device),
            return_complex=True,
        )
        mel = X.abs().square().T @ mel_bands(
            X.shape[0], c.bands, 16000, y.device
        )
        features = mel.clamp_min(1e-8).log()
        if c.kind == "mfcc":
            features = _dct(features)[:, :c.coefficients]
        return features


def extract_google_reference(
    data: GoogleResearchData, file_ids: list[str], config: ReferenceFeature,
    device: torch.device, cache_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """No fallback: raise if caller supplies KaggleData."""
    if not isinstance(data, GoogleResearchData) or data.source != "google_research":
        raise PermissionError("Magnitude/mel/MFCC references are Google research ONLY")
    result = {}
    frontend = GoogleOnlyReferenceExtractor(config)
    with torch.no_grad():
        for file_id in sorted(set(file_ids)):
            result[file_id] = frontend(data.load(file_id).to(device)).cpu()
    return result

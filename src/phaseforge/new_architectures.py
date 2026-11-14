"""Phase-only speaker-embedding backbones.

These models consume PHASE FEATURES [B,T,D], never magnitude/MFCCs.
They are independently implemented *adaptations* of published model families;
BCNN/KWT/ECAPA variants are not claims of exact paper reproduction.
Masks mark padded frames; output vectors are L2-normalized.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Architecture:
    name: str = "xvector"
    channels: int = 64
    embedding: int = 64
    depth: int = 3
    heads: int = 4
    dropout: float = 0.1
    res2_scale: int = 4

    def validate(self):
        if self.channels < 8 or self.embedding < 2 or self.depth < 1:
            raise ValueError("Invalid network capacity")
        if self.name not in MODEL_NAMES:
            raise ValueError(f"Unknown neural architecture {self.name}")
        if self.name in ("kwt", "conformer") and self.channels % self.heads:
            raise ValueError("Transformer channels must be divisible by heads")


def masked_stats(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B,C,T] -> weighted mean and standard deviation."""
    weights = valid[:, None, :].to(x.dtype)
    denominator = weights.sum(-1).clamp_min(1)
    mean = (x * weights).sum(-1) / denominator
    variance = ((x - mean[:, :, None]).square() * weights).sum(-1) / denominator
    return torch.cat((mean, variance.clamp_min(1e-6).sqrt()), -1)


class AttentiveStatistics(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv1d(channels * 3, channels, 1),
                                       nn.Tanh(), nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        statistics = masked_stats(x, valid)
        c = x.shape[1]
        context = torch.cat((x, statistics[:, :c, None].expand_as(x),
                             statistics[:, c:, None].expand_as(x)), dim=1)
        logits = self.attention(context).masked_fill(~valid[:, None, :], -1e4)
        weights = logits.softmax(-1)
        mean = (weights * x).sum(-1)
        std = ((x - mean[:, :, None]).square() * weights).sum(-1).clamp_min(1e-6).sqrt()
        return torch.cat((mean, std), -1)


def conv_block(ci: int, co: int, kernel: int = 3, dilation: int = 1) -> nn.Module:
    return nn.Sequential(nn.Conv1d(ci, co, kernel, dilation=dilation,
                                    padding=dilation * (kernel // 2), bias=False),
                         nn.BatchNorm1d(co), nn.GELU())


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                    nn.Conv1d(channels, max(4, channels // 8), 1),
                                    nn.ReLU(), nn.Conv1d(max(4, channels // 8), channels, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        return x * self.layers(x)


class Residual(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, se: bool = False):
        super().__init__()
        self.body = nn.Sequential(conv_block(channels, channels, 3, dilation),
                                  conv_block(channels, channels, 3),
                                  SqueezeExcite(channels) if se else nn.Identity())

    def forward(self, x):
        return F.gelu(x + self.body(x))


class BraidedBead(nn.Module):
    """Multi-scale cross-fed convolutional bead inspired by BCNN's braided design.

    Both branches receive original input, exchange projected features, and
    residual-fuse. This is a phase-1D adaptation, NOT the original mel-2D BCNN.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.left = conv_block(channels, channels, 3)
        self.right = conv_block(channels, channels, 5)
        self.cross_left = conv_block(channels * 2, channels, 3)
        self.cross_right = conv_block(channels * 2, channels, 5)
        self.fuse = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, x):
        a, b = self.left(x), self.right(x)
        cross = torch.cat((a, b), dim=1)
        a = self.cross_left(cross)
        b = self.cross_right(cross)
        return F.gelu(x + self.fuse(torch.cat((a, b), dim=1)))


class BraidedBead2D(nn.Module):
    """BCNN-style 2D three-source braided bead with 3x3/5x5 and 1x1 joins.

    Phase time-frequency matrix replaces the paper's mel-spectrogram image.
    This preserves its characteristic branch/concat topology without claiming
    the same trained filter counts or original magnitude input.
    """
    def __init__(self, channels: int):
        super().__init__()
        def pair(ci, kernel):
            return nn.Sequential(
                nn.Conv2d(ci, channels, kernel, padding=kernel//2, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(),
                nn.Conv2d(channels, channels, 1), nn.ReLU()
            )
        self.branch3 = pair(channels, 3)
        self.branch5 = pair(channels, 5)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.branchpool = pair(channels, 1)
        self.cross3 = pair(2 * channels, 3)
        self.cross5 = pair(2 * channels, 5)
        self.crosspool = pair(2 * channels, 1)
        self.merge = nn.Sequential(nn.Conv2d(2 * channels, channels, 1),
                                   nn.BatchNorm2d(channels), nn.ReLU())

    def forward(self, x):
        a = self.branch3(x)
        b = self.branch5(x)
        c = self.branchpool(self.pool(x))
        ab = torch.cat((a, b), dim=1)
        d = self.cross3(ab)
        e = self.cross5(ab)
        f = self.crosspool(self.pool(torch.cat((c, e), dim=1)))
        return F.relu(x + self.merge(torch.cat((d, f), dim=1)))


class SERes2Block(nn.Module):
    """ECAPA-style channel-split residual/dilated convolution + SE."""
    def __init__(self, channels: int, dilation: int = 2, scale: int = 4):
        super().__init__()
        if channels % scale:
            raise ValueError("res2 channels must divide res2_scale")
        width = channels // scale
        self.pre = conv_block(channels, channels, 1)
        self.parts = nn.ModuleList(
            [conv_block(width, width, 3, dilation) for _ in range(scale - 1)]
        )
        self.post = conv_block(channels, channels, 1)
        self.se = SqueezeExcite(channels)

    def forward(self, x):
        chunks = self.pre(x).chunk(len(self.parts) + 1, 1)
        y = [chunks[0]]
        for i, layer in enumerate(self.parts):
            y.append(layer(chunks[i + 1] + (y[-1] if i else 0)))
        return F.gelu(x + self.se(self.post(torch.cat(y, 1))))


class DepthwiseTemporal(nn.Module):
    """MobileNet/DS-CNN depthwise and pointwise phase-feature convolutions."""
    def __init__(self, channels: int, kernel: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2,
                      groups=channels, bias=False), nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 1), nn.BatchNorm1d(channels), nn.GELU(),
        )

    def forward(self, x):
        return x + self.layers(x)


class TemporalConformer(nn.Module):
    """Macaron feed-forward, MHSA, depthwise convolution and residual layer."""
    def __init__(self, channels: int, heads: int, dropout: float):
        super().__init__()
        def ff():
            return nn.Sequential(nn.LayerNorm(channels),
                                 nn.Linear(channels, channels * 4), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(channels * 4, channels))
        self.ff1, self.ff2 = ff(), ff()
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1), nn.GLU(dim=1),
            nn.Conv1d(channels, channels, 5, padding=2, groups=channels),
            nn.BatchNorm1d(channels), nn.SiLU(), nn.Conv1d(channels, channels, 1)
        )
        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x, padding_mask):
        x = x + self.ff1(x) * .5
        y = self.norm_attn(x)
        y = self.attn(y, y, y, key_padding_mask=padding_mask, need_weights=False)[0]
        x = x + y
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.final_norm(x + .5 * self.ff2(x))


class SelectiveGatedState(nn.Module):
    """Small recurrent selective state-space ablation; NOT full Mamba."""
    def __init__(self, channels: int):
        super().__init__()
        self.in_proj = nn.Linear(channels, channels * 3)
        self.gate = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, valid):
        prev = torch.zeros_like(x[:, 0])
        outputs = []
        for t in range(x.shape[1]):
            a, b, c = self.in_proj(x[:, t]).chunk(3, dim=-1)
            forget = torch.sigmoid(a)
            state = forget * prev + (1 - forget) * torch.tanh(b)
            prev = torch.where(valid[:, t, None], state, prev)
            outputs.append(x[:, t] + torch.sigmoid(self.gate(x[:, t])) *
                           torch.tanh(prev + c))
        return self.norm(torch.stack(outputs, dim=1))


MODEL_NAMES = (
    "bcnn", "xvector", "ecapa", "resnet", "se_resnet",
    "ds_cnn", "tc_resnet", "crnn", "kwt", "conformer", "ssm_lite",
    "cnn", "res2_ecapa", "bcnn_2d",
)


class PhaseSpeakerEncoder(nn.Module):
    """One consistent embedding API across all 13 phase-only backbones."""
    def __init__(self, input_dim: int, config: Architecture):
        super().__init__()
        config.validate()
        self.cfg = config
        self.stem = conv_block(input_dim, config.channels, 5)
        c = config.channels
        self.two_dim_stem = (
            nn.Sequential(
                nn.Conv2d(1, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c), nn.ReLU()
            ) if config.name == "bcnn_2d" else None
        )
        name = config.name
        self.blocks = nn.ModuleList()
        self.sequence = None
        if name == "bcnn_2d":
            self.blocks = nn.ModuleList(BraidedBead2D(c) for _ in range(config.depth))
        elif name == "bcnn":
            self.blocks = nn.ModuleList(BraidedBead(c) for _ in range(config.depth))
        elif name in ("ecapa", "res2_ecapa"):
            self.blocks = nn.ModuleList(
                SERes2Block(c, dilation=2+i, scale=config.res2_scale)
                for i in range(config.depth)
            )
        elif name in ("resnet", "se_resnet", "tc_resnet"):
            self.blocks = nn.ModuleList(Residual(c, dilation=(2**i if name == "tc_resnet" else 1),
                                                 se=name == "se_resnet")
                                        for i in range(config.depth))
        elif name == "ds_cnn":
            self.blocks = nn.ModuleList(DepthwiseTemporal(c) for _ in range(config.depth))
        elif name in ("xvector", "cnn"):
            self.blocks = nn.ModuleList(conv_block(c, c, 3, 2**(i % 3))
                                        for i in range(config.depth))
        elif name == "crnn":
            self.sequence = nn.GRU(c, c // 2, num_layers=config.depth, bidirectional=True,
                                   batch_first=True)
        elif name == "kwt":
            layer = nn.TransformerEncoderLayer(
                c, config.heads, dim_feedforward=c*4, batch_first=True,
                dropout=config.dropout, activation="gelu", norm_first=True
            )
            self.sequence = nn.TransformerEncoder(layer, config.depth)
            self.pos_scale = nn.Parameter(torch.tensor(1.0))
        elif name == "conformer":
            self.blocks = nn.ModuleList(
                TemporalConformer(c, config.heads, config.dropout)
                for _ in range(config.depth)
            )
        elif name == "ssm_lite":
            self.sequence = SelectiveGatedState(c)
        if name in ("ecapa", "res2_ecapa"):
            self.aggregate = conv_block(c * config.depth, c, 1)
        else:
            self.aggregate = nn.Identity()
        self.pooling = AttentiveStatistics(c)
        self.projection = nn.Sequential(nn.BatchNorm1d(c * 2), nn.Linear(c * 2, config.embedding))

    def forward(self, frames: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 3 or valid.shape != frames.shape[:2]:
            raise ValueError("Encoder requires [batch,time,phase] and [batch,time] mask")
        if not valid.any(1).all():
            raise ValueError("Every utterance must have at least one valid frame")
        if self.cfg.name == "bcnn_2d":
            # [B,T,D] -> phase time-frequency image [B,1,D,T]
            z = self.two_dim_stem(frames.transpose(1, 2)[:, None])
            for bead in self.blocks:
                z = bead(z)
            hidden = z.mean(dim=2)
            pooled = self.pooling(hidden, valid)
            return F.normalize(self.projection(pooled), dim=-1)
        hidden = self.stem(frames.transpose(1, 2))
        branches = []
        if self.cfg.name in ("kwt", "conformer", "ssm_lite", "crnn"):
            seq = hidden.transpose(1, 2)
            if self.cfg.name == "kwt":
                t = torch.arange(seq.shape[1], device=seq.device, dtype=seq.dtype)
                freqs = torch.exp(-math.log(10000) * torch.arange(
                    0, seq.shape[-1], 2, device=seq.device, dtype=seq.dtype) / seq.shape[-1])
                pe = torch.zeros_like(seq)
                pe[:, :, 0::2] = torch.sin(t[:, None] * freqs)
                pe[:, :, 1::2] = torch.cos(t[:, None] * freqs[:pe[:, :, 1::2].shape[-1]])
                seq = self.sequence(seq + self.pos_scale * pe,
                                    src_key_padding_mask=~valid)
            elif self.cfg.name == "conformer":
                for block in self.blocks:
                    seq = block(seq, ~valid)
            elif self.cfg.name == "crnn":
                seq, _ = self.sequence(seq)
            else:
                seq = self.sequence(seq, valid)
            hidden = seq.transpose(1, 2)
        else:
            for block in self.blocks:
                hidden = block(hidden)
                branches.append(hidden)
            if self.cfg.name in ("ecapa", "res2_ecapa"):
                hidden = self.aggregate(torch.cat(branches, dim=1))
        pooled = self.pooling(hidden, valid)
        return F.normalize(self.projection(pooled), dim=-1)

"""Trainable speaker verification losses for phase-only encoders.

Every label comes from BACKGROUND speaker IDs. No Google-pretrained weights.
GE2E uses leave-one-out same-speaker centroids (Wan et al.); paired
verification uses labels derived from identity equality, not test labels.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    def __init__(self, embedding: int, speakers: int, margin: float = .2, scale: float = 20):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(speakers, embedding) * .03)
        self.margin = margin
        self.scale = scale

    def forward(self, e: torch.Tensor, labels: torch.Tensor):
        cos = F.linear(F.normalize(e), F.normalize(self.weight)).clamp(-.999999, .999999)
        target = torch.cos(torch.acos(cos) + self.margin)
        amended = cos.scatter(1, labels[:, None], target.gather(1, labels[:, None]))
        return F.cross_entropy(amended * self.scale, labels)


class GE2ELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(2.3))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def forward(self, embeddings: torch.Tensor, n: int, m: int):
        """[N*M,D] in speaker-major order with M>=2; exclude query from own centroid."""
        if m < 2 or n < 2 or embeddings.shape[0] != n * m:
            raise ValueError("GE2E requires N>=2 speakers with M>=2 takes each")
        x = embeddings.reshape(n, m, -1)
        normal = F.normalize(x, dim=-1)
        global_c = F.normalize(x.mean(1), dim=-1)
        own_c = F.normalize((x.sum(1)[:, None] - x) / (m - 1), dim=-1)
        logits = torch.einsum("nmd,sd->nms", normal, global_c)
        diag = (normal * own_c).sum(-1)
        diag_ids = torch.arange(n, device=embeddings.device)
        logits[diag_ids, :, diag_ids] = diag
        logits = self.log_scale.exp().clamp(max=100) * logits + self.bias
        targets = torch.arange(n, device=embeddings.device).repeat_interleave(m)
        return F.cross_entropy(logits.reshape(n * m, n), targets)


def supervised_contrastive(e: torch.Tensor, labels: torch.Tensor, temperature: float = .1):
    e = F.normalize(e, dim=-1)
    logits = e @ e.T / temperature
    eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
    positive = labels[:, None].eq(labels[None]) & ~eye
    if not positive.any(1).all():
        raise ValueError("SupCon needs two utterances for every speaker")
    logits = logits.masked_fill(eye, -1e4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positive).sum() / positive.sum()


def batch_hard_triplet(e: torch.Tensor, labels: torch.Tensor, margin: float = .2):
    distance = torch.cdist(F.normalize(e), F.normalize(e), p=2)
    eye = torch.eye(len(e), device=e.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None]) & ~eye
    different = ~labels[:, None].eq(labels[None])
    if not same.any(1).all() or not different.any(1).all():
        raise ValueError("Triplet requires repeated speaker labels and multiple speakers")
    hard_pos = distance.masked_fill(~same, -1e5).max(1).values
    hard_neg = distance.masked_fill(~different, 1e5).min(1).values
    return F.relu(hard_pos - hard_neg + margin).mean()


def prototypical_loss(e: torch.Tensor, n: int, m: int, scale: float = 12.):
    if n < 2 or m < 2:
        raise ValueError("Prototypical speaker loss requires n,m>=2")
    x = e.reshape(n, m, -1)
    references = F.normalize(x[:, :-1].mean(1), dim=-1)
    queries = F.normalize(x[:, -1], dim=-1)
    scores = scale * queries @ references.T
    return F.cross_entropy(scores, torch.arange(n, device=e.device))


def latent_predictive_loss(prediction: torch.Tensor, teacher: torch.Tensor):
    return (2 - 2 * (F.normalize(prediction, dim=-1) *
                     F.normalize(teacher.detach(), dim=-1)).sum(-1)).mean()


class PairVerifier(nn.Module):
    """Google end-to-end text-dependent *inspired* learned pairwise scorer."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                     nn.Linear(hidden // 2, 1))

    def forward(self, enrol: torch.Tensor, query: torch.Tensor):
        return self.network(torch.cat((torch.abs(enrol - query), enrol * query), -1)).squeeze(-1)


def pairwise_bce(
    scorer: PairVerifier, embeddings: torch.Tensor, labels: torch.Tensor,
    max_pairs: int = 1024,
):
    """Balanced same/different speaker pairs in a background training batch."""
    same = labels[:, None].eq(labels[None])
    tri = torch.triu_indices(len(labels), len(labels), offset=1, device=labels.device)
    pos = tri[:, same[tri[0], tri[1]]]
    neg = tri[:, ~same[tri[0], tri[1]]]
    if not pos.shape[1] or not neg.shape[1]:
        raise ValueError("Pairwise training needs positives and impostors")
    count = min(pos.shape[1], neg.shape[1], max_pairs // 2)
    positive = pos[:, torch.randperm(pos.shape[1], device=labels.device)[:count]]
    negative = neg[:, torch.randperm(neg.shape[1], device=labels.device)[:count]]
    i = torch.cat((positive[0], negative[0]))
    j = torch.cat((positive[1], negative[1]))
    targets = torch.cat((embeddings.new_ones(count), embeddings.new_zeros(count)))
    return F.binary_cross_entropy_with_logits(scorer(embeddings[i], embeddings[j]), targets)

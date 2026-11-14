"""GMM supervector, actual i-vector total-variability EM and PLDA/SVM research backends.

All UBM/total-variability/PLDA/SVM parameters are learned exclusively on
the background population. Small-data fallbacks are explicit, never disguised
as externally pretrained speaker models. Inputs are phase-derived frames.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from .gmm import EMConfig, MAPConfig, TorchGMM


@torch.no_grad()
def bw_statistics(ubm: TorchGMM, frames: torch.Tensor):
    if ubm.config.covariance != "diag":
        raise ValueError("i-vector TV and supervectors require diagonal-covariance UBM")
    n, first, _, _ = ubm.sufficient_statistics(frames)
    centered = first - n[:, None] * ubm.mu
    return n, centered


@torch.no_grad()
def map_supervector(
    ubm: TorchGMM, frames: torch.Tensor, relevance: float = 8,
    occupancy_weighted: bool = True,
):
    """Posterior mean-supervector shift from global UBM, variance whitening."""
    if relevance <= 0:
        raise ValueError("MAP relevance must be positive")
    n, f = bw_statistics(ubm, frames)
    alpha = n / (n + relevance)
    shift = f / n.clamp_min(1e-8)[:, None] * alpha[:, None]
    shift = shift / ubm.cov.sqrt()
    if occupancy_weighted:
        shift = shift * (n / n.sum().clamp_min(1)).sqrt()[:, None]
    return shift.reshape(-1)


class TotalVariability:
    """Diagonal-GMM i-vector EM with Gaussian latent factors.

    E step: C_u=(I+sum_c N_uc T_c' Sigma_c^-1 T_c)^-1
            w_u=C_u sum_c T_c' Sigma_c^-1 F_uc
    M step: T_c=(sum_u F_uc w_u') (sum_u N_uc E[w_u w_u'])^-1.
    """
    def __init__(self, ubm: TorchGMM, dimension: int = 16, iterations: int = 8,
                 seed: int = 2026, reg: float = .001):
        if ubm.config.covariance != "diag":
            raise ValueError("Total-variability requires diagonal UBM")
        if dimension < 1 or iterations < 1 or reg <= 0:
            raise ValueError("Invalid i-vector configuration")
        self.ubm, self.dimension, self.iterations = ubm, dimension, iterations
        self.reg, self.seed = reg, seed
        self.T = None
        self.history = []

    @torch.no_grad()
    def posterior(self, n: torch.Tensor, f: torch.Tensor):
        K, D, R = self.T.shape
        inv_var = self.ubm.cov.reciprocal()
        weighted = self.T * inv_var[:, :, None]
        identity = torch.eye(R, device=n.device, dtype=n.dtype)
        precision = identity + torch.einsum("k,kdr,kds->rs", n, self.T, weighted)
        rhs = torch.einsum("kd,kdr->r", f, weighted)
        chol = torch.linalg.cholesky(precision + self.reg * identity)
        mean = torch.cholesky_solve(rhs[:, None], chol).squeeze(1)
        covariance = torch.cholesky_inverse(chol)
        return mean, covariance

    @torch.no_grad()
    def fit(self, utterances: dict[str, torch.Tensor]) -> "TotalVariability":
        if len(utterances) < 3:
            raise ValueError("Total variability needs at least 3 background recordings")
        data = [bw_statistics(self.ubm, x) for x in utterances.values()]
        K, D = self.ubm.mu.shape
        device, dtype = self.ubm.mu.device, self.ubm.mu.dtype
        g = torch.Generator(device=device).manual_seed(self.seed)
        self.T = torch.randn(K, D, self.dimension, device=device, dtype=dtype,
                             generator=g) * .025
        self.history = []
        for _ in range(self.iterations):
            cov_sums = torch.zeros(K, self.dimension, self.dimension, device=device, dtype=dtype)
            cross_sums = torch.zeros(K, D, self.dimension, device=device, dtype=dtype)
            average_norm = 0.
            for n, f in data:
                w, c = self.posterior(n, f)
                second = c + w[:, None] * w[None]
                cov_sums += n[:, None, None] * second[None]
                cross_sums += torch.einsum("kd,r->kdr", f, w)
                average_norm += float(w.square().mean())
            eye = torch.eye(self.dimension, device=device, dtype=dtype)
            # Each mixture component uses its own weighted second moments.
            self.T = torch.linalg.solve(
                cov_sums + self.reg * eye[None], cross_sums.transpose(1, 2)
            ).transpose(1, 2)
            self.history.append(average_norm / len(data))
            if not torch.isfinite(self.T).all():
                raise FloatingPointError("Non-finite total-variability matrix")
        return self

    @torch.no_grad()
    def transform(self, frames: torch.Tensor):
        if self.T is None:
            raise RuntimeError("Fit total variability on background before inference")
        n, f = bw_statistics(self.ubm, frames)
        mean, _ = self.posterior(n, f)
        return F.normalize(mean, dim=-1)


class TwoCovariancePLDA:
    """Regularized Gaussian between/within-speaker PLDA LLR backend.

    Independent embeddings: covariance C=B+W.
    Same speaker: joint covariance [[C,B],[B,C]].
    Scoring is log p(x,y|same) - log p(x)p(y), with train-only moments.
    """
    def __init__(self, shrinkage: float = .1):
        if not 0 <= shrinkage < 1:
            raise ValueError("PLDA shrinkage in [0,1)")
        self.shrinkage = shrinkage

    @torch.no_grad()
    def fit(self, embeddings: dict[str, torch.Tensor], labels: dict[str, str]):
        by_speaker = defaultdict(list)
        for file, vector in embeddings.items():
            by_speaker[labels[file]].append(vector)
        if len(by_speaker) < 2 or sum(len(x) > 1 for x in by_speaker.values()) < 2:
            raise ValueError("PLDA needs >=2 speakers and >=2 utterances per speaker")
        X = torch.stack([embeddings[file] for file in sorted(embeddings)])
        self.mean = X.mean(0)
        speaker_means = torch.stack([
            torch.stack(v).mean(0) for v in by_speaker.values()
        ])
        centered = X - self.mean
        total = centered.T @ centered / max(1, len(X) - 1)
        within = torch.zeros_like(total)
        count = 0
        for vectors in by_speaker.values():
            x = torch.stack(vectors)
            dev = x - x.mean(0)
            within += dev.T @ dev
            count += max(0, len(x) - 1)
        within /= max(1, count)
        between = (total - within).clone()
        eigen, eigenvec = torch.linalg.eigh((between + between.T) * .5)
        between = (eigenvec * eigen.clamp_min(1e-5)) @ eigenvec.T
        dim = X.shape[1]
        identity = torch.eye(dim, device=X.device, dtype=X.dtype)
        scale = total.diagonal().mean().clamp_min(1e-3)
        self.W = (1 - self.shrinkage) * within + self.shrinkage * scale * identity
        self.B = between + 1e-3 * scale * identity
        self.C = self.W + self.B
        joint = torch.cat([
            torch.cat((self.C, self.B), -1),
            torch.cat((self.B, self.C), -1)
        ], dim=0)
        self.joint_chol = torch.linalg.cholesky(joint + 1e-4 * identity.repeat(2, 2))
        self.single_chol = torch.linalg.cholesky(self.C + 1e-4 * identity)
        return self

    @torch.no_grad()
    def _gauss(self, x: torch.Tensor, chol: torch.Tensor):
        q = torch.linalg.solve_triangular(chol, x.unsqueeze(-1), upper=False)
        D = x.shape[-1]
        return -.5 * (q.square().sum((-2, -1)) +
                      2 * chol.diagonal().log().sum() + D * math.log(2 * math.pi))

    @torch.no_grad()
    def score(self, enrol: torch.Tensor, trial: torch.Tensor):
        if self.joint_chol is None:
            raise RuntimeError("Fit PLDA before scoring")
        x, y = enrol - self.mean, trial - self.mean
        joined = torch.cat((x, y))
        return (self._gauss(joined, self.joint_chol)
                - self._gauss(x, self.single_chol)
                - self._gauss(y, self.single_chol))


class PairwiseLinearSVM:
    """Hinge-trained pairwise SAME/DIFFERENT speaker linear verifier.

    Uses absolute difference and elementwise-product of speaker vectors.
    No held-out speaker labels used to learn the weights.
    """
    def __init__(self, epochs: int = 100, lr: float = .02, weight_decay: float = .01):
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.W = None

    @staticmethod
    def pair_features(a, b):
        return torch.cat(((a - b).abs(), a * b), -1)

    def fit(self, embeddings: dict[str, torch.Tensor], labels: dict[str, str],
            seed: int = 2026):
        files = sorted(embeddings)
        g = torch.Generator(device=embeddings[files[0]].device).manual_seed(seed)
        pairs_pos, pairs_neg = [], []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                if labels[files[i]] == labels[files[j]]:
                    pairs_pos.append((i, j))
                else:
                    pairs_neg.append((i, j))
        if not pairs_pos or not pairs_neg:
            raise ValueError("Pairwise SVM requires same and different speaker pairs")
        n = min(len(pairs_pos), len(pairs_neg), 5000)
        select = lambda pairs: [pairs[k] for k in
                               torch.randperm(len(pairs), generator=g)[:n].tolist()]
        selected = select(pairs_pos) + select(pairs_neg)
        X = torch.stack([
            self.pair_features(embeddings[files[i]], embeddings[files[j]])
            for i, j in selected
        ])
        y = torch.cat((X.new_ones(n), -X.new_ones(n)))
        w = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
        b = torch.zeros((), device=X.device, dtype=X.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([w, b], lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            margin = 1 - y * (X @ w + b)
            loss = margin.relu().mean() + self.weight_decay * w.square().sum() / 2
            loss.backward()
            optimizer.step()
        self.W, self.bias = w.detach(), b.detach()
        return self

    @torch.no_grad()
    def score(self, enrol, trial):
        return self.pair_features(enrol, trial) @ self.W + self.bias


def supervised_lda(embeddings: dict[str, torch.Tensor], labels: dict[str, str],
                   n_components: int, reg: float = .01):
    """Train-only Fisher discriminant projection (whiten within, rank between)."""
    by_speaker = defaultdict(list)
    for file, x in embeddings.items():
        by_speaker[labels[file]].append(x)
    X = torch.stack(list(embeddings.values()))
    mean = X.mean(0)
    within = torch.zeros(X.shape[1], X.shape[1], device=X.device)
    between = torch.zeros_like(within)
    for vectors in by_speaker.values():
        z = torch.stack(vectors)
        m = z.mean(0)
        centered = z - m
        within += centered.T @ centered
        delta = m - mean
        between += len(z) * torch.outer(delta, delta)
    eig, vec = torch.linalg.eigh(within + reg * torch.eye(X.shape[1], device=X.device))
    white = (vec * eig.clamp_min(1e-6).rsqrt()) @ vec.T
    spectrum, components = torch.linalg.eigh(white @ between @ white)
    W = white @ components[:, torch.argsort(spectrum, descending=True)[:n_components]]
    return mean, W

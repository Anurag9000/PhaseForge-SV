"""PyTorch Gaussian-mixture EM, covariance studies and MAP speaker adaptation."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

_LOG_2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class EMConfig:
    mixtures: int = 32
    covariance: str = "diag"  # diag, spherical, tied, full
    init: str = "kmeans"
    iterations: int = 35
    tolerance: float = 2e-4
    reg: float = .02
    min_occupancy: float = 2.0
    kmeans_steps: int = 12
    init_subsample: int = 25000
    dead_policy: str = "cyclic"  # cyclic or hard_examples
    init_variance_source: str = "full"  # full or initialization
    dead_weight_fraction: float = 0.05
    legacy_pre_mstep_likelihood: bool = False
    em_impl: str = "reference"  # reference | fused; fused reuses E-step LL
    chunk: int = 2048
    seed: int = 2026

    def validate(self) -> None:
        if self.mixtures < 1 or self.iterations < 1 or self.chunk < 1 or self.reg <= 0:
            raise ValueError("Invalid EM configuration")
        if self.covariance not in ("diag", "spherical", "tied", "full"):
            raise ValueError("Invalid covariance")
        if self.init not in ("random", "kmeans"):
            raise ValueError("Invalid initializer")
        if self.init_subsample < self.mixtures:
            raise ValueError("Initialization subset must contain at least K frames")
        if self.dead_policy not in ("cyclic", "hard_examples"):
            raise ValueError("Unknown dead-component replacement policy")
        if self.init_variance_source not in ("full", "initialization"):
            raise ValueError("Invalid initial covariance population")
        if not 0 < self.dead_weight_fraction <= 1:
            raise ValueError("Dead component weight fraction must be in (0,1]")
        if self.em_impl not in ("reference", "fused"):
            raise ValueError("EM implementation must be reference or fused")


@dataclass(frozen=True)
class MAPConfig:
    relevance: float = 8.0
    adapt_variances: bool = False
    adapt_weights: bool = False
    variance_floor: float = .02

    def validate(self) -> None:
        if self.relevance <= 0 or self.variance_floor <= 0:
            raise ValueError("Positive relevance and variance floor required")


class TorchGMM:
    """Fitted parameters are torch tensors; no sklearn/SciPy/NumPy.

    EM accumulates sufficient statistics by chunk, supporting large background pools.
    """

    def __init__(self, config: EMConfig):
        config.validate()
        self.config = config
        self.mu: torch.Tensor | None = None
        self.cov: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.history: list[float] = []
        self.dead_components: list[int] = []

    def _initialize(self, X: torch.Tensor) -> None:
        K, D = self.config.mixtures, X.shape[1]
        if len(X) < K:
            raise ValueError("GMM has more components than training frames")
        generator = torch.Generator(device=X.device).manual_seed(self.config.seed)
        init_count = min(len(X), self.config.init_subsample)
        subset = X[
            torch.linspace(0, len(X) - 1, init_count, device=X.device).long()
        ]
        choose = torch.randperm(
            len(subset), generator=generator, device=X.device
        )[:K]
        means = subset[choose].clone()
        if self.config.init == "kmeans":
            for _ in range(self.config.kmeans_steps):
                nearest = torch.cdist(subset, means).argmin(-1)
                counts = torch.bincount(nearest, minlength=K).to(X.dtype)
                sums = torch.zeros_like(means)
                sums.index_add_(0, nearest, subset)
                updated = torch.where(counts[:, None] > 0, sums / counts[:, None].clamp_min(1),
                                      means)
                if torch.allclose(updated, means, atol=1e-4, rtol=1e-4):
                    break
                means = updated
        variance_population = (
            subset if self.config.init_variance_source == "initialization" else X
        )
        variance = variance_population.var(0, unbiased=False).clamp_min(
            self.config.reg
        )
        if self.config.covariance == "diag":
            cov = variance[None].expand(K, -1).clone()
        elif self.config.covariance == "spherical":
            cov = variance.mean().expand(K).clone()
        elif self.config.covariance == "tied":
            cov = torch.diag(variance)
        else:
            cov = torch.diag(variance).unsqueeze(0).expand(K, D, D).clone()
        self.mu = means
        self.cov = cov
        self.weights = torch.full((K,), 1 / K, device=X.device, dtype=X.dtype)

    def log_components(self, X: torch.Tensor) -> torch.Tensor:
        if self.mu is None or self.cov is None or self.weights is None:
            raise RuntimeError("Fit the GMM before scoring")
        K, D = self.mu.shape
        if X.ndim != 2 or X.shape[1] != D:
            raise ValueError("Mismatched input features")
        weights = self.weights.clamp_min(1e-12).log()
        covtype = self.config.covariance
        if covtype == "diag":
            precision = self.cov.reciprocal()
            scores = X @ (self.mu * precision).T - .5 * (X.square() @ precision.T)
            bias = -.5 * (
                (self.mu.square() * precision + self.cov.log()).sum(-1)
                + D * _LOG_2PI
            )
        elif covtype == "spherical":
            precision = (1 / self.cov)[:, None]
            scores = X @ (self.mu * precision).T - .5 * (
                X.square().sum(-1, keepdim=True) @ precision.T
            )
            bias = -.5 * (
                (self.mu.square() * precision).sum(-1) + D * self.cov.log()
                + D * _LOG_2PI
            )
        else:
            C = self.cov if covtype == "full" else self.cov[None].expand(K, D, D)
            identity = torch.eye(D, device=X.device, dtype=X.dtype)
            chol, info = torch.linalg.cholesky_ex(C)
            if bool((info != 0).any()):
                chol = torch.linalg.cholesky(C + identity[None] * self.config.reg)
            difference = X[None] - self.mu[:, None]  # K,N,D
            solution = torch.linalg.solve_triangular(
                chol, difference.transpose(1, 2), upper=False
            )
            quadratic = solution.square().sum(1).T  # N,K
            logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            return -.5 * (quadratic + logdet[None] + D * _LOG_2PI) + weights
        return scores + bias[None] + weights[None]

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.log_components(X), -1)

    def responsibilities(self, X: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.log_components(X), -1)

    def sufficient_statistics(
        self, X: torch.Tensor, with_full_second: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        K, D = self.mu.shape
        n = torch.zeros(K, device=X.device, dtype=X.dtype)
        first = torch.zeros(K, D, device=X.device, dtype=X.dtype)
        second = torch.zeros(K, D, device=X.device, dtype=X.dtype)
        cross = torch.zeros(K, D, D, device=X.device, dtype=X.dtype) if with_full_second else None
        for block in X.split(self.config.chunk):
            resp = self.responsibilities(block)
            n += resp.sum(0)
            first += resp.T @ block
            second += resp.T @ block.square()
            if cross is not None:
                cross += torch.einsum("nk,nd,ne->kde", resp, block, block)
        return n, first, second, cross

    def _fused_stats_and_ll(
        self, X: torch.Tensor, with_full_second: bool = False,
    ):
        """E-step statistics and pre-M-step LL using one log-density evaluation.

        This is a *numerically matched* EM E-step optimization. The convergence
        history in fused mode records PRE-update log-likelihood; the reference
        mode records POST-update log-likelihood. Fixed-iteration comparisons
        are therefore the most controlled performance experiment.
        """
        K, D = self.mu.shape
        n = torch.zeros(K, device=X.device, dtype=X.dtype)
        first = torch.zeros(K, D, device=X.device, dtype=X.dtype)
        second = torch.zeros(K, D, device=X.device, dtype=X.dtype)
        cross = (torch.zeros(K, D, D, device=X.device, dtype=X.dtype)
                 if with_full_second else None)
        total_log_likelihood = torch.zeros((), device=X.device, dtype=X.dtype)
        for block in X.split(self.config.chunk):
            components = self.log_components(block)
            logp = torch.logsumexp(components, dim=1)
            resp = torch.exp(components - logp[:, None])
            total_log_likelihood += logp.sum()
            n += resp.sum(0)
            first += resp.T @ block
            second += resp.T @ block.square()
            if cross is not None:
                cross += torch.einsum("nk,nd,ne->kde", resp, block, block)
        return n, first, second, cross, float(total_log_likelihood / len(X))


    def fit(self, X: torch.Tensor) -> "TorchGMM":
        self.config.validate()
        if X.ndim != 2 or not torch.isfinite(X).all():
            raise ValueError("Expected finite [frames, dimensions] tensor")
        X = X.detach()
        self._initialize(X)
        c = self.config
        total, D, K = len(X), X.shape[1], c.mixtures
        eye = torch.eye(D, device=X.device, dtype=X.dtype)
        floor = X.var(0, unbiased=False).clamp_min(c.reg)
        self.history = []
        self.dead_components = []
        for iteration in range(c.iterations):
            if c.em_impl == "fused":
                n, first, second, cross, pre_update_ll = self._fused_stats_and_ll(
                    X, with_full_second=(c.covariance == "full")
                )
            else:
                pre_update_ll = (
                    self.mean_log_prob(X) if c.legacy_pre_mstep_likelihood else None
                )
                n, first, second, cross = self.sufficient_statistics(
                    X, with_full_second=(c.covariance == "full")
                )
            safe = n.clamp_min(1e-8)
            new_mu = first / safe[:, None]
            if c.covariance == "diag":
                new_cov = (second / safe[:, None] - new_mu.square()).clamp_min(c.reg)
                new_cov = torch.maximum(new_cov, floor[None] * .002)
            elif c.covariance == "spherical":
                new_cov = (second / safe[:, None] - new_mu.square()).mean(1).clamp_min(c.reg)
            elif c.covariance == "tied":
                moment = X.T @ X / total
                new_cov = moment - (n[:, None] * new_mu).T @ new_mu / total
                new_cov = (new_cov + new_cov.T) / 2 + eye * c.reg
            else:
                new_cov = cross / safe[:, None, None] - (
                    new_mu[:, :, None] * new_mu[:, None, :]
                )
                new_cov = (new_cov + new_cov.transpose(1, 2)) / 2 + c.reg * eye[None]
            dead = n < c.min_occupancy
            self.dead_components.append(int(dead.sum()))
            if dead.any():
                indices = torch.where(dead)[0]
                if c.dead_policy == "hard_examples":
                    # Match the earlier notebook's low-UBM-likelihood replacement,
                    # rather than recycling arbitrary training frames.
                    with torch.no_grad():
                        hard = torch.argsort(self.log_prob(X))[:len(indices)]
                        replacements = X[hard]
                else:
                    replacements = X[(indices + iteration * 131) % len(X)]
                new_mu[dead] = replacements
                if c.covariance == "diag":
                    new_cov[dead] = floor
                elif c.covariance == "spherical":
                    new_cov[dead] = floor.mean()
                elif c.covariance == "full":
                    new_cov[dead] = torch.diag(floor)
                n[dead] = max(c.min_occupancy, len(X) / K * c.dead_weight_fraction)
            self.mu = new_mu
            self.cov = new_cov
            self.weights = n / n.sum()
            log_likelihood = (
                pre_update_ll if pre_update_ll is not None
                else self.mean_log_prob(X)
            )
            self.history.append(log_likelihood)
            minimum_history = (
                4 if c.legacy_pre_mstep_likelihood or c.em_impl == "fused" else 3
            )
            if len(self.history) >= minimum_history:
                improvement = self.history[-1] - self.history[-2]
                if abs(improvement) <= c.tolerance * max(1.0, abs(self.history[-2])):
                    break
        return self

    @torch.no_grad()
    def mean_log_prob(self, X: torch.Tensor) -> float:
        if len(X) == 0:
            raise ValueError("Empty validation features")
        logsum = torch.zeros((), device=X.device, dtype=X.dtype)
        for block in X.split(self.config.chunk):
            logsum += self.log_prob(block).sum()
        return float(logsum / len(X))

    def parameter_count(self) -> int:
        K, D = self.mu.shape
        ncov = {
            "diag": K * D,
            "spherical": K,
            "tied": D * (D + 1) // 2,
            "full": K * D * (D + 1) // 2,
        }[self.config.covariance]
        return K - 1 + K * D + ncov

    def information_criteria(self, X: torch.Tensor) -> dict[str, float]:
        n, p = len(X), self.parameter_count()
        log_likelihood = self.mean_log_prob(X) * n
        return {"aic": 2 * p - 2 * log_likelihood,
                "bic": math.log(n) * p - 2 * log_likelihood,
                "parameters": p, "frames": n}

    def adapt(self, frames: torch.Tensor, cfg: MAPConfig) -> "TorchGMM":
        cfg.validate()
        if frames.ndim != 2 or not len(frames):
            raise ValueError("Speaker enrolment has no frames")
        counts, first, second, _ = self.sufficient_statistics(frames)
        alpha = counts / (counts + cfg.relevance)
        sample_means = first / counts.clamp_min(1e-12)[:, None]
        adapted = TorchGMM(self.config)
        adapted.mu = alpha[:, None] * sample_means + (1 - alpha[:, None]) * self.mu
        adapted.cov = self.cov.clone()
        adapted.weights = self.weights.clone()
        if cfg.adapt_variances:
            if self.config.covariance not in ("diag", "spherical"):
                raise ValueError("Variance MAP currently supports diagonal/spherical GMMs")
            if self.config.covariance == "diag":
                old_second = self.cov + self.mu.square()
                new_second = (alpha[:, None] * second / counts.clamp_min(1e-12)[:, None]
                              + (1 - alpha[:, None]) * old_second)
                adapted.cov = (new_second - adapted.mu.square()).clamp_min(cfg.variance_floor)
            else:
                old_second = self.cov + self.mu.square().mean(1)
                empirical = second.sum(1) / counts.clamp_min(1e-12) / self.mu.shape[1]
                new_second = alpha * empirical + (1 - alpha) * old_second
                adapted.cov = (new_second - adapted.mu.square().mean(1)).clamp_min(
                    cfg.variance_floor
                )
        if cfg.adapt_weights:
            proportions = counts / counts.sum().clamp_min(1e-12)
            adapted.weights = (alpha * proportions + (1 - alpha) * self.weights)
            adapted.weights = adapted.weights.clamp_min(1e-10)
            adapted.weights = adapted.weights / adapted.weights.sum()
        return adapted

    def snapshot(self) -> dict:
        return {"config": asdict(self.config), "mu": self.mu.cpu(), "cov": self.cov.cpu(),
                "weights": self.weights.cpu(), "history": self.history,
                "dead_components": self.dead_components}

    @classmethod
    def from_snapshot(cls, data: dict, device: str = "cpu") -> "TorchGMM":
        model = cls(EMConfig(**data["config"]))
        model.mu = data["mu"].to(device)
        model.cov = data["cov"].to(device)
        model.weights = data["weights"].to(device)
        model.history = list(data.get("history", []))
        model.dead_components = list(data.get("dead_components", []))
        return model

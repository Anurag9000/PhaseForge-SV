"""Background-fitted dimensionality transformations, all implemented using torch."""
from __future__ import annotations

import torch


class PCA:
    def __init__(self, dim: int | None = None, whiten: bool = False):
        self.dim, self.whiten = dim, whiten

    def fit(self, frames: torch.Tensor) -> "PCA":
        if frames.ndim != 2 or len(frames) < 2:
            raise ValueError("PCA requires at least two frames")
        self.mean = frames.mean(0)
        X = frames - self.mean
        eigenvalues, eigenvectors = torch.linalg.eigh(X.T @ X / (len(X) - 1))
        indices = torch.argsort(eigenvalues, descending=True)
        count = min(self.dim or frames.shape[-1], frames.shape[-1])
        self.values = eigenvalues[indices][:count].clamp_min(1e-7)
        self.components = eigenvectors[:, indices][:, :count]
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) @ self.components
        return z / self.values.sqrt().clamp_min(1e-4) if self.whiten else z


class ICA:
    """Symmetric FastICA (tanh score); fit only on permitted UBM background frames."""

    def __init__(self, dim: int, iterations: int = 150, seed: int = 2026):
        self.dim, self.iterations, self.seed = dim, iterations, seed

    @staticmethod
    def _sym_orth(W: torch.Tensor) -> torch.Tensor:
        val, vec = torch.linalg.eigh(W @ W.T)
        return (vec @ torch.diag(val.clamp_min(1e-8).rsqrt()) @ vec.T) @ W

    def fit(self, x: torch.Tensor) -> "ICA":
        self.pca = PCA(self.dim, whiten=True).fit(x)
        z = self.pca(x)
        gen = torch.Generator(device=z.device).manual_seed(self.seed)
        W = self._sym_orth(torch.randn(z.shape[-1], z.shape[-1], generator=gen, device=z.device))
        for _ in range(self.iterations):
            y = z @ W.T
            new = (torch.tanh(y).T @ z) / len(z) - (
                (1 - torch.tanh(y).square()).mean(0)[:, None] * W
            )
            new = self._sym_orth(new)
            error = (torch.diag(new @ W.T).abs() - 1).abs().max()
            W = new
            if float(error) < 1e-5:
                break
        self.W = W
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.pca(x) @ self.W.T


def fit_background_transform(
    features: dict[str, torch.Tensor], background_ids: list[str],
    method: str, dim: int, max_frames: int = 50000,
    seed: int = 2026,
) -> tuple[dict[str, torch.Tensor], PCA | ICA | None]:
    """Fit exclusively on background IDs provided by caller; no target/test statistics."""
    if not background_ids:
        raise ValueError("Need background feature IDs")
    if method == "none":
        return features, None
    X = torch.cat([features[ident] for ident in background_ids], 0)
    if len(X) > max_frames:
        X = X[torch.linspace(0, len(X) - 1, max_frames, device=X.device).long()]
    if method == "pca":
        transform = PCA(dim).fit(X)
    elif method == "whiten":
        transform = PCA(dim, whiten=True).fit(X)
    elif method == "ica":
        transform = ICA(dim, seed=seed).fit(X)
    else:
        raise ValueError(method)
    return {ident: transform(x) for ident, x in features.items()}, transform

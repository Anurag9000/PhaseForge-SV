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



class BackgroundCMVN:
    """Global mean/std normalization fitted ONLY on background feature frames."""

    def __init__(self, min_std: float = .06):
        if min_std <= 0:
            raise ValueError("min_std must be positive")
        self.min_std = min_std

    def fit(self, background: torch.Tensor) -> "BackgroundCMVN":
        if background.ndim != 2 or len(background) < 2:
            raise ValueError("Global CMVN requires at least 2 background frames")
        self.mean = background.mean(0)
        self.std = background.std(0, unbiased=False).clamp_min(self.min_std)
        return self

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std


class GlobalProjectedTransform:
    """First apply background-only global CMVN, then fit PCA/ICA to its output."""

    def __init__(self, global_norm: BackgroundCMVN, projection: PCA | ICA):
        self.global_norm = global_norm
        self.projection = projection

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(self.global_norm(features))



class GlobalLocalCenteredProjection:
    """Global background CMVN -> utterance scalar mean -> background-fitted PCA."""

    def __init__(self, global_norm: BackgroundCMVN, projection: PCA):
        self.global_norm, self.projection = global_norm, projection

    def __call__(self, features: torch.Tensor):
        z = self.global_norm(features)
        z = z - z.mean()  # MATLAB-style scalar cepstral mean (all channels/frames).
        return self.projection(z)


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
    if method == "global_local_pca":
        # Unlike legacy PCA, this newer MathWorks-inspired transform must not
        # concatenate every original-Google background frame before clipping
        # the training pool. Bound the frame budget BEFORE allocating.
        if max_frames < 2:
            raise ValueError("Need at least two background frames for global/local PCA")
        selected = list(dict.fromkeys(background_ids))
        if len(selected) > max_frames:
            selected = [
                selected[i] for i in torch.linspace(
                    0, len(selected) - 1, max_frames
                ).long().tolist()
            ]
        quota = max(1, (max_frames + len(selected) - 1) // len(selected))
        pieces = []
        for file in selected:
            frame = features[file]
            count = min(len(frame), quota)
            pieces.append(frame[
                torch.linspace(
                    0, len(frame) - 1, count, device=frame.device
                ).long()
            ])
        X = torch.cat(pieces, dim=0)
        if len(X) > max_frames:
            X = X[torch.linspace(
                0, len(X) - 1, max_frames, device=X.device
            ).long()]
        normalizer = BackgroundCMVN().fit(X)
        local = []
        for file in selected:
            z = normalizer(features[file])
            count = min(len(z), quota)
            indices = torch.linspace(
                0, len(z) - 1, count, device=z.device
            ).long()
            local.append(z[indices] - z.mean())
        pool = torch.cat(local, dim=0)
        if len(pool) > max_frames:
            pool = pool[torch.linspace(
                0, len(pool) - 1, max_frames, device=pool.device
            ).long()]
        transform = GlobalLocalCenteredProjection(
            normalizer, PCA(dim).fit(pool)
        )
        return {ident: transform(x) for ident, x in features.items()}, transform
    X = torch.cat([features[ident] for ident in background_ids], 0)
    if len(X) > max_frames:
        X = X[torch.linspace(0, len(X) - 1, max_frames, device=X.device).long()]
    if method == "pca":
        transform = PCA(dim).fit(X)
    elif method == "whiten":
        transform = PCA(dim, whiten=True).fit(X)
    elif method == "ica":
        transform = ICA(dim, seed=seed).fit(X)
    elif method == "global":
        transform = BackgroundCMVN().fit(X)
    elif method in ("global_pca", "global_whiten", "global_ica"):
        normalizer = BackgroundCMVN().fit(X)
        whitened = normalizer(X)
        if method == "global_ica":
            projection = ICA(dim, seed=seed).fit(whitened)
        else:
            projection = PCA(dim, whiten=(method == "global_whiten")).fit(whitened)
        transform = GlobalProjectedTransform(normalizer, projection)
    else:
        raise ValueError(method)
    return {ident: transform(x) for ident, x in features.items()}, transform

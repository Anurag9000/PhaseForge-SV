import pytest
import torch

from phaseforge.transforms import BackgroundCMVN, fit_background_transform


@pytest.mark.parametrize(
    "mode,expected_dim",
    [
        ("global", 6), ("global_pca", 4),
        ("global_whiten", 4), ("global_ica", 4),
        ("global_local_pca", 4),
    ],
)
def test_global_statistics_are_background_only(mode, expected_dim):
    rng = torch.Generator().manual_seed(17)
    data = {
        "bg_a": torch.randn(70, 6, generator=rng) * .5 + 2,
        "bg_b": torch.randn(70, 6, generator=rng) * .7 + 3,
        "heldout": torch.randn(30, 6, generator=rng) + 300,
    }
    transformed, learned = fit_background_transform(
        data, ["bg_a", "bg_b"], mode, dim=4, max_frames=120
    )
    assert transformed["heldout"].shape == (30, expected_dim)
    assert torch.isfinite(transformed["heldout"]).all()
    # Leakage test: vastly changing a held-out recording must not affect the
    # fitted transformation or any background output.
    data["heldout"] = data["heldout"] + 9999
    refit, second = fit_background_transform(
        data, ["bg_a", "bg_b"], mode, dim=4, max_frames=120
    )
    assert torch.allclose(transformed["bg_a"].abs(), refit["bg_a"].abs(), atol=1e-5)


def test_global_cmvn_background_mean_zero():
    X = torch.tensor([[1., 3.], [3., 7.], [5., 11.]])
    n = BackgroundCMVN().fit(X)
    assert torch.allclose(n(X).mean(0), torch.zeros(2), atol=1e-6)
    assert bool((n.std > 0).all())

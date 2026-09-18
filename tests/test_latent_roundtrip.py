import pytest
import torch

from latent.bounds import LatentBox
from latent.reshape import flatten, restore


def test_full_dimension_round_trip():
    z0 = torch.randn(1, 128, 3, dtype=torch.float64)
    box = LatentBox(z0, radius=0.25)
    x = torch.rand(z0.numel(), dtype=torch.float64)
    z = box.to_native(x)
    assert z.shape == z0.shape and box.dimension == z0.numel()
    torch.testing.assert_close(box.to_unit(z), x, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(box.to_native(box.to_unit(z)), z, rtol=1e-12, atol=1e-12)
    assert torch.equal(restore(flatten(z), tuple(z.shape)), z)
    assert torch.equal(box.to_native(torch.full_like(x, 0.5)), z0)


def test_each_dimension_can_move_independently():
    box = LatentBox(torch.zeros(1, 8, 3), radius=0.5, scale=torch.linspace(0.5, 2, 24))
    center = torch.full((24,), 0.5, dtype=torch.float64)
    for dimension in range(24):
        point = center.clone()
        point[dimension] = 1
        delta = box.to_native(point).reshape(-1)
        assert torch.count_nonzero(delta) == 1
        assert delta[dimension] == 0.5 * box.scale[dimension]


@pytest.mark.parametrize("radius", [0, -1, float("inf"), float("nan")])
def test_invalid_radius_rejected(radius):
    with pytest.raises(ValueError):
        LatentBox(torch.zeros(1, 8, 3), radius)


def test_no_silent_projection_clipping_or_shape_change():
    box = LatentBox(torch.zeros(1, 8, 3), 1)
    for x in [torch.zeros(23), torch.full((24,), 1.1), torch.full((24,), float("nan"))]:
        with pytest.raises(ValueError):
            box.to_native(x)
    with pytest.raises(ValueError):
        box.to_unit(torch.ones(1, 8, 3) * 2)
    with pytest.raises(ValueError):
        restore(torch.zeros(8), (1, 8, 3))

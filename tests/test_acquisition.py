import pytest
import torch

from dsp.acquisition import acquisition
from dsp.model import fit_gp
from dsp.optimize_acq import AcquisitionSettings, optimize_acquisition
from dsp.sampling import initial_design, sobol_points


def test_global_and_local_full_dimension_acquisition():
    torch.set_num_threads(2)
    x = initial_design(10, 8)
    y = (1-(x-.4).square().mean(-1)).unsqueeze(-1)
    gp = fit_gp(x, y, maxiter=50)
    settings = AcquisitionSettings(raw_samples=32, num_restarts=4, maxiter=30)
    acq = acquisition(gp.model, float(y.max()))
    for bounds in [torch.tensor([[0.]*10, [1.]*10]).double(), torch.tensor([[.3]*10, [.7]*10]).double()]:
        candidate, info = optimize_acquisition(acq, bounds, x[y.argmax()], seed=4, settings=settings)
        assert candidate.shape == (10,)
        assert (candidate >= bounds[0]).all() and (candidate <= bounds[1]).all()
        assert info["optimization_succeeded"]
        assert info["global_restarts"] == info["local_restarts"] == 2
        assert info["q"] == 1
    candidate2, _ = optimize_acquisition(acq, bounds, x[y.argmax()], seed=4, settings=settings)
    torch.testing.assert_close(candidate, candidate2, rtol=0, atol=1e-12)


def test_sobol_common_initial_design_is_exactly_reproducible():
    a, b = initial_design(100, 16, 3), initial_design(100, 16, 3)
    assert torch.equal(a, b)
    assert torch.equal(a[0], torch.full((100,), .5, dtype=torch.float64))
    whole = sobol_points(100, 25, 3)
    assert torch.equal(whole[15:], sobol_points(100, 10, 3, skip=15))


def test_gradient_failure_is_recorded_without_losing_finite_raw_candidate():
    class BrokenGradient(torch.nn.Module):
        def forward(self, x):
            if x.requires_grad:
                raise RuntimeError("test nonfinite optimizer failure")
            return x.sum(dim=(-1, -2))
    bounds = torch.tensor([[0.]*10, [1.]*10]).double()
    candidate, info = optimize_acquisition(BrokenGradient(), bounds, bounds.mean(0), seed=1,
        settings=AcquisitionSettings(raw_samples=8, num_restarts=2))
    assert torch.isfinite(candidate).all()
    assert info["used_raw_fallback"] and not info["optimization_succeeded"]
    assert "optimizer failure" in info["optimization_error"]


def test_frozen_dimensions_are_rejected():
    bounds = torch.tensor([[0., .5], [1., .5]]).double()
    with pytest.raises(ValueError, match="nonzero interval"):
        optimize_acquisition(None, bounds, bounds.mean(0), seed=0)

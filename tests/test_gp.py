import pytest
import torch

from dsp.acquisition import acquisition
from dsp.model import build_gp, fit_gp
from dsp.sampling import initial_design


def test_fit_and_posterior_are_finite_with_constant_or_varying_observations():
    torch.set_num_threads(2)
    x = initial_design(10, 8)
    for y in [torch.zeros(8, 1, dtype=torch.float64), (1-(x-.3).square().mean(-1)).unsqueeze(-1)]:
        fitted = fit_gp(x, y, maxiter=50)
        posterior = fitted.model.posterior(x[:2])
        assert torch.isfinite(posterior.mean).all()
        assert torch.isfinite(posterior.variance).all()
        assert torch.isfinite(fitted.model.covar_module.lengthscale).all()
        assert torch.isfinite(acquisition(fitted.model, float(y.max()))(x[:2].unsqueeze(1))).all()


def test_rejects_projection_or_bad_normalization_at_model_boundary():
    x = initial_design(10, 4)
    with pytest.raises(ValueError):
        build_gp(x * 2, torch.ones(4, 1, dtype=torch.float64))
    with pytest.raises(ValueError):
        build_gp(x.float(), torch.ones(4, 1))

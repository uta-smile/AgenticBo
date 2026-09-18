import math

import pytest
import torch
from gpytorch.kernels import RBFKernel

from dsp.model import build_gp
from dsp.priors import lengthscale_prior_parameters
from dsp.sampling import initial_design


@pytest.mark.parametrize("dimension", [10, 256, 1024, 6392])
def test_instantiated_prior_matches_formula_in_every_native_dimension(dimension):
    x = initial_design(dimension, 4)
    model = build_gp(x, x.mean(dim=1, keepdim=True))
    kernel = model.covar_module
    assert type(kernel) is RBFKernel
    assert kernel.ard_num_dims == dimension
    assert kernel.lengthscale.numel() == dimension
    assert kernel.active_dims is None
    assert kernel.lengthscale_prior.loc.item() == pytest.approx(math.sqrt(2) + .5*math.log(dimension), abs=1e-12)
    assert kernel.lengthscale_prior.scale.item() == pytest.approx(math.sqrt(3), abs=1e-12)
    assert kernel.raw_lengthscale_constraint.lower_bound.item() == pytest.approx(.025)
    assert kernel.lengthscale.min().item() == pytest.approx(lengthscale_prior_parameters(dimension)["mode"], rel=1e-6)
    assert model.outcome_transform._is_trained
    torch.testing.assert_close(model.input_transform(x), x)


def test_vanilla_changes_only_the_lengthscale_prior_setup():
    x = initial_design(100, 4)
    y = x.mean(dim=1, keepdim=True)
    dsp, vanilla = build_gp(x, y), build_gp(x, y, dsp=False)
    assert type(dsp.covar_module) is type(vanilla.covar_module)
    assert vanilla.covar_module.lengthscale_prior.loc.item() == pytest.approx(math.sqrt(2), abs=1e-12)
    assert vanilla.covar_module.ard_num_dims == 100
    ds, vs = dsp.state_dict(), vanilla.state_dict()
    differences = {key for key in ds if not torch.equal(ds[key], vs[key])}
    assert differences <= {"covar_module.raw_lengthscale", "covar_module.lengthscale_prior._transformed_loc"}
    assert not any("outputscale" in key for key in ds)

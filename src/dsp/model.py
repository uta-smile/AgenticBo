from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_mll_scipy
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from dsp.priors import lengthscale_prior_parameters


@dataclass
class FittedGP:
    model: SingleTaskGP
    info: dict


def build_gp(x: torch.Tensor, y: torch.Tensor, *, dsp: bool = True) -> SingleTaskGP:
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or y.shape != (len(x), 1):
        raise ValueError("Expected X=[N,D], Y=[N,1], N>=2")
    if x.dtype != torch.float64 or y.dtype != torch.float64 or x.device != y.device:
        raise ValueError("GP inputs and outcomes must be float64 on the same device")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Require finite outcomes and full-D X in [0,1]")
    dimension = x.shape[1]
    params = lengthscale_prior_parameters(dimension, dsp=dsp)
    # Construct the prior in float64 initially; casting a float32 prior afterward
    # would already have rounded the dimension-dependent formula.
    prior = LogNormalPrior(torch.tensor(params["loc"], dtype=x.dtype, device=x.device),
                           torch.tensor(params["scale"], dtype=x.dtype, device=x.device))
    kernel = RBFKernel(ard_num_dims=dimension, lengthscale_prior=prior,
                       lengthscale_constraint=GreaterThan(0.025, transform=None, initial_value=prior.mode))
    noise_prior = LogNormalPrior(torch.tensor(-4.0, dtype=x.dtype, device=x.device),
                                 torch.tensor(1.0, dtype=x.dtype, device=x.device))
    likelihood = GaussianLikelihood(noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-4, transform=None, initial_value=noise_prior.mode))
    unit_bounds = torch.stack([torch.zeros(dimension, dtype=x.dtype, device=x.device),
                               torch.ones(dimension, dtype=x.dtype, device=x.device)])
    return SingleTaskGP(x, y, covar_module=kernel, likelihood=likelihood,
                        input_transform=Normalize(d=dimension, bounds=unit_bounds),
                        outcome_transform=Standardize(m=1)).to(x)


def fit_gp(x: torch.Tensor, y: torch.Tensor, *, dsp: bool = True, maxiter: int = 100) -> FittedGP:
    if maxiter < 1:
        raise ValueError("GP fitting needs at least one iteration")
    started = time.monotonic()
    model = build_gp(x, y, dsp=dsp)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_gpytorch_mll_scipy(mll, options={"maxiter": maxiter, "ftol": 1e-9})
    if result.status.name not in {"SUCCESS", "STOPPED"}:
        raise RuntimeError(f"GP fitting failed: {result.status.name}: {result.message}")
    if not math.isfinite(result.fval) or any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("GP fitting returned nonfinite parameters or objective")
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        posterior = model.posterior(x)
        if not torch.isfinite(posterior.mean).all() or not torch.isfinite(posterior.variance).all():
            raise RuntimeError("GP posterior is not finite after fitting")
    info = {"fit_status": result.status.name.lower(), "fit_converged": result.status.name == "SUCCESS",
            "fit_message": result.message, "fit_iterations": result.step,
            "fit_runtime_seconds": time.monotonic() - started,
            "fit_warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
            "prior": lengthscale_prior_parameters(x.shape[1], dsp=dsp)}
    return FittedGP(model, info)

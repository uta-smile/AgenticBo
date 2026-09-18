import torch

from dsp.model import FittedGP


def diagnostics(fitted: FittedGP, x: torch.Tensor) -> dict:
    model = fitted.model
    with torch.no_grad():
        lengthscales = model.covar_module.lengthscale.reshape(-1)
        order = torch.argsort(lengthscales, stable=True)
        kernel = model.covar_module(x).to_dense()
        off_diagonal = kernel[~torch.eye(len(x), dtype=torch.bool, device=x.device)]
        uncertainty = model.posterior(x).variance.sqrt()
    return {**fitted.info, "number_observations": len(x), "dimension": x.shape[1],
            "noise": float(model.likelihood.noise.detach().squeeze()),
            "lengthscale_min": float(lengthscales.min()), "lengthscale_median": float(lengthscales.median()),
            "lengthscale_max": float(lengthscales.max()),
            "shortest_dimensions": order[:10].tolist(), "longest_dimensions": order[-10:].flip(0).tolist(),
            "posterior_std_mean": float(uncertainty.mean()),
            "kernel_off_diagonal_mean": float(off_diagonal.mean()),
            "kernel_off_diagonal_max": float(off_diagonal.max())}

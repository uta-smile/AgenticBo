from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import torch
from botorch.optim import optimize_acqf

from dsp.sampling import sobol_points


@dataclass(frozen=True)
class AcquisitionSettings:
    raw_samples: int = 1024
    num_restarts: int = 16
    maxiter: int = 100
    batch_limit: int = 2
    local_fraction: float = 0.1

    def __post_init__(self):
        if self.raw_samples < 4 or not 2 <= self.num_restarts <= self.raw_samples:
            raise ValueError("Need at least four raw samples and two to raw_samples restarts")
        if self.maxiter < 1 or self.batch_limit < 1 or not 0 < self.local_fraction <= 1:
            raise ValueError("Invalid acquisition optimization settings")


def optimize_acquisition(acq, bounds: torch.Tensor, incumbent: torch.Tensor, *, seed: int,
                         settings: AcquisitionSettings = AcquisitionSettings()) -> tuple[torch.Tensor, dict]:
    started = time.monotonic()
    if (bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1
            or not torch.isfinite(bounds).all() or (bounds[0] >= bounds[1]).any()
            or (bounds < 0).any() or (bounds > 1).any()):
        raise ValueError("Every dimension must have a nonzero interval inside the normalized box")
    if incumbent.shape != bounds[0].shape or not torch.isfinite(incumbent).all():
        raise ValueError("Incumbent must have D finite coordinates")
    if bounds.dtype != torch.float64 or incumbent.dtype != torch.float64 or bounds.device != incumbent.device:
        raise ValueError("Acquisition bounds and incumbent must be float64 on the same device")
    d, width = bounds.shape[1], bounds[1] - bounds[0]
    global_count = settings.raw_samples // 2
    global_points = bounds[0] + width * sobol_points(d, global_count, seed, device=bounds.device)
    rng = torch.Generator(device=bounds.device).manual_seed(seed)
    local_points = incumbent + settings.local_fraction * width * torch.randn(
        settings.raw_samples - global_count, d, generator=rng, device=bounds.device, dtype=bounds.dtype)
    local_points = local_points.clamp(bounds[0], bounds[1])
    raw = torch.cat([global_points, local_points])
    with torch.no_grad():
        # Limit simultaneous posterior evaluation memory in the actual full D.
        scores = torch.cat([acq(chunk.unsqueeze(1)).reshape(-1) for chunk in raw.split(128)])
    if not torch.isfinite(scores).all():
        raise RuntimeError("Nonfinite acquisition values at raw initializations")
    global_restarts = settings.num_restarts // 2
    local_restarts = settings.num_restarts - global_restarts
    indices = torch.cat([scores[:global_count].topk(global_restarts).indices,
                         scores[global_count:].topk(local_restarts).indices + global_count])
    initial = raw[indices].unsqueeze(1)
    info = {"dimension": d, "q": 1, "raw_samples": len(raw), "num_restarts": len(initial),
            "global_restarts": global_restarts, "local_restarts": local_restarts,
            "raw_acquisition_min": float(scores.min()), "raw_acquisition_max": float(scores.max()),
            "optimization_succeeded": False, "used_raw_fallback": False,
            "nonfinite_gradient_entries": 0, "bound_violations": 0, "optimization_warnings": []}
    # Always retain the best raw candidate; gradient optimization can be worse.
    candidate, value = raw[scores.argmax()].clone(), scores.max().clone()
    try:
        checked = initial.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(acq(checked).sum(), checked)[0]
        info["nonfinite_gradient_entries"] = int((~torch.isfinite(gradient)).sum())
        if info["nonfinite_gradient_entries"]:
            raise RuntimeError("Nonfinite acquisition gradient at restart initialization")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimized, optimized_value = optimize_acqf(
                acq_function=acq, bounds=bounds, q=1, num_restarts=len(initial), raw_samples=None,
                batch_initial_conditions=initial, options={"maxiter": settings.maxiter,
                    "batch_limit": settings.batch_limit}, retry_on_optimization_warning=False)
        info["optimization_warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
        optimized = optimized.reshape(-1)
        if not torch.isfinite(optimized).all() or not torch.isfinite(optimized_value).all():
            raise RuntimeError("Nonfinite optimized acquisition candidate")
        violations = (optimized < bounds[0] - 1e-10) | (optimized > bounds[1] + 1e-10)
        info["bound_violations"] = int(violations.sum())
        if info["bound_violations"]:
            raise RuntimeError("Acquisition candidate violates the active full-D bounds")
        info["optimization_succeeded"] = True
        if optimized_value >= value:
            candidate, value = optimized.clamp(bounds[0], bounds[1]), optimized_value
        else:
            info["used_raw_fallback"] = True
    except (RuntimeError, ValueError) as exc:
        info.update(used_raw_fallback=True, optimization_error=f"{type(exc).__name__}: {exc}")
    info.update(acquisition_value=float(value.detach()), runtime_seconds=time.monotonic() - started)
    return candidate.detach(), info

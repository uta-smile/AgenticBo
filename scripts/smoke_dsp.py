#!/usr/bin/env python3
"""Test the full-dimensional DSP GP and acquisition without protein decodes."""
import argparse
import json
from pathlib import Path
import time

import torch

from dsp.acquisition import acquisition
from dsp.diagnostics import diagnostics
from dsp.model import fit_gp
from dsp.optimize_acq import AcquisitionSettings, optimize_acquisition
from dsp.priors import lengthscale_prior_parameters
from dsp.sampling import initial_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--latent-report", type=Path, help="Also test the measured D in an existing latent report")
    parser.add_argument("--output", type=Path, default=Path("outputs/dsp_smoke"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--maxiter", type=int, default=100)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    dimensions = list(args.dimensions)
    measured = None
    if args.latent_report:
        report = json.loads(args.latent_report.read_text())
        measured = report.get("flattened_D")
        shape = report.get("native_latent_shape")
        if not measured or not shape or int(torch.tensor(shape).prod()) != measured:
            parser.error("Latent report must contain a measured native shape and D")
        dimensions.append(measured)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    settings = AcquisitionSettings(raw_samples=args.raw_samples, num_restarts=args.restarts, maxiter=args.maxiter)
    for dimension in dict.fromkeys(dimensions):
        print(json.dumps({"starting_dimension": dimension, "prior": lengthscale_prior_parameters(dimension)}), flush=True)
        started = time.monotonic()
        x = initial_design(dimension, 16, args.seed)
        # Every coordinate contributes, with unequal optima; no active subspace.
        target = torch.linspace(0.1, 0.9, dimension, dtype=torch.float64)
        y = (1 - (x-target).square().mean(dim=-1)).unsqueeze(-1)
        fitted = fit_gp(x, y, maxiter=args.maxiter)
        acq = acquisition(fitted.model, float(y.max()))
        bounds = torch.stack([torch.zeros(dimension), torch.ones(dimension)]).double()
        candidate, optimization = optimize_acquisition(acq, bounds, x[y.argmax()], seed=args.seed, settings=settings)
        posterior = fitted.model.posterior(candidate.unsqueeze(0))
        result = {"dimension": dimension, "measured_report_dimension": dimension == measured,
                  "latent_report": str(args.latent_report) if dimension == measured else None,
                  "diagnostics": diagnostics(fitted, x), "optimization": optimization,
                  "candidate_posterior_mean": float(posterior.mean.detach().squeeze()),
                  "candidate_posterior_std": float(posterior.variance.detach().sqrt().squeeze()),
                  "runtime_seconds": time.monotonic() - started, "gp_device": "cpu", "gpu_memory_bytes": 0,
                  "passed": optimization["optimization_succeeded"] and not optimization["bound_violations"]
                      and not optimization["nonfinite_gradient_entries"] and bool(torch.isfinite(posterior.mean).all())}
        torch.save({"x": x, "y": y, "candidate": candidate, "model": fitted.model.state_dict(),
                    "lengthscales": fitted.model.covar_module.lengthscale.detach()}, args.output / f"D_{dimension}.pt")
        (args.output / f"D_{dimension}.json").write_text(json.dumps(result, indent=2) + "\n")
        reports.append(result)
        print(json.dumps({"dimension": dimension, "passed": result["passed"], "seconds": result["runtime_seconds"]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the paper's Branin/Ackley synthetic comparison with our three methods."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analysis.aggregate import load_runs
from analysis.plots import render
from dsp.optimize_acq import AcquisitionSettings
from latent.bounds import LatentBox
from runner.engine import run_methods
from runner.provenance import software_versions
from sara.agent import SaraController
from sara.client import SaraClient
from targets.manifest import sha256


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    budget: int


PROBLEMS = {
    "branin_2d": Problem("branin_2d", 2, 50),
    "ackley_10d": Problem("ackley_10d", 10, 150),
    "ackley_20d": Problem("ackley_20d", 20, 200),
}


def hidden_transform(x: torch.Tensor, shift: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    # A quarter-range shift and permutation prevent the textbook coordinate
    # system from being exposed while preserving a known optimum value.
    return (x[..., permutation] + shift).remainder(1.0)


def score(problem: Problem, x: torch.Tensor, shift: torch.Tensor, permutation: torch.Tensor) -> float:
    y = hidden_transform(x, shift, permutation)
    if problem.name == "branin_2d":
        x1, x2 = 15.0 * y[0] - 5.0, 15.0 * y[1]
        value = ((x2 - 5.1 * x1.square() / (4.0 * np.pi**2) + 5.0 * x1 / np.pi - 6.0) ** 2
                 + 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * torch.cos(x1) + 10.0)
        # Maximize a bounded monotonic transform of the usual minimization
        # objective. The known optimum is exp(-0.397887 / 50).
        return float(torch.exp(-value / 50.0))
    z = 10.0 * (y - 0.5)
    value = (-20.0 * torch.exp(-0.2 * torch.sqrt(torch.mean(z.square())))
             - torch.exp(torch.mean(torch.cos(2.0 * np.pi * z))) + np.e + 20.0)
    return float(torch.exp(-value / 10.0))


def optimum(problem: Problem) -> float:
    return float(np.exp(-0.397887 / 50.0)) if problem.name == "branin_2d" else 1.0


class SyntheticGenerator:
    def decode(self, z, directory):
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "synthetic_unit_cube.pt"
        torch.save(z.detach().cpu(), path)
        return path


class SyntheticOracle:
    objective = "tm"

    def __init__(self, problem, shift, permutation):
        self.problem, self.shift, self.permutation = problem, shift, permutation

    def score(self, path):
        x = torch.load(path, weights_only=True).reshape(-1).double()
        value = score(self.problem, x, self.shift, self.permutation)
        return {"valid": True, "tm": value, "lddt": value, "rmsd": 1.0 - value,
                "score_kind": "paper synthetic bounded objective; not protein TM-score"}


def seeded_hidden(problem: Problem, seed: int):
    generator = torch.Generator().manual_seed(811_003 + seed * 1_009 + len(problem.name))
    shift = torch.rand(problem.dimension, generator=generator, dtype=torch.float64) * 0.5 - 0.25
    permutation = torch.randperm(problem.dimension, generator=generator)
    digest = hashlib.sha256(shift.numpy().tobytes() + permutation.numpy().tobytes()).hexdigest()
    return shift, permutation, digest


def write_paper_summary(runs, output: Path, problem: Problem):
    by_method = {}
    for run in runs:
        curve = np.maximum.accumulate([trial["objective"] for trial in run["trials"]])
        regret = optimum(problem) - curve
        by_method.setdefault(run["method"], []).append(regret)
    rows, finals = [], []
    for method, curves in sorted(by_method.items()):
        values = np.stack(curves)
        for evaluation in range(values.shape[1]):
            column = values[:, evaluation]
            rows.append({"method": method, "evaluations": evaluation + 1,
                         "median_simple_regret": float(np.median(column)),
                         "q25_simple_regret": float(np.quantile(column, .25)),
                         "q75_simple_regret": float(np.quantile(column, .75)),
                         "seeds": values.shape[0]})
        column = values[:, -1]
        finals.append({"method": method, "evaluations": values.shape[1],
                       "median_simple_regret": float(np.median(column)),
                       "q25_simple_regret": float(np.quantile(column, .25)),
                       "q75_simple_regret": float(np.quantile(column, .75)),
                       "seeds": values.shape[0]})
    import csv
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("paper_simple_regret.csv", rows), ("paper_final_summary.csv", finals)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=sorted(PROBLEMS), default=sorted(PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--methods", nargs="+", choices=("sobol", "ax", "agentic_dsp"),
                        default=("sobol", "ax", "agentic_dsp"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_synthetic"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "agentic_dsp" in args.methods and not args.dry_run:
        client = SaraClient.from_environment()
        prompt = Path("prompts/SARA_SYNTHETIC_SYSTEM.md")
        controller_identity = {**client.identity, "system_prompt_sha256": sha256(prompt)}
        controller = SaraController(client, prompt_path=prompt, max_tools=200, max_rounds=202)
    else:
        client = controller_identity = controller = None
    config = []
    for problem_name in args.problems:
        problem = PROBLEMS[problem_name]
        for seed in args.seeds:
            config.append({"problem": problem.name, "dimension": problem.dimension,
                           "budget": problem.budget, "seed": seed})
    print(json.dumps({"problems": config, "methods": args.methods,
                      "paper_budgets": {k: v.budget for k, v in PROBLEMS.items()},
                      "unit_cube": True, "seeds": len(args.seeds)}, indent=2))
    if args.dry_run:
        return
    for problem_name in args.problems:
        problem = PROBLEMS[problem_name]
        problem_root = args.output / problem.name
        for seed in args.seeds:
            shift, permutation, digest = seeded_hidden(problem, seed)
            z0 = torch.full((1, problem.dimension), .5, dtype=torch.float32)
            token = f"task_{hashlib.sha256(f'{problem.name}:{seed}'.encode()).hexdigest()[:12]}"
            metadata = {"kind": "synthetic_paper_benchmark", "benchmark_protocol": "paper_table_4",
                        "problem_token": token, "dimension": problem.dimension, "budget": problem.budget,
                        "shared_initial_count": 5, "hidden_transform_sha256": digest,
                        "score_kind": "synthetic bounded objective; not protein TM-score",
                        "paper_alignment": {"unit_cube": True, "ax_sobol_initialization": 5,
                            "shared_initial_for_comparison": True, "qwen_llamacpp": "agentic_dsp" in args.methods},
                        "software_versions": software_versions()}
            run_methods(directory=problem_root / f"seed_{seed}", box=LatentBox(z0, .5),
                generator=SyntheticGenerator(), oracle=SyntheticOracle(problem, shift, permutation),
                metadata=metadata, target_id=token, target_length=problem.dimension, seed=seed,
                budget=problem.budget, initial=5, methods=tuple(args.methods), controller=controller,
                controller_identity=controller_identity,
                acquisition_settings=AcquisitionSettings(raw_samples=256, num_restarts=8, maxiter=60),
                fit_maxiter=80)
        runs = load_runs(problem_root)
        analysis = problem_root / "analysis"
        render(runs, analysis)
        write_paper_summary(runs, analysis, problem)


if __name__ == "__main__":
    main()

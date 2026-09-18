#!/usr/bin/env python3
"""Reproduce Figure 5(a-c): Branin-2D, Ackley-10D, Ackley-20D.

Methods:
- Sobol: scrambled Sobol for the full evaluation budget.
- Ax: 5 Sobol initialization trials, then model-based BO.
- Sara: starts proposing from evaluation 1; no shared Sobol warm-up.

The optimization framework maximizes objectives, while Branin and Ackley are
normally minimization problems. We therefore expose objective = -f(x), so
simple regret is recovered exactly from the maximizing incumbent.
"""

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


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hidden benchmark transformation
# ---------------------------------------------------------------------------

def hidden_transform(
    x: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """
    Hide the textbook optimum by shifting each normalized coordinate by
    at most +/- 1/4 of its range.

    NOTE:
    The paper states that the optimum is relocated per seed by up to
    one quarter of each dimension's range, but does not publish the exact
    implementation of that relocation. This periodic shift preserves the
    objective values and global optimum while keeping the domain [0, 1]^D.
    """
    return (x + shift).remainder(1.0)


def raw_objective(
    problem: Problem,
    x: torch.Tensor,
    shift: torch.Tensor,
) -> float:
    """
    Return the usual minimization objective f(x).

    Branin optimum:
        f* ~= 0.397887

    Ackley optimum:
        f* = 0
    """
    y = hidden_transform(x, shift)

    if problem.name == "branin_2d":
        x1 = 15.0 * y[0] - 5.0
        x2 = 15.0 * y[1]

        value = (
            (
                x2
                - 5.1 * x1.square() / (4.0 * np.pi**2)
                + 5.0 * x1 / np.pi
                - 6.0
            ) ** 2
            + 10.0
            * (1.0 - 1.0 / (8.0 * np.pi))
            * torch.cos(x1)
            + 10.0
        )

        return float(value)

    # Unit cube -> conventional Ackley [-5, 5]^D.
    z = 10.0 * (y - 0.5)

    value = (
        -20.0
        * torch.exp(
            -0.2 * torch.sqrt(torch.mean(z.square()))
        )
        - torch.exp(
            torch.mean(torch.cos(2.0 * np.pi * z))
        )
        + np.e
        + 20.0
    )

    return float(value)


def score(
    problem: Problem,
    x: torch.Tensor,
    shift: torch.Tensor,
) -> float:
    """
    Internal framework maximizes, so optimize -f(x).

    This is an affine sign change only; unlike exp(-f / scale), it does not
    distort the geometry of the benchmark seen by the GP.
    """
    return -raw_objective(problem, x, shift)


def optimum(problem: Problem) -> float:
    """
    Known optimum on our maximizing objective -f(x).
    """
    if problem.name == "branin_2d":
        return -0.397887

    return 0.0


# ---------------------------------------------------------------------------
# Synthetic generator / oracle adapters
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    def decode(self, z, directory):
        directory.mkdir(parents=True, exist_ok=False)

        path = directory / "synthetic_unit_cube.pt"
        torch.save(z.detach().cpu(), path)

        return path


class SyntheticOracle:
    """
    Adapter for the existing protein-oriented optimization pipeline.

    `tm` is used only as the framework's scalar objective slot. It is NOT
    a TM-score for these synthetic experiments.
    """

    objective = "tm"

    def __init__(
        self,
        problem: Problem,
        shift: torch.Tensor,
    ):
        self.problem = problem
        self.shift = shift

    def score(self, path):
        x = (
            torch.load(path, weights_only=True)
            .reshape(-1)
            .double()
        )

        raw = raw_objective(
            self.problem,
            x,
            self.shift,
        )

        objective = -raw

        return {
            "valid": True,

            # Framework maximizes this.
            "tm": objective,

            # Compatibility fields only; not biological metrics.
            "lddt": objective,
            "rmsd": raw,

            "score_kind": (
                "negative raw synthetic objective; "
                "not protein TM-score"
            ),
        }


# ---------------------------------------------------------------------------
# Per-seed hidden transformation
# ---------------------------------------------------------------------------

def seeded_hidden(
    problem: Problem,
    seed: int,
):
    """
    Deterministically derive a different hidden shift for every
    (problem, seed) pair.

    Hashing the problem name avoids accidental correlation between
    Ackley-10D and Ackley-20D.
    """
    key = f"{problem.name}:{seed}".encode()

    seed_bytes = hashlib.sha256(key).digest()[:8]

    derived_seed = (
        int.from_bytes(seed_bytes, "little")
        % (2**63 - 1)
    )

    generator = torch.Generator().manual_seed(
        derived_seed
    )

    shift = (
        torch.rand(
            problem.dimension,
            generator=generator,
            dtype=torch.float64,
        )
        * 0.5
        - 0.25
    )

    digest = hashlib.sha256(
        shift.numpy().tobytes()
    ).hexdigest()

    return shift, digest


# ---------------------------------------------------------------------------
# Figure-5 simple regret aggregation
# ---------------------------------------------------------------------------

def write_paper_summary(
    runs,
    output: Path,
    problem: Problem,
):
    """
    Figure 5 reports:
      median simple regret over 10 seeds
      shaded 25%-75% interquartile range
    """

    by_method = {}

    for run in runs:
        objectives = np.asarray(
            [
                trial["objective"]
                for trial in run["trials"]
            ],
            dtype=float,
        )

        if len(objectives) == 0:
            raise RuntimeError(
                f"No evaluations found for "
                f"{run['method']}"
            )

        # Framework maximizes -f(x).
        incumbent = np.maximum.accumulate(
            objectives
        )

        # For objective g=-f:
        #
        # regret
        #   = f_best - f*
        #   = (-g_best) - (-g*)
        #   = g* - g_best
        regret = optimum(problem) - incumbent

        # Remove tiny floating-point negatives only.
        if np.min(regret) < -1e-7:
            raise RuntimeError(
                f"{run['method']} produced a value "
                f"better than the known optimum. "
                f"Minimum regret={regret.min()}"
            )

        regret = np.maximum(regret, 0.0)

        # If Sara stops before the hard budget, the returned incumbent
        # remains the best known solution for the remainder of the curve.
        if len(regret) < problem.budget:
            regret = np.pad(
                regret,
                (0, problem.budget - len(regret)),
                mode="edge",
            )
        else:
            regret = regret[: problem.budget]

        by_method.setdefault(
            run["method"],
            [],
        ).append(regret)

    rows = []
    finals = []

    for method, curves in sorted(
        by_method.items()
    ):
        values = np.stack(curves)

        for evaluation in range(
            problem.budget
        ):
            column = values[:, evaluation]

            rows.append(
                {
                    "method": method,
                    "evaluations": evaluation + 1,
                    "median_simple_regret": float(
                        np.median(column)
                    ),
                    "q25_simple_regret": float(
                        np.quantile(column, 0.25)
                    ),
                    "q75_simple_regret": float(
                        np.quantile(column, 0.75)
                    ),
                    "seeds": values.shape[0],
                }
            )

        column = values[:, -1]

        finals.append(
            {
                "method": method,
                "evaluations": problem.budget,
                "median_simple_regret": float(
                    np.median(column)
                ),
                "q25_simple_regret": float(
                    np.quantile(column, 0.25)
                ),
                "q75_simple_regret": float(
                    np.quantile(column, 0.75)
                ),
                "seeds": values.shape[0],
            }
        )

    import csv

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, data in (
        (
            "paper_simple_regret.csv",
            rows,
        ),
        (
            "paper_final_summary.csv",
            finals,
        ),
    ):
        if not data:
            continue

        with (
            output / name
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(data[0]),
            )

            writer.writeheader()
            writer.writerows(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--problems",
        nargs="+",
        choices=sorted(PROBLEMS),
        default=sorted(PROBLEMS),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,

        # Figure 5 uses 10 seeds.
        default=list(range(10)),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "sobol",
            "ax",
            "agentic_dsp",
        ),
        default=(
            "sobol",
            "ax",
            "agentic_dsp",
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/paper_synthetic"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Sara client configuration.
    #
    # Client can be reused, but the SaraController itself is instantiated
    # fresh for each independent benchmark run below.
    # ------------------------------------------------------------------

    if (
        "agentic_dsp" in args.methods
        and not args.dry_run
    ):
        client = (
            SaraClient.from_environment()
        )

        prompt = Path(
            "prompts/SARA_SYNTHETIC_SYSTEM.md"
        )

        controller_identity = {
            **client.identity,
            "system_prompt_sha256": sha256(
                prompt
            ),
        }

    else:
        client = None
        prompt = None
        controller_identity = None

    # ------------------------------------------------------------------
    # Print experiment plan.
    # ------------------------------------------------------------------

    config = []

    for problem_name in args.problems:
        problem = PROBLEMS[problem_name]

        for seed in args.seeds:
            config.append(
                {
                    "problem": problem.name,
                    "dimension": problem.dimension,
                    "budget": problem.budget,
                    "seed": seed,
                }
            )

    print(
        json.dumps(
            {
                "problems": config,
                "methods": args.methods,
                "paper_budgets": {
                    key: value.budget
                    for key, value
                    in PROBLEMS.items()
                },
                "unit_cube": True,
                "shared_initial_evaluations": 0,
                "ax_sobol_initialization": 5,
                "sara_starts_at_evaluation": 1,
                "seeds": len(args.seeds),
            },
            indent=2,
        )
    )

    if args.dry_run:
        return

    # ------------------------------------------------------------------
    # Run each benchmark.
    # ------------------------------------------------------------------

    for problem_name in args.problems:
        problem = PROBLEMS[problem_name]

        problem_root = (
            args.output / problem.name
        )

        for seed in args.seeds:

            shift, digest = seeded_hidden(
                problem,
                seed,
            )

            # Unit-cube center; defines the LatentBox geometry only.
            # It is NOT evaluated as a shared initial trial.
            z0 = torch.full(
                (1, problem.dimension),
                0.5,
                dtype=torch.float32,
            )

            token = (
                "task_"
                + hashlib.sha256(
                    f"{problem.name}:{seed}".encode()
                ).hexdigest()[:12]
            )

            # Fresh controller for every independent seed.
            if client is not None:
                controller = SaraController(
                    client,
                    prompt_path=prompt,

                    # Keep the repo's intended per-decision
                    # tool-call limit.
                    max_tools=4,

                    # Enough controller rounds for the largest
                    # 200-evaluation campaign.
                    max_rounds=202,
                )
            else:
                controller = None

            metadata = {
                "kind": (
                    "synthetic_paper_benchmark"
                ),
                "benchmark_protocol": (
                    "figure_5_abc_reimplementation"
                ),
                "problem_token": token,
                "dimension": problem.dimension,
                "budget": problem.budget,

                # Important: no common 5-point warm start.
                "shared_initial_count": 0,

                "hidden_transform_sha256": digest,

                "score_kind": (
                    "negative raw synthetic objective; "
                    "not protein TM-score"
                ),

                "paper_alignment": {
                    "unit_cube": True,

                    # Ax must generate these itself.
                    "ax_sobol_initialization": 5,

                    # Sara receives no forced warm-up.
                    "sara_proposes_from_first_trial": True,

                    # Sobol runs for the whole budget.
                    "sobol_full_budget": True,

                    "shared_initial_for_comparison": False,

                    # Figure 5 aggregation.
                    "target_seed_count": 10,

                    # Local Sara implementation differs from
                    # the paper's default Opus controller.
                    "qwen_llamacpp": (
                        "agentic_dsp"
                        in args.methods
                    ),
                },

                "software_versions": (
                    software_versions()
                ),
            }

            run_methods(
                directory=(
                    problem_root
                    / f"seed_{seed}"
                ),

                box=LatentBox(
                    z0,
                    0.5,
                ),

                generator=SyntheticGenerator(),

                oracle=SyntheticOracle(
                    problem,
                    shift,
                ),

                metadata=metadata,

                target_id=token,
                target_length=problem.dimension,

                seed=seed,
                budget=problem.budget,

                # CRITICAL:
                # No framework-level shared initialization.
                #
                # Sobol starts at trial 1.
                # Sara starts at trial 1.
                # Ax must perform its OWN 5 Sobol trials.
                initial=0,

                methods=tuple(args.methods),

                controller=controller,

                controller_identity=(
                    controller_identity
                ),

                acquisition_settings=(
                    AcquisitionSettings(
                        raw_samples=256,
                        num_restarts=8,
                        maxiter=60,
                    )
                ),

                fit_maxiter=80,
            )

        # --------------------------------------------------------------
        # Aggregate 10 seeds into median + IQR curves.
        # --------------------------------------------------------------

        runs = load_runs(problem_root)

        analysis = (
            problem_root / "analysis"
        )

        render(
            runs,
            analysis,
        )

        write_paper_summary(
            runs,
            analysis,
            problem,
        )


if __name__ == "__main__":
    main()
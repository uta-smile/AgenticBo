"""Two experiment commands and one report command; paths are relative to --root."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(
        description="Run a synthetic benchmark or Boltz target-matching POC"
    )

    sub = p.add_subparsers(
        dest="command",
        required=True,
    )

    for name in ("benchmark", "protein"):
        s = sub.add_parser(name)

        s.add_argument(
            "--root",
            type=Path,
            default=ROOT,
        )

        s.add_argument(
            "--config",
            type=Path,
        )

        s.add_argument(
            "--output",
            type=Path,
        )

        s.add_argument(
            "--seeds",
            nargs="+",
            type=int,
        )

        s.add_argument(
            "--methods",
            nargs="+",
            choices=(
                "sobol",
                "ax",
                "dsp_gp",
                "agentic_dsp",
            ),
        )

        s.add_argument(
            "--budget",
            type=int,
        )

        s.add_argument(
            "--threads",
            type=int,
        )

        # Engineering watchdog only.
        # This is NOT a computational-tool-call limit.
        s.add_argument(
            "--max-rounds",
            type=int,
        )

        s.add_argument(
            "--save-gp",
            action=argparse.BooleanOptionalAction,
            default=None,
        )

        if name == "benchmark":
            s.add_argument(
                "--problems",
                nargs="+",
                choices=(
                    "branin_2d",
                    "ackley_10d",
                    "ackley_20d",
                ),
            )

        else:
            s.add_argument(
                "--targets",
                nargs="+",
                required=True,
            )

            s.add_argument(
                "--manifest",
                type=Path,
            )

            s.add_argument(
                "--initial",
                type=int,
            )

            s.add_argument(
                "--radius",
                type=float,
            )

            s.add_argument(
                "--objective",
                choices=(
                    "tm",
                    "sqrt_tm_times_lddt",
                ),
            )

            s.add_argument(
                "--reference-mode",
                choices=(
                    "strict",
                    "resolved",
                ),
            )

            s.add_argument(
                "--device",
            )

    s = sub.add_parser("report")

    s.add_argument(
        "directory",
        type=Path,
    )

    s.add_argument(
        "--output",
        type=Path,
    )

    return p


def load_config(args):
    root = args.root.resolve()

    default = (
        ROOT
        / "configs"
        / f"{args.command}.yaml"
    )

    config = yaml.safe_load(
        default.read_text(
            encoding="utf-8"
        )
    )

    if args.config:
        custom = yaml.safe_load(
            (root / args.config).read_text(
                encoding="utf-8"
            )
        )

        if (
            not isinstance(custom, dict)
            or set(custom) - set(config)
        ):
            raise ValueError(
                "Config must be a mapping with keys "
                "from the example config"
            )

        for key, value in custom.items():
            if (
                isinstance(config.get(key), dict)
                and isinstance(value, dict)
            ):
                if set(value) - set(config[key]):
                    raise ValueError(
                        f"Unknown {key} settings"
                    )

                config[key].update(value)

            else:
                config[key] = value

    # CLI arguments override YAML.
    for key in config:
        value = getattr(
            args,
            key,
            None,
        )

        if value is not None:
            config[key] = (
                str(value)
                if isinstance(value, Path)
                else value
            )

    if (
        not config["methods"]
        or len(set(config["methods"]))
        != len(config["methods"])
        or set(config["methods"])
        - {
            "sobol",
            "ax",
            "dsp_gp",
            "agentic_dsp",
        }
    ):
        raise ValueError(
            "Select distinct supported methods"
        )

    if (
        not config["seeds"]
        or len(set(config["seeds"]))
        != len(config["seeds"])
        or any(
            not isinstance(seed, int)
            or seed < 0
            for seed in config["seeds"]
        )
    ):
        raise ValueError(
            "Seeds must be distinct "
            "nonnegative integers"
        )

    if (
        config["budget"] is not None
        and config["budget"] < 1
    ):
        raise ValueError(
            "Budget must be positive"
        )

    if config["threads"] < 1:
        raise ValueError(
            "Threads must be positive"
        )

    if config["max_rounds"] < 1:
        raise ValueError(
            "max_rounds must be positive"
        )

    if args.command == "benchmark":
        from .evaluators import PROBLEMS

        if (
            not config["problems"]
            or len(set(config["problems"]))
            != len(config["problems"])
            or set(config["problems"])
            - set(PROBLEMS)
        ):
            raise ValueError(
                "Select distinct supported "
                "benchmark problems"
            )

    if args.command == "protein":
        if (
            len(set(args.targets))
            != len(args.targets)
        ):
            raise ValueError(
                "Select distinct targets"
            )

        radius = config["radius"]

        if (
            radius is None
            or not math.isfinite(radius)
            or not 0 < radius <= 2
        ):
            raise ValueError(
                "Protein POC requires explicit "
                "--radius in (0, 2]; "
                "it is uncalibrated"
            )

        if not (
            2
            <= config["initial"]
            <= config["budget"]
        ):
            raise ValueError(
                "Require 2 <= initial <= budget"
            )

    return root, config


def controller(config, kind):
    if "agentic_dsp" not in config["methods"]:
        return None, None

    from sara.agent import SaraController
    from sara.client import SaraClient

    prompt = (
        ROOT
        / "prompts"
        / (
            "SARA_SYNTHETIC_SYSTEM.md"
            if kind == "benchmark"
            else "SARA_SYSTEM.md"
        )
    )

    client = SaraClient.from_environment()

    identity = {
        **client.identity,
        "prompt_sha256": hashlib.sha256(
            prompt.read_bytes()
        ).hexdigest(),
        "max_rounds": config["max_rounds"],
    }

    return (
        SaraController(
            client,
            prompt_path=prompt,
            max_rounds=config["max_rounds"],
        ),
        identity,
    )


def run(args):
    root, config = load_config(args)

    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    import torch

    from dsp.optimize_acq import (
        AcquisitionSettings,
    )
    from latent.bounds import LatentBox
    from runner.engine import run_methods
    from runner.provenance import (
        software_versions,
        source_digest,
    )
    from .evaluators import (
        PROBLEMS,
        SyntheticEvaluator,
        ProteinEvaluator,
    )
    from .report import report

    torch.set_num_threads(
        config["threads"]
    )

    acq = AcquisitionSettings(
        **config["acquisition"]
    )

    agent, identity = controller(
        config,
        args.command,
    )

    if "ax" in config["methods"]:
        try:
            from ax.service.ax_client import AxClient  # noqa: F401

        except ImportError as exc:
            raise ImportError(
                'Ax is optional; install with '
                'uv pip install -e ".[ax]"'
            ) from exc

    output = (
        root
        / config["output"]
    )

    common = {
        "resolved_config": config,
        "software_versions": software_versions(),
        "source_sha256": source_digest(
            ROOT / "src"
        ),
        "kind": args.command,
    }

    targets = None

    if args.command == "protein":
        from targets.manifest import (
            load_manifest,
            validate_reference,
        )
        from generator.prepare import (
            prepare_target,
        )
        from generator.adapter import (
            Boltz2Adapter,
        )
        from oracle.oracle import (
            StructuralOracle,
        )

        targets = {
            target.target_id: target
            for target in load_manifest(
                root / config["manifest"],
                root,
            )
        }

        if set(args.targets) - set(targets):
            raise ValueError(
                "Unknown target; add "
                "sequence/reference pairs "
                "to the manifest"
            )

        for name in args.targets:
            validate_reference(
                targets[name],
                reference_mode=config[
                    "reference_mode"
                ],
            )

    names = (
        config["problems"]
        if args.command == "benchmark"
        else args.targets
    )

    for name in names:
        if (
            args.command == "benchmark"
            and name not in PROBLEMS
        ):
            raise ValueError(
                f"Unknown benchmark: {name}"
            )

        for seed in config["seeds"]:
            directory = (
                output
                / name
                / f"seed_{seed}"
            )

            metadata = dict(common)

            if args.command == "benchmark":
                evaluator = SyntheticEvaluator(
                    name,
                    seed,
                )

                box = LatentBox(
                    torch.full(
                        (evaluator.dimension,),
                        0.5,
                    ),
                    0.5,
                )

                budget = (
                    config["budget"]
                    or evaluator.budget
                )

                initial = 0

                token = hashlib.sha256(
                    f"opaque:{name}:{seed}".encode()
                ).hexdigest()[:16]

                target_id = token
                target_length = (
                    evaluator.dimension
                )

                metadata.update(
                    evaluator=evaluator.identity,
                    objective_name=(
                        "negative_raw_objective"
                    ),
                    alignment=(
                        "Local LLM, DSP/RBF, "
                        "Ax installed defaults; "
                        "periodic shift; "
                        "paper subset"
                    ),
                )

            else:
                target = targets[name]

                g = config["generator"]

                prepared = prepare_target(
                    target,
                    directory / "prepared",
                    root / g["cache"],
                )

                generator = Boltz2Adapter(
                    prepared,
                    root / g["checkpoint"],
                    device=config["device"],
                    sampling_steps=g[
                        "sampling_steps"
                    ],
                    recycling_steps=g[
                        "recycling_steps"
                    ],
                    sampler_seed=g[
                        "sampler_seed"
                    ],
                )

                z0 = torch.randn(
                    prepared.latent_shape,
                    generator=torch.Generator().manual_seed(
                        seed
                    ),
                    dtype=torch.float32,
                )

                box = LatentBox(
                    z0,
                    config["radius"],
                )

                oracle = StructuralOracle(
                    target.reference,
                    target.sequence,
                    target.reference_chain_id,
                    target.chain_id,
                    objective=config[
                        "objective"
                    ],
                    reference_mode=config[
                        "reference_mode"
                    ],
                )

                evaluator = ProteinEvaluator(
                    box,
                    generator,
                    oracle,
                )

                target_id = name
                target_length = target.length

                budget = config["budget"]
                initial = config["initial"]

                metadata.update(
                    target=target.fingerprint(),
                    generator=generator.config,
                    reference_coverage=(
                        oracle.reference_info
                    ),
                    objective_name=config[
                        "objective"
                    ],
                    calibration={
                        "status": (
                            "uncalibrated_pilot"
                        ),
                        "radius": config[
                            "radius"
                        ],
                    },
                )

            run_methods(
                directory=directory,
                box=box,
                evaluator=evaluator,
                metadata=metadata,
                target_id=target_id,
                target_length=target_length,
                seed=seed,
                budget=budget,
                initial=initial,
                methods=config["methods"],
                controller=agent,
                controller_identity=identity,
                acquisition_settings=acq,
                fit_maxiter=config[
                    "fit_maxiter"
                ],
                save_gp=config["save_gp"],
            )

            if targets is not None:
                del evaluator, generator

        report(
            output / name,
            output / name / "analysis",
        )


def main(argv=None):
    p = parser()

    args = p.parse_args(argv)

    try:
        if args.command == "report":
            from .report import report

            report(
                args.directory,
                args.output
                or args.directory
                / "analysis",
            )

        else:
            run(args)

    except (
        ValueError,
        FileNotFoundError,
        ImportError,
    ) as exc:
        p.exit(
            2,
            f"Error: {exc}\n",
        )
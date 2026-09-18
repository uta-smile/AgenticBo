#!/usr/bin/env python3
"""Measure whether Boltz outputs respond to native initial-noise changes.

This is a development diagnostic. It does not run BO. It decodes the same set
of full-dimensional Gaussian latents under several fixed sampler seeds, repeats
one latent/seed pair, and records structure scores and coordinate differences.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from dsp.sampling import sobol_points
from generator.adapter import Boltz2Adapter
from generator.prepare import prepare_target
from latent.bounds import GaussianLatentSpace
from oracle.oracle import StructuralOracle
from targets.manifest import load_manifest, validate_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("data/targets/manifest.csv"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/boltz"))
    parser.add_argument("--checkpoint", type=Path, default=Path(".cache/boltz/boltz2_conf.ckpt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-seed", type=int, default=0)
    parser.add_argument("--latent-count", type=int, default=4)
    parser.add_argument("--sampler-seeds", nargs="+", type=int, default=[1729, 1730, 1731])
    parser.add_argument("--latent-eps", type=float, default=1.0e-6)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampler-mode", choices=("stochastic", "pf_ode"), default="stochastic")
    parser.add_argument("--reference-mode", choices=("strict", "resolved"), default="resolved")
    args = parser.parse_args()

    if args.latent_count < 2 or not args.sampler_seeds:
        parser.error("Need at least two latents and one sampler seed")

    root = args.root.resolve()
    targets = {target.target_id: target for target in load_manifest(root / args.manifest, root)}
    if args.target not in targets:
        parser.error(f"Unknown target {args.target!r}")
    target = targets[args.target]
    validate_reference(target, reference_mode=args.reference_mode)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = prepare_target(target, output / "prepared", root / args.cache)
    box = GaussianLatentSpace(prepared.latent_shape, eps=args.latent_eps)
    oracle = StructuralOracle(
        target.reference,
        target.sequence,
        target.reference_chain_id,
        target.chain_id,
        objective="tm",
        reference_mode=args.reference_mode,
    )

    x = sobol_points(box.dimension, args.latent_count, args.latent_seed)
    latents = [box.to_native(point).to(torch.float32) for point in x]
    records = []

    generator = Boltz2Adapter(
        prepared,
        root / args.checkpoint,
        device=args.device,
        sampling_steps=args.sampling_steps,
        recycling_steps=args.recycling_steps,
        sampler_seed=args.sampler_seeds[0],
        sampler_mode=args.sampler_mode,
    )

    try:
        for sampler_seed in args.sampler_seeds:
            generator.sampler_seed = sampler_seed
            generator.config["sampler_seed"] = sampler_seed
            for latent_index, latent in enumerate(latents):
                directory = output / f"seed_{sampler_seed}" / f"latent_{latent_index:03d}"
                path = generator.decode(latent, directory)
                score = oracle.score(path)
                coords = torch.load(directory / "atom_coords.pt", weights_only=True)
                records.append({
                    "sampler_seed": sampler_seed,
                    "latent_index": latent_index,
                    "tm": score["tm"],
                    "lddt": score["lddt"],
                    "rmsd": score["rmsd"],
                    "structure": str(path),
                    "coords_path": str(directory / "atom_coords.pt"),
                    "coords_mean": float(coords.mean()),
                    "coords_std": float(coords.std()),
                })

        repeat_dir = output / "repeat" / "same_latent_same_seed"
        generator.sampler_seed = args.sampler_seeds[0]
        generator.config["sampler_seed"] = args.sampler_seeds[0]
        repeat_path = generator.decode(latents[0], repeat_dir)
        repeat_coords = torch.load(repeat_dir / "atom_coords.pt", weights_only=True)
        first = torch.load(
            output / f"seed_{args.sampler_seeds[0]}" / "latent_000" / "atom_coords.pt",
            weights_only=True,
        )

        report = {
            "target": args.target,
            "native_shape": list(prepared.latent_shape),
            "dimension": box.dimension,
            "latent_seed": args.latent_seed,
            "latent_count": args.latent_count,
            "sampler_seeds": args.sampler_seeds,
            "sampler_mode": args.sampler_mode,
            "repeat_max_abs_coordinate_difference": float((first - repeat_coords).abs().max()),
            "records": records,
            "generator_config": generator.config,
            "repeat_structure": str(repeat_path),
        }
    finally:
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

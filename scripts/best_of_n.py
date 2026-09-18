#!/usr/bin/env python3
"""Run an independent stochastic Boltz-2 best-of-N baseline for one target."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

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
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler-seed", type=int, default=1729)
    parser.add_argument("--latent-eps", type=float, default=1.0e-6)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--reference-mode", choices=("strict", "resolved"), default="resolved")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")

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
    generator = Boltz2Adapter(
        prepared,
        root / args.checkpoint,
        device=args.device,
        sampling_steps=args.sampling_steps,
        recycling_steps=args.recycling_steps,
        sampler_seed=args.sampler_seed,
        sampler_mode="stochastic",
    )
    rng = torch.Generator().manual_seed(args.seed)
    records = []
    try:
        for index in range(args.count):
            x = torch.rand(box.dimension, generator=rng, dtype=torch.float64)
            latent = box.to_native(x).to(torch.float32)
            # Each independent generation gets a distinct downstream seed.
            generator.sampler_seed = args.sampler_seed + index
            generator.config["sampler_seed"] = generator.sampler_seed
            directory = output / f"sample_{index:04d}"
            path = generator.decode(latent, directory)
            score = oracle.score(path)
            records.append({
                "index": index,
                "sampler_seed": generator.sampler_seed,
                "tm": score["tm"],
                "lddt": score["lddt"],
                "rmsd": score["rmsd"],
                "structure": str(path),
            })
    finally:
        config = generator.config
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(records, key=lambda row: row["tm"])
    report = {
        "target": args.target,
        "baseline": "best_k_of_n",
        "count": args.count,
        "latent_distribution": "truncated_standard_normal_icdf(Uniform[0,1]^D)",
        "independent_downstream_seeds": True,
        "generator_config": config,
        "best": best,
        "records": records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect Boltz2's real native noise before starting any optimization."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from generator.prepare import prepare_target
from latent.inspect import inspect_native, inspect_source
from targets.manifest import load_manifest, validate_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="Audit installed sampler; leaves target shape and D unknown")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("data/targets/manifest.csv"))
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=Path(".cache/boltz"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-decoder", action="store_true", help="Spend three logged development decodes: repeat z0 twice and perturb once")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--recycling-steps", type=int, default=3)
    args = parser.parse_args()
    if args.source_only:
        if args.target or args.verify_decoder:
            parser.error("--source-only cannot be combined with target decoding")
        output = args.output or args.root / "outputs/source_latent_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        report = inspect_source()
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if not args.target:
        parser.error("Supply --target from the manifest, or use --source-only")
    matches = [t for t in load_manifest(args.root / args.manifest, args.root) if t.target_id == args.target]
    if len(matches) != 1:
        parser.error("Target ID is not registered in the manifest")
    target = matches[0]
    validate_reference(target)
    if args.verify_decoder and target.split != "dev":
        parser.error("Decoder probes require a development target; benchmark generation calls belong to the benchmark budget")
    directory = args.output or args.root / "outputs" / "inspection" / target.target_id / f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    cache = args.root / args.cache
    prepared = prepare_target(target, directory / "prepared", cache)
    z_path = directory / "z0.pt"
    stamp_path = directory / "z0_provenance.json"
    provenance = {"seed": args.seed, "target": target.fingerprint(), "shape": list(prepared.latent_shape)}
    if z_path.exists():
        if not stamp_path.exists() or json.loads(stamp_path.read_text()) != provenance:
            raise ValueError("Existing z0 provenance mismatch; use a new output directory")
        z0 = torch.load(z_path, weights_only=True)
    else:
        if stamp_path.exists():
            raise ValueError("Incomplete z0 artifact; use a new output directory")
        z0 = torch.randn(prepared.latent_shape, generator=torch.Generator().manual_seed(args.seed), dtype=torch.float32)
        # Exclusive creation preserves initialization across methods and reruns.
        with z_path.open("xb") as handle:
            torch.save(z0, handle)
        stamp_path.write_text(json.dumps(provenance, indent=2) + "\n")
    report = {**inspect_native(z0, prepared.batch["atom_pad_mask"]), **provenance,
              "decoder_probe_calls": 0}
    report_path = directory / "latent_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.verify_decoder:
        from generator.adapter import Boltz2Adapter

        generator = Boltz2Adapter(prepared, cache / "boltz2_conf.ckpt", device=args.device,
                                 sampling_steps=args.sampling_steps, recycling_steps=args.recycling_steps)
        probe_dir = directory / "decoder_probe"
        probe_dir.mkdir(exist_ok=False)
        try:
            generator.decode(z0, probe_dir / "repeat_a")
            generator.decode(z0, probe_dir / "repeat_b")
            perturbed = z0 + 0.1 * torch.randn(z0.shape, generator=torch.Generator().manual_seed(args.seed + 1))
            generator.decode(perturbed, probe_dir / "perturbed")
            a, b, c = [torch.load(probe_dir / p / "atom_coords.pt", weights_only=True)
                       for p in ("repeat_a", "repeat_b", "perturbed")]
            report.update(generation_deterministic_adapter=torch.equal(a, b),
                          repeat_max_abs_difference=(a - b).abs().max().item(),
                          perturbation_max_abs_difference=(a - c).abs().max().item(),
                          valid_generation_under_perturbations="finite output for one perturbation; structural validity and radius calibration still required",
                          generator_config=generator.config,
                          status="native_shape_and_decoder_probed")
        except Exception as exc:
            report.update(status="decoder_probe_failed", decoder_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report["decoder_probe_calls"] = generator.calls
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

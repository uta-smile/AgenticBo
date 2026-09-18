#!/usr/bin/env python3
"""Calibrate and freeze one full-D radius using development proteins only."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from generator.adapter import Boltz2Adapter
from generator.prepare import prepare_target
from latent.bounds import LatentBox
from latent.inspect import inspect_native
from oracle.oracle import StructuralOracle
from oracle.rmsd import rmsd
from oracle.validation import read_protein
from targets.manifest import load_manifest, sha256, validate_reference
from runner.provenance import source_digest, software_versions
from runner.calibration import summarize_radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("data/targets/manifest.csv"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/boltz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-valid-rate", type=float, default=0.9)
    parser.add_argument("--min-geometry-rate", type=float, default=0.9)
    parser.add_argument("--min-median-rmsd", type=float, default=0.5)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not 20 <= args.samples <= 50:
        parser.error("Calibration requires 20–50 samples per radius")
    if not (0 <= args.min_valid_rate <= 1 and 0 <= args.min_geometry_rate <= 1
            and np.isfinite(args.min_median_rmsd) and args.min_median_rmsd > 0):
        parser.error("Require rates in [0,1] and a positive finite variation threshold")
    targets = [t for t in load_manifest(args.root / args.manifest, args.root) if t.split == "dev"]
    if not targets:
        parser.error("Register development FASTA/reference pairs before calibration")
    # Finish all setup validation before the first expensive call.
    for target in targets:
        validate_reference(target)
        StructuralOracle(target.reference, target.sequence, target.reference_chain_id, target.chain_id)
    output, cache = args.root / args.output, args.root / args.cache
    output.mkdir(parents=True, exist_ok=False)
    radii = [0.1, 0.25, 0.5, 1.0, 2.0]
    settings = {"radii": radii, "samples_per_radius": args.samples, "seed": args.seed,
                "min_valid_rate": args.min_valid_rate, "min_geometry_rate": args.min_geometry_rate,
                "min_median_center_rmsd_angstrom": args.min_median_rmsd,
                "geometry_rule": "at least 95% of adjacent C-alpha distances lie in [2.5,5.0] A",
                "selection_rule": "smallest radius passing all criteria on every development target",
                "target_fingerprints": [t.fingerprint() for t in targets],
                "center_decodes": 2,
                "oracle_source_sha256": source_digest(Path(__file__).resolve().parents[1] / "src" / "oracle"),
                "software_versions": software_versions(),
                "expected_calls": len(targets) * (2 + len(radii) * args.samples)}
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    summaries, generators = {}, {}
    attempts = 0
    journal_path = output / "trials.jsonl"
    for target in targets:
        directory = output / target.target_id
        prepared = prepare_target(target, directory / "prepared", cache)
        generator = Boltz2Adapter(prepared, cache / "boltz2_conf.ckpt", device=args.device)
        generators[target.target_id] = generator.config
        oracle = StructuralOracle(target.reference, target.sequence, target.reference_chain_id, target.chain_id)
        rng = torch.Generator().manual_seed(args.seed)
        z0 = torch.randn(prepared.latent_shape, generator=rng)
        torch.save(z0, directory / "z0.pt")
        latent_report = inspect_native(z0, prepared.batch["atom_pad_mask"])
        latent_report_path = directory / "latent_report.json"
        latent_report_path.write_text(json.dumps(latent_report, indent=2) + "\n")
        # Reuse normalized points across radii for a paired development comparison.
        points = torch.rand(args.samples, z0.numel(), generator=rng, dtype=torch.float64)
        torch.save(points, directory / "normalized_points.pt")

        def evaluate(latent, name, radius, index, center_coords=None):
            nonlocal attempts
            trial_dir = directory / name
            attempts += 1
            record = {"call_id": attempts, "target_id": target.target_id, "radius": radius,
                      "sample_index": index, "status": "started", "output": str(trial_dir)}
            # An interrupted call remains visible and is never silently discarded.
            with journal_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            coords = None
            try:
                generated = generator.decode(latent, trial_dir)
                result = oracle.score(generated)
                if result["valid"]:
                    coords = read_protein(generated, target.chain_id, target.sequence).ca
            except Exception as exc:
                result = oracle.invalid(f"{type(exc).__name__}: {exc}")
            geometry = False
            if coords is not None:
                distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                geometry = float(np.mean((distances >= 2.5) & (distances <= 5.0))) >= 0.95
            variation = rmsd(coords, center_coords) if coords is not None and center_coords is not None else None
            record.update(status="completed", **result, sensible_geometry=geometry, center_rmsd_angstrom=variation)
            with journal_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record, coords

        _, center = evaluate(z0, "center", 0.0, -1)
        _, repeated_center = evaluate(z0, "center_repeat", 0.0, -2)
        deterministic = (center is not None and repeated_center is not None and torch.equal(
            torch.load(directory / "center" / "atom_coords.pt", weights_only=True),
            torch.load(directory / "center_repeat" / "atom_coords.pt", weights_only=True)))
        latent_report.update(generation_deterministic_adapter=deterministic, generator_config=generator.config,
                             status="development_native_shape_and_repeatability_measured")
        latent_report_path.write_text(json.dumps(latent_report, indent=2) + "\n")
        if not deterministic:
            raise RuntimeError("Development generator repeats failed or differ; inspect the two charged probes before calibrating deterministic BO")
        summaries[target.target_id] = {}
        for radius in radii:
            box = LatentBox(z0, radius)
            observations = []
            for index, x in enumerate(points):
                record, _ = evaluate(box.to_native(x), f"radius_{radius}/trial_{index:03d}", radius, index, center)
                observations.append(record)
            summary = summarize_radius(observations, settings)
            summaries[target.target_id][str(radius)] = summary
            print(target.target_id, radius, json.dumps(summary), flush=True)
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    eligible = [r for r in radii if all(s[str(r)]["qualifies"] for s in summaries.values())]
    report = {"settings": settings, "summaries": summaries, "generators": generators,
              "calls": attempts, "selected_radius": min(eligible) if eligible else None,
              "journal_sha256": sha256(journal_path)}
    report_path = output / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not eligible:
        raise SystemExit("No radius passed. No frozen radius was created; inspect development results.")
    frozen = {"radius": min(eligible), "calibration_report_sha256": sha256(report_path),
              "generator_configs": generators, "development_targets": settings["target_fingerprints"],
              "scope": "one shared full-dimensional radius for all benchmark targets and methods"}
    (output / "frozen_radius.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()

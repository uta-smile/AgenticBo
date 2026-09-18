#!/usr/bin/env python3
"""Three real checkpoint decodes of a tiny synthetic fixture, not a benchmark."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from generator.adapter import Boltz2Adapter
from generator.prepare import prepare_target
from latent.inspect import inspect_native
from targets.manifest import Target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/generator_smoke"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/boltz"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fasta, reference = args.output / "synthetic.fasta", args.output / "synthetic.pdb"
    fasta.write_text(">synthetic_adapter_fixture\nACDEFGHIK\n")
    names = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS"]
    reference.write_text("".join(
        f"ATOM  {i:5d}  CA  {name} A{i:4d}    {i*3.8:8.3f}{0:8.3f}{0:8.3f}  1.00 20.00           C\n"
        for i, name in enumerate(names, 1)) + "END\n")
    target = Target("synthetic", fasta.resolve(), reference.resolve(), "A", "A", 9,
                    "dev", None, "synthetic adapter fixture; linear reference is not ground truth")
    prepared = prepare_target(target, args.output / "prepared", args.cache)
    z = torch.randn(prepared.latent_shape, generator=torch.Generator().manual_seed(0))
    torch.save(z, args.output / "z0.pt")
    report = inspect_native(z, prepared.batch["atom_pad_mask"])
    report.update(scope="tiny synthetic adapter smoke, not protein benchmark or calibration",
                  expensive_calls=0, passed=False)
    report_path = args.output / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    generator = None
    try:
        generator = Boltz2Adapter(prepared, args.cache / "boltz2_conf.ckpt", device=args.device,
                                 sampling_steps=3, recycling_steps=0)
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        for name, latent in [("repeat_a", z), ("repeat_b", z),
                             ("perturbed", z + 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(1)))]:
            print(f"Decoding {name} at D={z.numel()}", flush=True)
            generator.decode(latent, args.output / name)
        a, b, c = [torch.load(args.output / name / "atom_coords.pt", weights_only=True)
                   for name in ("repeat_a", "repeat_b", "perturbed")]
        report.update(repeat_equal=torch.equal(a, b), repeat_max_abs_difference=(a-b).abs().max().item(),
                      status="native_shape_and_decoder_probed_on_synthetic_fixture",
                      generation_deterministic_adapter=torch.equal(a, b),
                      perturbation_max_abs_difference=(a-c).abs().max().item(),
                      frozen_parameters=all(not p.requires_grad for p in generator.model.parameters()),
                      generator_config=generator.config)
        report["passed"] = report["repeat_equal"] and report["frozen_parameters"] and not torch.equal(a, c)
        if str(args.device).startswith("cuda"):
            report["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(args.device)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["expensive_calls"] = generator.calls if generator else 0
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

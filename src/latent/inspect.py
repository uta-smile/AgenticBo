"""Report measured native shape; keep unverified decoder properties explicit."""
from __future__ import annotations

import math
from importlib.metadata import version

import torch

from generator.native_noise import sampler_source_audit


def inspect_source() -> dict:
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    return {
        "status": "source_inspected_target_not_measured", "generator": "boltz2",
        "boltz_version": version("boltz"), "native_latent_shape": None, "flattened_D": None,
        "shape_rule": "[1, atom_pad_mask.shape[-1], 3]", "dtype": "torch.float32",
        "distribution": "independent standard Normal(0,1) before init_sigma scaling",
        "latent_role": "per-atom initial diffusion noise, including native padding",
        "bounded": False, "native_bounds": None,
        "initial_coordinate_scale": "sigma_max * sigma_data (2560 at pinned defaults)",
        "arbitrary_perturbations_interface": True, "valid_generation_under_perturbations": None,
        "generation_deterministic_upstream": False,
        "generation_deterministic_adapter": None,
        "randomness_policy": "fixed sampler seed; preserve and consume original initialization draw",
        "notes": ["D is target-specific and must be measured from actual featurized atom_pad_mask.",
                  "The generator centers and augments coordinates; no dimensionality is removed by the optimizer.",
                  "Only initial noise is optimized. All subsequent diffusion noise and augmentation draws are frozen.",
                  "Repeat decoding is required to establish empirical determinism."],
        **sampler_source_audit(AtomDiffusion.sample),
    }


def inspect_native(latent: torch.Tensor, atom_mask: torch.Tensor) -> dict:
    if tuple(latent.shape) != (*atom_mask.shape, 3) or latent.shape[0] != 1:
        raise ValueError("Latent and native atom mask shapes differ")
    if not torch.isfinite(latent).all() or not latent.numel():
        raise ValueError("Latent must be nonempty and finite")
    report = inspect_source()
    report.update({"status": "native_shape_measured_decoder_unverified",
                   "native_latent_shape": list(latent.shape), "flattened_D": latent.numel(),
                   "dtype": str(latent.dtype), "mean": latent.mean().item(),
                   "std": latent.std(correction=0).item(), "min": latent.min().item(), "max": latent.max().item(),
                   "active_atoms": int(atom_mask.sum()), "padded_atoms": atom_mask.shape[-1],
                   "dsp_loc": math.sqrt(2) + 0.5 * math.log(latent.numel()),
                   "dsp_scale": math.sqrt(3)})
    return report

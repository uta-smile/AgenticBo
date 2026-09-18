"""Persistent frozen Boltz2 model, using FP32 for Maxwell GPUs."""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from generator.native_noise import fixed_randomness, inject_native_noise, sampler_source_audit
from generator.prepare import PreparedTarget
from targets.manifest import sha256


class Boltz2Adapter:
    def __init__(self, prepared: PreparedTarget, checkpoint: Path, *, device: str = "cuda:0",
                 sampling_steps: int = 200, recycling_steps: int = 3, sampler_seed: int = 1729):
        from boltz.main import Boltz2, Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2

        if sampling_steps < 2 or recycling_steps < 0:
            raise ValueError("Need at least two diffusion steps and nonnegative recycling")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing Boltz2 checkpoint: {checkpoint}")
        # Required before CUDA context creation for deterministic cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.prepared, self.latent_shape = prepared, prepared.latent_shape
        self.device = torch.device(device)
        self.sampler_seed = sampler_seed
        self.calls = 0
        self.config = {
            "checkpoint_sha256": sha256(checkpoint), "sampling_steps": sampling_steps,
            "adapter_source_sha256": sha256(Path(__file__)),
            "native_noise_source_sha256": sha256(Path(__file__).with_name("native_noise.py")),
            "recycling_steps": recycling_steps, "sampler_seed": sampler_seed,
            "precision": "float32", "use_kernels": False, "diffusion_samples": 1,
        }
        self.model = Boltz2.load_from_checkpoint(
            checkpoint, strict=True, map_location="cpu", ema=False, use_kernels=False,
            compile_pairformer=False, compile_structure=False, compile_confidence=False,
            compile_affinity=False, compile_msa=False, compile_templates=False,
            predict_args={"recycling_steps": recycling_steps, "sampling_steps": sampling_steps,
                          "diffusion_samples": 1, "max_parallel_samples": 1,
                          "write_confidence_summary": True, "write_full_pae": False, "write_full_pde": False},
            diffusion_process_args=asdict(Boltz2DiffusionParams()),
            pairformer_args=asdict(PairformerArgsV2()),
            msa_args=asdict(MSAModuleArgs(subsample_msa=False, use_paired_feature=True)),
            steering_args=asdict(BoltzSteeringParams()),
        ).eval().float().requires_grad_(False).to(self.device)
        self.config.update(sampler_source_audit(self.model.structure_module.sample))

    def decode(self, latent: torch.Tensor, output_dir: Path) -> Path:
        from boltz.data.write.writer import BoltzWriter

        if tuple(latent.shape) != self.latent_shape or not torch.isfinite(latent).all():
            raise ValueError("Latent must have the complete native shape and finite coordinates")
        output_dir.mkdir(parents=True, exist_ok=False)
        batch = {key: value.detach().clone().to(self.device) if isinstance(value, torch.Tensor)
                 else copy.deepcopy(value) for key, value in self.prepared.batch.items()}
        self.calls += 1
        started = time.monotonic()
        receipt = {"call": self.calls, "status": "started", "generator": self.config}
        receipt_path = output_dir / "generation.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        try:
            with fixed_randomness(self.sampler_seed), torch.inference_mode(), \
                    inject_native_noise(self.model.structure_module, latent):
                # No Lightning Trainer: its Boltz2 CLI default is bf16-mixed.
                prediction = self.model.predict_step(batch, 0)
            if prediction.get("exception"):
                raise RuntimeError("Boltz2 generation failed (including possible GPU out of memory)")
            coords = prediction["coords"]
            if not torch.isfinite(coords).all():
                raise RuntimeError("Boltz2 returned nonfinite atom coordinates")
            torch.save(coords.detach().cpu(), output_dir / "atom_coords.pt")
            writer = BoltzWriter(data_dir=self.prepared.directory / "processed" / "structures",
                                 output_dir=output_dir, output_format="mmcif", boltz2=True)
            writer.write_on_batch_end(None, self.model, prediction, [0], batch, 0, 0)
            structures = list(output_dir.glob("*/*_model_0.cif"))
            if len(structures) != 1:
                raise RuntimeError("Expected exactly one generated mmCIF")
            receipt.update(status="complete", structure=str(structures[0]))
            return structures[0]
        except Exception as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            receipt["runtime_seconds"] = time.monotonic() - started
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

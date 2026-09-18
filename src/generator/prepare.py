"""Prepare actual Boltz2 features from a validated single-chain target."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch

from generator.native_noise import SUPPORTED_BOLTZ, fixed_randomness
from targets.manifest import Target, write_boltz_input


@dataclass
class PreparedTarget:
    target: Target
    directory: Path
    batch: dict

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        mask = self.batch["atom_pad_mask"]
        if mask.ndim != 2 or mask.shape[0] != 1 or mask.shape[1] == 0:
            raise ValueError("Expected one nonempty atom mask")
        return (*mask.shape, 3)


def prepare_target(target: Target, directory: Path, cache: Path) -> PreparedTarget:
    from boltz.data import const
    from boltz.main import Boltz2InferenceDataModule, Manifest, process_inputs

    if version("boltz") != SUPPORTED_BOLTZ:
        raise RuntimeError(f"Expected boltz=={SUPPORTED_BOLTZ}")
    directory, cache = directory.resolve(), cache.resolve()
    mol_dir = cache / "mols"
    missing = [name for name in const.canonical_tokens if not (mol_dir / f"{name}.pkl").is_file()]
    if missing:
        raise FileNotFoundError(f"Boltz canonical molecule files are missing under {mol_dir}; fetch the official Boltz2 assets first")
    fingerprint = target.fingerprint()
    stamp = directory / "target.json"
    if directory.exists():
        if not stamp.exists() or json.loads(stamp.read_text()) != fingerprint:
            raise ValueError("Prepared target provenance differs or is missing; use a new output directory")
    else:
        directory.mkdir(parents=True)
        stamp.write_text(json.dumps(fingerprint, indent=2) + "\n")
    input_path = directory / f"{target.target_id}.yaml"
    write_boltz_input(target, input_path)
    process_inputs(data=[input_path], out_dir=directory, ccd_path=cache / "ccd.pkl",
                   mol_dir=mol_dir, msa_server_url="", msa_pairing_strategy="greedy",
                   use_msa_server=False, boltz2=True, preprocessing_threads=1)
    processed = directory / "processed"
    manifest = Manifest.load(processed / "manifest.json")
    if len(manifest.records) != 1 or manifest.records[0].id != target.target_id:
        raise RuntimeError("Boltz input preparation failed; no substitute target is allowed")
    module = Boltz2InferenceDataModule(
        manifest=manifest, target_dir=processed / "structures", msa_dir=processed / "msa",
        mol_dir=mol_dir, num_workers=0, constraints_dir=processed / "constraints",
        template_dir=processed / "templates", extra_mols_dir=processed / "mols")
    with fixed_randomness(42):
        batch = next(iter(module.predict_dataloader()))
    if len(batch["record"]) != 1 or batch["record"][0].id != target.target_id:
        raise RuntimeError("Featurizer returned the wrong target")
    if int(batch["token_pad_mask"].sum()) != target.length:
        raise ValueError("Boltz token count differs from the target length")
    prepared = PreparedTarget(target, directory, batch)
    prepared.latent_shape  # Validate the actual sampler input shape.
    return prepared

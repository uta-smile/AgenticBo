"""Read user FASTA/reference pairs without leaking references into generation."""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path) -> tuple[str, str]:
    records: list[tuple[str, list[str]]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if not line[1:].strip():
                raise ValueError(f"Empty FASTA header: {path}")
            records.append((line[1:].strip(), []))
        elif not records:
            raise ValueError(f"Sequence before FASTA header: {path}")
        else:
            records[-1][1].append(line.upper())
    if len(records) != 1:
        raise ValueError(f"Expected exactly one FASTA record: {path}")
    name, lines = records[0]
    sequence = "".join(lines)
    if not sequence or set(sequence) - AMINO_ACIDS:
        raise ValueError(f"Expected a nonempty standard protein sequence: {path}")
    return name, sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Target:
    target_id: str
    fasta: Path
    reference: Path
    chain_id: str
    reference_chain_id: str
    length: int
    split: str
    msa: Path | None
    provenance: str

    @property
    def sequence(self) -> str:
        return read_fasta(self.fasta)[1]

    def fingerprint(self) -> dict:
        return {
            "target_id": self.target_id,
            "fasta_sha256": sha256(self.fasta),
            "reference_sha256": sha256(self.reference),
            "msa_sha256": sha256(self.msa) if self.msa else None,
            "chain_id": self.chain_id,
            "reference_chain_id": self.reference_chain_id,
            "length": self.length,
            "split": self.split,
        }


def load_manifest(path: Path, root: Path) -> list[Target]:
    required = {"target_id", "fasta", "reference", "chain_id", "reference_chain_id", "length", "split", "msa", "provenance"}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"Manifest columns must be {sorted(required)}")
        rows = list(reader)
    targets = []
    ids = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed manifest row")
        row = {key: value.strip() for key, value in row.items()}
        target_id = row["target_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", target_id) or target_id in ids:
            raise ValueError(f"Invalid or duplicate target ID: {target_id}")
        ids.add(target_id)
        if row["split"] not in {"dev", "test"}:
            raise ValueError("Split must be dev or test")
        if not row["chain_id"] or not row["reference_chain_id"]:
            raise ValueError("Both chain IDs must be explicit")
        paths = {}
        for key in ("fasta", "reference", "msa"):
            if not row[key] and key == "msa":
                paths[key] = None
                continue
            paths[key] = (root / row[key]).resolve()
            if not paths[key].is_file():
                raise ValueError(f"Missing {key}: {paths[key]}")
        if paths["reference"].suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError("Reference must be PDB or mmCIF")
        target = Target(target_id=target_id, **paths, chain_id=row["chain_id"],
                        reference_chain_id=row["reference_chain_id"], length=int(row["length"]),
                        split=row["split"], provenance=row["provenance"])
        if len(target.sequence) != target.length:
            raise ValueError(f"FASTA length differs from manifest for {target_id}")
        targets.append(target)
    dev_sequences = {t.sequence for t in targets if t.split == "dev"}
    test_sequences = {t.sequence for t in targets if t.split == "test"}
    if dev_sequences & test_sequences:
        raise ValueError("Development and test sequences overlap")
    return targets


def validate_reference(target: Target, *, reference_mode: str = "strict") -> dict:
    """Require complete, unambiguous residue correspondence for the first POC."""
    import numpy as np
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.SeqUtils import seq1

    if reference_mode=="resolved":
        from oracle.reference import read_resolved_reference
        _,_,info=read_resolved_reference(target.reference,target.reference_chain_id,target.sequence)
        return {"residues":info["resolved_residues"],"reference_chain_id":target.reference_chain_id,
                "reference_coverage":info}
    if reference_mode!="strict":
        raise ValueError("Reference mode must be strict or resolved")

    parser = PDBParser(QUIET=True, PERMISSIVE=False) if target.reference.suffix.lower() == ".pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure(target.target_id, str(target.reference))
    models = list(structure)
    if len(models) != 1:
        raise ValueError("Reference must contain exactly one model")
    model = models[0]
    if target.reference_chain_id not in model:
        raise ValueError(f"Missing reference chain {target.reference_chain_id}")
    residues = [r for r in model[target.reference_chain_id] if r.id[0] == " "]
    sequence = "".join(seq1(r.resname) for r in residues)
    if sequence != target.sequence:
        raise ValueError("Reference residue sequence must exactly match FASTA")
    if any("CA" not in r for r in residues):
        raise ValueError("Missing reference C-alpha coordinates")
    if any(r.is_disordered() or r["CA"].is_disordered() for r in residues):
        raise ValueError("Resolve alternate reference conformations before benchmarking")
    coords = np.asarray([r["CA"].coord for r in residues])
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite reference coordinates")
    return {"residues": len(residues), "reference_chain_id": target.reference_chain_id,
            "residue_ids": [[r.id[1], r.id[2]] for r in residues]}


def write_boltz_input(target: Target, output: Path) -> None:
    import yaml

    # The reference is intentionally absent from this schema.
    payload = {"version": 1, "sequences": [{"protein": {
        "id": target.chain_id, "sequence": target.sequence,
        "msa": str(target.msa) if target.msa else "empty",
    }}]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))

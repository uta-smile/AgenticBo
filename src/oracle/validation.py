from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1


@dataclass(frozen=True)
class ProteinStructure:
    sequence: str
    ca: np.ndarray
    atoms: dict[tuple[int, str], np.ndarray]
    residue_names: tuple[str, ...]


def read_protein(path: Path, chain_id: str, sequence: str) -> ProteinStructure:
    if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError("Structure must be PDB or mmCIF")
    parser = PDBParser(QUIET=True, PERMISSIVE=False) if path.suffix.lower() == ".pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", str(path))
    models = list(structure)
    if len(models) != 1 or chain_id not in models[0]:
        raise ValueError("Expected exactly one model with the requested chain")
    residues = [r for r in models[0][chain_id] if r.id[0] == " "]
    observed = "".join(seq1(r.resname) for r in residues)
    if not residues or observed != sequence:
        raise ValueError("Residue mapping does not match the fixed target sequence")
    atoms, ca = {}, []
    for index, residue in enumerate(residues):
        if residue.is_disordered() or "CA" not in residue:
            raise ValueError("Ambiguous residue or missing C-alpha")
        for atom in residue:
            if atom.is_disordered():
                raise ValueError("Resolve alternate atom conformations before scoring")
            coordinates = np.asarray(atom.coord, dtype=np.float64)
            if not np.isfinite(coordinates).all():
                raise ValueError("Nonfinite structure coordinates")
            if atom.element not in {"H", "D"}:
                atoms[index, atom.name] = coordinates
        ca.append(atoms[index, "CA"])
    return ProteinStructure(observed, np.asarray(ca), atoms, tuple(r.resname for r in residues))

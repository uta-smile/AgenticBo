from __future__ import annotations

from pathlib import Path
import time
from Bio.PDB.PDBExceptions import PDBConstructionException, PDBException

from oracle.lddt import LDDT
from oracle.rmsd import rmsd
from oracle.tm_score import tm_score
from oracle.validation import read_protein
from oracle.objectives import DEFAULT_OBJECTIVE, objective_value, validate_objective
from oracle.reference import read_resolved_reference, select_residues


class StructuralOracle:
    def __init__(self, reference_path: Path, sequence: str, reference_chain: str, generated_chain: str,
                 *, objective: str = DEFAULT_OBJECTIVE, reference_mode: str = "strict"):
        self.objective = validate_objective(objective)
        if len(sequence) < 3:
            raise ValueError("Target must have at least three residues for TM-align")
        self.sequence, self.generated_chain = sequence, generated_chain
        # Bad reference data is a setup failure, never a zero-score observation.
        if reference_mode=="resolved":
            self.reference,self.reference_indices,self.reference_info=read_resolved_reference(reference_path,reference_chain,sequence)
        elif reference_mode=="strict":
            self.reference = read_protein(reference_path, reference_chain, sequence)
            self.reference_indices=tuple(range(len(sequence)))
            self.reference_info={"mode":"strict","full_sequence_length":len(sequence),
                "resolved_residues":len(sequence),"coverage":1.,"tm_normalization_length":len(sequence)}
        else:
            raise ValueError("Reference mode must be strict or resolved")
        self.lddt = LDDT(self.reference)

    def score(self, generated_path: Path) -> dict:
        started = time.monotonic()
        try:
            model = read_protein(generated_path, self.generated_chain, self.sequence)
            model = select_residues(model,self.reference_indices)
            tm = tm_score(model.ca, self.reference.ca, self.reference.sequence)
            lddt, details = self.lddt.score(model)
            result = {"valid": True, "tm": tm, "lddt": lddt,
                      "rmsd": rmsd(model.ca, self.reference.ca), "gdt_ha": None,
                      "objective": objective_value(tm, lddt, self.objective), "lddt_details": details,
                      "tm_definition": "TM-align on reference-covered positions, normalized by resolved reference length"}
        except (OSError, ValueError, KeyError, IndexError, PDBException, PDBConstructionException) as exc:
            result = self.invalid(f"{type(exc).__name__}: {exc}")
        result["oracle_runtime_seconds"] = time.monotonic() - started
        result["reference_coverage"] = self.reference_info
        return result

    @staticmethod
    def invalid(reason: str) -> dict:
        return {"valid": False, "tm": 0.0, "lddt": 0.0, "objective": 0.0,
                "rmsd": None, "gdt_ha": None, "error": reason}

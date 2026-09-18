"""All-heavy-atom distance conservation, without stereochemical filtering.

Reference pairs are from different residues, within 15 A. Missing model atoms
count as nonconserved. Equivalent side-chain names are optimized with local
residue swaps until no improving swap remains; convergence is reported.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from oracle.validation import ProteinStructure

EQUIVALENT_NAMES = {
    "ASP": [("OD1", "OD2")], "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")], "LEU": [("CD1", "CD2")], "VAL": [("CG1", "CG2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}


class LDDT:
    def __init__(self, reference: ProteinStructure, inclusion_radius: float = 15.0):
        self.keys = sorted(reference.atoms)
        self.reference = reference
        coords = np.asarray([reference.atoms[key] for key in self.keys])
        pairs = cKDTree(coords).query_pairs(inclusion_radius, output_type="ndarray")
        residues = np.asarray([key[0] for key in self.keys])
        pairs = pairs[residues[pairs[:, 0]] != residues[pairs[:, 1]]]
        if not len(pairs):
            raise ValueError("Reference has no inter-residue contacts for lDDT")
        distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        keep = distances < inclusion_radius
        self.pairs, self.distances = pairs[keep], distances[keep]
        if not len(self.pairs):
            raise ValueError("Reference has no contacts strictly within the lDDT radius")
        self.thresholds = np.asarray([0.5, 1.0, 2.0, 4.0])
        lookup = {key: i for i, key in enumerate(self.keys)}
        self.swaps = []
        for index, name in enumerate(reference.residue_names):
            swap = [(lookup[index, left], lookup[index, right])
                    for left, right in EQUIVALENT_NAMES.get(name, [])
                    if (index, left) in lookup and (index, right) in lookup]
            if swap:
                # Aromatic ring swaps must move both CD and CE names together.
                changed = np.asarray([atom for pair in swap for atom in pair])
                affected = np.isin(self.pairs, changed).any(axis=1)
                self.swaps.append((swap, affected))

    def _counts(self, coords, present, subset=None):
        pairs = self.pairs if subset is None else self.pairs[subset]
        reference_distances = self.distances if subset is None else self.distances[subset]
        distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        errors = np.abs(distances - reference_distances)
        valid = present[pairs[:, 0]] & present[pairs[:, 1]]
        return ((errors[:, None] < self.thresholds) & valid[:, None]).sum(axis=1)

    def score(self, model: ProteinStructure) -> tuple[float, dict]:
        if model.sequence != self.reference.sequence:
            raise ValueError("lDDT requires the fixed target residue mapping")
        present = np.asarray([key in model.atoms for key in self.keys])
        coords = np.asarray([model.atoms.get(key, np.zeros(3)) for key in self.keys])
        counts = self._counts(coords, present)
        swaps = 0
        # Each accepted swap strictly increases the integer conserved-distance
        # count, so this deterministic finite procedure terminates.
        while True:
            improved = False
            for swap, affected in self.swaps:
                old = int(counts[affected].sum())
                for left, right in swap:
                    coords[[left, right]], present[[left, right]] = coords[[right, left]], present[[right, left]]
                proposed = self._counts(coords, present, affected)
                if proposed.sum() > old:
                    counts[affected] = proposed
                    swaps += 1
                    improved = True
                else:
                    for left, right in swap:
                        coords[[left, right]], present[[left, right]] = coords[[right, left]], present[[right, left]]
            if not improved:
                break
        return float(counts.sum() / (4 * len(self.pairs))), {
            "definition": "all-heavy-atom inter-residue distance lDDT; no stereochemical checks",
            "reference_atoms": len(self.keys), "missing_model_atoms": int((~present).sum()),
            "reference_pairs": len(self.pairs), "symmetry_swaps": swaps,
            "symmetry_method": "deterministic improving residue swaps to a local optimum",
        }

import math

import numpy as np
import pytest

from oracle.lddt import LDDT
from oracle.oracle import StructuralOracle
from oracle.rmsd import rmsd
from oracle.validation import ProteinStructure


def write_structure(path, rotation=None, offset=None, missing_ca=False):
    names = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS"]
    lines = []
    serial = 0
    for residue, name in enumerate(names, 1):
        origin = np.array([residue * 2.0, math.sin(residue)*2, math.cos(residue)*2])
        for atom, shift in [("N", [-0.5, 0, 0]), ("CA", [0, 0, 0]), ("C", [0.5, 0, 0])]:
            if missing_ca and residue == 3 and atom == "CA":
                continue
            serial += 1
            coord = origin + shift
            if rotation is not None:
                coord = coord @ rotation
            if offset is not None:
                coord += offset
            x, y, z = coord
            lines.append(f"ATOM  {serial:5d} {atom:^4s} {name} A{residue:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom[0]}\n")
    path.write_text("".join(lines) + "END\n")


def test_identity_and_rigid_motion_score_perfectly(tmp_path):
    reference, generated = tmp_path / "ref.pdb", tmp_path / "gen.pdb"
    write_structure(reference)
    write_structure(generated, rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), offset=[4, -3, 2])
    oracle = StructuralOracle(reference, "ACDEFGHIK", "A", "A")
    for path in [reference, generated]:
        result = oracle.score(path)
        assert result["valid"]
        assert result["tm"] == pytest.approx(1, abs=1e-6)
        assert result["lddt"] == 1
        assert result["rmsd"] < 1e-5
        assert result["objective"] == pytest.approx(1, abs=1e-6)


def test_invalid_outputs_return_zero_and_bad_reference_fails_setup(tmp_path):
    reference, generated = tmp_path / "ref.pdb", tmp_path / "gen.pdb"
    write_structure(reference)
    oracle = StructuralOracle(reference, "ACDEFGHIK", "A", "A")
    write_structure(generated, missing_ca=True)
    for path in [generated, tmp_path / "missing.pdb"]:
        result = oracle.score(path)
        assert result["valid"] is False
        assert result["tm"] == result["lddt"] == result["objective"] == 0
    generated.write_text("ATOM      1  CA  ALA A   1         nan     inf     nan\nEND\n")
    assert oracle.score(generated)["valid"] is False
    with pytest.raises(ValueError):
        StructuralOracle(reference, "AAAAAAAAA", "A", "A")


def minimal_structure(atoms):
    return ProteinStructure("AA", np.zeros((2, 3)), {k: np.array(v, dtype=float) for k, v in atoms.items()}, ("ALA", "ALA"))


def test_lddt_known_distance_errors_and_missing_atoms():
    reference = minimal_structure({(0, "CA"): [0, 0, 0], (1, "CA"): [10, 0, 0]})
    scorer = LDDT(reference)
    displaced = minimal_structure({(0, "CA"): [0, 0, 0], (1, "CA"): [11, 0, 0]})
    # Error exactly 1 A: conserved at 2 and 4, not at 0.5 and 1.
    assert scorer.score(displaced)[0] == 0.5
    assert scorer.score(minimal_structure({(0, "CA"): [0, 0, 0]}))[0] == 0


def test_lddt_equivalent_side_chain_names():
    atoms = {(0, "CA"): np.array([0., 0., 0.]), (0, "OD1"): np.array([1., 2., 0.]),
             (0, "OD2"): np.array([1., -2., 0.]), (1, "CA"): np.array([3., 3., 0.])}
    reference = ProteinStructure("DA", np.zeros((2, 3)), atoms, ("ASP", "ALA"))
    swapped = dict(atoms)
    swapped[0, "OD1"], swapped[0, "OD2"] = atoms[0, "OD2"], atoms[0, "OD1"]
    result, details = LDDT(reference).score(ProteinStructure("DA", np.zeros((2, 3)), swapped, ("ASP", "ALA")))
    assert result == 1
    assert details["symmetry_swaps"] == 1


def test_rmsd_does_not_allow_reflection():
    coords = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    reflected = coords.copy()
    reflected[:, 0] *= -1
    assert rmsd(reflected, coords) > 0.1


def test_tm_objective_retains_lddt_diagnostics(tmp_path):
    reference, generated = tmp_path / "ref.pdb", tmp_path / "gen.pdb"
    write_structure(reference)
    # Remove non-CA atoms: TM stays perfect while lDDT penalizes missing atoms.
    generated.write_text("".join(line for line in reference.read_text().splitlines(True)
                                 if line[12:16].strip()=="CA" or line.startswith("END")))
    combined=StructuralOracle(reference,"ACDEFGHIK","A","A").score(generated)
    oracle=StructuralOracle(reference,"ACDEFGHIK","A","A",objective="tm")
    tm=oracle.score(generated)
    assert tm["valid"] and tm["objective"]==pytest.approx(1)
    assert tm["lddt"]==combined["lddt"]<1
    assert combined["objective"]<tm["objective"]
    assert oracle.score(tmp_path/"missing.pdb")["objective"]==0
    with pytest.raises(ValueError,match="Unknown objective"):
        StructuralOracle(reference,"ACDEFGHIK","A","A",objective="typo")

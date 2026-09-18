import csv
from pathlib import Path

import pytest
import yaml

from targets.manifest import load_manifest, read_fasta, validate_reference, write_boltz_input


def target_files(tmp_path):
    (tmp_path / "test.fasta").write_text(">normal_header\nACD\n")
    lines = [f"ATOM  {i:5d}  CA  {name} A{i:4d}    {i*3.8:8.3f}{0:8.3f}{0:8.3f}  1.00 20.00           C\n"
             for i, name in enumerate(["ALA", "CYS", "ASP"], 1)]
    (tmp_path / "test.pdb").write_text("".join(lines) + "END\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("target_id,fasta,reference,chain_id,reference_chain_id,length,split,msa,provenance\n"
                        "test,test.fasta,test.pdb,A,A,3,dev,,synthetic fixture\n")
    return load_manifest(manifest, tmp_path)[0]


def test_plain_fasta_manifest_and_reference(tmp_path):
    target = target_files(tmp_path)
    assert target.sequence == "ACD"
    assert validate_reference(target)["residues"] == 3
    assert len(target.fingerprint()["reference_sha256"]) == 64


def test_generation_yaml_has_sequence_but_no_reference(tmp_path):
    target = target_files(tmp_path)
    output = tmp_path / "boltz.yaml"
    write_boltz_input(target, output)
    assert yaml.safe_load(output.read_text()) == {
        "version": 1, "sequences": [{"protein": {"id": "A", "sequence": "ACD", "msa": "empty"}}]}
    assert str(target.reference) not in output.read_text()


@pytest.mark.parametrize("contents", ["ACD", ">a\nACD\n>b\nACD\n", ">a\nACX", ">a\n", ">\nACD"])
def test_invalid_fasta_rejected(tmp_path, contents):
    path = tmp_path / "bad.fasta"
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_fasta(path)


def test_reference_sequence_mismatch_rejected(tmp_path):
    target = target_files(tmp_path)
    target.reference.write_text(target.reference.read_text().replace("CYS", "GLY"))
    with pytest.raises(ValueError, match="sequence"):
        validate_reference(target)


def test_missing_ca_rejected(tmp_path):
    target = target_files(tmp_path)
    target.reference.write_text(target.reference.read_text().replace(" CA  CYS", " N   CYS"))
    with pytest.raises(ValueError, match="C-alpha"):
        validate_reference(target)


def test_duplicate_target_rejected(tmp_path):
    target_files(tmp_path)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("a") as f:
        f.write("test,test.fasta,test.pdb,A,A,3,test,,\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(manifest, tmp_path)

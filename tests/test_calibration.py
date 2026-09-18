import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from generator.prepare import PreparedTarget
from targets.manifest import Target, sha256
from scripts import calibrate_bounds
from runner.protocol import verify_calibration


def test_constant_generator_cannot_freeze_radius_and_failures_remain_charged(tmp_path,monkeypatch):
    fasta,reference=tmp_path/"fixture.fasta",tmp_path/"fixture.pdb"
    fasta.write_text(">fixture\nACD\n")
    reference.write_text("".join(f"ATOM  {i:5d}  CA  {name} A{i:4d}    {i*3.8:8.3f}{0:8.3f}{0:8.3f}  1.00 20.00           C\n"
        for i,name in enumerate(["ALA","CYS","ASP"],1))+"END\n")
    target=Target("fixture",fasta,reference,"A","A",3,"dev",None,"test fixture")
    monkeypatch.setattr(calibrate_bounds,"load_manifest",lambda *args:[target])
    def prepare(target,directory,cache):
        directory.mkdir(parents=True)
        return PreparedTarget(target,directory,{"atom_pad_mask":torch.ones(1,3)})
    monkeypatch.setattr(calibrate_bounds,"prepare_target",prepare)
    class ConstantGenerator:
        calls=0
        def __init__(self,*args,**kwargs):
            self.config={"test_double":True}
        def decode(self,z,directory):
            type(self).calls+=1
            if type(self).calls%7==0:
                raise RuntimeError("controlled failure")
            directory.mkdir(parents=True)
            shutil.copyfile(reference,directory/"output.pdb")
            torch.save(torch.zeros(1,3,3),directory/"atom_coords.pt")
            return directory/"output.pdb"
    monkeypatch.setattr(calibrate_bounds,"Boltz2Adapter",ConstantGenerator)
    monkeypatch.setattr(sys,"argv",["calibrate_bounds.py","--root",str(tmp_path),"--samples","20",
                                   "--min-valid-rate","0.8","--min-geometry-rate","0.8"])
    with pytest.raises(SystemExit,match="No radius passed"):
        calibrate_bounds.main()
    output=tmp_path/"outputs/calibration"
    report=json.loads((output/"calibration_report.json").read_text())
    assert report["calls"]==ConstantGenerator.calls==102
    assert report["selected_radius"] is None and not (output/"frozen_radius.json").exists()
    records=[json.loads(line) for line in (output/"trials.jsonl").read_text().splitlines()]
    assert len(records)==204
    assert sum(r["status"]=="completed" and not r["valid"] for r in records)==14
    assert all(s["median_center_rmsd"]==0 for s in report["summaries"]["fixture"].values())


def test_successful_calibration_freezes_one_radius_and_checks_generator_provenance(tmp_path,monkeypatch):
    fasta,reference=tmp_path/"fixture.fasta",tmp_path/"fixture.pdb"
    fasta.write_text(">fixture\nACD\n")
    names=["ALA","CYS","ASP"]
    base=torch.tensor([[3.8,0,0],[7.6,0,0],[11.4,0,0]],dtype=torch.float64)
    def write(path,coords):
        path.write_text("".join(f"ATOM  {i:5d}  CA  {name} A{i:4d}    {float(c[0]):8.3f}{float(c[1]):8.3f}{float(c[2]):8.3f}  1.00 20.00           C\n"
            for i,(name,c) in enumerate(zip(names,coords),1))+"END\n")
    write(reference,base)
    target=Target("fixture",fasta,reference,"A","A",3,"dev",None,"test fixture")
    monkeypatch.setattr(calibrate_bounds,"load_manifest",lambda *args:[target])
    def prepare(target,directory,cache):
        directory.mkdir(parents=True)
        return PreparedTarget(target,directory,{"atom_pad_mask":torch.ones(1,3)})
    monkeypatch.setattr(calibrate_bounds,"prepare_target",prepare)
    config={"checkpoint_sha256":"c"*64,"sampling_steps":200,"recycling_steps":3,"sampler_seed":1729,
        "precision":"float32","use_kernels":False,"diffusion_samples":1,"sample_source_sha256":"d"*64,
        "adapter_source_sha256":"a"*64,"native_noise_source_sha256":"b"*64,"test_double":True}
    class DeformingGenerator:
        def __init__(self,*args,**kwargs):
            self.config=config
        def decode(self,z,directory):
            directory.mkdir(parents=True)
            coords=base+.05*z.reshape(3,3).double()
            write(directory/"output.pdb",coords)
            torch.save(coords.unsqueeze(0),directory/"atom_coords.pt")
            return directory/"output.pdb"
    monkeypatch.setattr(calibrate_bounds,"Boltz2Adapter",DeformingGenerator)
    monkeypatch.setattr(sys,"argv",["calibrate_bounds.py","--root",str(tmp_path),"--samples","20","--min-median-rmsd","0.0001"])
    calibrate_bounds.main()
    frozen=tmp_path/"outputs/calibration/frozen_radius.json"
    assert verify_calibration(frozen,[target],config)==.1
    with pytest.raises(ValueError,match="generator differs"):
        verify_calibration(frozen,[target],{**config,"sampler_seed":123})
    report_path=frozen.parent/"calibration_report.json"
    original_report=report_path.read_text()
    original_frozen=frozen.read_text()
    report=json.loads(original_report)
    report["summaries"]["fixture"]["0.1"]["median_center_rmsd"]+=1
    report_path.write_text(json.dumps(report))
    frozen_data=json.loads(original_frozen)
    frozen_data["calibration_report_sha256"]=sha256(report_path)
    frozen.write_text(json.dumps(frozen_data))
    with pytest.raises(ValueError,match="per-trial measurements"):
        verify_calibration(frozen,[target],config)
    report_path.write_text(original_report)
    frozen.write_text(original_frozen)
    repeat_path=frozen.parent/"fixture/center_repeat/atom_coords.pt"
    repeat_coords=torch.load(repeat_path,weights_only=True)
    torch.save(repeat_coords+1,repeat_path)
    with pytest.raises(ValueError,match="repeat coordinates differ"):
        verify_calibration(frozen,[target],config)
    torch.save(repeat_coords,repeat_path)
    (frozen.parent/"trials.jsonl").write_text("")
    with pytest.raises(ValueError,match="journal"):
        verify_calibration(frozen,[target],config)

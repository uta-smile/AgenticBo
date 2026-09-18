import json
from pathlib import Path
import shutil
import sys
import pytest
from types import SimpleNamespace

from runner import run_suite
from analysis import aggregate,plots


@pytest.mark.parametrize("objective",["sqrt_tm_times_lddt","tm"])
def test_suite_schedules_fifteen_runs_without_overlapping_a_gpu(tmp_path,monkeypatch,objective):
    config=tmp_path/"configs"
    config.mkdir()
    shutil.copyfile("configs/poc_v1.yaml",config/"poc_v1.yaml")
    frozen=tmp_path/"outputs/calibration/frozen_radius.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}")
    targets=[SimpleNamespace(target_id=f"target_{i}",sequence="A"*119+letter,split="test",length=120)
             for i,letter in enumerate("ACDEF")]
    monkeypatch.setattr(run_suite,"load_manifest",lambda *args:targets)
    monkeypatch.setattr(run_suite,"validate_reference",lambda target:None)
    monkeypatch.setattr(run_suite.SaraClient,"from_environment",lambda:object())
    monkeypatch.setattr(run_suite,"process_identity",lambda pid:{"pid":pid,"test_double":True})
    monkeypatch.setattr(run_suite.time,"sleep",lambda seconds:None)
    monkeypatch.setattr(aggregate,"load_runs",lambda *args,**kwargs:[])
    monkeypatch.setattr(plots,"render",lambda *args:{"test_double":True})
    active=set()
    launched=[]
    class Worker:
        def __init__(self,command,*,cwd,env,stdout,stderr):
            self.gpu=env["CUDA_VISIBLE_DEVICES"]
            assert self.gpu not in active
            active.add(self.gpu)
            assert len(active)<=4
            launched.append(command)
            self.pid=1000000+len(launched)
            self.polls=0
        def poll(self):
            self.polls+=1
            if self.polls==1:
                return None
            active.remove(self.gpu)
            return 0
    monkeypatch.setattr(run_suite.subprocess,"Popen",Worker)
    monkeypatch.setattr(sys,"argv",["run_suite.py","--root",str(tmp_path),"--objective",objective])
    assert run_suite.main()==0
    assert len(launched)==15 and not active
    assert all(command[command.index("--objective")+1]==objective for command in launched)
    jobs=json.loads((tmp_path/"outputs/poc/suite_status.json").read_text())
    assert all(job["status"]=="completed" for job in jobs)

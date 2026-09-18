import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("ax")

from analysis.aggregate import load_runs
from analysis.plots import render
from dsp.optimize_acq import AcquisitionSettings
from latent.bounds import LatentBox
from runner.engine import run_methods


class Generator:
    calls=0

    def decode(self,z,path):
        self.calls+=1
        path.mkdir(parents=True)
        torch.save(z,path/"z.pt")
        return path/"z.pt"


class Oracle:
    objective="tm"

    def score(self,path):
        z=torch.load(path,weights_only=True)
        value=float(torch.exp(-(z-.15).square().mean()))
        return {"valid":True,"tm":value,"lddt":value*.5,"rmsd":1-value}


def test_real_ax_proposals_share_budget_tm_scores_and_resume(tmp_path):
    torch.set_num_threads(2)
    generator=Generator()
    kwargs=dict(directory=tmp_path,box=LatentBox(torch.zeros(1,1,3),.25),
        generator=generator,oracle=Oracle(),metadata={"kind":"synthetic_ax_test"},
        target_id="fixture",target_length=1,seed=7,budget=7,initial=5,
        methods=("sobol","dsp_gp","ax"),
        acquisition_settings=AcquisitionSettings(raw_samples=16,num_restarts=2,maxiter=20),fit_maxiter=30)
    result=run_methods(**kwargs)
    assert result["completed"] and generator.calls==11
    assert result["total_physical_calls"]==5+3*(7-5)
    runs=load_runs(tmp_path)
    assert {r["method"] for r in runs}=={"sobol","dsp_gp","ax"}
    assert len({r["shared_initial_sha256"] for r in runs})==1
    for run in runs:
        assert len(run["trials"])==7
        assert all(t["objective"]==t["tm"] for t in run["trials"])
    diagnostic=json.loads((tmp_path/"ax/ax/diagnostics_0005.json").read_text())
    assert diagnostic["kernel"]=="RBFKernel"
    assert diagnostic["acquisition"]=="qLogNoisyExpectedImprovement"
    assert (tmp_path/"ax/ax/before_eval_0006.json").is_file()
    report=render(runs,tmp_path/"analysis")
    assert report["objective"]=="tm" and len(report["summary"])==3
    assert (tmp_path/"analysis/best_tm.png").is_file()
    assert run_methods(**kwargs)["completed"] and generator.calls==11

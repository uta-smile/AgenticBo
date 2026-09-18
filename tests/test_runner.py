import json
from pathlib import Path
import sqlite3

import torch
import pytest

from dsp.optimize_acq import AcquisitionSettings
from dsp.state import OptimizationState
from latent.bounds import LatentBox
from runner.engine import METHODS,run_methods
from sara.agent import SaraController
from analysis.aggregate import load_runs,aggregate


class SyntheticGenerator:
    def __init__(self):
        self.calls=0
    def decode(self,z,path):
        self.calls+=1
        path.mkdir(parents=True)
        torch.save(z,path/"synthetic.pt")
        return path/"synthetic.pt"


class SyntheticOracle:
    def __init__(self, objective="sqrt_tm_times_lddt"):
        self.objective=objective

    def score(self,path):
        z=torch.load(path,weights_only=True)
        value=float(torch.exp(-(z-.2).square().mean()))
        return {"valid":True,"tm":value,"lddt":value*.5,"rmsd":1-value}


class AdvisoryTestClient:
    def complete(self,messages,tools,seed):
        context=next(json.loads(m["content"]) for m in reversed(messages) if m["role"]=="user" and m["content"].startswith("{") and "advisory_candidate" in json.loads(m["content"]))
        candidate_id=context["advisory_candidate"]["candidate_id"]
        return {"role":"assistant","content":None,"tool_calls":[{"id":"test_evaluate","type":"function",
            "function":{"name":"EVALUATE","arguments":json.dumps({"candidate_id":candidate_id})}}]}, {
                "prompt_tokens":10,"completion_tokens":5,"total_tokens":15}


@pytest.mark.parametrize("objective",["sqrt_tm_times_lddt","tm"])
def test_all_four_methods_share_initialization_finish_budget_and_resume_without_decodes(tmp_path,objective):
    torch.set_num_threads(2)
    generator=SyntheticGenerator()
    controller=SaraController(AdvisoryTestClient(),prompt_path=Path("prompts/SARA_SYSTEM.md"))
    kwargs=dict(directory=tmp_path,box=LatentBox(torch.zeros(1,2,3),.5),generator=generator,oracle=SyntheticOracle(objective),
        metadata={"kind":"synthetic_test_not_protein_results"},target_id="fixture",target_length=2,
        seed=3,budget=6,initial=4,methods=METHODS,controller=controller,
        acquisition_settings=AcquisitionSettings(raw_samples=16,num_restarts=4,maxiter=25),fit_maxiter=40)
    report=run_methods(**kwargs)
    assert report["completed"] and generator.calls==report["total_physical_calls"]==12
    results=[json.loads((tmp_path/m/"results.json").read_text()) for m in METHODS]
    initial_scores=[[t["objective"] for t in r["trials"][:4]] for r in results]
    assert all(values==initial_scores[0] for values in initial_scores)
    assert len({r["shared_initial_sha256"] for r in results})==1
    assert all(r["physical_method_calls"]==2 and len(r["trials"])==6 for r in results)
    for result in results:
        for trial in result["trials"]:
            expected=trial["tm"] if objective=="tm" else (trial["tm"]*trial["lddt"])**.5
            assert trial["objective"]==pytest.approx(expected)
    assert all(t["agent_decision"]["accepted_dsp_suggestion"] for t in results[-1]["trials"][4:])
    assert (tmp_path/"agentic_dsp"/"agent_trace.jsonl").is_file()
    report2=run_methods(**kwargs)
    assert report2["completed"] and generator.calls==12
    runs=load_runs(tmp_path)
    analysis=aggregate(runs,tmp_path/"analysis")
    assert analysis["runs"]==4
    assert analysis["objective"]==objective
    assert (tmp_path/"analysis"/"thresholds.csv").is_file()
    trace=[json.loads(line) for line in (tmp_path/"agentic_dsp"/"agent_trace.jsonl").read_text().splitlines()]
    assert all(row["context"]["objective"]==objective for row in trace if row["event"]=="turn_start")
    mixed=[{**runs[0],"context":{**runs[0]["context"],"objective":"tm"}},
           {**runs[1],"context":{**runs[1]["context"],"objective":"sqrt_tm_times_lddt"}}]
    with pytest.raises(ValueError,match="different optimization objectives"):
        aggregate(mixed,tmp_path/"mixed")
    kwargs["oracle"]=SyntheticOracle("tm" if objective!="tm" else "sqrt_tm_times_lddt")
    with pytest.raises(ValueError,match="Run context changed"):
        run_methods(**kwargs)
    assert generator.calls==12


class EvaluateThenStopClient(AdvisoryTestClient):
    def complete(self,messages,tools,seed):
        previous=[m for m in messages if m['role']=='tool' and 'objective' in m['content']]
        if not previous:
            return super().complete(messages,tools,seed)
        assert json.loads(previous[-1]['content'])['valid']
        return {'role':'assistant','content':None,'tool_calls':[{'id':'stop','type':'function',
            'function':{'name':'STOP','arguments':json.dumps({'reason':'Fixture termination after observing evaluation feedback'})}}]}, {
                'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}


def test_stop_history_resume_and_analysis(tmp_path):
    from analysis.plots import render
    torch.set_num_threads(2)
    generator=SyntheticGenerator()
    kwargs=dict(directory=tmp_path,box=LatentBox(torch.zeros(1,2,3),.5),generator=generator,oracle=SyntheticOracle('tm'),
        metadata={'kind':'synthetic_test_not_protein_results'},target_id='fixture',target_length=2,
        seed=3,budget=7,initial=4,methods=('agentic_dsp','sobol'),
        controller=SaraController(EvaluateThenStopClient(),prompt_path=Path('prompts/SARA_SYSTEM.md')),
        acquisition_settings=AcquisitionSettings(raw_samples=16,num_restarts=4,maxiter=20),fit_maxiter=30)
    report=run_methods(**kwargs)
    assert report['completed'] and generator.calls==8
    result=json.loads((tmp_path/'agentic_dsp/results.json').read_text())
    assert result['budget_used']==5 and result['termination']=='agent_stop'
    assert result['incumbent']['objective']==max(t['tm'] for t in result['trials'])
    assert run_methods(**kwargs)['completed'] and generator.calls==8
    runs=load_runs(tmp_path)
    assert render(runs,tmp_path/'analysis')['runs']==2
    import sqlite3
    with sqlite3.connect(tmp_path/'agentic_dsp/state.sqlite') as db:
        history=json.loads(db.execute("SELECT value FROM settings WHERE key='agent_history'").fetchone()[0])
        assert len(history)==2
        assert history[-1]['messages'][-1]['role']=='tool'
        db.execute("DELETE FROM settings WHERE key='stop_decision'")
    with pytest.raises(ValueError,match='termination differs'):
        load_runs(tmp_path)


def test_random_baseline_continues_after_shared_initialization(tmp_path):
    generator = SyntheticGenerator()
    kwargs = dict(
        directory=tmp_path,
        box=LatentBox(torch.zeros(1, 2, 3), .5),
        generator=generator,
        oracle=SyntheticOracle("tm"),
        metadata={"kind": "random_baseline_test"},
        target_id="fixture",
        target_length=2,
        seed=13,
        budget=6,
        initial=4,
        methods=("random",),
    )
    report = run_methods(**kwargs)
    assert report["completed"] and generator.calls == 6
    with sqlite3.connect(tmp_path / "random" / "state.sqlite") as db:
        rows = db.execute("SELECT info FROM candidates ORDER BY id").fetchall()
    info = [json.loads(row[0]) for row in rows]
    assert [row["source"] for row in info[-2:]] == ["random", "random"]
    assert [row["random_index"] for row in info[-2:]] == [0, 1]

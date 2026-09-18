import json
from pathlib import Path

import pytest
import torch

from dsp.optimize_acq import AcquisitionSettings
from dsp.sampling import initial_design
from dsp.state import OptimizationState
from dsp.tools import OptimizationBackend
from latent.bounds import LatentBox
from sara.agent import SaraController
from sara.schemas import validate_action


def backend_at(path):
    torch.set_num_threads(2)
    state=OptimizationState(path,dimension=6,budget=8,metadata={})
    x=initial_design(6,4)
    state.import_initial(x,[{"valid":True,"tm":float(v),"lddt":float(v),"rmsd":1.} for v in x.mean(-1)],"a"*64)
    return OptimizationBackend(state,LatentBox(torch.zeros(1,2,3),.5),None,None,
        acquisition_settings=AcquisitionSettings(raw_samples=16,num_restarts=4,maxiter=20),fit_maxiter=30)


class TooManyCalls:
    def complete(self,*args):
        return {"role":"assistant","content":None,"tool_calls":[{"id":str(i),"type":"function",
            "function":{"name":"diagnostics","arguments":"{}"}} for i in range(5)]},None


def test_agent_tool_limit_does_not_consume_any_oracle_budget(tmp_path):
    backend=backend_at(tmp_path/"state")
    agent=SaraController(TooManyCalls(),prompt_path=Path("prompts/SARA_SYSTEM.md"))
    with pytest.raises(RuntimeError,match="configured computational-tool limit"):
        agent.select(backend,target_id="fixture",target_length=2,trace_path=tmp_path/"trace.jsonl",seed=0)
    assert backend.state.used==4


class MultipleEvaluations:
    def complete(self,messages,*args):
        candidate=json.loads(messages[1]["content"])["advisory_candidate"]["candidate_id"]
        return {"role":"assistant","content":None,"tool_calls":[{"id":str(i),"type":"function",
            "function":{"name":"EVALUATE","arguments":json.dumps({"candidate_id":candidate})}} for i in range(2)]},None


def test_agent_cannot_evaluate_multiple_candidates_in_a_turn(tmp_path):
    backend=backend_at(tmp_path/"state")
    agent=SaraController(MultipleEvaluations(),prompt_path=Path("prompts/SARA_SYSTEM.md"))
    with pytest.raises(RuntimeError,match="sole tool call"):
        agent.select(backend,target_id="fixture",target_length=2,trace_path=tmp_path/"trace.jsonl",seed=0)
    assert backend.state.used==4


@pytest.mark.parametrize("name,args",[("project",{"dimension":8}),("evaluate",{"candidate_id":"cand_1"}),
    ("set_search_radius",{"radius":0}),("set_search_radius",{"radius":float("nan")}),
    ("suggest",{"dimensions":[0,1]}),("set_acquisition",{"name":"fabricate"})])
def test_disallowed_actions_are_rejected(name,args):
    with pytest.raises(ValueError):
        validate_action(name,args)


def test_stop_is_terminal_and_returns_valid_incumbent(tmp_path):
    backend=backend_at(tmp_path/'state')
    candidate=backend.state.add_candidate(torch.full((6,),.4))
    before=backend.state.used
    stopped=backend.state.record_stop({'action':'STOP','reason':'Test terminal state'})
    assert stopped['incumbent']['valid'] and backend.state.used==before
    with pytest.raises(RuntimeError,match='stopped'):
        backend.state.reserve(candidate)
    with pytest.raises(ValueError,match='already stopped'):
        backend.state.record_stop({'action':'STOP','reason':'Again'})


@pytest.mark.parametrize('args',[{}, {'reason':''}, {'reason':'  '}, {'reason':'ok','candidate_id':'cand_1'}])
def test_invalid_stop_arguments(args):
    with pytest.raises(ValueError):
        validate_action('STOP',args)

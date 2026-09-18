import json

import pytest
import torch

from dsp.sampling import initial_design
from dsp.state import BudgetExhausted, OptimizationState
from dsp.tools import OptimizationBackend
from latent.bounds import LatentBox


def state_at(path, budget=5):
    return OptimizationState(path, dimension=6, budget=budget, metadata={"target_id":"fixture"})


def score(value=.25):
    return {"valid":True, "tm":value, "lddt":value, "rmsd":1.0}


class FailingGenerator:
    def __init__(self):
        self.calls = 0
    def decode(self,z,path):
        self.calls += 1
        raise RuntimeError("controlled generation failure")


def test_failed_generation_consumes_budget_and_cannot_exceed_it(tmp_path):
    state = state_at(tmp_path, budget=2)
    generator = FailingGenerator()
    backend = OptimizationBackend(state,LatentBox(torch.zeros(1,2,3),.25),generator,None)
    for _ in range(2):
        trial = backend.evaluate(state.add_candidate(torch.rand(6)))
        assert trial["valid"] is False and trial["objective"] == 0
    assert generator.calls == state.used == 2
    with pytest.raises(BudgetExhausted):
        backend.evaluate(state.add_candidate(torch.rand(6)))
    assert generator.calls == 2
    state.close()
    reopened = state_at(tmp_path,budget=2)
    assert reopened.used == 2 and reopened.remaining == 0


def test_pending_live_call_is_never_restarted_or_recovered(tmp_path):
    state = state_at(tmp_path)
    candidate = state.add_candidate(torch.full((6,),.5))
    trial = state.reserve(candidate)
    with pytest.raises(RuntimeError, match="pending"):
        state.reserve(state.add_candidate(torch.zeros(6)))
    with pytest.raises(RuntimeError, match="live or unverified"):
        state.recover_interrupted()
    state.finish(trial,score())
    with pytest.raises(ValueError, match="already consumed"):
        state.reserve(candidate)
    assert state.used == 1


def test_verified_dead_process_remains_a_charged_zero_score(tmp_path):
    state = state_at(tmp_path)
    trial = state.reserve(state.add_candidate(torch.zeros(6)))
    owner = json.loads(state.db.execute("SELECT owner FROM trials").fetchone()[0])
    owner["start"] = "not-the-current-process-start-time"
    state.db.execute("UPDATE trials SET owner=?",(json.dumps(owner),))
    assert state.recover_interrupted() == [trial]
    assert state.used == 1 and state.trials()[0]["objective"] == 0


def test_shared_observations_are_identical_and_charged_to_every_method(tmp_path):
    x, results = initial_design(6,3,seed=9), [score(.1),score(.2),score(.3)]
    a,b = state_at(tmp_path/"a"),state_at(tmp_path/"b")
    for state in (a,b):
        state.import_initial(x,results,"a"*64)
        assert state.used == 3 and state.remaining == 2
        assert not any(t["physical_call"] for t in state.trials())
        with pytest.raises(ValueError,match="empty run"):
            state.import_initial(x,results,"a"*64)
    for xa,xb in zip(a.observations(),b.observations()):
        assert torch.equal(xa,xb)


def test_resume_cannot_change_dimension_budget_or_target(tmp_path):
    state_at(tmp_path).close()
    with pytest.raises(ValueError,match="differs"):
        state_at(tmp_path,budget=6)


def test_shared_import_is_atomic_on_invalid_scores(tmp_path):
    state = state_at(tmp_path)
    with pytest.raises(ValueError):
        state.import_initial(initial_design(6,3),[score(),score(2),score()],"f"*64)
    assert state.used == 0
    assert state.db.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0


def test_recent_improvements_must_beat_earlier_incumbent(tmp_path, monkeypatch):
    state = state_at(tmp_path, budget=12)
    values = [.9, .1] + [.2 + i*.05 for i in range(10)]
    state.import_initial(initial_design(6,12), [score(v) for v in values], "b"*64)
    backend = OptimizationBackend(state, LatentBox(torch.zeros(1,2,3),.25), None, None)
    monkeypatch.setattr(backend, "fit", lambda: None)
    monkeypatch.setattr("dsp.tools.diagnostics", lambda *args: {})
    assert backend.diagnostics()["recent_improvement_rate"] == 0.0

import torch

from dsp.optimize_acq import AcquisitionSettings
from dsp.sampling import initial_design
from dsp.state import OptimizationState
from dsp.tools import OptimizationBackend
from latent.bounds import LatentBox


def test_tools_never_consume_generation_budget_or_expose_vectors(tmp_path):
    torch.set_num_threads(2)
    state = OptimizationState(tmp_path,dimension=6,budget=10,metadata={})
    x=initial_design(6,4)
    values=1-(x-.3).square().mean(-1)
    state.import_initial(x,[{"valid":True,"tm":float(v),"lddt":float(v),"rmsd":1.} for v in values],"d"*64)
    backend=OptimizationBackend(state,LatentBox(torch.zeros(1,2,3),.25),None,None,
        acquisition_settings=AcquisitionSettings(raw_samples=16,num_restarts=4,maxiter=20),fit_maxiter=30)
    original=state.used
    global_candidate=backend.suggest()
    backend.set_search_radius(.2)
    local=backend.suggest_local(radius=.1)
    center=state.candidate(state.incumbent()["candidate_id"])
    assert ((state.candidate(local["candidate_id"])-center).abs() <= .1+1e-12).all()
    assert backend.predict(local["candidate_id"])["std"] >= 0
    backend.acquisition_score(global_candidate["candidate_id"])
    backend.set_acquisition("ucb", beta=4)
    backend.reset_bounds()
    info=backend.diagnostics()
    assert info["dimension"]==6 and len(info["shortest_dimensions"])==6
    assert "vector" not in str(backend.incumbent())
    assert len(backend.trials()) == 4
    assert state.used==original

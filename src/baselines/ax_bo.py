"""Versioned Ax baseline with the same evaluated initialization as our methods.

Ax owns its model and acquisition defaults. The common runner supplies the
initial observations, so Ax starts its model phase immediately. This is a
matched-initialization adaptation, not an exact reproduction of paper Appendix B.
The Ax client remains alive during one continuous method segment and is rebuilt
from the SQLite ledger only after a process restart.
"""
import json
from importlib.metadata import version
import time

import torch


def ax_identity():
    return {"ax_platform":version("ax-platform"),"botorch":version("botorch"),
            "protocol":"shared_initial_observations_then_ax_defaults",
            "noise_sem":0.0,"device":"cpu","dtype":"float64"}


class AxSession:
    """Keep one Ax generation strategy alive for a method segment."""

    def __init__(self):
        self.client = None
        self.pending = None

    def _create(self, state, *, seed):
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties

        client=AxClient(random_seed=seed+1009*state.used,verbose_logging=False,
                        torch_device=torch.device("cpu"))
        names=[f"x_{i}" for i in range(state.dimension)]
        client.create_experiment(name="protein_native_noise",
            parameters=[{"name":name,"type":"range","bounds":[0.,1.],"value_type":"float"} for name in names],
            objectives={"objective":ObjectiveProperties(minimize=False)},
            choose_generation_strategy_kwargs={"num_initialization_trials":0})
        # Rebuild Ax only when this process is new or has resumed.
        for trial in state.trials():
            if trial["status"]!="completed":
                raise ValueError("Resolve pending trials before asking Ax for a candidate")
            values=state.candidate(trial["candidate_id"])
            _,index=client.attach_trial(dict(zip(names,values.tolist())))
            client.complete_trial(index,raw_data={"objective":(trial["objective"],0.0)})
        self.client=client
        self.names=names

    def suggest(self, state, *, seed):
        if self.client is None:
            self._create(state,seed=seed)
        if self.pending is not None:
            raise RuntimeError("Ax has an uncompleted proposal")
        started=time.monotonic()
        parameters,index=self.client.get_next_trial()
        x=torch.tensor([parameters[name] for name in self.names],dtype=torch.float64)
        bridge_model=self.client.generation_strategy.model.model
        gp=bridge_model.surrogate.model
        info={"source":"ax","ax":ax_identity(),"ax_trial_index":index,
              "kernel":type(gp.covar_module).__name__,
              "acquisition":bridge_model.botorch_acqf_class.__name__,
              "generation_strategy":str(self.client.generation_strategy),
              "proposal_runtime_seconds":time.monotonic()-started}
        directory=state.directory/"ax"
        directory.mkdir(exist_ok=True)
        self.client.save_to_json_file(str(directory/f"before_eval_{state.used+1:04d}.json"))
        (directory/f"diagnostics_{state.used:04d}.json").write_text(json.dumps(info,indent=2)+"\n")
        candidate_id=state.add_candidate(x,info)
        self.pending=(index,candidate_id)
        return candidate_id

    def complete(self, candidate_id, result):
        if self.pending is None or self.pending[1]!=candidate_id:
            raise RuntimeError("Ax completion does not match its pending candidate")
        self.client.complete_trial(self.pending[0],raw_data={"objective":(result["objective"],0.0)})
        self.pending=None


def suggest_ax(state, *, seed):
    """Compatibility helper for one-shot callers; the runner uses AxSession."""
    return AxSession().suggest(state,seed=seed)

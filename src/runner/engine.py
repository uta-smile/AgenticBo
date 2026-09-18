from __future__ import annotations

import json
# import fcntl
from filelock import FileLock, Timeout
from pathlib import Path
import time

import torch

from baselines.sobol import suggest_sobol
from dsp.optimize_acq import AcquisitionSettings
from dsp.priors import lengthscale_prior_parameters
from dsp.sampling import initial_design
from dsp.state import OptimizationState
from dsp.tools import OptimizationBackend
from targets.manifest import sha256
from oracle.objectives import DEFAULT_OBJECTIVE, validate_objective
from sara.history import finish_history

METHODS = ("sobol","vanilla_gp","dsp_gp","agentic_dsp")
AVAILABLE_METHODS = (*METHODS,"ax")


def _run_methods(*, directory: Path, box, generator, oracle, metadata: dict, target_id: str,
                target_length: int, seed: int, budget: int = 80, initial: int = 16,
                methods=METHODS, controller=None, controller_identity=None, acquisition_settings=AcquisitionSettings(), fit_maxiter=100):
    if not 2 <= initial <= budget or not methods or len(set(methods)) != len(methods) or set(methods)-set(AVAILABLE_METHODS):
        raise ValueError("Invalid budget, initial design, or methods")
    if "agentic_dsp" in methods and controller is None:
        raise ValueError("Agentic DSP requires a real controller; no automatic fixed-policy substitute")
    directory=Path(directory)
    context={"target_id":target_id,"target_length":target_length,"seed":seed,"metadata":metadata,
             "objective":validate_objective(getattr(oracle,"objective",DEFAULT_OBJECTIVE)),
             "native_shape":list(box.native_shape),"native_radius":box.radius,
             "budget":budget,"initial":initial,"acquisition_settings":acquisition_settings.__dict__,
             "fit_maxiter":fit_maxiter}
    if "ax" in methods:
        from baselines.ax_bo import ax_identity
        context["ax"]=ax_identity()
    directory.mkdir(parents=True,exist_ok=True)
    prior_report={"actual":lengthscale_prior_parameters(box.dimension),
                  "examples":{str(d):lengthscale_prior_parameters(d) for d in (256,1024,6392)}}
    (directory/"dsp_prior_report.json").write_text(json.dumps(prior_report,indent=2)+"\n")
    context_path=directory/"run_context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("Run context changed; use a new output directory")
    if not context_path.exists():
        context_path.write_text(json.dumps(context,indent=2,allow_nan=False)+"\n")
    z0_path=directory/"z0.pt"
    if z0_path.exists():
        if not torch.equal(torch.load(z0_path,weights_only=True),box.z0.reshape(box.native_shape)):
            raise ValueError("Run z0 changed; all methods must use the exact same initialization")
    else:
        torch.save(box.z0.reshape(box.native_shape),z0_path)
    shared=OptimizationState(directory/"shared_initial",dimension=box.dimension,budget=initial,metadata=context)
    shared.recover_interrupted()
    design=initial_design(box.dimension,initial,seed)
    initial_backend=OptimizationBackend(shared,box,generator,oracle,seed=seed)
    for index,point in enumerate(design):
        if index < shared.used:
            continue
        candidate_id=f"cand_{index+1:06d}"
        try:
            existing=shared.candidate(candidate_id)
            if not torch.equal(existing,point):
                raise ValueError("Saved initial candidate differs from seeded full-D Sobol design")
        except ValueError as exc:
            if str(exc)!="Unknown candidate ID":
                raise
            candidate_id=shared.add_candidate(point,{"source":"initial_center" if index==0 else "initial_sobol"})
        initial_backend.evaluate(candidate_id)
    initial_x,initial_y=shared.observations()
    if not torch.equal(initial_x,design) or len(initial_y)!=initial:
        raise RuntimeError("Common initial observations are incomplete or differ from the frozen design")
    initial_results=shared.trials()
    artifact=directory/"shared_initial"/"observations.pt"
    payload={"x":initial_x,"results":initial_results}
    if artifact.exists():
        old=torch.load(artifact,weights_only=True)
        if not torch.equal(old["x"],initial_x) or old["results"]!=initial_results:
            raise ValueError("Shared initial artifact differs from its authoritative ledger")
    else:
        torch.save(payload,artifact)
    initial_hash=sha256(artifact)
    shared.close()
    reports=[]
    for method in methods:
        state=OptimizationState(directory/method,dimension=box.dimension,budget=budget,metadata={**context,"method":method})
        state.recover_interrupted()
        if state.used==0:
            state.import_initial(initial_x,initial_results,initial_hash)
        if state.get_setting("initial_source_sha256")!=initial_hash:
            raise ValueError("Method does not share the exact frozen initial evaluations")
        if method=="agentic_dsp":
            previous_identity=state.get_setting("controller_identity")
            if previous_identity is not None and previous_identity!=controller_identity:
                raise ValueError("Controller identity changed during an agentic run")
            state.set_setting("controller_identity",controller_identity)
        backend=OptimizationBackend(state,box,generator,oracle,dsp=method!="vanilla_gp",seed=seed,
            acquisition_settings=acquisition_settings,fit_maxiter=fit_maxiter)
        ax_session=None
        if method=="ax":
            from baselines.ax_bo import AxSession
            ax_session=AxSession()
        finish_history(state)
        started=time.monotonic()
        while state.remaining and state.get_setting("stop_decision") is None:
            old_best=state.incumbent()["objective"]
            agent_info=None
            if method=="sobol":
                candidate_id=suggest_sobol(state,initial_count=initial,seed=seed)
            elif method=="ax":
                candidate_id=ax_session.suggest(state,seed=seed)
            elif method=="agentic_dsp":
                candidate_id,agent_info=controller.select(backend,target_id=target_id,target_length=target_length,
                    trace_path=state.directory/"agent_trace.jsonl",seed=seed+1009*state.used)
                if candidate_id is None:
                    decision=state.record_stop(agent_info)
                    finish_history(state,decision)
                    print(json.dumps({"method":method,"target":target_id,"seed":seed,**decision}),flush=True)
                    break
            else:
                candidate_id=backend.suggest()["candidate_id"]
            result=backend.evaluate(candidate_id,agent_decision=agent_info)
            if ax_session is not None:
                ax_session.complete(candidate_id,result)
            if agent_info is not None:
                finish_history(state,result)
                agent_info.update(trial_id=result["trial_id"],objective=result["objective"],
                    improvement_after_override=max(0.0,result["objective"]-old_best) if agent_info["overrode_dsp_suggestion"] else None)
                with (state.directory/"agent_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(agent_info,allow_nan=False)+"\n")
            print(json.dumps({"method":method,"target":target_id,"seed":seed,"used":state.used,
                "budget":budget,"objective":result["objective"],"best":state.incumbent()["objective"]}),flush=True)
        if method not in {"sobol","ax"}:
            backend.fit()  # Preserve final GP/ARD diagnostics even after the last evaluation.
        trials=state.trials()
        stop=state.get_setting("stop_decision")
        report={"method":method,"target_id":target_id,"seed":seed,"dimension":box.dimension,
                "budget":budget,"initial":initial,"completed":(len(trials)==budget or stop is not None) and all(t["status"]=="completed" for t in trials),
                "budget_used":len(trials),"budget_exhausted":len(trials)==budget,
                "termination":"agent_stop" if stop is not None else "budget_exhausted","stop_decision":stop,
                "physical_method_calls":sum(t["physical_call"] for t in trials),
                "shared_initial_sha256":initial_hash,"incumbent":stop["incumbent"] if stop else backend.incumbent(),
                "controller_identity":state.get_setting("controller_identity"),
                "resume_segment_runtime_seconds":time.monotonic()-started,"trials":trials}
        (state.directory/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        reports.append(report)
        state.close()
    summary={"target_id":target_id,"seed":seed,"dimension":box.dimension,"initial":initial,"budget":budget,
             "methods":list(methods),"completed":all(r["completed"] for r in reports),
             "shared_initial_physical_calls":initial,"shared_initial_sha256":initial_hash,
             "total_physical_calls":initial+sum(r["physical_method_calls"] for r in reports)}
    summary["charged_method_evaluations"]=sum(r["budget_used"] for r in reports)
    (directory/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def run_methods(**kwargs):
    directory = Path(kwargs["directory"])
    directory.mkdir(parents=True, exist_ok=True)

    lock = FileLock(directory / "run.lock", timeout=0)

    try:
        with lock:
            return _run_methods(**kwargs)

    except Timeout as exc:
        raise RuntimeError(
            "This target/seed run already has a live runner"
        ) from exc

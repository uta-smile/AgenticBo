#!/usr/bin/env python3
"""Verify live Sara tool calling against real GP tools, without protein decodes."""
import argparse
import json
from pathlib import Path
import time

import torch

from dsp.optimize_acq import AcquisitionSettings
from dsp.sampling import initial_design
from dsp.state import OptimizationState
from dsp.tools import OptimizationBackend
from latent.bounds import LatentBox
from sara.agent import SaraController
from sara.client import SaraClient
from sara.schemas import TOOLS, validate_action
from targets.manifest import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/sara_llamacpp_smoke"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    client=SaraClient.from_environment()
    started=time.monotonic()
    report={"scope":"live LLM and real GP; synthetic observations; zero protein decodes",
            "client":client.identity,"passed":False,"protein_generator_calls":0}
    try:
        state=OptimizationState(args.output/"state",dimension=6,budget=6,metadata={"objective":"tm","kind":"synthetic_llm_smoke"})
        x=initial_design(6,5,seed=0)
        results=[{"valid":True,"tm":float(torch.exp(-(point-.3).square().mean())),
                  "lddt":.5,"rmsd":1.} for point in x]
        fixture=args.output/"synthetic_observations.pt"
        torch.save({"x":x,"results":results},fixture)
        state.import_initial(x,results,sha256(fixture))
        backend=OptimizationBackend(state,LatentBox(torch.zeros(1,2,3),.25),None,None,
            acquisition_settings=AcquisitionSettings(raw_samples=32,num_restarts=2,maxiter=30),fit_maxiter=40)
        candidate=backend.suggest()["candidate_id"]
        messages=[{"role":"system","content":"This is a tool-interface test. Use exactly the requested tool and candidate ID."},
                  {"role":"user","content":f"Call predict for candidate_id={candidate}."}]
        probe=[]
        for name in ("predict","EVALUATE","STOP"):
            message,usage=client.complete(messages,[tool for tool in TOOLS if tool["function"]["name"]==name],seed=0)
            calls=message.get("tool_calls") or []
            if len(calls)!=1 or calls[0]["function"]["name"]!=name:
                raise RuntimeError(f"Expected exactly one {name} tool call")
            arguments=json.loads(calls[0]["function"]["arguments"])
            validate_action(name,arguments)
            if name!="STOP" and arguments["candidate_id"]!=candidate:
                raise RuntimeError("Model changed the candidate ID")
            probe.append({"tool":name,"arguments":arguments,"usage":usage})
            messages.append(message)
            if name=="predict":
                messages.append({"role":"tool","tool_call_id":calls[0]["id"],"content":json.dumps(backend.predict(candidate))})
                messages.append({"role":"user","content":f"Prediction received. Now call EVALUATE for candidate_id={candidate}. This test only checks selection; no generation will run."})
            elif name=="EVALUATE":
                messages.append({"role":"tool","tool_call_id":calls[0]["id"],"content":json.dumps({"serialization_only":True})})
                messages.append({"role":"user","content":"Now call STOP with reason: interface test complete."})
        before=state.used
        controller=SaraController(client,prompt_path=Path("prompts/SARA_SYSTEM.md"))
        chosen,metrics=controller.select(backend,target_id="synthetic_interface_fixture",target_length=2,
            trace_path=args.output/"agent_trace.jsonl",seed=0)
        if chosen is not None:
            state.candidate(chosen)
        elif metrics.get("action")!="STOP":
            raise RuntimeError("Expected EVALUATE or STOP")
        if state.used!=before:
            raise RuntimeError("Selection unexpectedly consumed an evaluation")
        report.update(passed=True,tool_roundtrip=probe,selected_candidate_id=chosen,
                      controller_metrics=metrics,synthetic_observations=before)
        state.close()
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()

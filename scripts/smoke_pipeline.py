#!/usr/bin/env python3
"""Exercise all four methods and plots with an explicitly synthetic oracle/controller."""
import argparse
import json
from pathlib import Path

import torch

from analysis.aggregate import load_runs
from analysis.plots import render
from dsp.optimize_acq import AcquisitionSettings
from latent.bounds import LatentBox
from runner.engine import run_methods, METHODS, AVAILABLE_METHODS
from oracle.objectives import OBJECTIVES, DEFAULT_OBJECTIVE
from sara.agent import SaraController


class SyntheticGenerator:
    def decode(self,z,directory):
        directory.mkdir(parents=True,exist_ok=False)
        path=directory/"analytic_test_tensor.pt"
        torch.save(z,path)
        return path


class AnalyticOracle:
    """These interface score fields are mathematical stand-ins, not TM/lDDT measurements."""
    def score(self,path):
        z=torch.load(path,weights_only=True).reshape(-1)
        target=torch.linspace(-.3,.3,z.numel(),dtype=z.dtype)
        global_score=float(torch.exp(-(z-target).square().mean()))
        local_score=float(torch.exp(-((z[1:]-z[:-1])-(target[1:]-target[:-1])).square().mean()/2))
        return {"valid":True,"tm":global_score,"lddt":local_score,
                "rmsd":float((z-target).square().mean().sqrt()),
                "score_kind":"analytic stand-ins; not structural protein scores"}


class ScriptedTestClient:
    """Protocol test double; deliberately never represented as a live LLM."""
    def complete(self,messages,tools,seed):
        context=next(json.loads(m["content"]) for m in reversed(messages) if m["role"]=="user" and m["content"].startswith("{") and "advisory_candidate" in json.loads(m["content"]))
        if messages[-1]["role"]=="tool":
            candidate=json.loads(messages[-1]["content"])["candidate_id"]
            name,arguments="EVALUATE",{"candidate_id":candidate}
        elif context["budget_used"]%2==0:
            name,arguments="suggest_local",{"center":"incumbent","radius":.25}
        else:
            name,arguments="EVALUATE",{"candidate_id":context["advisory_candidate"]["candidate_id"]}
        return {"role":"assistant","content":"Scripted integration test action","tool_calls":[{
            "id":f"test_{len(messages)}","type":"function","function":{"name":name,"arguments":json.dumps(arguments)}}]},None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-report",type=Path,default=Path("outputs/generator_smoke/smoke_report.json"))
    parser.add_argument("--output",type=Path,default=Path("outputs/synthetic_pipeline"))
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--budget",type=int,default=20)
    parser.add_argument("--initial",type=int,default=5)
    parser.add_argument("--methods",nargs="+",choices=AVAILABLE_METHODS,default=METHODS)
    parser.add_argument("--objective",choices=OBJECTIVES,default=DEFAULT_OBJECTIVE)
    args=parser.parse_args()
    torch.set_num_threads(4)
    report=json.loads(args.latent_report.read_text())
    shape=tuple(report["native_latent_shape"])
    if not shape or int(torch.tensor(shape).prod())!=report["flattened_D"]:
        parser.error("A measured native shape is required")
    z0=torch.randn(shape,generator=torch.Generator().manual_seed(args.seed))
    controller=SaraController(ScriptedTestClient(),prompt_path=Path("prompts/SARA_SYSTEM.md"))
    oracle=AnalyticOracle()
    oracle.objective=args.objective
    summary=run_methods(directory=args.output/"runs"/f"seed_{args.seed}",box=LatentBox(z0,.5),
        generator=SyntheticGenerator(),oracle=oracle,methods=args.methods,
        metadata={"kind":"synthetic_engineering_smoke","controller":"scripted_test_double",
                  "oracle":"analytic score stand-ins, not protein TM/lDDT","shape_source":str(args.latent_report)},
        target_id="synthetic",target_length=shape[-2],seed=args.seed,budget=args.budget,initial=args.initial,
        controller=controller,acquisition_settings=AcquisitionSettings(raw_samples=512,num_restarts=10))
    render(load_runs(args.output/"runs"),args.output/"analysis")
    print(json.dumps({"scope":"synthetic engineering only",**summary},indent=2))


if __name__=="__main__":
    main()

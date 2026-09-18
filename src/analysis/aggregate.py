import csv
import json
from itertools import combinations
from pathlib import Path
import sqlite3

import numpy as np
import torch

from analysis.metrics import agent_metrics,best_curve,evaluations_to_threshold
from runner.engine import METHODS, AVAILABLE_METHODS
from targets.manifest import sha256
from oracle.objectives import DEFAULT_OBJECTIVE


def run_objective(runs):
    objectives={r["context"].get("objective",DEFAULT_OBJECTIVE) for r in runs}
    if len(objectives)!=1:
        raise ValueError("Do not pool runs with different optimization objectives")
    return objectives.pop()


def load_runs(root:Path,*,require_poc=False):
    runs=[]
    for path in sorted(root.rglob("results.json")):
        run=json.loads(path.read_text())
        context=json.loads((path.parent.parent/"run_context.json").read_text())
        if not run["completed"]:
            raise ValueError(f"Incomplete run: {path}")
        db_path=path.parent/"state.sqlite"
        with sqlite3.connect(db_path.resolve().as_uri()+"?mode=ro",uri=True) as db:
            rows=db.execute("SELECT id,candidate,status,result,physical_call FROM trials ORDER BY id").fetchall()
            metadata={k:json.loads(v) for k,v in db.execute("SELECT key,value FROM metadata")}
            settings={k:json.loads(v) for k,v in db.execute("SELECT key,value FROM settings")}
            vectors=[torch.from_numpy(np.frombuffer(row[0],dtype=np.float64).copy()) for row in db.execute(
                "SELECT candidates.vector FROM trials JOIN candidates ON candidates.id=trials.candidate ORDER BY trials.id")]
        stop=settings.get("stop_decision")
        if stop is not None and (run["method"]!="agentic_dsp" or stop.get("action")!="STOP" or
                not stop.get("reason", "").strip() or stop.get("budget_used")!=len(rows) or
                stop.get("budget_remaining")!=run["budget"]-len(rows) or run.get("stop_decision")!=stop):
            raise ValueError(f"Invalid STOP record: {path}")
        if run.get("stop_decision")!=stop or run.get("budget_used",len(rows))!=len(rows):
            raise ValueError(f"Exported termination differs from ledger: {path}")
        valid_count=(len(rows)==run["budget"] or stop is not None and run["initial"]<=len(rows)<run["budget"])
        if not valid_count or metadata["budget"]!=run["budget"] or any(r[2]!="completed" for r in rows):
            raise ValueError(f"Ledger does not prove exact completed budget: {path}")
        if metadata["run"]!={**context,"method":run["method"]} or metadata["dimension"]!=run["dimension"]:
            raise ValueError("Exported run identity differs from the ledger")
        canonical=[{**json.loads(r[3]),"trial_id":f"trial_{r[0]:06d}","candidate_id":f"cand_{r[1]:06d}",
                    "status":r[2],"physical_call":bool(r[4])} for r in rows]
        if canonical!=run["trials"]:
            raise ValueError(f"Exported results differ from the authoritative ledger: {path}")
        initial_path=path.parent.parent/"shared_initial"/"observations.pt"
        if sha256(initial_path)!=run["shared_initial_sha256"]:
            raise ValueError("Shared initial observations changed after the run")
        shared=torch.load(initial_path,weights_only=True)
        if not torch.equal(torch.stack(vectors[:run["initial"]]),shared["x"]):
            raise ValueError("Method initial vectors differ from the shared design")
        for observed,expected in zip(canonical[:run["initial"]],shared["results"]):
            if any(observed[k]!=expected[k] for k in ("objective","tm","lddt","rmsd","valid")):
                raise ValueError("Method initial scores differ from shared observations")
        if any(v.numel()!=run["dimension"] for v in vectors):
            raise ValueError("Trial vector lost native dimensions")
        if run["method"] not in {"sobol", "random", "ax", "ax_saasbo"}:
            expected={f"fit_{n:04d}.pt" for n in range(run["initial"],len(rows)+1)}
            actual={p.name for p in (path.parent/"gp").glob("fit_*.pt")}
            if not expected<=actual:
                raise ValueError("Per-iteration GP diagnostics are incomplete")
        run.update(directory=path.parent,context=context)
        runs.append(run)
    if not runs:
        raise ValueError("No completed method results found")
    run_objective(runs)
    groups={}
    for run in runs:
        groups.setdefault((run["target_id"],run["seed"]),[]).append(run)
    for group in groups.values():
        expected_methods=set(METHODS) if require_poc else {r["method"] for r in runs}
        if len(group)!=len(expected_methods) or {r["method"] for r in group}!=expected_methods:
            raise ValueError("Every analyzed target/seed must contain the same selected methods")
        if expected_methods-set(AVAILABLE_METHODS):
            raise ValueError("Unknown method in results")
        for key in ("shared_initial_sha256","dimension","budget","initial"):
            if len({r[key] for r in group})!=1:
                raise ValueError(f"Fairness invariant differs across methods: {key}")
        contexts=[r["context"] for r in group]
        if any(c!=contexts[0] for c in contexts[1:]):
            raise ValueError("Methods used different target/generator/bounds context")
    kinds={r["context"]["metadata"]["kind"] for r in runs}
    if len(kinds)!=1:
        raise ValueError("Do not pool synthetic, smoke, and benchmark results")
    if len({r["budget"] for r in runs})!=1:
        raise ValueError("Plot matched evaluation budgets together")
    if require_poc:
        if kinds!={"protein_benchmark"} or len(runs)!=60 or len(groups)!=15:
            raise ValueError("Full POC requires 60 real protein benchmark runs")
        targets={r["target_id"] for r in runs}
        seeds={r["seed"] for r in runs}
        if len(targets)!=5 or len(seeds)!=3 or set(groups)!={(t,s) for t in targets for s in seeds}:
            raise ValueError("Missing target/seed cells in the 5×3 design")
        if any(r["budget"]!=80 or r["initial"]!=16 or len(r["trials"])!=80 for r in runs):
            raise ValueError("Full POC requires budget 80 and initial 16")
        controllers=[r.get("controller_identity") for r in runs if r["method"]=="agentic_dsp"]
        if any(c is None for c in controllers) or len({json.dumps(c,sort_keys=True) for c in controllers})!=1:
            raise ValueError("Full POC requires one recorded live-controller identity across runs")
    return runs


def write_csv(path,rows):
    if not rows:
        return
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(runs,output:Path):
    objective=run_objective(runs)
    output.mkdir(parents=True,exist_ok=True)
    curves,thresholds,finals,agents=[],[],[],[]
    for run in runs:
        ids={"target_id":run["target_id"],"seed":run["seed"],"method":run["method"],"objective_mode":objective}
        metric_curves={k:best_curve(run["trials"],k) for k in ("tm","lddt","objective")}
        for i in range(len(run["trials"])):
            curves.append({**ids,"evaluations":i+1,**{k:float(v[i]) for k,v in metric_curves.items()}})
        finals.append({**ids,**{f"best_{k}":float(v[-1]) for k,v in metric_curves.items()},
                       "budget_used":len(run["trials"]),"budget":run["budget"],"stopped":bool(run.get("stop_decision")),
                       "valid_rate":sum(t["valid"] for t in run["trials"])/len(run["trials"])})
        for threshold in (.5,.7,.8,.9):
            crossing=evaluations_to_threshold(run["trials"],threshold)
            thresholds.append({**ids,"threshold":threshold,"evaluations":crossing,"censored":crossing is None,"budget":run["budget"]})
        if run["method"]=="agentic_dsp":
            agents.append({**ids,**agent_metrics(run["trials"],run.get("stop_decision"))})
    for name,rows in [("curves",curves),("thresholds",thresholds),("finals",finals),("agent_metrics",agents)]:
        write_csv(output/f"{name}.csv",rows)
    summary=[]
    methods=[m for m in AVAILABLE_METHODS if any(r["method"]==m for r in runs)]
    for method in methods:
        selected=[r for r in finals if r["method"]==method]
        row={"method":method,"runs":len(selected)}
        for metric in ("best_tm","best_lddt","best_objective","valid_rate","budget_used"):
            values=[r[metric] for r in selected]
            row.update({f"{metric}_mean":float(np.mean(values)),f"{metric}_std":float(np.std(values,ddof=1)) if len(values)>1 else 0.})
        summary.append(row)
    write_csv(output/"summary.csv",summary)
    paired=[]
    lookup={(r["target_id"],r["seed"],r["method"]):r for r in finals}
    cells=sorted({(r["target_id"],r["seed"]) for r in finals})
    selected_methods=[m for m in methods if any((target,seed,m) in lookup for target,seed in cells)]
    for target,seed in cells:
        row={"target_id":target,"seed":seed}
        for left,right in combinations(selected_methods,2):
            if (target,seed,left) not in lookup or (target,seed,right) not in lookup:
                continue
            for metric in ("best_objective","best_tm"):
                row[f"{left}_minus_{right}_{metric.removeprefix('best_')}"]=(
                    lookup[target,seed,left][metric]-lookup[target,seed,right][metric])
        paired.append(row)
    write_csv(output/"paired_differences.csv",paired)
    return {"kind":runs[0]["context"]["metadata"]["kind"],"objective":objective,"runs":len(runs),"summary":summary}

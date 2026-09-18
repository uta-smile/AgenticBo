import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path(".cache/matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.aggregate import aggregate,load_runs
from analysis.metrics import best_curve,evaluations_to_threshold
from runner.engine import AVAILABLE_METHODS

LABELS={"sobol":"Sobol","vanilla_gp":"Vanilla GP","dsp_gp":"DSP GP","agentic_dsp":"Agentic DSP","ax":"Ax 0.5 default"}
COLORS=dict(zip(AVAILABLE_METHODS,["#64748b","#d97706","#2563eb","#7c3aed","#dc2626"]))


def render(runs,output:Path):
    report=aggregate(runs,output)
    kind=report["kind"]
    methods=[m for m in AVAILABLE_METHODS if any(r["method"]==m for r in runs)]
    heading="Synthetic engineering test — metric stand-ins" if kind.startswith("synthetic") else (
        "Uncalibrated protein pilot" if kind=="protein_pilot" else "Protein smoke experiment" if kind=="protein_smoke" else "Protein target matching")
    objective_label="Best TM-score objective" if report["objective"]=="tm" else "Best composite objective"
    heading+=f" / objective: {report['objective']}"
    for metric,label in [("tm","Best TM-score"),("lddt","Best lDDT"),("objective",objective_label)]:
        fig,ax=plt.subplots(figsize=(8,5))
        for method in methods:
            selected=[r for r in runs if r["method"]==method]
            # Stop each observed curve at its actual evaluation count. Never invent trials.
            lengths={len(r["trials"]) for r in selected}
            if len(lengths)>1:
                for index,run in enumerate(selected):
                    values=best_curve(run["trials"],metric)
                    ax.plot(np.arange(1,len(values)+1),values,label=LABELS[method] if index==0 else None,color=COLORS[method],alpha=.6)
                    if run.get("stop_decision"):
                        ax.scatter(len(values),values[-1],marker="x",color=COLORS[method])
            else:
                values=np.stack([best_curve(r["trials"],metric) for r in selected])
                mean,std=values.mean(0),values.std(0)
                x=np.arange(1,len(mean)+1)
                ax.plot(x,mean,label=LABELS[method],color=COLORS[method])
                ax.fill_between(x,np.clip(mean-std,0,1),np.clip(mean+std,0,1),color=COLORS[method],alpha=.12)
                if any(r.get("stop_decision") for r in selected):
                    ax.scatter(x[-1],mean[-1],marker="x",color=COLORS[method])
        ax.set(xlabel="Charged oracle evaluations (includes shared initial observations)",ylabel=label,title=heading)
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output/f"best_{metric}.png",dpi=160); plt.close(fig)
    budget=runs[0]["budget"]
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    for ax,threshold in zip(axes.flat,[.5,.7,.8,.9]):
        for index,method in enumerate(methods):
            selected=[r for r in runs if r["method"]==method]
            jitter=np.linspace(-.15,.15,len(selected))
            for offset,run in zip(jitter,selected):
                crossing=evaluations_to_threshold(run["trials"],threshold)
                ax.scatter(index+offset,crossing if crossing else budget+1,marker="o" if crossing else "x",color=COLORS[method],s=28)
        ax.axhline(budget+.5,linestyle=":",color="gray")
        ax.set(title=f"TM ≥ {threshold} (× = not reached)",ylabel="Evaluations",xticks=range(len(methods)),xticklabels=[LABELS[m] for m in methods])
        ax.tick_params(axis="x",labelrotation=20)
    fig.suptitle(heading); fig.tight_layout(); fig.savefig(output/"evaluations_to_tm_threshold.png",dpi=160); plt.close(fig)
    targets=sorted({r["target_id"] for r in runs})
    fig,ax=plt.subplots(figsize=(max(8,len(targets)*1.8),5))
    for index,method in enumerate(methods):
        means,stds=[],[]
        for target in targets:
            values=[max(t["tm"] for t in r["trials"]) for r in runs if r["target_id"]==target and r["method"]==method]
            means.append(np.mean(values)); stds.append(np.std(values))
        ax.errorbar(np.arange(len(targets))+(index-(len(methods)-1)/2)*.15,means,yerr=stds,fmt="o",capsize=3,label=LABELS[method],color=COLORS[method])
    ax.set(xticks=range(len(targets)),xticklabels=targets,ylabel="Final best observed TM-score",title=heading)
    ax.legend(); fig.tight_layout(); fig.savefig(output/"final_tm_by_target.png",dpi=160); plt.close(fig)
    length_dir=output/"lengthscales"; length_dir.mkdir(exist_ok=True)
    kernel_rows=[]
    for run in runs:
        if run["method"] in {"sobol","ax"}:
            continue
        fits=[torch.load(path,weights_only=True) for path in sorted((run["directory"]/"gp").glob("fit_*.pt"))]
        if not fits:
            raise ValueError("Missing GP/ARD diagnostics for a GP method")
        for fit in fits:
            if fit["lengthscales"].numel()!=run["dimension"]:
                raise ValueError("GP diagnostics do not retain the full native ARD dimension")
            kernel_rows.append((run["method"],fit["observations"],fit["diagnostics"]["kernel_off_diagonal_mean"]))
        if run["method"] in {"dsp_gp","agentic_dsp"}:
            values=[fit["lengthscales"].reshape(-1).numpy() for fit in fits]
            low,high=min(v.min() for v in values),max(v.max() for v in values)
            bins=np.geomspace(low*.99,high*1.01,41)
            hist=np.stack([np.histogram(v,bins=bins)[0]/len(v) for v in values],axis=1)
            fig,ax=plt.subplots(figsize=(9,5))
            xs=np.array([f["observations"] for f in fits])
            xedges=np.concatenate([xs-.5,[xs[-1]+.5]])
            mesh=ax.pcolormesh(xedges,bins,hist,shading="flat",cmap="viridis")
            ax.set_yscale("log")
            ax.set(xlabel="Observed evaluations",ylabel="ARD lengthscale",title=f"{heading}\n{run['target_id']} / seed {run['seed']} / {LABELS[run['method']]}")
            fig.colorbar(mesh,ax=ax,label="Fraction of native dimensions")
            fig.tight_layout(); fig.savefig(length_dir/f"{run['target_id']}_seed_{run['seed']}_{run['method']}.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method in [m for m in methods if m not in {"sobol","ax"}]:
        xs=sorted({n for m,n,v in kernel_rows if m==method})
        ys=[np.mean([v for m,n,v in kernel_rows if m==method and n==x]) for x in xs]
        ax.plot(xs,ys,label=LABELS[method],color=COLORS[method])
    ax.set(xlabel="Observed evaluations",ylabel="Mean off-diagonal RBF correlation",title=heading)
    ax.legend(); fig.tight_layout(); fig.savefig(output/"kernel_correlations.png",dpi=160); plt.close(fig)
    (output/"analysis_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description="Generate all required statistics and plots from completed four-method ledgers")
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--require-poc",action="store_true")
    args=parser.parse_args()
    report=render(load_runs(args.root,require_poc=args.require_poc),args.output)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()

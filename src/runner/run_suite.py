"""Schedule independent target/seed runs, one process per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from sara.client import SaraClient
from targets.manifest import load_manifest,validate_reference
from dsp.state import process_identity,process_alive
from oracle.objectives import DEFAULT_OBJECTIVE, OBJECTIVES, validate_objective


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config",type=Path,default=Path("configs/poc_v1.yaml"))
    parser.add_argument("--output",type=Path,default=Path("outputs/poc"))
    parser.add_argument("--frozen-radius",type=Path,default=Path("outputs/calibration/frozen_radius.json"))
    parser.add_argument("--gpus",type=int,nargs="+",default=[0,1,2,3])
    parser.add_argument("--objective",choices=OBJECTIVES,help="Override the objective for every target/seed")
    args=parser.parse_args()
    args.root=args.root.resolve()
    config=yaml.safe_load((args.root/args.config).read_text())
    try:
        objective=validate_objective(args.objective or config.get("objective",DEFAULT_OBJECTIVE))
    except ValueError as exc:
        parser.error(str(exc))
    targets=load_manifest(args.root/config["manifest"],args.root)
    selected=[t for t in targets if t.split=="test"]
    if len(selected)!=5 or len({t.sequence for t in selected})!=5 or any(not 100<=t.length<=200 for t in selected):
        parser.error("The POC suite requires five distinct test proteins of 100–200 residues")
    if config["budget"]!=80 or config["initial"]!=16 or len(set(config["seeds"]))!=3:
        parser.error("The final POC requires three seeds, budget 80, and initial 16")
    if set(config["methods"])!={"sobol","vanilla_gp","dsp_gp","agentic_dsp"}:
        parser.error("All four mandatory methods are required")
    if len(set(args.gpus))!=len(args.gpus) or any(g<0 for g in args.gpus):
        parser.error("GPU IDs must be distinct and nonnegative")
    for target in targets:
        validate_reference(target)
    SaraClient.from_environment()
    if not (args.root/args.frozen_radius).is_file():
        parser.error("A calibrated frozen radius is required")
    output=args.root/args.output
    output.mkdir(parents=True,exist_ok=True)
    suite_lock=(output/"suite.lock").open("a")
    try:
        fcntl.flock(suite_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("This suite already has a live scheduler")
    previous_status=output/"suite_status.json"
    if previous_status.exists():
        for previous in json.loads(previous_status.read_text()):
            if previous["status"]=="running":
                owner=previous.get("process_identity")
                if owner is None or process_alive(owner) is not False:
                    parser.error(f"Worker PID {previous.get('pid')} is live or unverified; monitor its existing log instead of restarting")
    jobs=[{"target":t.target_id,"seed":seed,"status":"queued"} for t in selected for seed in config["seeds"]]
    running={}
    logs=output/"logs"
    logs.mkdir(exist_ok=True)
    def save():
        temporary=output/"suite_status.json.tmp"
        temporary.write_text(json.dumps(jobs,indent=2)+"\n")
        temporary.replace(output/"suite_status.json")
    while any(j["status"] in {"queued","running"} for j in jobs):
        for gpu in args.gpus:
            if gpu in running:
                continue
            job=next((j for j in jobs if j["status"]=="queued"),None)
            if job is None:
                break
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8")
            log_path=logs/f"{job['target']}_seed_{job['seed']}.log"
            handle=log_path.open("a")
            command=[sys.executable,"-m","runner.run_target","--root",str(args.root),"--config",str(args.config),
                "--output",str(args.output),"--frozen-radius",str(args.frozen_radius),"--target",job["target"],"--seed",str(job["seed"]),"--device","cuda:0"]
            command.extend(["--objective",objective])
            process=subprocess.Popen(command,cwd=args.root,env=env,stdout=handle,stderr=subprocess.STDOUT)
            handle.close()
            try:
                owner=process_identity(process.pid)
            except FileNotFoundError:
                owner=None
            job.update(status="running",gpu=gpu,pid=process.pid,process_identity=owner,log=str(log_path))
            running[gpu]=(process,job)
        for gpu,(process,job) in list(running.items()):
            code=process.poll()
            if code is not None:
                job.update(status="completed" if code==0 else "failed",exit_code=code)
                del running[gpu]
        save()
        if running:
            time.sleep(1)
    if all(j["status"]=="completed" for j in jobs):
        from analysis.aggregate import load_runs
        from analysis.plots import render
        render(load_runs(output/"benchmark",require_poc=True),output/"analysis")
        return 0
    return 1


if __name__=="__main__":
    raise SystemExit(main())

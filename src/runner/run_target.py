import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

import torch
import yaml

from dsp.optimize_acq import AcquisitionSettings
from generator.adapter import Boltz2Adapter
from generator.prepare import prepare_target
from latent.bounds import LatentBox
from latent.inspect import inspect_native
from oracle.oracle import StructuralOracle
from oracle.objectives import DEFAULT_OBJECTIVE, OBJECTIVES, validate_objective
from runner.engine import METHODS,AVAILABLE_METHODS,run_methods
from runner.protocol import verify_calibration
from runner.provenance import source_digest,software_versions
from sara.agent import SaraController
from sara.client import SaraClient
from targets.manifest import load_manifest,validate_reference,sha256


def main():
    parser=argparse.ArgumentParser(description="Run all four methods with the same Boltz2 target, z0, radius, and initial observations")
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config",type=Path,default=Path("configs/poc_v1.yaml"))
    parser.add_argument("--generator-config",type=Path,default=Path("configs/generator.yaml"))
    parser.add_argument("--gp-config",type=Path,default=Path("configs/dsp_gp.yaml"))
    parser.add_argument("--frozen-radius",type=Path,default=Path("outputs/calibration/frozen_radius.json"))
    parser.add_argument("--target",required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--output",type=Path,default=Path("outputs/poc"))
    parser.add_argument("--smoke",action="store_true",help="Use budget 20 and initial 5, preserving generator and calibrated radius")
    parser.add_argument("--methods",nargs="+",choices=AVAILABLE_METHODS)
    parser.add_argument("--budget",type=int,help="Total charged evaluations per method, including initialization (pilot/smoke only)")
    parser.add_argument("--initial",type=int,help="Shared initial evaluations, included in budget (pilot/smoke only)")
    parser.add_argument("--max-computational-tools",type=int,
                        help="Sara computational calls allowed per campaign step; defaults to the total evaluation budget")
    parser.add_argument("--pilot",action="store_true",help="Exploratory single-target run outside the calibrated POC protocol")
    parser.add_argument("--radius",type=float,help="Explicit uncalibrated native-noise radius, required for --pilot")
    parser.add_argument("--objective",choices=OBJECTIVES,help="Override the configured optimization objective")
    parser.add_argument("--reference-mode",choices=("strict","resolved"),help="Score a complete reference or its mapped resolved positions")
    args=parser.parse_args()
    args.root=args.root.resolve()
    if args.threads<1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    config=yaml.safe_load((args.root/args.config).read_text())
    reference_mode=args.reference_mode or config.get("reference_mode","strict")
    if args.pilot and args.smoke:
        parser.error("Choose either --pilot or --smoke")
    if (args.budget is not None or args.initial is not None) and not (args.pilot or args.smoke):
        parser.error("Custom budgets require --pilot or --smoke; the full POC stays at 80/16")
    if args.pilot:
        if args.radius is None or not 0<args.radius<=2:
            parser.error("--pilot requires an explicit --radius in (0,2]; this is not calibrated")
    elif args.radius is not None:
        parser.error("An explicit radius is allowed only with --pilot")
    budget=args.budget if args.budget is not None else (20 if args.smoke or args.pilot else config["budget"])
    initial=args.initial if args.initial is not None else (5 if args.smoke or args.pilot else config["initial"])
    if not 2<=initial<=budget:
        parser.error("Require 2 <= initial <= budget")
    max_computational_tools=args.max_computational_tools if args.max_computational_tools is not None else budget
    if max_computational_tools<0:
        parser.error("--max-computational-tools must be nonnegative")
    try:
        objective=validate_objective(args.objective or config.get("objective",DEFAULT_OBJECTIVE))
    except ValueError as exc:
        parser.error(str(exc))
    generator_config=yaml.safe_load((args.root/args.generator_config).read_text())
    gp_config=yaml.safe_load((args.root/args.gp_config).read_text())
    required_gp={"kernel":"rbf","ard":"full_native_dimension","outputscale":1.,"learn_outputscale":False,
                 "standardize_outcome":True,"normalize_input":"fixed_unit_box","dtype":"float64","device":"cpu"}
    if (any(gp_config.get(k)!=v for k,v in required_gp.items())
            or gp_config.get("lengthscale_prior")!={"loc":"sqrt(2) + 0.5 * log(D)","scale":"sqrt(3)","minimum":.025}
            or gp_config.get("noise_prior")!={"loc":-4.,"scale":1.,"minimum":.0001}
            or gp_config.get("vanilla_ablation")!={"lengthscale_loc":"sqrt(2)","lengthscale_scale":"sqrt(3)"}):
        parser.error("GP config must preserve the explicit full-D DSP protocol; only fit_maxiter is tunable here")
    targets=load_manifest(args.root/config["manifest"],args.root)
    matches=[t for t in targets if t.target_id==args.target]
    if len(matches)!=1:
        parser.error("Register this target's FASTA/reference files in the manifest")
    target=matches[0]
    if not (args.smoke or args.pilot) and (target.split!="test" or not 100<=target.length<=200
            or config["budget"]!=80 or config["initial"]!=16 or args.seed not in config["seeds"]):
        parser.error("Benchmark runs require a 100–200-residue test target, configured seed, budget 80, and initial 16")
    validate_reference(target,reference_mode=reference_mode)
    methods=args.methods or config["methods"]
    controller=None
    controller_identity=None
    if "agentic_dsp" in methods:
        client=SaraClient.from_environment()
        controller_identity={**client.identity,"system_prompt_sha256":sha256(args.root/"prompts/SARA_SYSTEM.md")}
        controller=SaraController(client,prompt_path=args.root/"prompts/SARA_SYSTEM.md",
                                  max_tools=max_computational_tools,
                                  max_rounds=max(6,max_computational_tools+2))
    frozen_path=args.root/args.frozen_radius
    if not args.pilot and not frozen_path.is_file():
        parser.error("Run development calibration before the protein optimization experiment")
    if (generator_config["precision"]!="32-true" or generator_config["use_kernels"] or generator_config["compile"]
            or generator_config["diffusion_samples"]!=1 or generator_config["max_parallel_samples"]!=1):
        parser.error("The Titan X profile requires FP32, kernels/compile disabled, and one diffusion sample")
    output=args.root/args.output/("pilot" if args.pilot else "smoke" if args.smoke else "benchmark")/target.target_id/f"seed_{args.seed}"
    previous=output/"run_context.json"
    if previous.exists():
        context=json.loads(previous.read_text())
        if context.get("objective",DEFAULT_OBJECTIVE)!=objective or context["budget"]!=budget or context["initial"]!=initial:
            parser.error("Objective or budget differs from existing run; use a new output directory")
    prepared=prepare_target(target,output/"prepared",args.root/generator_config["cache"])
    generator=Boltz2Adapter(prepared,args.root/generator_config["checkpoint"],device=args.device,
        sampling_steps=generator_config["sampling_steps"],recycling_steps=generator_config["recycling_steps"],
        sampler_seed=generator_config["sampler_seed"])
    radius=args.radius if args.pilot else verify_calibration(frozen_path,targets,generator.config)
    z0_path=output/"z0.pt"
    z0=(torch.load(z0_path,weights_only=True).float() if z0_path.exists() else
        torch.randn(prepared.latent_shape,generator=torch.Generator().manual_seed(args.seed),dtype=torch.float32))
    report=inspect_native(z0,prepared.batch["atom_pad_mask"])
    (output/"latent_report.json").write_text(json.dumps(report,indent=2)+"\n")
    oracle=StructuralOracle(target.reference,target.sequence,target.reference_chain_id,target.chain_id,
                            objective=objective,reference_mode=reference_mode)
    acq=config["acquisition"]
    if acq["name"]!="log_ei" or acq["q"]!=1:
        parser.error("Fixed baseline acquisition must be sequential LogEI")
    run_methods(directory=output,box=LatentBox(z0,radius),generator=generator,oracle=oracle,
        metadata={"kind":"protein_pilot" if args.pilot else "protein_smoke" if args.smoke else "protein_benchmark","target":target.fingerprint(),
                  "reference_coverage":oracle.reference_info,
                  "generator":generator.config,"calibration":({"status":"uncalibrated_pilot","radius":radius} if args.pilot else json.loads(frozen_path.read_text())),
                  "software_versions":software_versions(),"source_sha256":source_digest(args.root/"src"),
                  "torch_threads":args.threads,"gp_config":gp_config},
        target_id=target.target_id,target_length=target.length,seed=args.seed,
        budget=budget,initial=initial,
        methods=methods,controller=controller,controller_identity=controller_identity,fit_maxiter=gp_config["fit_maxiter"],
        acquisition_settings=AcquisitionSettings(raw_samples=acq["raw_samples"],num_restarts=acq["num_restarts"],
            maxiter=acq.get("maxiter",100),batch_limit=acq.get("batch_limit",2)))
    if len(methods)>=2:
        from analysis.aggregate import load_runs
        from analysis.plots import render
        render(load_runs(output),output/"analysis")


if __name__=="__main__":
    main()

import json
from pathlib import Path

import torch

from targets.manifest import sha256
from runner.provenance import source_digest,software_versions
from runner.calibration import summarize_radius


def verify_calibration(path:Path,targets:list,generator_config:dict):
    if not path.is_file():
        raise ValueError("Missing frozen_radius.json; calibrate on development targets before protein BO")
    frozen=json.loads(path.read_text())
    report_path=path.parent/"calibration_report.json"
    if not report_path.is_file() or sha256(report_path)!=frozen.get("calibration_report_sha256"):
        raise ValueError("Calibration report does not match the frozen radius provenance")
    report=json.loads(report_path.read_text())
    if report["settings"].get("oracle_source_sha256")!=source_digest(Path(__file__).resolve().parents[1]/"oracle"):
        raise ValueError("Structural oracle implementation changed after calibration")
    if report["settings"].get("software_versions")!=software_versions():
        raise ValueError("Scientific package versions changed after calibration")
    if frozen.get("development_targets")!=report["settings"].get("target_fingerprints") or frozen.get("generator_configs")!=report.get("generators"):
        raise ValueError("Frozen generator/target provenance differs from the calibration report")
    journal=path.parent/"trials.jsonl"
    if not journal.is_file() or sha256(journal)!=report.get("journal_sha256"):
        raise ValueError("Calibration trial journal is missing or changed")
    records=[json.loads(line) for line in journal.read_text().splitlines()]
    complete={r["call_id"]:r for r in records if r["status"]=="completed"}
    started={r["call_id"]:r for r in records if r["status"]=="started"}
    expected_calls=report["settings"]["expected_calls"]
    if (set(complete)!=set(started) or set(complete)!=set(range(1,expected_calls+1))
            or len(records)!=2*expected_calls or report["calls"]!=expected_calls):
        raise ValueError("Calibration has incomplete or missing expensive calls")
    if report["settings"].get("center_decodes")!=2:
        raise ValueError("Calibration must verify production-setting repeatability on development data")
    for call_id,row in complete.items():
        if any(row[k]!=started[call_id][k] for k in ("target_id","radius","sample_index","output")):
            raise ValueError("Calibration call identity changed between reservation and completion")
    if report["selected_radius"]!=frozen.get("radius") or frozen["radius"] not in [.1,.25,.5,1.,2.]:
        raise ValueError("Invalid frozen radius selection")
    if not 20<=report["settings"]["samples_per_radius"]<=50:
        raise ValueError("Calibration does not meet the required sample count")
    by_id={t.target_id:t for t in targets}
    development_ids={t["target_id"] for t in frozen["development_targets"]}
    if set(frozen["generator_configs"])!=development_ids:
        raise ValueError("Calibration lacks generator provenance for a development target")
    for fingerprint in frozen["development_targets"]:
        target=by_id.get(fingerprint["target_id"])
        if target is None or target.split!="dev" or target.fingerprint()!=fingerprint:
            raise ValueError("Calibration development targets are missing, changed, or mislabeled")
        if not report["summaries"][target.target_id][str(frozen["radius"])]["qualifies"]:
            raise ValueError("Selected radius did not qualify for every development target")
        subset=[r for r in complete.values() if r["target_id"]==target.target_id]
        expected={(0.0,-1),(0.0,-2)} | {(r,i) for r in [.1,.25,.5,1.,2.] for i in range(report["settings"]["samples_per_radius"])}
        if len(subset)!=len(expected) or {(r["radius"],r["sample_index"]) for r in subset}!=expected:
            raise ValueError("Calibration is missing a full-D radius/sample cell")
        for radius in [.1,.25,.5,1.,2.]:
            measurements=[r for r in subset if r["radius"]==radius]
            if summarize_radius(measurements,report["settings"])!=report["summaries"][target.target_id][str(radius)]:
                raise ValueError("Calibration summary differs from its per-trial measurements")
        latent_report=path.parent/target.target_id/"latent_report.json"
        if not latent_report.is_file() or json.loads(latent_report.read_text()).get("generation_deterministic_adapter") is not True:
            raise ValueError("Development decoder repeatability was not verified")
        centers=[r for r in subset if r["radius"]==0]
        coordinate_paths=[path.parent/target.target_id/name/"atom_coords.pt" for name in ("center","center_repeat")]
        if not all(r["valid"] for r in centers) or not all(p.is_file() for p in coordinate_paths):
            raise ValueError("Development repeatability evidence is missing or invalid")
        first,second=(torch.load(p,weights_only=True,map_location="cpu") for p in coordinate_paths)
        if not torch.isfinite(first).all() or not torch.equal(first,second):
            raise ValueError("Development repeat coordinates differ or are nonfinite")
    if not frozen["development_targets"]:
        raise ValueError("Calibration must include development data")
    eligible=[r for r in [.1,.25,.5,1.,2.] if all(report["summaries"][t][str(r)]["qualifies"] for t in development_ids)]
    if not eligible or min(eligible)!=frozen["radius"]:
        raise ValueError("Frozen radius is not the smallest qualifying development radius")
    # The same generator and diffusion schedule apply to every method and target.
    keys=("checkpoint_sha256","sampling_steps","recycling_steps","sampler_seed","precision","use_kernels","diffusion_samples","sample_source_sha256","adapter_source_sha256","native_noise_source_sha256")
    if "sampler_mode" in generator_config or any("sampler_mode" in config for config in frozen["generator_configs"].values()):
        keys += ("sampler_mode",)
    for config in frozen["generator_configs"].values():
        if any(k not in config or k not in generator_config or config[k]!=generator_config[k] for k in keys):
            raise ValueError("Benchmark generator differs from the calibrated generator")
    return frozen["radius"]

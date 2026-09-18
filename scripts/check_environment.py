#!/usr/bin/env python3
"""Run real CUDA operations and save the observed hardware/software report."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/hardware_report.json"))
    parser.add_argument("--require-gpus", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F

    report = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "devices": [], "packages": {}}
    for name in ["boltz", "numpy", "pytorch-lightning", "rdkit"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"],
                                capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = str(exc)
    for index in range(torch.cuda.device_count()):
        entry = {"index": index, "name": torch.cuda.get_device_name(index),
                 "capability": list(torch.cuda.get_device_capability(index)),
                 "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
        try:
            with torch.cuda.device(index):
                device = f"cuda:{index}"
                x = torch.randn(32, 32, device=device, requires_grad=True)
                y = x @ x.T
                y.sum().backward()
                singular_values = torch.linalg.svdvals(y.detach())
                distances = torch.cdist(x.detach(), x.detach())
                q = torch.randn(1, 2, 32, 16, device=device)
                attention = F.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize(index)
                entry["fp32_checks_passed"] = all(bool(t.isfinite().all()) for t in
                    (y, x.grad, singular_values, distances, attention))
                entry["bf16_supported"] = torch.cuda.is_bf16_supported(including_emulation=False)
        except Exception as exc:
            entry.update(fp32_checks_passed=False, error=f"{type(exc).__name__}: {exc}")
        report["devices"].append(entry)
    report["passed"] = (len(report["devices"]) >= args.require_gpus
                         and all(d["fp32_checks_passed"] for d in report["devices"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

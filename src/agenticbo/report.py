"""Small, ledger-checked reports. Historical outputs use the original reader."""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

from targets.manifest import sha256


def load_v2(root):
    runs = []
    for path in sorted(root.rglob("results.json")):
        context_path = path.parent.parent / "run_context.json"
        if not context_path.exists():
            continue
        context = json.loads(context_path.read_text())
        if context.get("format_version") != 2:
            raise ValueError("Report historical and v2 outputs separately")
        run = json.loads(path.read_text())
        with sqlite3.connect((path.parent / "state.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute("SELECT id,candidate,status,result,physical_call FROM trials ORDER BY id").fetchall()
            saved = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM metadata")}
            settings = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM settings")}
            vectors = [np.frombuffer(row[0], dtype=np.float64).copy() for row in db.execute(
                "SELECT candidates.vector FROM trials JOIN candidates ON candidates.id=trials.candidate ORDER BY trials.id")]
        trials = [{**json.loads(r[3] or "{}"), "trial_id": f"trial_{r[0]:06d}",
                   "candidate_id": f"cand_{r[1]:06d}", "status": r[2], "physical_call": bool(r[4])} for r in rows]
        if (not run["completed"] or any(t["status"] != "completed" for t in trials)
                or trials != run["trials"] or saved["run"] != {**context, "method": run["method"]}
                or saved["budget"] != run["budget"] or run["budget_used"] != len(trials)
                or settings.get("stop_decision") != run.get("stop_decision")
                or not (len(trials) == run["budget"] or run.get("stop_decision") and len(trials) < run["budget"])):
            raise ValueError(f"Incomplete or inconsistent run: {path}")
        if any(v.shape != (saved["dimension"],) or not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in vectors):
            raise ValueError(f"Invalid saved candidate vectors: {path}")
        if context["initial"]:
            artifact = path.parent.parent / "shared_initial" / "observations.pt"
            # Ax uses the same shared initial candidates/results, but its
            # backend does not persist the runner setting that the native
            # methods use to record that source hash.  The artifact hash and
            # candidate/result equality checks below still validate Ax's
            # shared initialization identity.
            if (sha256(artifact) != run["shared_initial_sha256"]
                    or (run["method"] not in {"ax", "ax_saasbo"}
                        and settings.get("initial_source_sha256") != run["shared_initial_sha256"])):
                raise ValueError("Shared initialization identity changed")
            shared = torch.load(artifact, weights_only=True)
            if not np.array_equal(np.asarray(vectors[:context["initial"]]), shared["x"].numpy()):
                raise ValueError("Shared initial candidates changed")
            for observed, expected in zip(trials[:context["initial"]], shared["results"]):
                if any(observed.get(k) != expected.get(k) for k in ("objective", "valid", "metrics")):
                    raise ValueError("Shared initial results changed")
        valid = [t for t in trials if t.get("valid")]
        best = max(valid, key=lambda t: t["objective"]) if valid else None
        if (run.get("incumbent") or {}).get("objective") != (best or {}).get("objective"):
            raise ValueError("Exported incumbent differs from valid observations")
        runs.append({**run, "context": context})
    return runs


def curve(run):
    """Failures consume x-axis positions but never improve the incumbent."""
    values, best = [], None
    for trial in run["trials"]:
        if trial.get("valid"):
            best = trial["objective"] if best is None else max(best, trial["objective"])
        values.append(np.nan if best is None else best)
    if not values:
        values = [np.nan]
    values.extend([values[-1]] * (run["budget"] - len(values)))
    result = np.asarray(values)
    if run["context"]["metadata"]["kind"] == "benchmark":
        result = run["context"]["metadata"]["evaluator"]["optimum"] - result
        if np.any(result < -1e-7):
            raise ValueError("Observed score exceeds known optimum")
        result = np.maximum(result, 0)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path, output: Path):
    contexts = list(root.rglob("run_context.json"))
    if not contexts:
        raise ValueError(f"No runs found in {root}")
    versions = {json.loads(p.read_text()).get("format_version", 1) for p in contexts}
    if versions == {1}:
        from analysis.aggregate import load_runs
        from analysis.plots import render
        return render(load_runs(root), output)
    if versions != {2}:
        raise ValueError("Report historical and v2 outputs separately")
    runs = load_v2(root)
    if not runs:
        raise ValueError("No completed method results found")
    groups = {}
    for run in runs:
        metadata = run["context"]["metadata"]
        name = metadata["evaluator"]["problem"] if metadata["kind"] == "benchmark" else run["target_id"]
        groups.setdefault(name, []).append(run)
    output.mkdir(parents=True, exist_ok=True)
    if len(groups) > 1:
        for name in groups:
            # CLI outputs have one directory per problem/target.
            report(root / name, output / name)
        return {"groups": list(groups)}

    signatures = {json.dumps({k: r["context"][k] for k in
                              ("budget", "initial", "acquisition_settings", "fit_maxiter", "methods")}, sort_keys=True)
                  + json.dumps({k: v for k, v in r["context"]["metadata"].items() if k != "evaluator"}, sort_keys=True)
                  for r in runs}
    if len(signatures) != 1:
        raise ValueError("Cannot pool runs with different experiment settings")
    expected = set(runs[0]["context"]["methods"])
    identities = {json.dumps(r["controller_identity"], sort_keys=True) for r in runs if r["method"] == "agentic_dsp"}
    if len(identities) > 1:
        raise ValueError("Cannot pool different controller identities")
    for seed in {r["seed"] for r in runs}:
        if {r["method"] for r in runs if r["seed"] == seed} != expected:
            raise ValueError("Each seed must contain all selected methods")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    rows, finals = [], []
    for method in sorted(expected):
        selected = [r for r in runs if r["method"] == method]
        values = np.stack([curve(r) for r in selected])
        median, q25, q75 = [], [], []
        for index in range(values.shape[1]):
            column = values[:, index]
            valid = column[np.isfinite(column)]
            # Do not quietly drop failed seeds from the aggregate statistic.
            stats = np.quantile(valid, [.5, .25, .75]) if len(valid) == len(selected) else [np.nan] * 3
            median.append(stats[0]); q25.append(stats[1]); q75.append(stats[2])
            rows.append({"method": method, "evaluations": index + 1, "median": stats[0],
                         "q25": stats[1], "q75": stats[2], "seeds": len(selected),
                         "seeds_with_valid_incumbent": len(valid)})
        ax.plot(range(1, len(median) + 1), median, label=method)
        ax.fill_between(range(1, len(median) + 1), q25, q75, alpha=.15)
        for run in selected:
            finals.append({"method": method, "seed": run["seed"], "budget": run["budget"],
                           "evaluations_used": run["budget_used"], "termination": run["termination"],
                           "valid_evaluations": sum(t["valid"] for t in run["trials"]),
                           "best_objective": (run["incumbent"] or {}).get("objective"),
                           "final_curve_value": curve(run)[-1]})
    metadata = runs[0]["context"]["metadata"]
    metric = "Simple regret" if metadata["kind"] == "benchmark" else f"Best {metadata['objective_name']}"
    ax.set(xlabel="Charged evaluations", ylabel=metric, title=next(iter(groups)))
    ax.legend(); fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160); plt.close(fig)
    write_csv(output / "curves.csv", rows)
    write_csv(output / "summary.csv", finals)
    result = {"format_version": 2, "runs": len(runs), "metric": metric,
              "early_stop_policy": "Carry incumbent forward; actual evaluation counts are in summary.csv"}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

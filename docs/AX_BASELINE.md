# Ax comparison and paper correspondence

The reference is `2608.00316v2.pdf`, especially Section 6 and Appendices A/B.
The paper counts real black-box evaluations, including initialization, and plots
best value or regret against that count. Proposals, GP predictions, and acquisition
queries do not consume that budget. Failed generator attempts in this repository
also consume budget and receive zero; they are never silently retried for free.

The paper specifies five initial Sobol trials for Ax, then default model-based BO.
It describes a Matérn GP and log expected improvement. Table 4 uses different total
budgets: Branin 50, Hartmann 100, Ackley 150/200, LCBench 80, reaction yield 40.
It has no protein benchmark, protein search radius, or universal N.

## Installed baseline

`ax-platform==0.5.0` is an optional dependency compatible with the repository's
existing `botorch==0.13.0`, `gpytorch==1.14`, and Titan X PyTorch installation.
Installation: `uv pip install --python .venv/bin/python --cache-dir .uv-cache
--constraint configs/constraints.txt '.[ax]'`.
The verified installed package set is saved in `requirements-ax-titan-x.lock`.

This version's actual default model is `SingleTaskGP` with an **RBF kernel**, and
the acquisition is `qLogNoisyExpectedImprovement`. These settings are recorded
from the fitted Ax model, not inferred from the paper. This is a versioned Ax
default baseline, **not an exact reproduction of the paper's Matérn baseline**.
There are no custom Ax kernel, prior, or acquisition overrides. Observations use
SEM=0 because the generator's downstream randomness is fixed.

## Comparable initialization and N

All selected methods receive the same saved z0, native dimensions, normalized
bounds, oracle, initial structures, and per-method budget N. By default the common
initial design includes the center plus Sobol points. This differs from the paper's
five pure Sobol Ax points and Sara's self-chosen opening; it preserves our controlled
comparison. Use `--initial 5` for a five-evaluation common opening.

Ax receives these observed points and immediately enters model-based search. Its
own additional Sobol warm-up is disabled to avoid spending initialization twice.
Ax remains alive during one continuous method segment, and each completed result
is fed back into that same generation strategy before the next proposal. After a
process restart, the authoritative SQLite ledger is replayed into a fresh Ax
client; a JSON snapshot and actual model/acquisition diagnostics are saved. A
changed objective, budget, or Ax version cannot silently reinterpret an existing
run.

For M methods, N evaluations each, and I shared initial evaluations:

* Charged method evaluations: M × N.
* Physical generator calls: I + M × (N − I).
* Development calibration/probes are separately logged and excluded from N.

The ordinary POC is N=80, I=16. Smoke/pilot defaults are N=20, I=5.
`--budget N --initial I` makes both explicit for smoke/pilot runs.

## Running

For a registered target with calibration:

```bash
scripts/run_poc.sh --target TARGET --smoke --objective tm \
  --budget 20 --initial 5 --methods sobol dsp_gp ax --output outputs/ax_tm
```

To include the agent, add `agentic_dsp` and configure a live Sara endpoint.

An exploratory target outside the original 100–200-residue POC can use
`--pilot --radius 0.25` instead of `--smoke`. The radius is explicitly uncalibrated;
outputs are labeled `protein_pilot` and cannot satisfy `--require-poc`. This is a
pilot setting, not a paper-derived protein radius. Use identical settings across
methods and a new output root for each experiment.

The standard four-method/five-target POC remains unchanged. Selected-method pilot
plots support Ax and show best TM-score against charged evaluations. A single seed
on one target is a pilot, not evidence of general superiority.

## Verification and 22PE input status

`outputs/ax_synthetic_tm` contains a completed N=20/I=5 comparison of Ax,
DSP-GP, and Sobol at D=288, using explicitly analytic score stand-ins. This
verified actual Ax fitting/proposals, shared initialization, accounting, and plots;
it did not measure protein TM-scores. The full test suite passes (64 tests).

The supplied `inputs/22PE.fasta` has 349 residues and is registered in the manifest.
`--reference-mode resolved` maps the 342 resolved residues (positions 3–344),
chooses one alternate conformer per residue by atom coverage then occupancy,
and scores only those positions. Generation still uses all 349 residues.
TM-align is normalized by 342 reference residues. Original input files are preserved.
No real 22PE run is claimed. See [the experiment commands](22PE_TM.md).

# Run instructions

Run commands from the repository root. The commands source `.env` through the
wrapper scripts, so Sara uses the configured llama.cpp endpoint and model.

There is currently no experiment running. Start the Sara server in a separate
terminal only when a run includes `agentic_dsp`:

```bash
SARA_GPU=3 scripts/serve_sara_llamacpp.sh
```

The server is optional for Ax, Sobol, and vanilla GP runs.

## Default calibrated protocol

The default configuration follows the paper-style comparison: four methods,
the configured composite objective `sqrt_tm_times_lddt`, N=80 total evaluations
per method, and 16 shared initial observations. It requires a registered
100–200-residue test target and a completed development calibration:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_poc.sh \
  --target TARGET_ID --seed 0 \
  --output outputs/poc
```

The default methods are `sobol vanilla_gp dsp_gp agentic_dsp`. Replace
`TARGET_ID` with a target in `data/targets/manifest.csv`.

Once all five test targets and calibration are available, launch the complete
five-target × three-seed suite across the four GPUs:

```bash
scripts/run_suite.sh --objective sqrt_tm_times_lddt \
  --gpus 0 1 2 3 --output outputs/poc_suite
```

Use `--objective tm` for the equivalent full TM-score suite. The suite always
uses the configured four methods, N=80, and 16 initial observations; method
subsets and custom N are supported by the single-target pilot command below.

## TM-score-only protocol

For TM-score as the only optimization objective, pass `--objective tm`:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_poc.sh \
  --target TARGET_ID --seed 0 --objective tm \
  --output outputs/poc_tm
```

The objective is recorded in the run metadata and cannot be mixed with composite
objective runs during analysis.

## Selecting methods

`--methods` accepts one or more of `sobol`, `vanilla_gp`, `dsp_gp`, `agentic_dsp`,
and `ax`. For example, the paper comparison requested for 22PE is Ax versus
Sara versus Sobol, TM-score only:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_poc.sh \
  --target 22PE --reference-mode resolved --objective tm \
  --methods ax agentic_dsp sobol \
  --pilot --radius 0.25 --budget 20 --initial 5 --seed 0 \
  --output outputs/22PE_tm_retry
```

This is an explicitly uncalibrated pilot. It uses full 349-residue generation,
scores the 342 resolved reference positions, and allows Sara to stop before the
maximum N=20. Ax and Sobol continue to N=20. Repeat with `--seed 1` and
`--seed 2` using separate output roots when comparing seeds.

Useful subsets include:

```bash
# Sara only, TM-score pilot
CUDA_VISIBLE_DEVICES=0 scripts/run_poc.sh \
  --target 22PE --reference-mode resolved --objective tm \
  --methods agentic_dsp --pilot --radius 0.25 --budget 20 --initial 5 \
  --seed 0 --output outputs/22PE_sara_tm

# Ax versus Sobol, no LLM server required
CUDA_VISIBLE_DEVICES=2 scripts/run_poc.sh \
  --target 22PE --reference-mode resolved --objective tm \
  --methods ax sobol --pilot --radius 0.25 --budget 20 --initial 5 \
  --seed 0 --output outputs/22PE_ax_sobol_tm
```

The pilot output is written below
`<output>/pilot/22PE/seed_<seed>/`. Re-running the same command resumes its
ledger; a recorded Sara STOP is terminal. Use a new output directory for a new
campaign.

Sara’s computational-tool limit defaults to the total evaluation budget for the
run (`20` for this pilot, `80` for the full protocol). Override it explicitly
with `--max-computational-tools N` if desired. This is a per-campaign-step
limit for free probe/reconfigure/propose calls; EVALUATE and STOP are terminal
actions and do not count toward it.

## Analysis and inspection

After at least two methods finish, generate comparison CSVs and plots:

```bash
.venv/bin/python -m analysis.plots outputs/22PE_tm_retry \
  --output outputs/22PE_tm_retry_analysis
```

The reports preserve the maximum budget N and the actual evaluations used.
Sara early stopping is shown as an early endpoint; no unevaluated scores are
fabricated.

The main files are:

- `summary.csv`: per-method mean and standard deviation for best TM-score,
  objective, lDDT, validity rate, and evaluations used.
- `finals.csv`: one row per target/seed/method, including budget used and STOP
  status.
- `curves.csv`: best-so-far TM-score, lDDT, and objective at every real
  evaluation.
- `paired_differences.csv`: every available pairwise method difference for both
  objective and TM-score, including Sobol–Ax and Sobol–Sara.
- `thresholds.csv`: evaluations required to reach each TM-score
  threshold, with censored runs marked when a threshold is not reached.

To check the registered FASTA/reference mapping before spending GPU time:

```bash
.venv/bin/python scripts/validate_targets.py --reference-mode resolved
```

For 22PE, resolved mode retains all 349 residues for generation and scores
reference positions 3–344 (342 residues). The 0.25 pilot radius is provisional;
it is not the calibrated radius used by the full protocol.

## Paper-style synthetic checks

The separate [paper synthetic harness](benchmarks/paper_synthetic/README.md)
reproduces the paper's Branin/Ackley dimensions and evaluation budgets without
Boltz or protein data:

```bash
benchmarks/paper_synthetic/run.sh \
  --problems branin_2d ackley_10d ackley_20d \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Use `--dry-run` first or select one problem and two seeds for a quick check.

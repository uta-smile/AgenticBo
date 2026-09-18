# Agentic BO: benchmark first, Boltz next

A proof of concept of an LLM controlling a Bayesian optimizer: inspect
diagnostics, request candidates, adjust search strategy, then evaluate.
One optimization loop supports synthetic benchmarks and Boltz structure matching.

## Install

Use Python 3.10–3.12. From the repository root:

```sh
uv venv --python 3.12
uv pip install -e ".[ax,test]"
```

Activate the environment (`.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux). Commands below use that environment.
Synthetic benchmarks do not require Boltz, checkpoints, or a GPU.

For proteins, also install `uv pip install -e ".[boltz]"` and run
`python scripts/fetch_boltz_assets.py`. Install a PyTorch build suitable for your
GPU. The adapter uses FP32 with optional kernels and compilation disabled.
Existing Titan X locks describe the original Linux machine, not Windows setup.

## 1. Benchmark

Start with a short run without an LLM:

```sh
python -m agenticbo benchmark --problems branin_2d --seeds 0 --methods sobol dsp_gp --budget 8 --output outputs/benchmark_smoke
```

For Sara, configure `SARA_BASE_URL` (including `/v1`), `SARA_MODEL`, and optionally
`SARA_API_KEY` in your environment or `.env`. The endpoint must support chat
completion tool calls. See [the llama.cpp guide](docs/SARA_LLAMA_CPP.md).
Then run the full subset:

```sh
python -m agenticbo benchmark
```

[configs/benchmark.yaml](configs/benchmark.yaml) defaults to Branin-2D (50
evaluations), Ackley-10D (150), Ackley-20D (200), three seeds (0, 1, 2), and Sobol/Ax/Sara.
Each method owns its history. Ax uses five initial Sobol trials; Sara chooses
evaluations from the start, with space-filling backend proposals until five
valid observations support GP fitting. Fixed DSP is an optional comparison.

This is a **paper-aligned subset**, not an exact numerical reproduction of
`2608.00316v2.pdf`. It uses your configured LLM, the existing DSP/RBF backend,
and Ax 0.5 defaults instead of the paper's complete Sara/lenz setup. The local
hidden transformation is a seeded periodic shift; its boundary behavior is a
local implementation choice. Sara receives no function name, formula, shift,
or optimum. Other paper benchmarks, LLAMBO, and Centaur are outside this POC.

## 2. New protein dataset

Put sequences in `inputs/`, reference PDB/mmCIF structures in `reference/`, and
register pairs in a CSV with the existing manifest columns:

```csv
target_id,fasta,reference,chain_id,reference_chain_id,length,split,msa,provenance
myprotein,inputs/myprotein.fasta,reference/myprotein.cif,A,A,150,test,,source description
```

Replace stale example rows whose files you do not have, or use a separate manifest.

```sh
python -m agenticbo protein --targets myprotein --radius 0.25
```

Use `--manifest path/to/manifest.csv` for another dataset. The example radius is
an explicit exploratory choice, **not a calibrated recommendation**. Defaults in
[configs/protein.yaml](configs/protein.yaml): one seed, 20 evaluations,
five shared initial observations, TM-score optimization, and Sobol/DSP/Sara.
Add `--methods sobol dsp_gp` to run without an LLM, or
`--reference-mode resolved` for a supported partially resolved reference.

Candidates change the full initial Boltz diffusion-noise tensor. Weights,
conditioning, and later sampler randomness stay fixed. References are used only
for scoring. lDDT and RMSD remain diagnostics. There is no five-target or
100–200-residue restriction; runtime and memory still depend on target size.

## Results and resume

```sh
python -m agenticbo report outputs/benchmark
```

Outputs are grouped by problem/target and seed. `run_context.json` records the
resolved configuration and identities; SQLite stores candidate vectors and
trials. Method results, agent traces, and protein structures remain on disk.
Reports contain `comparison.png`, `curves.csv`, and per-seed `summary.csv`.
Synthetic plots show simple regret; protein plots show the best objective.
Stopped runs carry their incumbent forward in aggregate curves, with actual
evaluation counts recorded separately. Missing valid incumbents stay missing.

Repeat the same command to resume. Changed settings require a new output
directory. Failed evaluations consume budget but never improve the incumbent
or train the GP. Live or unverifiable pending calls are not rerun.
`--save-gp` enables detailed GP snapshots; they are off by default.

Use `--config file.yaml` for partial overrides; CLI options take precedence.
Paths are relative to `--root` (the checkout by default). New runs use output
format 2; `report` reads historical outputs separately. No in-place migration
of existing results is performed.

## Code and verification

- `src/agenticbo/`: CLI, evaluators, concise reporting.
- `src/runner/`, `src/dsp/`, `src/sara/`, `src/baselines/`: shared loop, ledger,
  numerical backend, controller, and comparisons.
- `src/generator/`, `src/latent/`, `src/oracle/`, `src/targets/`: native-noise
  mapping, Boltz, and structural scoring.

```sh
python -m pytest -q
```

The full historical test suite also needs the Boltz/structural extra. For a
synthetic-only installation, run `python -m pytest -q tests/test_poc.py`.

Unit and scripted-controller tests are engineering checks, not paper results.
A live LLM run and real checkpoint decode are separate integration checks.
The [historical protocol](docs/LEGACY_PROTOCOL.md), calibration scripts, and
multi-GPU suite remain advanced tools outside the default POC. Old shell
entry points are compatibility utilities; use `python -m agenticbo` for new work.

# Agentic BO

An LLM controlling a Bayesian optimizer: it inspects diagnostics, requests
candidates, adjusts search strategy, then evaluates. One optimization loop serves
two purposes:

1. **Measure the paper's synthetic benchmarks** (`2608.00316v2.pdf`).
2. **Run protein experiments** — native-noise search against a frozen Boltz-2
   decoder, scored by structural similarity to a reference.

Before quoting any number as a paper result, read
[docs/PAPER_COMPARISON.md](docs/PAPER_COMPARISON.md). This is a paper-aligned
subset, not an exact reproduction.

## Setup

This machine is an awkward host: Ubuntu 16.04 (glibc 2.23, gcc 5.5) with four
GTX TITAN X (Maxwell, compute capability 5.2) on driver 465.19.01. Modern Python
sdists will not compile on it, and the driver caps CUDA at 11.x. Both install
paths below account for that; neither will work if you "upgrade" PyTorch past
cu118. See [docs/DOCKER.md](docs/DOCKER.md) for the full reasoning.

### Docker (recommended)

```sh
docker-compose build
docker-compose run --rm bo python -m pytest -q
```

Prefix any command below with `docker-compose run --rm bo`. Select GPUs with
`GPUS=0`. Source and outputs are bind-mounted, so edits need no rebuild.

### Host virtualenv

Order matters: PyTorch from the cu118 index first, then the verified lock, then
the project. Resolving from `pyproject.toml` alone pulls sdists this host cannot
build.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python -r requirements-ax-titan-x.lock
uv pip install --python .venv/bin/python -e . --no-deps
```

`requirements-ax-titan-x.lock` includes Ax, Boltz and the structural extras.
Use `requirements-titan-x.lock` for the same environment without Ax.

For proteins, also fetch the decoder weights:

```sh
.venv/bin/python scripts/fetch_boltz_assets.py
```

## 1. Benchmarks

A short run without an LLM:

```sh
python -m agenticbo benchmark --problems branin_2d --seeds 0 \
  --methods sobol dsp_gp --budget 8 --output outputs/smoke
```

The full subset needs a Sara endpoint. Configure `SARA_BASE_URL` (including
`/v1`), `SARA_MODEL` and optionally `SARA_API_KEY` in `.env`; the endpoint must
support chat-completion tool calls. See
[docs/SARA_LLAMA_CPP.md](docs/SARA_LLAMA_CPP.md).

```sh
python -m agenticbo benchmark
```

[configs/benchmark.yaml](configs/benchmark.yaml) defaults to Branin-2D (50
evaluations), Ackley-10D (150), Ackley-20D (200), and Sobol/Ax/Sara. Each method
owns its history. Ax uses five initial Sobol trials; Sara chooses evaluations
from the start, with space-filling backend proposals until five valid
observations support GP fitting. Fixed DSP is an optional comparison.

## 2. Protein experiments

Put sequences in `inputs/`, reference PDB/mmCIF structures in `reference/`, and
register the pair in the manifest:

```csv
target_id,fasta,reference,chain_id,reference_chain_id,length,split,msa,provenance
myprotein,inputs/myprotein.fasta,reference/myprotein.cif,A,A,150,test,,source description
```

```sh
python -m agenticbo protein --targets myprotein
```

Use `--manifest path/to/manifest.csv` for another dataset, `--methods sobol dsp_gp`
to run without an LLM, and `--reference-mode resolved` for a partially resolved
reference. Defaults in [configs/protein.yaml](configs/protein.yaml): one seed, 20
evaluations, five shared initial observations, TM-score, Sobol/DSP/Sara.

Candidates set the full initial Boltz diffusion-noise tensor in a normalized
Gaussian latent space; no radius argument is required. Weights, conditioning and
later sampler randomness stay fixed. References are used only for scoring. lDDT
and RMSD remain diagnostics. Runtime and memory scale with target size.

A deterministic decoder comparison, plus the independent stochastic baseline:

```sh
python -m agenticbo protein --targets 22PE \
  --methods random sobol ax_saasbo dsp_gp --budget 20 --sampler-mode pf_ode \
  --output outputs/22PE_pf_ode
python scripts/best_of_n.py --target 22PE --count 20 --output outputs/22PE_best_of_20
```

> `inputs/22PE.fasta` and `reference/22PE.cif` are gitignored and absent from a
> fresh clone. Restore them before running the registered `22PE` target.

## Results and resume

```sh
python -m agenticbo report outputs/smoke
```

Outputs are grouped by problem/target and seed. `run_context.json` records the
resolved configuration and identities; SQLite stores candidate vectors and
trials. Reports contain `comparison.png`, `curves.csv` and per-seed `summary.csv`.
Synthetic plots show simple regret; protein plots show the best objective.
Stopped runs carry their incumbent forward, with actual evaluation counts
recorded separately.

Repeat the same command to resume. Changed settings require a new output
directory. Failed evaluations consume budget but never improve the incumbent or
train the GP. `--save-gp` enables detailed GP snapshots; off by default.

Use `--config file.yaml` for partial overrides; CLI options take precedence.
Paths are relative to `--root` (the checkout by default).

## Layout

- `src/agenticbo/` — CLI, evaluators, reporting. Single entry point.
- `src/runner/`, `src/dsp/`, `src/sara/`, `src/baselines/` — shared loop, ledger,
  GP backend, controller, comparisons.
- `src/generator/`, `src/latent/`, `src/oracle/`, `src/targets/` — native-noise
  mapping, Boltz, structural scoring.
- `src/analysis/` — aggregation, metrics, plots.
- `scripts/` — asset fetch, the best-of-N baseline, latent/decoder diagnostics,
  and the llama.cpp server launcher.

```sh
python -m pytest -q
```

Unit and scripted-controller tests are engineering checks, not paper results. A
live LLM run and a real checkpoint decode are separate integration checks.

**Three tests currently fail** (`tests/test_agent.py` x2,
`tests/test_poc.py::test_config_overrides_and_radius_required`). They assert
contracts the code intentionally dropped — a required `radius` argument, a
computational-tool cap, and one tool call per turn. They predate this refactor
and are not regressions; they need rewriting against the current agent contract.

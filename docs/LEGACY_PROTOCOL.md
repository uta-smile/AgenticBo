# Historical protein research protocol

This records the earlier full research workflow and historical verification
claims, not verification of this checkout or machine. Paths are relative to the
repository root. Use the root README for the supported POC workflow.

Work in progress toward the full research protocol in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).
The frozen generator is **Boltz2**. Search varies every coordinate of its initial
diffusion noise, with no projection or reduced search dimension.

## Current status

Implemented: the frozen Boltz2 adapter, full native-noise search, structural
oracle, development calibration, explicit DSP GP, all four methods, bounded
Sara tool loop, failure-inclusive budget ledger, four-GPU suite scheduler,
plots, and reproducibility comparison.

**Engineering verification has passed; the protein experiment is pending.**
Three real checkpoint decodes passed on a nine-residue synthetic fixture.
All four methods completed a synthetic 20-evaluation run, and a fresh repeat
produced identical candidates and scores. That synthetic controller was scripted.
A separate live Qwen3.5-9B/llama.cpp probe now verifies tool calling and a Sara
decision against real GP tools; no protein benchmark result is claimed.

22PE is registered for a TM-only pilot with resolved-reference scoring. See
[22PE experiment commands and paper action mapping](docs/22PE_TM.md).
Copy-paste commands for default, TM-only, and method-subset runs are in
[runInstruction.md](runInstruction.md).
The full protein suite still requires development-target calibration and additional targets. The local Sara server setup is
documented in [docs/SARA_LLAMA_CPP.md](docs/SARA_LLAMA_CPP.md).

## Your four Titan X GPUs

Verified hardware: four NVIDIA GeForce GTX TITAN X cards, about 12 GB per GPU,
compute capability 5.2, driver 465.19.01. PyTorch **2.6.0+cu118** passes FP32 matrix,
gradient, distance, SVD, and attention checks on all four cards. See
[outputs/hardware_report.json](outputs/hardware_report.json).

Use FP32 and disable optional Boltz kernels and compilation. These cards do not
support BF16. The adapter calls the model directly to avoid Boltz's CLI default
`bf16-mixed` Trainer. The scheduler runs one independent target/seed job per
GPU, each executing the four methods with sequential evaluations. Memory remains 12 GB per run; four GPUs
do not automatically form a 48 GB device. The tiny checkpoint smoke peaked at
2.065 GB of allocated GPU memory; real benchmark targets still need measurement.

The CUDA build choice follows [NVIDIA's CUDA 11 minor compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[NVIDIA's legacy GPU list](https://developer.nvidia.com/cuda/gpus/legacy), and
[PyTorch's 2.6 installation instructions](https://docs.pytorch.org/get-started/previous-versions/).
Runtime checks are the evidence for this machine's compatibility.

## Setup

The project `.venv` is installed on this machine. For a fresh environment:

```bash
uv venv .venv --python 3.12 --cache-dir .uv-cache
uv pip install --python .venv/bin/python --cache-dir .uv-cache torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python --cache-dir .uv-cache -r requirements-titan-x.lock -e .
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/fetch_boltz_assets.py
```

Use the SSD cache: this host's `/tmp` filesystem is nearly full. The asset command
downloads the official structure checkpoint and molecule archive, records their
SHA-256 hashes, and extracts the molecule data. It does not fetch affinity weights.
GPU commands in this agent session require execution outside the filesystem
sandbox, which does not expose `/dev/nvidia*`.

## Add your proteins

1. Put FASTA files in **[inputs/](inputs/README.md)**.
2. Put ground-truth PDB/mmCIF files in **[reference/](reference/README.md)**.
3. Add pairs to **[data/targets/manifest.csv](data/targets/manifest.csv)**.

Example manifest row (replace these file names and length with actual data):

```csv
target_id,fasta,reference,chain_id,reference_chain_id,length,split,msa,provenance
dev01,inputs/dev01.fasta,reference/dev01.cif,A,A,150,dev,,record source and selection here
```

Paths are relative to the repository root unless absolute. Each FASTA must contain
one protein. The reference chain's complete amino acid sequence must match that
FASTA and contain every Cα coordinate. Explicit local A3M paths are accepted in
`msa`; blank means fixed single-sequence conditioning. No sequence is submitted
to an MSA server. The ground truth is never included in generator input YAML.

Use separate development proteins for radius calibration and five test proteins
of 100–200 residues for the POC. Record provenance to assess training-set overlap;
being outside Boltz's training set cannot be inferred from a filename.

```bash
.venv/bin/python scripts/validate_targets.py
.venv/bin/python scripts/validate_targets.py --poc
```

## Phase 1: measure the native latent

[Boltz 2.2.1's sampler](https://github.com/jwohlwend/boltz/blob/v2.2.1/src/boltz/model/modules/diffusionv2.py)
initializes atom coordinates as `init_sigma * torch.randn(shape)`. We expose that
standard-normal draw as the native search variable. For one prediction:

```text
shape = [1, padded_atoms, 3]
D = 3 * padded_atoms
z ~ Normal(0, 1)
initial atom coordinates = init_sigma * z
```

All native padding entries are retained: the upstream sampler itself centers and
augments its coordinates, and removing padding would change its native interface.
The optimizer flattens only for GP input and reshapes before decoding.

The adapter injects noise at exactly the audited initialization expression.
It consumes the original RNG draw before replacement so later random draws stay
aligned. All later diffusion noise and augmentation are fixed by a sampler seed.
This defines a deterministic conditional map in principle; actual checkpoint
repeatability is tested explicitly, not assumed from setting the seed.

Source inspection alone needs neither targets nor weights:

```bash
.venv/bin/python scripts/inspect_latent.py --source-only
```

It writes `outputs/source_latent_report.json`, with `D: null` until actual features
are available. To measure a registered target (molecule assets required):

```bash
.venv/bin/python scripts/inspect_latent.py --target dev01 --seed 0
```

This saves `outputs/inspection/dev01/seed_0/latent_report.json`, an immutable
`z0.pt`, and target/source fingerprints. D is measured from Boltz's actual
featurized atom mask, never estimated from sequence length or chosen in config.

To test the frozen checkpoint and perturbation interface:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/inspect_latent.py --target dev01 --seed 0 --verify-decoder
```

This spends **three logged development decodes**: two repeats and one perturbation.
It reports repeat equality and coordinate changes, and keeps each generated CIF.
Finite coordinates alone do not prove structural validity or radius calibration.
The probe refuses test targets and refuses to overwrite an existing probe.

## Structural scoring and calibration

The oracle reports TM-align normalized by reference length, heavy-atom lDDT,
fixed-correspondence Cα RMSD, and the geometric mean objective. lDDT uses available
reference heavy atoms in different residues within 15 Å, distance thresholds
0.5/1/2/4 Å, penalties for missing generated atoms, and deterministic improving
swaps of equivalent side-chain atom names to a local optimum. Stereochemical
filtering is disabled. This implements the distance conservation definition from
the [lDDT paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3799472/); it is not claimed
to be bit-identical to OpenStructure's full scoring pipeline. The optional GDT-HA
diagnostic is not implemented. Invalid generated structures score zero.

```bash
.venv/bin/python scripts/smoke_oracle.py --fasta inputs/dev01.fasta --reference reference/dev01.cif --generated outputs/prediction.cif
.venv/bin/python scripts/calibrate_bounds.py --device cuda:0
```

Calibration uses only development targets, 30 shared normalized points at each
of r = 0.1/0.25/0.5/1/2 plus two repeated center decodes per target
(**152 calls per development target**). Center repeats must produce bit-identical
coordinates at the production diffusion settings before calibration continues. Every attempted decode,
including failures, is journaled. The declared defaults require at least 90%
valid outputs, 90% passing a Cα continuity check, and median center-relative RMSD
of at least 0.5 Å on every development target. The smallest passing radius is
saved with provenance in `outputs/calibration/frozen_radius.json`. If no radius
qualifies, none is frozen. These geometry checks are calibration diagnostics,
not all-atom physical validation. No calibration experiment has run yet.

## Optimization and controller

The GP operates in the complete normalized box, on CPU in float64. It uses an
explicit ARD RBF kernel with one lengthscale per native coordinate:

```text
DSP:     LogNormal(sqrt(2) + 0.5*log(D), sqrt(3))
Vanilla: LogNormal(sqrt(2), sqrt(3))
minimum lengthscale: 0.025
outputscale: fixed at 1
outcomes: standardized
noise: learned, with a small-noise prior
```

Sequential LogEI uses mixed global Sobol and local initializations with 16
restarts and 1,024 raw candidates. Failures and any raw-candidate fallback are
recorded. Full lengthscales, model state, kernel correlations, candidate
uncertainty, distances, timings, and structural scores are retained on disk.

Configure `SARA_BASE_URL`, `SARA_MODEL`, and optionally `SARA_API_KEY` in your shell;
see [.env.example](.env.example). The URL includes `/v1`. The endpoint must support
chat-completion tool calls. The controller can use at most four computational
tool calls, then select one backend candidate ID with `EVALUATE`. It can adjust
full-D local bounds or switch LogEI/UCB. Action traces, token usage when supplied
by the endpoint, and suggestion overrides are saved. The project `.env` selects
Qwen3.5-9B served by llama.cpp on `http://127.0.0.1:8080/v1`; the shell runners
load that file. Start it with `scripts/serve_sara_llamacpp.sh` when needed.
The verified profile uses GPU 3, so schedule Boltz on GPUs 0–2 while it is running.
See [the local server guide](docs/SARA_LLAMA_CPP.md) for model provenance and the
live tool-calling smoke test.

## Run the protein experiments

After development calibration and controller configuration:

```bash
# One registered target/seed; 20 evaluations per method, initial 5.
CUDA_VISIBLE_DEVICES=0 scripts/run_poc.sh --target test01 --seed 0 --smoke

# Five registered test targets × three seeds × four methods, budget 80/initial 16.
scripts/run_suite.sh --gpus 0 1 2
```

The runner verifies calibration provenance and matching generator settings,
recomputes radius summaries from per-trial measurements, checks the repeated
center coordinates, and requires the smallest qualifying radius.
Each target/seed shares one saved z0, calibrated bounds, and the exact same
initial observations across methods. Initial structures are generated once and
charged to every method: the full suite has **4,800 method-budget evaluations**
and **4,080 physical benchmark generator calls** (excluding development work
and any separate smoke experiment). Failed or interrupted evaluations consume
budget. SQLite records candidate vectors and reservations before decoding;
resume refuses to duplicate a live or unverifiable evaluation.

Reports appear under `outputs/poc/benchmark/<target>/seed_<seed>/`; suite logs
and status are under `outputs/poc/`. Completed suites automatically produce the
six requested plot types, kernel correlations, CSV summaries, and agent metrics.
To regenerate them:

```bash
.venv/bin/python -m analysis.plots outputs/poc/benchmark --output outputs/poc/analysis --require-poc
```

### TM-score-only runs

The default objective is `sqrt_tm_times_lddt`. To optimize TM-score alone, pass
`--objective tm` to either runner, or set `objective: tm` in the run YAML:

```bash
scripts/run_poc.sh --target test01 --seed 0 --smoke --objective tm --output outputs/poc_tm
scripts/run_suite.sh --gpus 0 1 2 3 --objective tm --output outputs/poc_tm
```

These commands still require registered targets, development calibration, and a
live controller for the agentic method. lDDT and RMSD remain recorded diagnostics.
The GP, incumbent selection, and Sara optimize the selected objective. Use a
separate output directory for each objective: resuming with a changed objective
is rejected, and reports refuse to pool different objectives.

## Verification evidence

Ax is available as an optional comparison method (`--methods sobol dsp_gp ax`).
See [the Ax protocol notes](docs/AX_BASELINE.md) for installation, explicit
`--budget N --initial I` controls, pilot runs, and differences from the PDF's
baseline. Ax 0.5.0 uses the existing BoTorch stack; its actual defaults are RBF
and noisy LogEI, so this is not an exact reproduction of the paper's Matérn Ax.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_dsp.py --latent-report outputs/generator_smoke/smoke_report.json
.venv/bin/python scripts/smoke_pipeline.py
```

All 65 tests pass with the optional Ax dependency installed. They cover native noise and RNG preservation, all-coordinate round trips,
structural scoring, DSP priors, finite GP/acquisition results, failed-call
budgeting, shared initialization, resume, bounded agent actions, calibration
provenance, aggregation, and scheduling 15 jobs across four GPU slots.

| Evidence | Measured scope |
| --- | --- |
| [Hardware report](outputs/hardware_report.json) | FP32 checks on all four GPUs |
| [Boltz2 smoke](outputs/generator_smoke/smoke_report.json) | Three actual checkpoint decodes; nine residues; shape [1,96,3], D=288; three diffusion steps, zero recycling |
| [GP smoke](outputs/dsp_smoke/summary.json) | Synthetic objectives at D=10, 100, 1000, and measured fixture D=288 |
| [Acquisition stress](outputs/dsp_stress_6392/summary.json) | Synthetic D=6392; finite full-D candidate and gradients |
| [Pipeline plots](outputs/synthetic_pipeline/analysis/analysis_report.json) | Four methods × 20 logical evaluations; analytic score stand-ins and scripted controller |
| [Fresh-repeat comparison](outputs/synthetic_reproduction/reproducibility.json) | Zero candidate/score differences across all four synthetic methods |
| [Live Sara probe](outputs/sara_llamacpp_smoke/report.json) | Qwen3.5-9B on GPU 3; predict/tool-result/EVALUATE round trip and one real controller selection with GP tools; synthetic observations, zero protein decodes |

For fresh-repeat verification, use a new output directory and compare runs:

```bash
.venv/bin/python scripts/smoke_pipeline.py --output outputs/synthetic_repeat_new
.venv/bin/python -m analysis.reproduce outputs/synthetic_pipeline/runs outputs/synthetic_repeat_new/runs --output outputs/synthetic_repeat_new/reproducibility.json
```

Production target dimensions, memory, runtime, repeatability, calibrated radius,
live controller behavior across protein runs, and scientific results remain unmeasured. The tiny
checkpoint smoke does not establish behavior for 100–200-residue proteins with
200 diffusion steps and three recycling steps. Optional GDT-HA is unimplemented.

[configs/poc_v1.yaml](configs/poc_v1.yaml) preserves the full protocol; its radius
is intentionally unset until calibration. [docs/STATUS.md](docs/STATUS.md) records
completed checks and the remaining experimental gates.

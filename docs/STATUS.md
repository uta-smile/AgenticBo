# Implementation and verification status

The full objective is preserved in [REQUIREMENTS.md](REQUIREMENTS.md).
The implementation is available, but the research POC is not complete.

## Implemented

- Local Qwen3.5-9B Q4_K_M served by a pinned llama.cpp source build (CUDA 11.3, sm_52) on GPU 3. Shell runners load `.env`; stable model/server identity is recorded through `SARA_RUNTIME_MANIFEST`. See `SARA_LLAMA_CPP.md`.

- Optional Ax 0.5.0 baseline, matched shared initialization, selectable pilot/smoke evaluation budgets, and selected-method comparison plots. The installed Ax defaults are RBF/noisy LogEI, not an exact reproduction of the PDF's stated Matérn baseline; see `AX_BASELINE.md`.

- Selectable TM-score-only optimization (`--objective tm` or YAML `objective: tm`), with lDDT/RMSD diagnostics, objective-aware Sara context, persistent scoring, and reports. The combined objective remains the default. Synthetic regressions cover both modes; no real TM-only protein experiment has run.

- Frozen Boltz2 2.2.1 FP32 adapter and audited native-noise injection, preserving every coordinate and the downstream RNG sequence.
- FASTA/reference validation, measured feature-derived dimensions, fixed conditioning, immutable z0, and full-dimensional normalization.
- Reference-normalized TM-align, heavy-atom distance lDDT, Cα RMSD, and geometric mean objective; invalid structures score zero.
- Development-only radius calibration with two production-setting repeat probes, 20–50 points at each prescribed radius, failure-inclusive journals, and provenance-checked frozen radius.
- Explicit DSP ARD RBF GP and short-prior vanilla ablation, CPU float64, standardization, learned small noise, mixed-start sequential acquisition optimization, and full diagnostics.
- Sobol, vanilla GP, fixed DSP GP, and agentic DSP methods with identical initial observations and exact budget enforcement.
- Sara chat-completion client, bounded strategy tools, candidate IDs, action/token logs, and immutable controller identity for recorded agentic runs.
- Persistent SQLite state, atomic generation reservations, process-identity checks for interrupted work, and run/suite locks.
- Four-GPU scheduling: 15 target/seed jobs, at most one job per GPU, all four methods per job.
- Budget/provenance-checked aggregation, all six requested plot types, kernel correlations, agent metrics, and fresh-repeat comparison.

## Verified evidence

- Full test suite with Ax: 65 passed. Python compilation and shell syntax checks pass. Dependency compatibility checks passed with Ax installed.
- Live Qwen3.5-9B probe passed: predict/tool-result/EVALUATE round trip and a natural Sara controller turn using real GP tools on synthetic observations. Sara requested local search and selected the resulting candidate. Zero protein decodes; 60.6 seconds for the full probe; approximately 22–24 generated tokens/second. Server uses about 5.4 GB on GPU 3. Evidence: `outputs/sara_llamacpp_smoke/report.json` and `outputs/sara_runtime.json`.
- Ax, DSP-GP, and Sobol completed N=20/I=5 synthetic TM-field stand-in runs at D=288: 60 charged evaluations, 50 physical analytic calls. Artifacts are under `outputs/ax_synthetic_tm`; these are not protein scores.
- User-supplied 22PE audited: 349-residue FASTA, 342 resolved reference residues (3–344), 15 residues with alternate atoms. Full-sequence features expose D=8352. Original files are preserved; resolved-reference scoring is implemented and selected explicitly with `--reference-mode resolved` and zero real 22PE decodes have run.
- All four GTX TITAN X GPUs pass FP32 matrix, backward, distance, SVD, and attention checks using PyTorch 2.6.0+cu118. Each GPU has about 12 GB; BF16 is unsupported.
- Official Boltz assets downloaded with SHA-256 receipts.
- Three actual checkpoint decodes passed for a nine-residue synthetic fixture at three diffusion steps and zero recycling: same z was bit-identical; perturbation changed output. D=288, shape [1,96,3], 70 active atoms; peak GPU allocation 2,064,936,960 bytes.
- Real GP/acquisition code passes synthetic checks at D=10, 100, 1000, 288, and 6392. The latter is a stress dimension, not a measured benchmark protein dimension.
- All four methods completed a synthetic budget-20/initial-5 run. The 80 logical evaluations used 65 analytic calls because initial observations were shared. A fresh repeat had zero differences in candidate vectors and scores.
- Plot generation and saved full ARD diagnostics verified on those synthetic runs. The controller was a scripted test double, not live Sara/Qwen.
- Calibration tests cover rejection of unvarying structures, charged failures, successful freezing with a deforming test double, and rejection of changed provenance. The runner recomputes summaries from per-trial RMSD/validity/geometry measurements, verifies repeated center coordinates, and enforces the smallest qualifying radius. Calibration and runner regression tests pass after this audit change. Scheduler tests use mock workers.

## Remaining experimental gates

1. Register separate development proteins and five diverse test proteins of 100–200 residues in `data/targets/manifest.csv`, with FASTA, matching reference chain, split, and provenance. The manifest currently has no target rows.
2. Inspect actual development features/native D and run production-setting decoder checks. Validate the oracle on selected reference structures. The lDDT implementation is not claimed identical to OpenStructure's full stereochemical/symmetry pipeline; optional GDT-HA is absent.
3. Run real development calibration and freeze r. Defaults spend 152 logged calls per development target. No real calibration or frozen radius exists yet.
4. Repeat GP/acquisition stress checks at actual benchmark dimensions.
5. Live Sara/Qwen interface verification is complete. Next validate sustained controller behavior in the protein pilot; keep GPU 3 reserved for the local LLM server.
6. Complete and inspect one real target/seed at budget 20/initial 5 across all four methods.
7. Run five targets × three seeds × four methods at budget 80/initial 16, then audit numerical/acquisition success and all failures.
8. Generate real structural plots and analyses, reproduce fresh seeded runs, and assess the scientific result regardless of which method wins.

22PE reference mapping is ready for an uncalibrated pilot. Separate development data is still needed for calibration.
Synthetic tests and the tiny Boltz smoke establish engineering behavior only;
they do not satisfy the requested protein benchmark or scientific success criteria.

Sara now supports budget-free STOP with a persisted feasible incumbent, terminal resume behavior, and deliberation history across evaluations. Reports distinguish the maximum budget from evaluations used. See [22PE TM commands](22PE_TM.md).

Verification after the action/reference changes: 72 tests passed, including STOP/resume, cross-step evaluation feedback, early-stop analysis, and the actual 22PE residue mapping. No real 22PE generation has run.

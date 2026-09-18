# Paper-style synthetic benchmark harness

This folder is separate from the protein runner. It benchmarks the existing
Sobol, Ax, and Agentic DSP implementations on the three unconstrained synthetic
problems used in the paper:

| Problem | Dimension | Budget per seed |
|---|---:|---:|
| Branin | 2 | 50 |
| Ackley | 10 | 150 |
| Ackley | 20 | 200 |

The harness uses the normalized unit cube, ten independent seeds by default, a
seeded hidden coordinate permutation and shift up to a quarter of each range,
and no natural-language benchmark name in Sara's context. It reports simple
regret (`known optimum - best observed score`) by evaluation count with median,
25th percentile, and 75th percentile across seeds.

Run it only after starting the local Sara server. From the repository root:

```bash
SARA_GPU=3 scripts/serve_sara_llamacpp.sh
```

In another terminal:

```bash
benchmarks/paper_synthetic/run.sh
```

The full paper-shaped run is expensive for Sara: 3 problems × 10 seeds and up
to 200 evaluations per method. Select a smaller smoke run first:

```bash
benchmarks/paper_synthetic/run.sh \
  --problems branin_2d --seeds 0 1 \
  --output outputs/paper_synthetic_smoke
```

Run one problem across all ten seeds with:

```bash
benchmarks/paper_synthetic/run.sh \
  --problems ackley_10d --seeds 0 1 2 3 4 5 6 7 8 9 \
  --output outputs/paper_synthetic_ackley10
```

Each problem writes per-seed ledgers under `<output>/<problem>/seed_<seed>/`.
Aggregate files are written under `<output>/<problem>/analysis/`, including
`paper_simple_regret.csv`, `paper_final_summary.csv`, and the existing method
comparison CSVs and plots.

The paper reports Ax with five Sobol initialization evaluations while Sara can
choose its opening strategy. This repository's shared runner deliberately gives
all selected methods the same five initial evaluations so Ax, Sara, and Sobol
are directly comparable. The script records that choice in metadata. It also
uses Qwen3.5:9b through llama.cpp, our current Ax 0.5 defaults, and our DSP
RBF backend; these differ from the paper's Claude Opus 4.8, Ax/Matérn report,
and lenz Matérn backend. The benchmark is therefore a paper-aligned regression
comparison, not a claim of exact numerical reproduction.

The synthetic score is an opaque maximization score in `[0, 1]`; it is not a
protein TM-score and no Boltz generation is performed.

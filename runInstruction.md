# Run instructions

The supported commands and setup are in [README.md](README.md):

```sh
python -m agenticbo benchmark --problems branin_2d --seeds 0 --methods sobol dsp_gp --budget 8 --output outputs/benchmark_smoke
python -m agenticbo protein --targets YOUR_TARGET --methods sobol dsp_gp --budget 20 --output outputs/protein_smoke
python -m agenticbo report outputs/benchmark_smoke
```

## Deterministic PF-ODE protein comparison

Run the optimizers on the same deterministic Boltz-2 decoder:

```sh
python -m agenticbo protein --targets 22PE --methods random sobol ax_saasbo dsp_gp --budget 20 --sampler-mode pf_ode --output outputs/22PE_pf_ode
```

Run the independent stochastic Best-K-of-N baseline:

```sh
python scripts/best_of_n.py --target 22PE --count 20 --output outputs/22PE_best_of_20
```

<!-- python -m agenticbo protein \
  --targets 22PE \
  --methods sobol ax_saasbo dsp_gp \
  --seeds 1265 \
  --budget 20 \
  --initial 5 \
  --sampler-mode stochastic \
  --reference-mode resolved \
  --output outputs/22PE -->


<!-- python benchmarks/paper_synthetic/run.py \
  --problems ackley_10d ackley_20d \
  --seeds 32 \
  --methods sobol ax agentic_dsp \
  --output outputs/benchmarkv2 -->

Protein generation uses the full-dimensional normalized Gaussian latent space;
no radius argument is required. Register your sequence and reference first.
See [the historical protocol](docs/LEGACY_PROTOCOL.md) for advanced
calibration and multi-GPU experiments.

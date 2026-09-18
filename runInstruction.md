# Run instructions

The supported commands and setup are in [README.md](README.md):

```sh
python -m agenticbo benchmark --problems branin_2d --seeds 0 --methods sobol dsp_gp --budget 8 --output outputs/benchmark_smoke
python -m agenticbo protein --targets YOUR_TARGET --radius 0.25
python -m agenticbo report outputs/benchmark_smoke
```

<!--   python -m agenticbo protein \
  --targets 22PE \
  --methods sobol ax agentic_dsp \
  --seeds 1265 \
  --budget 20 \
  --initial 5 \
  --radius 0.25 \
  --reference-mode resolved \
  --output outputs/22PE -->


<!-- python benchmarks/paper_synthetic/run.py \
  --problems ackley_10d ackley_20d \
  --seeds 32 \
  --methods sobol ax agentic_dsp \
  --output outputs/benchmarkv2 -->

The protein radius is an explicit exploratory choice. Register your sequence
and reference first. See [the historical protocol](docs/LEGACY_PROTOCOL.md) for
advanced calibration and multi-GPU experiments.

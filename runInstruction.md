# Run instructions

The supported commands and setup are in [README.md](README.md):

```sh
python -m agenticbo benchmark --problems branin_2d --seeds 0 --methods sobol dsp_gp --budget 8 --output outputs/benchmark_smoke
python -m agenticbo protein --targets YOUR_TARGET --radius 0.25
python -m agenticbo report outputs/benchmark_smoke
```

The protein radius is an explicit exploratory choice. Register your sequence
and reference first. See [the historical protocol](docs/LEGACY_PROTOCOL.md) for
advanced calibration and multi-GPU experiments.

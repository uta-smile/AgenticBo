# 22PE: Ax, Sara, and Sobol on TM-score

Start the Qwen3.5:9b llama.cpp server in a separate terminal if it is not already running:

```bash
SARA_GPU=3 scripts/serve_sara_llamacpp.sh
```

The existing `.env` selects `http://127.0.0.1:8080/v1` and `qwen3.5:9b`.
Run the three methods sequentially on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_poc.sh \
  --target 22PE --reference-mode resolved --objective tm \
  --methods ax agentic_dsp sobol \
  --pilot --radius 0.25 --budget 20 --initial 5 --seed 0 \
  --output outputs/22PE_tm_retry
```

This is an **uncalibrated pilot**: radius 0.25 is provisional. All methods use the
same full native search dimensions, bounds, five initial observations, oracle,
and maximum budget N=20 (including initialization). Sara can STOP early; Ax and
Sobol run to N. Compare both best observed TM-score and actual evaluations used.
With no early stop there are 60 charged method evaluations and 50 physical
protein generations, because the initial five are generated once and shared.
Failed generations also consume budget. Repeat with different `--seed` values
for a comparison across seeds. This command has not yet produced real 22PE results.

The command writes results and plots beneath
`outputs/22PE_tm_retry/pilot/22PE/seed_0/`. Repeating it resumes the same experiment;
a recorded STOP remains terminal. Use a new output path to start a new campaign.
Curves end at actual observations; an × endpoint marks STOP. When seeds stop at
different counts, individual seed curves are shown. To aggregate completed seeds:

```bash
.venv/bin/python -m analysis.plots outputs/22PE_tm_retry \
  --output outputs/22PE_tm_retry_analysis
```

Full 349-residue generation is preserved. The reference declares the same sequence,
but has coordinates for positions 3–344 only. Resolved mode selects those 342
positions from each full generated structure and normalizes TM-align by 342.
It chooses one alternate conformer per reference residue using atom coverage,
then occupancy, then alphabetical ties. Coverage and excluded positions are
recorded. lDDT and RMSD are diagnostics; only TM-score is optimized.

## Mapping to the paper, equations 4.2–4.4

| Paper action | Implemented tools | Effect on evaluation budget |
|---|---|---|
| probe | `predict`, `acquisition_score`, `diagnostics`, `incumbent`, `trials` | None |
| reconfigure | `set_search_radius`, `reset_bounds`, `set_acquisition` | None; configuration persists |
| propose | `suggest`, `suggest_local` | None |
| evaluate | `EVALUATE(candidate_id)` | One charged expensive evaluation |
| stop | `STOP(reason)` | None; returns best valid observed solution, or none if all invalid |

State contains the trial dataset D, persistent search configuration c, target
information K, and deliberation history H. Each computational response informs
the next LLM decision. Completed tool exchanges and measured evaluation responses
carry forward across campaign steps and restarts. Full completed history is saved
in SQLite; a bounded recent view and older action summaries enter the model context.

This is a constrained single-objective adaptation: setup and TM-score are configured
by the experiment, with no agent changes to objectives, constraints, original
bounds, or GP prior. There is no multi-objective Pareto mode. Each step permits up
to four explicit computational calls, followed by EVALUATE or STOP. An initial
advisory GP proposal is also supplied automatically. Surrogate predictions are
estimates, not substitute measured evaluations. See [Ax details](AX_BASELINE.md)
for backend and initialization differences from the paper.

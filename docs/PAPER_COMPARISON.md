# What this code measures, and where it differs from the paper

Reference: `2608.00316v2.pdf`, especially Section 6 and Appendices A/B.

This repository is a **paper-aligned subset**, not an exact numerical
reproduction. Read this before quoting any number from a run as a paper result.

## Budget accounting

The paper counts real black-box evaluations, including initialization, and plots
best value or regret against that count. Proposals, GP predictions and
acquisition queries do not consume budget.

This code follows that convention, with one addition: a failed generator attempt
also consumes budget and receives zero. It is never silently retried for free.

For M methods, N evaluations each, and I shared initial evaluations:

- Charged method evaluations: `M x N`
- Physical generator calls: `I + M x (N - I)`

The initial design is generated once and shared across methods, so physical
calls are fewer than charged evaluations.

## Known differences from the paper

| Paper | This code | Why it matters |
|---|---|---|
| Matern GP baseline | Ax 0.5.0 default: `SingleTaskGP`, **RBF** kernel, `qLogNoisyExpectedImprovement` | Recorded from the fitted Ax model, not inferred. There are no custom kernel/prior/acquisition overrides. This is a versioned Ax default baseline. |
| Five pure Sobol points for Ax, Sara opens for itself | All methods share the same initial design (center + Sobol) | Preserves a controlled comparison. Use `--initial 5` for a five-evaluation common opening. |
| Full Sara/lenz setup | Your configured local LLM plus the DSP/RBF backend in `src/dsp/` | The controller is whatever `SARA_MODEL` points at. |
| Per-problem budgets: Branin 50, Hartmann 100, Ackley 150/200, LCBench 80, reaction yield 40 | Branin 50, Ackley-10D 150, Ackley-20D 200 | Hartmann, LCBench and reaction yield are not implemented. |
| No protein benchmark | Boltz-2 native-noise protein matching | The protein work is an **extension**, not a paper reproduction. There is no paper-derived protein radius or universal N. |
| LLAMBO, Centaur comparisons | Absent | Out of scope here. |

The hidden transformation used by the synthetic problems is a locally chosen
seeded periodic shift; its boundary behaviour is an implementation choice. Sara
receives no function name, formula, shift or optimum.

Ax observations use `SEM=0` because the generator's downstream randomness is
fixed. Ax's own Sobol warm-up is disabled so initialization is not spent twice.
Ax stays alive for one continuous method segment, and each result is fed back
into the same generation strategy before the next proposal. After a restart the
SQLite ledger is replayed into a fresh Ax client. A changed objective, budget or
Ax version cannot silently reinterpret an existing run.

## Agent action mapping, paper equations 4.2-4.4

| Paper action | Implemented tools | Budget cost |
|---|---|---|
| probe | `predict`, `acquisition_score`, `diagnostics`, `incumbent`, `trials` | none |
| reconfigure | `set_search_radius`, `reset_bounds`, `set_acquisition` | none; configuration persists |
| propose | `suggest`, `suggest_local` | none |
| evaluate | `EVALUATE(candidate_id)` | one charged evaluation |
| stop | `STOP(reason)` | none; returns the best valid observation, or none if all were invalid |

State holds the trial dataset D, the persistent search configuration c, target
information K, and deliberation history H. Completed tool exchanges and measured
responses carry across campaign steps and restarts. The full history is in
SQLite; a bounded recent view plus older action summaries enter the model
context.

This is a constrained single-objective adaptation. The experiment fixes the
objective, constraints, original bounds and GP prior; the agent does not change
them. There is no multi-objective Pareto mode. Surrogate predictions are
estimates and never substitute for a measured evaluation.

## Status of protein results

The registered target `22PE` is 349 residues; `--reference-mode resolved` scores
the 342 resolved positions (3-344), choosing one alternate conformer per residue
by atom coverage, then occupancy, then alphabetical tie-break. Generation always
uses the full 349 residues. TM-align is normalized by the 342 reference residues.
lDDT and RMSD are diagnostics; only the selected objective is optimized.

A single seed on one target is a pilot, not evidence of general superiority.

# Agentic DSP-BO for Direct Protein 3D Latent-Space Target Matching

## 1. Main research question

Build a proof of concept answering:

> Can an LLM-controlled Bayesian optimizer using a dimensionality-scaled Gaussian Process efficiently optimize the **full native latent space** of a frozen protein 3D generator and recover a target protein structure under a limited evaluation budget?

The main contribution being tested is:

**Agentic high-dimensional Bayesian optimization.**

The experiment is NOT initially testing:

* a new protein generator,
* a new latent representation,
* dimensionality reduction,
* protein sequence design,
* biological function,
* or global protein-generation SOTA.

Everything except the optimization strategy should remain frozen.

---

# 2. Core optimization problem

Given:

$$
X^*
$$

a target/reference protein structure,

and a frozen generator:

$$
G(z)
$$

with native latent:

$$
z\in\mathbb R^D,
$$

find:

$$
z^*=
\arg\max_z f(G(z),X^*)
$$

where \(f\) measures structural similarity.

The pipeline is:

```text
full native latent z ∈ R^D
          ↓
frozen protein 3D generator
          ↓
generated structure X
          ↓
structural oracle
          ↓
TM-score
lDDT
RMSD
GDT-HA (optional)
          ↓
scalar BO objective
          ↓
DSP Gaussian Process
          ↓
Agentic BO
          ↓
next full D-dimensional latent
```

There is **no lower-dimensional projection**.

---

# 3. Important distinction: normalized coordinates are NOT a subspace

The GP should not necessarily receive raw native latent coordinates.

Define:

$$
x\in[0,1]^D
$$

with exactly the same dimensionality as \(z\).

Then map one-to-one:

$$
x\leftrightarrow z.
$$

Example:

```text
native latent:
D = 6,392

GP variable:
x.shape = [6,392]

decoder variable:
z.shape = [6,392]
```

No dimensions are removed.

This is only coordinate normalization.

The high-dimensional BO paper's DSP formulation is designed around appropriately scaled coordinates; current BoTorch documentation likewise recommends normalized covariates.

---

# 4. First task: inspect the generator's actual latent

Before implementing BO, create:

```text
scripts/inspect_latent.py
```

Report:

```text
native latent shape
flattened dimensionality D
dtype
mean
std
min
max
latent sampling distribution
whether latent is global or per-residue
whether latent is bounded
whether decoder accepts arbitrary perturbations
whether generation is deterministic
```

Example:

```text
latent shape:
[256, 64]

flattened D:
16384

distribution:
approximately N(0,1)

generator deterministic:
yes
```

If the latent is a tensor:

$$
z\in\mathbb R^{L\times d},
$$

flatten it only for BO:

$$
z_{\text{flat}}\in\mathbb R^{Ld}.
$$

Before decoding, restore the original tensor shape.

Do not change dimensionality.

---

# 5. Define the direct full-dimensional search box

BO requires finite bounds.

If the generator already defines valid latent bounds, use those.

Otherwise, if latent coordinates are approximately standardized Gaussian variables, construct a full-dimensional box around an initial latent \(z_0\):

$$
z_j
=
z_{0,j}
+
r\,s_j(2x_j-1)
$$

where:

$$
x_j\in[0,1]
$$

and:

* \(z_0\) = initial latent,
* \(s_j\) = latent scale for dimension \(j\),
* \(r\) = search-radius multiplier.

If latent coordinates are already standardized:

$$
s_j=1.
$$

Therefore:

$$
z_j\in[z_{0,j}-r,\;z_{0,j}+r].
$$

This remains a **full \(D\)-dimensional search**.

Every coordinate can independently change.

---

# 6. Calibrate the direct latent radius

Do not arbitrarily pick an enormous box.

Use development targets only.

Test:

```text
r = 0.1
r = 0.25
r = 0.5
r = 1.0
r = 2.0
```

For each radius:

```text
sample 20–50 full-D latent points
↓
decode structures
↓
measure validity
↓
measure structural variation
```

Choose a radius satisfying:

```text
high valid-generation rate
meaningful variation in structures
not all outputs nearly identical
not mostly catastrophic structures
```

Freeze `r` before benchmark experiments.

Do not tune it independently for individual test targets.

---

# 7. Structural oracle

Each expensive evaluation is:

$$
z
\rightarrow
G(z)
\rightarrow
\hat X
\rightarrow
\text{metrics}(\hat X,X^*).
$$

Compute:

### Primary

$$
TM\text{-score}
$$

and:

$$
lDDT.
$$

### Secondary diagnostics

```text
RMSD
GDT-HA
```

### Optional later

```text
MolProbity
clash score
```

if reliable all-atom structures are available.

Do not use:

```text
pLDDT
GFP brightness
scTM
binding affinity
```

for this target-matching POC.

---

# 8. Internal BO objective

Use:

$$
\boxed{
S(z)=\sqrt{TM(z)\times lDDT(z)}
}
$$

with both metrics in:

$$
[0,1].
$$

Reason:

```text
TM-score
→ global fold agreement

lDDT
→ local geometry agreement
```

The geometric mean penalizes a structure that performs strongly on only one.

Example:

```text
TM   = 0.90
lDDT = 0.40

objective = sqrt(0.90 × 0.40)
          = 0.60
```

rather than treating the high TM-score as sufficient.

This composite is an **internal optimization objective**, not a standard protein benchmark score.

Always report TM and lDDT individually.

---

# 9. Structural validation

Before scoring:

```text
parse structure successfully
expected chain exists
expected residue mapping works
required Cα coordinates exist
coordinates contain no NaN/Inf
structure is non-empty
residue count is sensible
```

If invalid:

```text
valid = false
TM = 0
lDDT = 0
objective = 0
```

The evaluation still consumes budget.

Never silently discard failed generations.

---

# 10. DSP Gaussian Process

This is the key replacement for the previous projected-space GP.

The surrogate operates directly on:

$$
x\in[0,1]^D.
$$

For each ARD dimension:

$$
\boxed{
\ell_i
\sim
\operatorname{LogNormal}
\left(
\sqrt2+\frac12\log D,\sqrt3
\right)
}
$$

for:

$$
i=1,\ldots,D.
$$

The characteristic prior lengthscale therefore increases approximately as:

$$
\sqrt D.
$$

This is the central DSP modification proposed by Hvarfner et al.; the paper demonstrates it on real-world problems extending to the 6,392-dimensional Humanoid task.

---

# 11. Exact GP implementation

For reproducibility, instantiate the components explicitly even though modern BoTorch already has these defaults.

Use:

```text
SingleTaskGP

kernel:
ARD RBF initially

lengthscale prior:
LogNormal(
    loc = sqrt(2) + 0.5 * log(D),
    scale = sqrt(3)
)

minimum lengthscale:
0.025

outputscale:
fixed at 1 / no learned ScaleKernel

outcome:
standardized

noise:
small learned noise
```

Current BoTorch implements the DSP kernel as:

```text
get_covar_module_with_dim_scaled_prior(...)
```

and currently defaults to RBF; the implementation uses exactly:

```text
loc = sqrt(2) + 0.5 * log(D)
scale = sqrt(3)
```

with a lower lengthscale constraint of `0.025`.

The authors also found RBF competitive or preferable across their high-dimensional experiments; BoTorch subsequently adopted RBF plus the DSP prior in its defaults.

---

# 12. Example DSP values

Automatically log these at startup.

For:

$$
D=256
$$

compute:

```text
loc = sqrt(2) + 0.5 log(256)
```

For:

$$
D=1024
$$

compute the corresponding value.

For:

$$
D=6392
$$

approximately:

```text
loc ≈ 5.796
scale ≈ 1.732
prior mode ≈ 16.4
```

Never hard-code the Humanoid value.

Always calculate:

```python
loc = sqrt(2) + 0.5 * log(D)
scale = sqrt(3)
```

from the actual protein latent dimensionality.

---

# 13. GP input/output transforms

Input:

$$
z\rightarrow x\in[0,1]^D.
$$

Output:

$$
S\rightarrow
\frac{S-\mu_S}{\sigma_S}.
$$

Use:

```text
Normalize(d=D)
Standardize(m=1)
```

or explicitly equivalent transformations.

The decoder receives native \(z\).

The GP receives normalized \(x\).

---

# 14. Acquisition function

Primary acquisition:

```text
LogExpectedImprovement
```

or its numerically stable BoTorch equivalent.

Because this POC is sequential:

```text
q = 1
```

one candidate is committed per expensive evaluation.

If generation/evaluation is deterministic:

```text
LogEI
```

is sufficient.

If meaningful stochasticity exists:

```text
LogNEI
```

should be considered.

The official high-dimensional BO repository supports qLogNEI experiments directly.

---

# 15. Acquisition optimization is a separate high-dimensional problem

Do not assume fitting the DSP GP solves everything.

You still need:

$$
\arg\max_x\alpha(x).
$$

Implement acquisition candidate generation robustly.

Use a mixture of:

```text
global Sobol initializations
+
candidate initializations around incumbent
+
multi-start gradient optimization
```

Do not rely on one L-BFGS initialization.

Initial POC settings can be approximately:

```text
raw acquisition candidates: 512–2048
restarts: 10–20
q: 1
```

Scale these based on \(D\) and GPU memory.

Record acquisition-optimization failures.

---

# 16. Initial experimental design

Every optimization method must start from exactly the same evaluations.

POC-v1:

```text
total expensive budget:
80

initial design:
16

adaptive evaluations:
64
```

Use seeded Sobol sampling in the full normalized box:

$$
[0,1]^D.
$$

Also explicitly include the center corresponding to:

$$
z=z_0.
$$

If 80 generator calls are too expensive, smoke-test first with:

```text
budget = 20
initial = 5
adaptive = 15
```

but final POC should use a larger budget.

---

# 17. Mandatory baselines

The key experiment should have four methods.

## A. Sobol search

No surrogate.

```text
full D-dimensional latent
same bounds
same budget
```

Purpose:

> Does optimization beat space-filling search?

---

## B. Vanilla GP-BO

Use conventional/default short-scale prior intentionally.

Same:

```text
full latent
same acquisition
same initial observations
same budget
```

Purpose:

> Does DSP actually matter for our protein latent problem?

---

## C. DSP GP-BO

Use:

$$
\ell_i
\sim
LogNormal
\left(
\sqrt2+\frac12\log D,\sqrt3
\right).
$$

Fixed acquisition strategy.

Purpose:

> Can high-dimensional direct BO work without an LLM?

---

## D. Agentic DSP-BO

Use the exact same DSP GP.

Sara/LLM controls strategy.

Purpose:

> Does agentic orchestration provide value beyond DSP itself?

---

# 18. The critical ablation

Your most important comparison is:

```text
Vanilla GP
      vs
DSP GP
      vs
Agentic DSP GP
```

This isolates two separate contributions.

### Question 1

```text
Vanilla GP
vs
DSP GP
```

answers:

> Does the dimensionality-aware prior make direct protein latent optimization viable?

### Question 2

```text
DSP GP
vs
Agentic DSP GP
```

answers:

> Does the agent improve optimization after the high-dimensional surrogate has already been fixed?

This is much stronger than comparing Agentic BO only to random search.

---

# 19. Agent role

The LLM does NOT directly manipulate protein coordinates.

It controls the BO strategy.

At every iteration it receives a compact state containing:

```text
target ID
target length

D
latent bounds

budget used
budget remaining

current incumbent objective
TM
lDDT
RMSD

recent trials

GP diagnostics

acquisition state
```

Do not send a 6,392-dimensional vector as natural-language text unless needed.

Store full vectors in backend state and refer to them by IDs.

Example:

```text
trial_17
incumbent_3
candidate_5
```

---

# 20. Why IDs matter in high dimensions

Do NOT make Qwen print:

```text
[0.1742, -0.8831, ... 6392 numbers ...]
```

The LLM should make strategic decisions, not serialize huge tensors.

Instead:

```text
Sara:
suggest()

backend:
candidate_id = cand_023
```

Then Sara can say:

```text
EVALUATE(candidate_id="cand_023")
```

The backend owns the actual vector.

This drastically reduces context length and numerical errors.

---

# 21. Lenz/DSP backend tools

Expose:

```text
suggest()
```

Return the best DSP acquisition candidate.

---

```text
suggest_local(
    center="incumbent",
    radius=...
)
```

Still searches all \(D\) dimensions, but within a smaller full-D box.

This is NOT a subspace.

---

```text
predict(candidate_id)
```

Return:

```text
posterior mean
posterior std
```

---

```text
acquisition_score(candidate_id)
```

---

```text
incumbent()
```

Return:

```text
trial ID
objective
TM
lDDT
RMSD
```

---

```text
diagnostics()
```

Return summarized high-dimensional diagnostics:

```text
GP fit status
number observations
noise
posterior uncertainty
acquisition range

lengthscale summary:
    minimum
    median
    maximum

10 dimensions with shortest lengthscales

10 dimensions with longest lengthscales

recent improvement rate
```

Do NOT dump all thousands of ARD lengthscales into the LLM context.

---

```text
trials(last_n=...)
```

Return a compact history.

---

```text
set_search_radius(radius)
```

Changes the active full-D box around the incumbent.

Example:

```text
radius = 1.0
radius = 0.5
radius = 0.25
```

All \(D\) dimensions remain searchable.

---

```text
reset_bounds()
```

Restore full calibrated latent box.

---

```text
set_acquisition(...)
```

Initially allow:

```text
log_ei
ucb
```

---

# 22. Agent actions that are NOT allowed

Do not let the agent:

```text
delete dimensions
project latent space
construct PCA subspaces
freeze thousands of dimensions
change generator weights
edit structures directly
fabricate scores
evaluate multiple candidates in one turn
change the structural oracle
```

The primary experiment must remain **direct full-space optimization**.

---

# 23. Agent loop

For iteration \(t\):

```text
build compact optimization context
        ↓
Sara reasons about optimization strategy
        ↓
up to 4 computational tool calls
        ↓
Sara selects candidate ID
        ↓
EVALUATE(candidate_id)
        ↓
denormalize x → native z
        ↓
generator(z)
        ↓
CIF/PDB
        ↓
TM + lDDT + diagnostics
        ↓
observe result
        ↓
refit DSP GP
        ↓
new agent turn
```

Only:

```text
EVALUATE
```

consumes the protein-generation budget.

---

# 24. Agent strategic freedom

Sara may decide to:

```text
accept DSP suggestion

request local suggestion

compare candidate predictions

change exploration/exploitation

tighten full-D search radius

reopen global search

switch LogEI ↔ UCB

return to global search after stagnation
```

But the GP, bounds and oracle remain controlled by backend code.

---

# 25. Agent system instruction

Use approximately:

```text
You are the optimization controller for a high-dimensional
protein latent-space search.

You optimize the full native latent dimensionality.

Do not create dimensionality-reduced projections or subspaces.

The DSP Gaussian Process provides probabilistic guidance,
but its suggestions are advisory.

Your objective is to maximize the structural target-matching
score under a limited expensive generation budget.

Use:
- current incumbent,
- GP uncertainty,
- lengthscale diagnostics,
- recent improvement,
- acquisition behavior,
- remaining budget

to decide when to explore globally or refine locally.

Only EVALUATE consumes oracle budget.

Never invent structural scores.
```

---

# 26. Repository structure

```text
agentic-protein-dsp-bo/
│
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── poc_v1.yaml
│   ├── generator.yaml
│   ├── dsp_gp.yaml
│   └── targets.yaml
│
├── prompts/
│   └── SARA_SYSTEM.md
│
├── data/
│   └── targets/
│       ├── manifest.csv
│       └── structures/
│
├── src/
│   │
│   ├── generator/
│   │   ├── base.py
│   │   └── adapter.py
│   │
│   ├── latent/
│   │   ├── inspect.py
│   │   ├── bounds.py
│   │   ├── normalize.py
│   │   └── reshape.py
│   │
│   ├── oracle/
│   │   ├── oracle.py
│   │   ├── validation.py
│   │   ├── tm_score.py
│   │   ├── lddt.py
│   │   ├── rmsd.py
│   │   └── gdt.py
│   │
│   ├── dsp/
│   │   ├── model.py
│   │   ├── priors.py
│   │   ├── acquisition.py
│   │   ├── optimize_acq.py
│   │   ├── diagnostics.py
│   │   ├── state.py
│   │   └── tools.py
│   │
│   ├── sara/
│   │   ├── client.py
│   │   ├── schemas.py
│   │   └── agent.py
│   │
│   ├── baselines/
│   │   ├── sobol.py
│   │   ├── vanilla_gp.py
│   │   └── dsp_gp.py
│   │
│   ├── runner/
│   │   ├── run_target.py
│   │   └── run_suite.py
│   │
│   └── analysis/
│       ├── aggregate.py
│       ├── metrics.py
│       └── plots.py
│
├── scripts/
│   ├── inspect_latent.py
│   ├── calibrate_bounds.py
│   ├── smoke_oracle.py
│   ├── smoke_dsp.py
│   ├── run_poc.sh
│   └── run_suite.sh
│
├── tests/
│   ├── test_latent_roundtrip.py
│   ├── test_normalization.py
│   ├── test_oracle.py
│   ├── test_dsp_prior.py
│   ├── test_gp.py
│   ├── test_budget.py
│   └── test_acquisition.py
│
└── outputs/
```

Notice what is gone:

```text
basis.py
projection.py
PCA
8D subspace
```

They are intentionally absent.

---

# 27. DSP unit test

Given dimensionality \(D\), verify:

```python
expected_loc = sqrt(2) + 0.5 * log(D)
expected_scale = sqrt(3)
```

For example:

```text
D = 6392

loc ≈ 5.796
scale ≈ 1.732
```

Inspect the instantiated kernel and verify those values.

Also verify:

```text
ARD dimensions = D
```

not:

```text
ARD dimensions = reduced dimension
```

---

# 28. GP sanity tests

Before protein experiments, create synthetic functions in:

```text
D = 10
D = 100
D = 1000
```

Verify:

```text
DSP GP trains
posterior predictions finite
lengthscales finite
acquisition finite
candidate generation succeeds
```

Then test the actual protein latent dimension without decoding proteins:

```text
D = native protein latent D
```

using a cheap synthetic objective.

This separates:

```text
GP engineering failure
```

from:

```text
protein generator failure
```

---

# 29. Acquisition optimizer stress test

In the actual full \(D\):

```text
fit synthetic DSP GP
↓
optimize LogEI
```

Record:

```text
runtime
GPU memory
optimization warnings
NaN gradients
candidate bound violations
```

Do this before expensive protein runs.

In a very large latent space, acquisition optimization may become the main computational bottleneck even when GP fitting works.

---

# 30. Target selection

POC-v1:

```text
5 held-out proteins
single-chain
100–200 residues
minimal missing coordinates
structurally diverse
```

Freeze:

```text
target ID
reference CIF
sequence if required
length
split
```

Targets should preferably be outside the generator's training set.

---

# 31. Initial latent

For each:

```text
target
seed
```

generate exactly one:

$$
z_0.
$$

Save it permanently.

Every optimizer receives the same:

```text
z0
bounds
initial Sobol evaluations
```

Do not regenerate `z0` separately for each method.

---

# 32. POC-v1 configuration

Start with:

```text
targets:
5

seeds:
3

native latent dimension:
whatever the model exposes

search dimension:
exactly native D

total budget:
80

initial Sobol:
16

adaptive:
64

objective:
sqrt(TM × lDDT)

methods:
1. Sobol
2. Vanilla GP-BO
3. DSP GP-BO
4. Agentic DSP-BO
```

Total:

```text
5 targets
× 3 seeds
× 4 methods
=
60 runs
```

For debugging first:

```text
1 target
1 seed
20 evaluations
```

---

# 33. Fairness requirements

All four methods must use:

```text
same target

same z0

same full latent dimensionality

same search bounds

same normalization

same initial 16 observations

same generator

same structural oracle

same total expensive budget
```

For GP-based methods, where applicable:

```text
same acquisition optimizer
same q=1
same runtime precision
```

The only difference between vanilla GP and DSP GP should initially be the GP prior setup.

The only difference between DSP GP and Agentic DSP should be agentic orchestration.

---

# 34. Main plots

Generate:

### Plot A

```text
best TM-score
vs
oracle evaluations
```

### Plot B

```text
best lDDT
vs
oracle evaluations
```

### Plot C

```text
best composite objective
vs
oracle evaluations
```

### Plot D

```text
evaluations required for
TM >= threshold
```

Possible thresholds:

```text
0.5
0.7
0.8
0.9
```

where attainable.

### Plot E

```text
final TM by target/method
```

### Plot F

```text
DSP GP lengthscale distribution
over optimization iterations
```

This last plot is particularly important because your research is explicitly using the DSP hypothesis.

---

# 35. Diagnostics worth logging

At every iteration record:

```text
minimum ARD lengthscale
median ARD lengthscale
maximum ARD lengthscale

prior mode

observation noise

posterior uncertainty at candidate

acquisition value

distance candidate → incumbent

distance candidate → z0

objective

TM

lDDT

RMSD

GP fit runtime

acquisition optimization runtime

generator runtime
```

This lets you analyze whether DSP is actually maintaining useful correlations in the protein latent space.

---

# 36. Additional scientific analysis

One interesting diagnostic is kernel correlation.

For observed points calculate representative:

$$
k(x_i,x_j).
$$

Compare:

```text
Vanilla GP
vs
DSP GP
```

If vanilla GP produces something close to:

$$
K\approx I
$$

while DSP retains nontrivial off-diagonal correlations, that directly supports the mechanism proposed by the high-dimensional BO paper.

This would make a strong figure for your eventual research report.

---

# 37. Agent-specific metrics

Also log:

```text
tool calls per oracle evaluation

number of local-search decisions

number of global resets

number of acquisition switches

number of times agent accepts DSP suggestion

number of times agent overrides suggestion

improvement after agent override

tokens per optimization iteration
```

Then you can study *why* agentic BO helps or fails.

---

# 38. Success criteria

The POC is successful technically when:

```text
full native latent is optimized directly

no projection/subspace is used

DSP prior matches formula

ARD dimensionality equals native D

structural oracle is deterministic/reproducible

every expensive call is budgeted

all methods share initial observations

all methods complete full budget

GP fitting remains numerically stable

acquisition optimization succeeds

results reproduce from seeds
```

Scientific success does NOT require Agentic BO to win.

The experiment is valid even if:

```text
DSP BO ≈ Agentic DSP BO
```

or:

```text
DSP BO > Agentic DSP BO.
```

That would still answer the research question.

---

# 39. Desired positive result

An illustrative outcome would be:

```text
Best TM @80

Sobol              0.62
Vanilla GP          0.64
DSP GP              0.78
Agentic DSP GP      0.84
```

and:

```text
evaluations to TM >= 0.8

Sobol               >80
Vanilla GP           >80
DSP GP                61
Agentic DSP GP        39
```

This would support two claims:

1. DSP makes direct high-dimensional latent BO viable.
2. Agentic strategy control further improves sample efficiency.

---

# 40. What NOT to claim

Even with a positive result, initially say:

> Agentic DSP-BO improves sample-efficient target recovery in the native latent space of a frozen protein structure generator.

Do NOT immediately say:

> We beat protein-generation SOTA.

That requires an external standardized benchmark and identical protocol.

---

# 41. Later ablations

Only after the full-dimensional POC works, test:

```text
RBF vs Matérn

TM-only
vs
sqrt(TM × lDDT)

40 vs 80 vs 160 evaluations

different full-space radii

DSP
vs
TuRBO

DSP
vs
SAASBO

Agentic DSP
vs
Agentic TuRBO
```

A projected 8D/16D method can eventually be included as an ablation:

```text
full DSP
vs
low-dimensional projection
```

but it should no longer be the primary method.

---

# 42. Implementation sequence for the coding agent

## Phase 1

Inspect native latent.

Deliver:

```text
latent_report.json
```

Do not proceed until \(D\), shape and latent distribution are understood.

## Phase 2

Implement and validate:

```text
TM
lDDT
RMSD
composite objective
```

## Phase 3

Implement full-dimensional latent normalization and round-trip:

```text
z → x → z
```

Test numerical equality.

## Phase 4

Calibrate full-D search radius on development targets.

Freeze it.

## Phase 5

Implement DSP GP.

Verify exact prior numerically.

## Phase 6

Test DSP GP on high-dimensional synthetic objectives.

## Phase 7

Stress-test acquisition optimization at the real latent dimensionality.

## Phase 8

Implement Sobol baseline.

## Phase 9

Implement vanilla GP baseline.

## Phase 10

Implement fixed DSP GP-BO.

## Phase 11

Implement backend tools.

## Phase 12

Connect Sara/Qwen.

## Phase 13

Run:

```text
1 target
1 seed
20 evaluations
```

and manually inspect all trials.

## Phase 14

Run POC:

```text
5 targets
3 seeds
80 evaluations
4 methods
```

## Phase 15

Generate all statistics and plots automatically.

---

# 43. Definition of done

```text
[ ] Native latent D identified.

[ ] No dimensionality reduction exists in primary pipeline.

[ ] Full latent box calibrated.

[ ] z ↔ normalized x transformation tested.

[ ] DSP loc calculated from actual D.

[ ] DSP scale = sqrt(3).

[ ] ARD kernel has exactly D lengthscales.

[ ] Output standardization enabled.

[ ] Structural oracle tests pass.

[ ] Acquisition optimization works at actual D.

[ ] Sobol completes.

[ ] Vanilla GP completes.

[ ] DSP GP completes.

[ ] Agentic DSP completes.

[ ] Same initialization used by all methods.

[ ] Exact evaluation budget enforced.

[ ] 5 targets × 3 seeds complete.

[ ] TM curves generated.

[ ] lDDT curves generated.

[ ] DSP kernel diagnostics generated.

[ ] Agent action traces saved.

[ ] Results reproducible from configuration and seed.
```

---

# 44. Core POC architecture

The final experiment should be exactly:

$$
\boxed{
x\in[0,1]^D
\leftrightarrow
z\in\mathbb R^D
\xrightarrow{G}
\hat X
\xrightarrow{\text{TM+lDDT}}
S
\xrightarrow{\text{DSP GP}}
\text{Agent}
}
$$

where:

$$
\boxed{
D=\text{actual native latent dimensionality}
}
$$

and:

$$
\boxed{
\ell_i\sim
LogNormal
\left(
\sqrt2+\frac12\log D,
\sqrt3
\right)
}
$$

with **no PCA, no projection and no reduced search space**.

The core scientific comparison is:

$$
\boxed{
\text{Vanilla GP}
\rightarrow
\text{DSP GP}
\rightarrow
\text{Agentic DSP GP}
}
$$

under identical target structures, native latent space, generator and expensive-evaluation budget.

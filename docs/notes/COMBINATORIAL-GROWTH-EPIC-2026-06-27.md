---
title: "Combinatorial-growth epic: keep a stable core as models × backends × features multiply"
description: "The epic decomposition for handling fak's combinatorial growth — a generated coverage matrix, a cheap per-model conformance contract, a model leaf scaffold, a researcher experiment registry, and a growth-debt metric folded into the existing control-pane — so a new model/backend/feature never silently leaves the support grid undefined."
---

# Combinatorial-growth epic: a stable core as the cross-product grows

_Grounded at `HEAD = 3d4597a6` (2026-06-27). Every `file:line` below was read while
writing this; the numbers are derived from the tree, not invented._

## The problem, stated precisely

fak grows along several axes at once:

- **Models** — 16 architecture families on disk today (`tensor_resolver.go` family
  switch, `resolveSpecFor` `tensor_resolver.go:121`), each added as a ~4-file core
  edit (config detection → tensor-name resolver → weight materializer → `newModel`
  loader `weights.go:252`).
- **Backends** — `cpu` (reference), `cuda`/HAL, `metal`, `vulkan` (`cmd/fak/serve.go`
  `--backend`).
- **Performance features** — block-topology dispatch, MoE, MLA/DSA/MSA sparse
  attention, quant (Q8/Q4_K/Q5_K/GPTQ/AWQ/EXL2/MXFP4), tiered KV residency,
  spec-decode, RadixAttention prefix reuse.
- **Researchers** — many sessions/hosts run experiments concurrently
  (`experiments/`, `experiments/nightrun/<host>/`).

The cross-product is **combinatorial**, but the *correctness* of each cell is tracked
two bad ways today:

1. **By hand-written prose that rots.** The model-arch seam status
   ([`docs/notes/model-arch-seam-status-487.md`](model-arch-seam-status-487.md)) is the
   best artifact we have — a per-family × per-backend × per-oracle grid. It opens with
   *"the epic thesis is stale"* and is grounded against a now-removed roadmap doc. A
   prose table is a snapshot; the tree moves underneath it.
2. **By tests that SKIP.** All 25 oracle tests (`oracle_test.go`) skip under
   `-short` (CI) and skip when a checkpoint is absent (always, in CI). So **non-Llama
   numeric correctness is asserted, not proven** in CI. A divergence in Gemma-3 HAL
   decode ships undetected.

And the holes are real and countable: `requirePreNorm` (`kv.go:151`) **panics** on any
non-PreNorm topology hitting an accelerated path (HAL decode `kv.go:162`, HAL/Metal
prefill, quant-batch). That's **3 non-Llama topologies × 4 accelerated paths = 12 panic
cells**, affecting ≥4 shipped families (Gemma, OLMo2, Cohere, GPT-NeoX/Falcon). A second
super-seam, `requireGLMDsaSession` (`kv.go:189`), fails GLM-5.2 closed on Metal. There
are **87 `panic(`/boundary markers across 27 files** in `internal/model/` — each one a
cell of the cross-product that is honestly-unsupported but **invisible** unless you read
the source or hit it at runtime.

**The thread we keep losing:** when a new model or backend drops, nobody can answer
"which cells of the support grid did this just change, and which are still undefined?"
without re-reading the source. We re-derive the map by hand, it goes stale, and the next
new thing starts the cycle over.

## The frame (what makes this distinct, not a duplicate)

fak already has the *measuring* discipline for growth — it just hasn't been pointed at
the correctness cross-product:

- **`steerability_scorecard.py`** owns "effort stays flat as size grows" — growth-invariant
  *structure* (file/func size, hub-share, churn). It does **not** own the support **grid**.
- **`industry_scorecard.py`** (#1050) owns "fak vs SOTA, the field's dimensions" — it
  measures *coverage of the industry*, not *which of our own cells work*.
- **The model-arch seam epic** (#487) owns *building* the seam past Llama-only (S1–S8);
  several children (#474 per-family oracle matrix, #86 GLM-DSA backend, #489 MXFP4) are
  the per-cell implementation work.
- **`scorecard_control_pane.py`** + `scorecard_baseline.json` already fold N `*-debt`
  integers into one ratcheted portfolio gate (`--check`, `--pin`, per-metric early-warning).

This epic owns the **seam between them**: turn the rotting prose grid into a
**generated, deterministic, gated artifact**, give every model a **cheap conformance
contract that actually runs in CI** (the brittleness trap is exactly the SKIP-when-no-checkpoint
gap), make adding a model a **scaffolded leaf** instead of a 4-file core edit, and give
researchers a **shared experiment registry** so growth in the contributor pool doesn't
collide. All folded into the **existing** control-pane — one new `growth_debt` number, not
a parallel universe.

> One-line test for every child below: *does it make a NEW model/backend/feature change
> the grid in a way that is visible, gated, and cheap — without a core edit and without a
> brittle weight-backed test?*

## Child tickets

_Epic: **#1079**. Children: C1 **#1080** · C2 **#1081** · C3 **#1082** · C4 **#1083** · C5 **#1084**._

### C1 (keystone, #1080) — the generated coverage matrix + `growth-debt` metric

**The one deliverable everything else hangs on.** A deterministic generator
(`fak coverage-matrix`, pure logic in `internal/covmatrix/`, thin shell in
`cmd/fak/covmatrix.go` per the Go-not-Python rule, AGENTS.md) that **derives the
model × backend × feature support grid FROM the tree** — not from a hand-written table:

- **Models** from the `resolveSpecFor` family switch (`tensor_resolver.go:121`) +
  `topologyForFamily` (`weights.go:732`).
- **Backends** from the `--backend` enum + the HAL/Metal/quant dispatch in
  `internal/model/kernel.go` / `kv.go`.
- **Support level per cell** from the boundary markers themselves: a cell is
  `SUPPORTED` / `PROOF-PATH-ONLY` / `PANICS` / `UNTESTED` derived from the
  `requirePreNorm` / `requireGLMDsaSession` call sites and the topology each family
  lowers to. (These 87 markers ARE the ground truth; the generator reads them.)
- **Oracle presence** per family from the `TestOptional*Oracle*` test set
  (`oracle_test.go`) — present-and-runs-in-CI vs present-only-with-checkpoint vs absent.

Emits the canonical control-pane payload (`schema/ok/verdict/.../corpus.growth_debt/kpis`)
and a committed snapshot (`docs/coverage-matrix.md` + `tools/coverage_matrix.snapshot.json`).
`growth_debt` = count of cells that are **silently undefined** — a (model, backend) pair
the dispatch can reach with neither a panic-fence NOR a passing conformance row. A panic
is *honest* (debt 0); a silent reachable wrong-result path is debt. **Wire `growth_debt`
into `scorecard_control_pane.py`'s `SCORECARDS` list and pin it in
`scorecard_baseline.json`.** This replaces the rotting #487 prose grid with a generated one
and makes "what did this change touch?" a `git diff` of the snapshot.

*Acceptance:* `fak coverage-matrix --json` is deterministic (two clones, one commit,
identical); the snapshot regenerates with no diff on a clean tree; `growth_debt` appears
in the control-pane fold; the generated grid agrees with the hand-checked #487 table at
`HEAD` (validation that the derivation is faithful).

### C2 (#1081) — the weight-free conformance contract (the anti-brittleness child)

The trap the goal names explicitly: integration tests are "easy to get brittle or false."
The current oracle tests are *false-green* in CI (they SKIP). Fix the **failure mode, not
the symptom**: every registered family gets a **deterministic, weight-free conformance
row** that runs in CI on a tiny synthetic fixture (the pattern already exists —
`newSyntheticMLA` `kvlayout_test.go:119`, `TestStandardLayoutNoOp` `kvlayout_test.go:14`,
the `Float32bits`-equality gate `arch_test.go:62`). The contract asserts the **invariants
that don't need real weights**: config derivation lowers to the declared topology; the
mechanical axes are no-ops on Llama (already proven, generalize the harness); the
proof-path forward is finite/shape-correct; and — the key one — **a cell the matrix marks
`PANICS` actually panics** (the fence is real), while a cell marked `SUPPORTED` runs on its
synthetic fixture. This is a *table test over the registry*, so adding a family adds a row,
not a test file. The expensive real-checkpoint HF oracle (#474) stays as the separate
`needs-runtime-witness` gate it already is — we don't fake it, we **bound** what CI can
honestly prove and assert exactly that.

*Acceptance:* a table test iterates every family in the resolver switch; CI exercises a
real (synthetic) forward for each `SUPPORTED` cell and asserts the fence for each `PANICS`
cell; deleting a family's resolver case OR mis-declaring its topology reds the trunk;
runtime under `-short` stays in the make-test-fast budget (~2s).

### C3 (#1082) — `fak new-model` leaf scaffold (additive, not a core edit)

Adding a family is a 4-file core edit today with **no scaffold** (AGENTS.md cites
`tools/new_leaf.py` for *leaves*, but a model family is a config+resolver+materializer
edit, not a tiered leaf). Ship `fak new-model <family>` that stamps the conforming
skeleton: a `Config` family-detection helper, a `resolveSpecFor` case + spec stub, a
materializer stub, the `newModel` wire, AND a C2 conformance row pre-filled with the
declared topology — so the **first** thing a new family gets is a failing-until-implemented
conformance test, not silence. The scaffold encodes the seam discipline (#487's pattern)
as a verb, the way `fak commit` encoded the commit discipline.

*Acceptance:* `fak new-model foo --topology postnorm` produces a buildable tree where the
new family appears in the coverage matrix as `UNIMPLEMENTED` (red conformance row), and the
diff touches only the seam files + one new conformance row — never an unrelated core file.

### C4 (#1083) — the researcher experiment registry (growth in the people axis)

Today experiments are isolated by filesystem convention (`experiments/<name>/`,
`experiments/nightrun/<host>/backlog.json`) with no shared registry — cross-host collision
avoidance is ad-hoc (Slack, memory). As researchers multiply this is the "losing the
thread" axis for *people*. Ship a thin, append-only experiment ledger (Go, over the
existing `internal/nightrun/ledger.go` shape) that records `{experiment-id, owner, host,
models, backends, started, artifact-path}` and a `fak experiments` read verb — so a new
researcher can answer "what's running, on what, by whom" and "has this cell already been
measured" without a filesystem walk or a Slack ping. Bind it to `dos_arbitrate`'s
lane-disjointness so two researchers measuring the *same* (model, backend) cell are warned,
not silently duplicated. **Keep it append-only and provenance-first** — it records what
ran, it does not orchestrate.

*Acceptance:* `fak experiments list` reads every host ledger into one view; a new
experiment self-registers; querying a (model, backend) cell returns prior runs +
artifact paths; the BENCHMARK-AUTHORITY rows can cite a registry id.

### C5 (#1084) — the growth-gardening cadence (keep it from re-rotting)

The control-pane only stays honest if it's *run*. Add the growth axes to the existing
`/loop` gardening cadence and the `score-2x` conductor: (a) the coverage matrix
regenerates and `growth_debt` is re-checked on every model/backend/feature PR (a CI step,
like `--check`); (b) a `--stale` lens lists cells whose oracle is older than N days or whose
support level is `PROOF-PATH-ONLY` past a grace window (the honest-but-incomplete cells
that #487's S4 residual carries forever); (c) document the **stable-core ritual** — a new
backend ships only once its column of the matrix is generated, its `SUPPORTED` cells have
conformance rows, and `growth_debt` did not rise. This is the RSI discipline (#1021 dojo
loop, the scorecard family) pointed at kernel growth instead of doc/code quality.

*Acceptance:* a documented "adding a model/backend" ritual in `EXTENDING.md`;
`growth_debt` in the `--check` ratchet; a `--stale` cell list; the cadence wired into the
gardening bundle.

## Dependencies (extends, does not duplicate)

| This epic's child | Builds on / feeds | Relationship |
|---|---|---|
| C1 coverage matrix | #487 seam status doc; #307/#305/#303 track epics | **Replaces** the hand-written grid with a generated one; the track epics get a live dashboard |
| C2 conformance contract | #474 per-family oracle matrix; #303 Testing/Quality | C2 is the **weight-free** floor; #474 stays the **weight-backed** witness — disjoint, not overlapping |
| C2 / C1 | #86 (GLM-DSA backend), #489 (MXFP4), the S4 residual | Each per-cell fix flips a matrix cell `PANICS`→`SUPPORTED`; the matrix makes that progress legible |
| C3 new-model scaffold | #1026 (Ornith), `new_leaf.py` pattern | The scaffold is the **repeatable** form of what #1026 did by hand |
| C4 experiment registry | `internal/nightrun`, BENCHMARK-AUTHORITY, `dos_arbitrate` | New read/registry surface over existing ledgers; no orchestration overlap |
| C5 gardening cadence | `scorecard_control_pane.py`, `score-2x`, #1021 | Adds one metric to the existing fold; reuses the ratchet |

## What this epic explicitly does NOT do

- It does **not** implement any per-cell fix (generalizing the HAL/Metal hot-path copies
  past PreNorm is #487/S4 work; MXFP4 is #489; GLM-DSA backend is #86). It makes that work
  **legible and gated**.
- It does **not** add a weight-backed oracle (that's #474). C2 is deliberately weight-free
  to stay un-brittle and CI-cheap.
- It does **not** replace `steerability`/`industry` scorecards. It adds the one axis they
  don't cover — the correctness cross-product — to the same fold.

---

_Filed as the combinatorial-growth epic deliverable. The keystone is C1: once the grid is
generated instead of hand-written, every other child has a place to land and a number to
move._

---
title: "fak proof method: claim to deterministic witness"
description: "How fak proves each sub-module math-correct: a theorem, a proof, a machine-rerun deterministic witness, and a DOS commit binding."
---

# The fak math-proof method — claim → proof → deterministic witness → DOS binding

This directory is fak's **dedicated math-proof section**: one place that states, for
every sub-module, the *theorem* the module must satisfy to be "math correct," gives the
proof, and binds that proof to a **deterministic witness** that a machine re-runs — never
to an agent's say-so.

It is the missing companion to [`../../SUBSYSTEM-CHECKS.md`](../../SUBSYSTEM-CHECKS.md).
That ledger answers *"is this boundary alive and mechanically checked?"* This section
answers the strictly harder question one layer down: *"is the **math** the boundary
computes actually **correct** — and how do we know, deterministically, rather than by
trusting whoever wrote the code or the prose?"*

> **The one rule.** A claim is never `PROVEN` because an author (human or agent) says so.
> It is `PROVEN` only when a **deterministic witness** — a test the toolchain re-runs to
> the same verdict every time — runs **green** and that green corroborates the stated
> theorem, **and** the commit that shipped it is corroborated by [DOS](#5-dos-as-the-meta-witness)
> (`dos commit-audit` / `dos verify`), so the meta-claim "the proof shipped" is itself
> witnessed, not narrated. Anything else is `REFUTED`, `OPEN`, or `SCOPED-OUT` — and is
> labelled as such, honestly.

---

## 1. What "math correct" means here — three correctness regimes

"Math correct" is not one thing across 47 leaves. A GEMM kernel, a radix prefix tree, and
an information-flow gate are correct in three different senses. We name the regime so the
witness is the *right kind* of witness:

| Regime | What "correct" means | Modules (examples) | Witness kind |
|---|---|---|---|
| **N — Numerical / linear-algebra** | The computed tensor equals the mathematically-defined function, within a stated error model (exact for integer paths, ULP/cosine-bounded for float). | `model` (attention, RoPE, RMSNorm, SwiGLU, logits), `compute`/`metalgemm` (GEMM/HAL), `model` quant (Q4_K/Q8/AWQ dequant), `tokenizer`, `ggufload` | Oracle parity · metamorphic relations · bit-identity |
| **A — Algebraic / structural** | A data structure or algorithm preserves its defining invariants (bijection, conservation, longest-prefix, total order, idempotence). | `radixkv`, `kvmmu`, `cachemeta`, `recall`, `contextq`, `blob`, `preflight`, `abi`/`architest` | Round-trip · invariant · property · structural contract |
| **C — Crypto / integrity** | A guarantee resists an *adversary*, not just average input: tamper-evidence, unforgeability, collision-binding, provable equivalence after deletion. | `journal`, `deletioncert`, `provenance`, `ifc` (lattice non-interference) | Hash-chain integrity · unforgeability · `max\|Δ\|=0` equivalence |
| **D — Decision-procedure soundness** | A gate's verdict is *sound* (never admits what the spec forbids) and *fails closed*; the composition order is well-defined. | `adjudicator`, `policy`, `ctxmmu`, `normgate`, `canon`, `plancfi`, `grammar`, `witness`, `ratelimit`, `gateway`, `steward`, `shipgate` | Soundness test · fail-closed test · monotone-fold order |

A few leaves carry two regimes (e.g. `deletioncert` is C *and* depends on an N-claim
`max|Δ|=0`); the per-module doc names every theorem and its regime.

---

## 2. The proof object — what every `<module>.md` contains

Each per-module proof file is a list of **proof obligations**, where each obligation is:

```
THEOREM   a precise, falsifiable statement of the property (∀ inputs in domain D, P holds)
REGIME    N | A | C | D
PROOF     the argument: the mechanism in the code that makes the theorem hold, with file:line
WITNESS   the deterministic test(s) that re-check it — exact `go test -run` command
VERDICT   PROVEN | REFUTED | OPEN | SCOPED-OUT   (from an actual run, dated)
DOS       the commit/phase the witness shipped in, bound via dos commit-audit / verify
```

The PROOF is a real argument, not a restatement. For an integer kernel it is "the
reduction is computed in `int32` with no intermediate rounding, so it is bit-identical to
the reference sum (file:line)." For a gate it is "every admit path routes through
`fold()` which is fail-closed because the zero value is `Deny` (file:line)." The WITNESS is
what stops the PROOF from being wishful.

---

## 3. The witness taxonomy — the deterministic tools we discharge with

These are the only things that earn a `PROVEN`. Each is **deterministic**: same input →
same verdict, no network, no wall-clock, no RNG without a fixed seed.

1. **Exact oracle parity.** Compare against a reference implementation computed
   independently — for fak the PyTorch/HF export (`internal/model/export_oracle.py` →
   `.cache/oracle-*`). Float forward passes are checked by **cosine ≈ 1.0 + per-position
   argmax match + greedy-id match** (a single ULP of drift cannot flip an argmax that the
   oracle pins); integer paths are checked **bit-exactly**. This is the gold witness when a
   reference exists.

2. **Metamorphic relations (MR).** The established technique for numerical kernels that
   have *no* cheap exact oracle: assert a *relation between two runs* that the true
   function must satisfy, sidestepping the oracle problem (Chen et al.; "Metamorphic
   Testing for Deep Learning Operators," ACM 2024, which formalizes the error-of-
   metamorphosis / error-of-difference metrics for exactly this float-precision setting).
   The relations fak uses:
   - `softmax(x + c·1) = softmax(x)` (shift-invariance) and `Σ softmax(x) = 1`, all ≥ 0 (row-stochastic).
   - RoPE preserves the per-head vector norm and depends only on *relative* position `(m−n)`.
   - RMSNorm is invariant to input scaling up to the learned gain; `LayerNorm` is shift+scale equivariant.
   - GEMM is bilinear: `A(B+C)=AB+AC`, `(αA)B = α(AB)` — checked against the naive triple loop.
   - Residual is exact addition; the causal mask makes the attention matrix strictly lower-triangular.

3. **Bit-identity for integer reductions.** Where a kernel reduces in integer arithmetic
   (the Q4_K → NEON `SDOT` decode path), correctness is *bit-exact* equality to the integer
   reference; only the final float de-affine combine carries a bounded float error. This is
   the strongest witness available and is used wherever the math is exact-representable.

4. **Round-trip / involution.** `decode∘encode = id`, `pageIn∘pageOut = id`,
   `verify∘mint = accept`, `dequant∘quant ≈ id` within the quant error — asserted
   **byte-identical** (`...RoundTripBitExact`, `...RoundTripByteIdentical`) where the map is
   lossless, or within the stated error where it is lossy.

5. **Invariant / property tests.** Monotonicity, idempotence, conservation (ref-counts),
   determinism, and total-order, via stdlib `testing` and `testing/quick` (Go's built-in
   property generator — **zero-dependency**, matching this repo's no-external-test-dep
   house style; `pgregory.me/rapid` and `leanovate/gopter` are richer but we keep the dep
   surface empty deliberately).

6. **Structural / contract.** `internal/architest` machine-checks the package DAG (no
   upward imports, fold-rank total order) — the witness that the *composition* the other
   proofs assume actually holds in the build.

7. **Crypto / integrity.** Tamper-evidence is witnessed by mutating a journalled row and
   asserting the hash chain breaks; unforgeability by attempting the forge in-test and
   asserting `Deny`; deletion-equivalence by the `max|Δ|=0` parity between the evicted
   context and one that never saw the span.

---

## 4. The verdict vocabulary — and the honesty rule

| Verdict | Meaning | Earned by |
|---|---|---|
| **PROVEN** | The theorem holds and is mechanically re-checkable. | A deterministic witness runs **green** and corroborates the theorem, **and** its ship commit is `diff-witnessed` by `dos commit-audit`. |
| **REFUTED** | The theorem as stated does **not** hold. | A witness runs **red**, or a counter-witness / contradiction is exhibited. A `REFUTED` row is a *finding*, recorded with the counterexample — not deleted. |
| **OPEN** | The theorem is stated but **not yet** discharged by any deterministic witness. | Honestly un-witnessed. The per-module doc says what witness *would* close it. We do not promote OPEN to PROVEN by argument alone. |
| **SCOPED-OUT** | Discharging it needs a frontier tool we deliberately do not run here. | E.g. full functional/memory-safety/data-race verification via **Gobra** (see §6). Named, with rationale and the upgrade path. |

**The honesty rule restated:** prove *or refute* at each step. An obligation we could not
witness is `OPEN`, never quietly `PROVEN`. This mirrors the existing
`SUBSYSTEM-CHECKS.md` discipline of a "what it does **not** prove" column.

---

## 5. DOS as the meta-witness

The proofs above check the *modules*. But "I wrote 30 proofs and ran their witnesses" is
itself a claim — and an agent that self-narrates "all green, all shipped" is exactly the
worker DOS exists not to believe. So the proof section is grounded one level up by the
**DOS trust substrate** (`dos-kernel`, the repo's own dependency):

- **`dos verify <plan> <phase>`** — the truth syscall: did the proof phase actually ship,
  by git evidence (registry row or ship-commit grep), or is the "done" a self-report
  (`source:"none"`)? Run against the **fleet repo root** workspace (the directory that holds
  `.git`, one level above `fak/`), because the `(fak <leaf>)` trailer grammar is defined there —
  `dos doctor` from inside `fak/` reports `git:false` and cannot bind.
- **`dos commit-audit <ref>`** — does the commit's **diff** witness its **subject**? A
  commit claiming "add proof + witness test" that touched only a README is `subject-only`,
  not `diff-witnessed`. This catches a proof doc that *asserts* a new test exists when the
  diff added none. Every proof-shipping commit in this section is audited; the verdict is
  recorded in the ledger's **DOS** column.
- **`dos review <range>`** — folds the per-commit audits into the *residual*: the set of
  claims the kernel could **not** corroborate, which is the only place a human's review
  attention buys anything the machine couldn't.

This is the recursion that makes the section trustworthy: **the math proves the modules;
DOS proves the math actually shipped.** Neither layer trusts the author.

---

## 6. SOTA tooling & helpers — what we use, what we scope out

Surveyed before building this section (GitHub + the verification literature, June 2026):

- **Gobra** — modular deductive verifier for Go (separation logic → Viper IVL → SMT;
  arXiv:2105.13840). The *frontier* for full functional correctness, memory safety,
  data-race freedom, and crash safety. **Scoped out** for the bulk here: it needs a
  JVM+Viper+Z3 toolchain and per-function specification annotations across a 17k-line model
  core, which is a project in itself. It is named in the per-module docs as the **upgrade
  path** for the three leaves where a runtime test is weakest — the *concurrency*-critical
  ones (`gpulease` advisory lock, `journal` append ordering, `kvmmu` page aliasing) — where
  a data-race-freedom proof would strictly dominate a property test.
- **Metamorphic testing for DL operators** (ACM 2024; Segura et al. survey) — the basis for
  witness kind §3.2, including the EM/ED error metrics that make a float MR a *quantitative*
  pass/fail rather than a brittle exact compare.
- **stdlib `testing` + `testing/quick`** — the zero-dep property generator we standardize on
  for new witnesses, so the proof section adds **no** dependency to `go.mod`. Richer
  generators (`pgregory.me/rapid`, `leanovate/gopter`) were evaluated and deliberately not
  adopted for that reason.
- **PyTorch/HF oracle export** — the existing `export_oracle.py` reference path that makes
  witness kind §3.1 possible without a second inference engine in the test loop.

---

## 7. How to reproduce every witness

The whole section is re-derivable from a clean checkout:

```bash
# Numerical / model correctness boundary (heavy; weight-backed rungs may need the oracle cache):
go test ./internal/model ./internal/compute ./internal/modelengine ./internal/tokenizer ./internal/ggufload
# Algebraic / cache / KV invariants:
go test ./internal/radixkv ./internal/kvmmu ./internal/cachemeta ./internal/recall ./internal/contextq ./internal/blob
# Crypto / integrity:
go test ./internal/journal ./internal/deletioncert ./internal/provenance ./internal/ifc
# Decision-procedure soundness:
go test ./internal/adjudicator ./internal/policy ./internal/ctxmmu ./internal/normgate ./internal/canon \
        ./internal/plancfi ./internal/grammar ./internal/witness ./internal/ratelimit ./internal/gateway \
        ./internal/steward ./internal/shipgate ./internal/preflight
# Structural contract (the composition the other proofs assume):
go test ./internal/architest ./internal/abi
```

On this macOS node these run natively (`go test`); on the Windows host run them through WSL
via `.\fak\test.ps1` (see the root `CLAUDE.md` for why native `go test` is unreliable there).
The per-module docs give the **exact `-run` target** for each individual theorem.

See [`README.md`](README.md) for the master ledger: every theorem, its witness, its live
verdict, and its DOS binding, in one table.

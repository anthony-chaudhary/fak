---
title: "Level 3 — Prove correct: the correctness gate"
description: "How a new kernel earns the right to ship: declare a correctness class (Reference or Approx) and pass a deterministic witness the harness re-runs."
slug: kernel-development-level-3
keywords:
  - kernel correctness gate
  - correctness class
  - Reference vs Approx
  - deterministic witness
  - logit cosine gate
  - argmax-exact
  - max abs delta
  - prove or refute never launder
  - compute HAL backend
date: 2026-11-18
---

# Level 3 — Prove correct: the correctness gate

The correctness gate is how a new kernel earns the right to ship: it declares a correctness class, Reference or Approx, and passes a deterministic witness the harness re-runs. The model proposes; the test disposes.

*You are on **Level 3 of 6** of the [kernel-development ladder](README.md).*

> **What this is.** The correctness rung. Rung 2 got your kernel *attached* to the
> registration seam; this rung is the gate that decides whether it is allowed to run.
> The rule is one line: **the model proposes, the test disposes.** You do not get to
> say your kernel is right — you declare a *correctness class*, and a deterministic
> command the harness re-runs decides. By the end you can name the two classes, write
> the witness, and read a pass/fail without fooling yourself.

## Questions this page answers

- How do I know my kernel is *right*, not just plausible?
- What is a **correctness class**, and which one is mine?
- What is a **witness**, and how is it different from a claim?
- Why can't I just eyeball the output and call it good?
- My new Q8 matmul returns slightly different floats — is that a failure?
- What does a passing witness *not* prove?

## Why can't I just eyeball the output?

Because the author is the worst possible judge. You wrote the kernel, so you already
believe it works; reading its output confirms the belief you started with. A kernel
that is *plausible-but-wrong* — an off-by-one row stride, a non-linear activation
slipped into a dot product, a reduction that reorders floats — produces output that
*looks* fine and is wrong in the last digits or in one row out of a thousand.

fak's answer is the same discipline as [verify, don't trust](../verify-dont-trust.md):
replace "take the author's word" with "check an artifact the author cannot forge."
For a kernel, that artifact is a **witness** — a deterministic test the toolchain
re-runs. `EXTENDING.md` states the contract plainly: an optimization that "should be
equivalent" is not accepted on your word; it ships a witness the harness re-checks.

## What is the difference between a claim and a witness?

A **claim** is a sentence you write. A **witness** is a command that runs.

| | Claim | Witness |
|---|---|---|
| Authored by | you | the test harness |
| Form | prose ("this is equivalent to the reference") | a deterministic test that exits 0 or 1 |
| Reproducible | no — it is an opinion | yes — same input, same bytes, on any machine |
| Fails when | never (it is not checked) | the property it asserts is violated |
| Lives in | a commit message | `internal/<pkg>/proofs_witness_test.go` |

The witness is **deterministic**: no wall-clock, no RNG in the gate. That is what
licenses the claim that a Mac (arm64) and a Windows box (x86_64) agree bit-for-bit —
the property is re-derived, not remembered. `docs/proofs/00-METHOD.md` gives the one
rule: a claim is never `PROVEN` because an author says so; it is `PROVEN` only when a
deterministic witness runs green. Anything else is `REFUTED`, `OPEN`, or `SCOPED-OUT`
— and a `REFUTED` witness that records a real gap is a first-class, mergeable result.
**Prove or refute, never launder.**

## What is a correctness class?

A correctness class is a *typed promise* about how close your kernel's arithmetic is
to the reference. It is not a label you pick for style — it is the contract that
decides which gate applies to you. In this repo it is a real Go type,
`compute.CorrectnessClass`, with exactly two values (`internal/compute/compute.go`):

```go
const (
	Reference CorrectnessClass = iota // held to max|Δ| = 0 + the HF argmax oracle
	Approx                            // held to argmax-exact + a per-backend cosine floor
)
```

Every backend declares its class in one method — `Class() CorrectnessClass` — and the
harness reads it. The enforcement is mechanical: `compute.RequireReference(b)` returns
true only for a `Reference` backend, and the harness calls it *before* any
`max|Δ| = 0` assertion. So it is **impossible** to hold a device to bit-identity it
cannot meet, or to quietly promote a device to reference. Declare your class honestly
and the right gate applies itself.

## Which class is mine?

Ask one question: *does my kernel reproduce the reference arithmetic byte-for-byte?*

```
   Does my kernel produce the SAME BYTES as the reference path?
        │
        ├── YES ──►  Reference   (max|Δ| = 0, plus the argmax oracle)
        │            e.g. the CPU reference itself: internal/compute/cpuref.go
        │
        └── NO  ──►  Approx      (argmax-exact + a cosine floor you declare)
                     e.g. the Q8 lane, and every device backend (CUDA, Vulkan, Metal)
```

Almost every kernel you will write is **Approx**. The moment you reorder a float
reduction, narrow an operand to fp16, or quantize a weight, you have left bit-identity
behind — and that is fine, as long as you say so and clear the looser gate.

## What does each class actually promise?

| Class | What it promises | What it does NOT promise | How it is witnessed |
|---|---|---|---|
| **Reference** | Byte-identical output to the reference reduction order; same argmax token | That it is fast, or that it is the only correct order | `max\|Δ\| = 0` against the reference, plus the Hugging-Face argmax oracle; gated by `RequireReference` |
| **Approx** | Same argmax token (argmax-exact) and a logit/GEMM cosine at or above a **declared per-backend floor** | Bit-identity; that a *localized* fault is caught by the scalar cosine alone | A cosine floor per dtype (e.g. `cudaQ8CosineMin = 0.999`, `cudaQ4KCosineMin = 0.995`, `cudaFP16CosineMin = 0.997`) — and, beside it, an error-*distribution* gate |

The floors are **recorded, not guessed**. `internal/compute/cuda_accuracy_gates.go`
sets each one from the format's own structure: Q8_0 keeps a per-block f32 scale beside
its 8-bit codes, so only the in-block code rounds and the lane stays tight (0.999);
fp16 rounds *both* operands to a 10-bit mantissa with no scale to preserve magnitude,
so its floor is set looser (0.997). The comment on each constant is explicit that it
**records the threshold; it does not assert the path passes it** — the realized cosine
is measured on a GPU node. Do not read a pass from the constant alone.

## The witness loop

```
   ┌──────────────┐   ┌──────────────────┐   ┌───────────────────────┐
   │ write kernel │──►│ declare class in │──►│ run the deterministic │
   │  (your .go)  │   │  Class()         │   │  witness (go test)    │
   └──────────────┘   └──────────────────┘   └───────────┬───────────┘
                                                          │
                        ┌─────────────────────────────────┘
                        ▼
              ┌──────────────────────┐        ┌──────────────────────┐
              │ compare against the  │───────►│  PASS  → ship it     │
              │ reference path       │        │  FAIL  → fix or      │
              │ (cpuref / oracle)    │        │         re-class     │
              └──────────────────────┘        └──────────────────────┘
```

The loop is closed by the harness, not by you. You write the kernel and the witness;
the toolchain re-runs the witness on every change. A kernel whose witness goes red
cannot land — that is the whole point of the gate.

## Predict, then run: a real witness

Let us run a real, fast, no-GPU witness and read its verdict. The test is
`TestQ8DispatchIsApproxAndGated` in `internal/compute/compute_test.go`. It proves the
dtype-selects-the-kernel collapse: a Q8 weight routes `MatMul` to the int8 path, and
the result clears the **Approx** gate against the f32 reference — argmax-exact plus a
cosine floor of **0.995** — while the f32 path stays `Reference`.

**Predict first.** The Q8 lane is `Approx`, so the cosine will be high but *not* 1.0;
the test asserts `cos >= 0.995` and also that the Q8 path is deterministic (same input
⇒ same bytes). I expect **PASS**, in well under a second, with no GPU.

Now run it from the repo root:

```bash
go test ./internal/compute/ -run 'TestQ8DispatchIsApproxAndGated' -count=1 -v
```

**What actually happened** (run on this host, 2026-11-18):

```
=== RUN   TestQ8DispatchIsApproxAndGated
--- PASS: TestQ8DispatchIsApproxAndGated (0.00s)
PASS
ok  	github.com/anthony-chaudhary/fak/internal/compute	0.052s
```

The prediction held. How to read it: `--- PASS` is the verdict; `(0.00s)` is the
runtime; `ok` is the package result. A failure would print `--- FAIL` and the
`t.Fatalf` message — here, `Q8 vs f32 cosine %.5f < 0.995 (the Approx gate)` — naming
the exact floor that was missed. That message *is* the witness talking: it tells you
which class you are in and by how much you missed.

Two more witnesses worth running, both instant and GPU-free:

```bash
go test ./internal/compute/ -run 'TestCorrectnessClassEnforcement' -count=1 -v
go test ./internal/compute/ -run 'TestGEMMBilinear' -count=1 -v
```

`TestCorrectnessClassEnforcement` asserts the type contract itself: `cpu-ref` must be
`Reference` (eligible for bit-identity rungs) and an `Approx` device must **not** be.
`TestGEMMBilinear` is a *metamorphic* witness — it does not compare against a stored
answer, it asserts that `MatMul` is bilinear (`A(x1+x2) ≈ Ax1+Ax2`, and three more
legs) up to a `1e-5` tolerance, with a non-vacuity guard so the identity cannot hold
merely because everything is zero. Both PASS on this host.

## Worked example: my Q8 matmul returns slightly different floats

This is the question every kernel author asks. Walk it through.

**Is it a failure?** Not by itself. Slightly different floats are the *expected*
signature of an `Approx` kernel — you quantized the weights, so the arithmetic is
genuinely not the reference arithmetic. The failure would be *claiming* `Reference`
and then missing `max|Δ| = 0`, or missing the `Approx` floor.

**Which class does it belong to?** `Approx`. A Q8 weight is a quantized lane; the
reference is the f32 path. Declare `Class() CorrectnessClass { return compute.Approx }`.

**What must the witness assert?** Two things, and the second is the one people forget:

1. **argmax-exact** — the same token comes out. A kernel that drifts the logits but
   still picks the same argmax is fine; one that flips the argmax is a correctness
   defect, not drift.
2. **a cosine at or above the declared floor** — and the floor must match the format.

| Lane | Recorded cosine floor | Why that number |
|---|---|---|
| Q8_0 (int8, per-block f32 scale) | **0.999** (`cudaQ8CosineMin`) | only the in-block code rounds; the dot is integer-exact before one f32 scale multiply |
| fp16 (10-bit mantissa, no scale) | **0.997** (`cudaFP16CosineMin`) | both operands round before the multiply; drift compounds along the contraction |
| Q4_K (4-bit k-quant) | **0.995** (`cudaQ4KCosineMin`) | 16 levels, and the per-sub-block scale is itself quantized to 6 bits |
| Q8 lane, CPU reference test | **0.995** (`compute_test.go`) | the gate `TestQ8DispatchIsApproxAndGated` asserts |

A scalar cosine is a whole-vector reduction, so a *handful* of catastrophically-wrong
elements barely moves it. That is why the repo also grades the error **distribution**
beside the cosine (`internal/compute/approx_distribution_test.go`): a close-fraction
floor of **0.999** (an element is "close" when `|a-b| <= absTol + relTol*max(|a|,|b|)`,
with `absTol = 1e-3`, `relTol = 1e-2`) plus a 99.9th-percentile relative-error tail
bound of **0.05**. A localized fault collapses the close-fraction and spikes the tail
even when the cosine still reads 0.997. Assert both.

## What a passing witness does NOT prove

This is the honesty discipline, and it is load-bearing. A green witness is a *bounded*
result:

- **It does not prove the kernel is fast.** Correctness and speed are separate gates.
  A kernel can pass every witness here and still be slower than the baseline — that is
  what rung 4 (the keep-bit) is for.
- **It does not prove the kernel is correct on inputs the witness never tried.** A
  witness checks the property it asserts, on the inputs it generates. A shape, a dtype,
  or a sequence length outside the witness's coverage is unwitnessed, not proven.
- **A recorded floor is not a measured pass.** The `cuda*CosineMin` constants *record*
  a threshold; they do not assert the path clears it. The realized cosine is measured
  on a GPU node. Reading a pass from the constant alone is exactly the laundering the
  method doc forbids.
- **It does not prove the *reference* is right.** The witness compares your kernel to
  the reference path. If the reference is wrong, a green witness proves only that you
  are wrong the same way.

## Checkpoint

1. What is the difference between a claim and a witness?
2. Name the two correctness classes and the one question that decides which is yours.
3. Your kernel reorders a float reduction. Which class, and what does the witness
   assert?
4. Why is a scalar cosine floor alone not enough to catch a localized fault?
5. Give one thing a passing witness does *not* prove.

### Answers

1. A claim is prose you author and nobody checks; a witness is a deterministic test
   the harness re-runs, which exits 0 or 1 and reproduces bit-for-bit across machines.
2. `Reference` and `Approx`. The deciding question: does your kernel produce the same
   *bytes* as the reference path? Yes ⇒ `Reference`; no ⇒ `Approx`.
3. `Approx`. The witness asserts argmax-exact (same token) plus a cosine at or above
   the declared per-backend floor — and, beside the cosine, the error-distribution
   gate (close-fraction ≥ 0.999, p999 relative error ≤ 0.05).
4. A cosine is a whole-vector reduction, so a small fraction of blown elements barely
   moves a value sitting near 1.0. The distribution gate counts the close-fraction and
   bounds the relative-error tail, so a localized fault collapses it.
5. Any one of: that it is fast; that it is correct on untried inputs; that a recorded
   floor is a measured pass; that the reference itself is right.

## See also

- [Level 2 — The registration seam](level-2-the-registration-seam.md): how the kernel
  attaches before it is proven.
- [Level 4 — Prove faster](level-4-prove-faster.md): the keep-bit, the second gate.
- [Verify, don't trust](../verify-dont-trust.md) — the repo-wide witness discipline
  this rung applies to kernels.
- [Hardware portability via the compute HAL](../hardware-portability.md) — the
  `internal/compute` seam, `Class()`, and the private `Tier()` probe.
- [`EXTENDING.md`](../../../EXTENDING.md) — "Gate 2 — Prove it's correct" in full.
- [`docs/proofs/00-METHOD.md`](../../proofs/00-METHOD.md) — the `PROVEN` / `REFUTED` /
  `OPEN` / `SCOPED-OUT` verdict vocabulary.

---

*[← Level 2](level-2-the-registration-seam.md) · [full ladder](README.md) · next → [Level 4](level-4-prove-faster.md)*

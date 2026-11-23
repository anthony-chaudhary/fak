---
title: "The evidence boundary"
description: "How to tell a measured number from a modeled one and from code that exists but is never called, applied to every precision claim in the stack."
slug: evidence-boundary
keywords:
  - measured vs modeled
  - evidence classes
  - designed not wired
  - observed evidence
  - decode roofline
  - kv cache codec
  - dynamic precision policy
  - uncalibrated defaults
  - fak mixed precision ladder
date: 2026-11-18
---

# The evidence boundary

The evidence boundary is the line between a measured number, a modeled one, and code that exists but is never called. This page applies it to every precision claim in the stack.

*You are on **Level 5 of 5** of the [mixed-precision ladder](README.md).*

> **In one breath:** A stack can measure a number, model it, or design it
> without wiring it. fak marks which is which. That honesty is the point of
> this rung.

**One line:** Every precision claim in this stack belongs to one of three classes, measured, modeled, or designed-not-wired, and the value of the stack is that it says which class each claim is in.

> **Audience.** You are a researcher or a frontier-tier reader. You have read
> the four rungs below and you want the honest edge of what the stack proves.
> You are suspicious of any number without a provenance.

> **Short answer.** There are three evidence classes, plus a fourth that is
> subtler. **Measured** means code ran on hardware and the envelope and sample
> count were recorded. **Modeled** means the number was derived by arithmetic
> from known inputs. **Designed but not wired** means the code, the reasoning
> and the accounting all exist and the live path never calls them. The fourth
> is **wired but uncalibrated**: the mechanism runs, and its tuning is an open
> per-model question. The stack has a typed slot for a measurement, and that
> slot is empty. Knowing which claim is which is the whole skill.

## Why does this rung exist?

Because the four rungs below made claims, and a claim is only as good as its
provenance.

A codebase that marks its own unwired code is **more** trustworthy, not less.
The alternative is a stack that presents a design document and a measurement in
the same font, and leaves you to guess which is which. That is the failure mode
this rung exists to prevent.

Here is the shape of the problem. A number can reach you three ways:

```text
  measured   ->  someone ran the code on hardware and wrote down the result
  modeled    ->  someone did arithmetic on known inputs and wrote down the result
  designed   ->  someone wrote the code and the reasoning, and nothing calls it
```

All three can be printed as "537 MB" or "3 to 4 times smaller." Only one of them
is a fact about a machine. The other two are facts about a derivation and a
plan. None of the three is a lie. But a reader who cannot tell them apart will
believe things that were never shown.

So this rung is not a criticism of fak. It is the opposite. The value being
demonstrated is that the stack says which of its own claims are which.

## What are the three evidence classes?

Three classes, and the stack has a name for each. A fourth shows up later on
this page, and it is the one that hides best.

| Class | What produced the number | How you recognize it | Example in fak |
|---|---|---|---|
| **Measured** | Code ran on hardware; envelope and sample count recorded | A hardware envelope and a non-zero sample count | The typed `observed` slot, which is empty |
| **Modeled** | Arithmetic on known inputs; no wall clock | A stated model basis, no hardware, no samples | 0.5625 bytes per weight for the 4-bit K-quant format |
| **Designed, not wired** | Code and reasoning exist; the live path does not call them | A comment that says so, plus a live path that ignores it | The asymmetric KV codec |

The first two are **disjoint by construction**, and that is enforced, not
promised. In `internal/mixedprecision/contract.go`, the evidence kind is a typed
value with exactly two members: `EvidenceModeled` and `EvidenceObserved`
(`contract.go:55-56`). The validator `validateEvidence` (`contract.go:318-340`)
requires an `observed` record to carry all four hardware fields, accelerator,
runtime, driver and operating system, plus a **non-zero** sample count. A
`modeled` record must carry a model basis and must **not** carry a hardware
envelope or a sample count. You cannot file a model estimate as a measurement,
because the type will not hold it.

The third class has no type. It is a property of the code, and you find it by
reading the comment and then checking whether anything calls the function.

## What has fak actually measured?

Nothing. Say it plainly, because it is the single most important fact on this
page.

The observed slot is **real and empty**. The machinery to record a measurement
exists, is typed, and is validated. No measurement has been recorded through it.

`internal/mixedprecision/README.md` states this directly. Lines 45 to 48 record
that no observed hardware or quality result ships with this leaf, and that
consequently the contract makes no throughput, latency, memory-footprint,
perplexity, accuracy, or task-quality claim. Lines 38 to 40 add that the
fixture's weight-bit total is modeled accounting, not a measured memory, speed,
or quality result. Lines 42 to 43 say the package validates and preserves
caller-supplied measurements and never manufactures them.

```text
  the observed slot:

    +-------------------------------------------+
    |  EvidenceObserved = "observed"            |   <- the type exists
    |  requires: accelerator, runtime, driver,  |   <- the validator exists
    |            OS, and samples > 0            |
    |                                           |
    |  records filed:  none                     |   <- the slot is empty
    +-------------------------------------------+
```

Why is an empty slot good news? Because it is a **seam**, not a gap. A future
lab run can attach observed evidence without reclassifying a model estimate as a
measurement. The alternative, a stack with no observed slot at all, would force
every future measurement to be smuggled in as prose. The empty slot is the
honest version of "we have not measured this yet."

## What has fak modeled, and is that honest?

Yes, and the honesty is in the disclosure. A modeled number is not a lie. It is
a derivation. What makes it honest is that it says which inputs it derived from,
and that it does not claim a wall clock it never took.

Four modeled results carry the ladder.

**The bytes-per-parameter figures.** `internal/model/quant.go` records 4 bytes
per weight for 32-bit float and 1.125 bytes per weight for the 8-bit block
format, which is 1 byte plus 4/32 for the per-32-block scale.
`internal/model/quant_q4k.go` records 0.5625 bytes per weight for the 4-bit
K-quant format, which is 144 bytes per 256-weight super-block. These are
arithmetic on a format layout. They are exact, and they are not a measurement of
anything running.

**The decode roofline.** `internal/model/decode_roofline.go` lines 5 to 7 state
that decode is memory-bandwidth-bound because each generated token streams every
weight matrix exactly once with no reuse, so per-token latency is floored by
weight bytes divided by achieved bandwidth. `internal/model/batch.go` lines 6 to
8 put a number on it: batch-1 decode runs at about 0.50 flop per byte, and each
generated token re-streams roughly 537 MB of 32-bit weights or about 150 MB at
8-bit. The 537 and 150 are arithmetic on the weight sizes. The roofline is a
derivation, not a stopwatch.

**The cache accounting.** `internal/model/asymmetric_kv.go` lines 27 to 31
record that at a quarter-million token context the 32-bit baseline is roughly
61.4 to 80.3 GB against roughly 15.6 to 20.08 GB for an asymmetric 8-bit-key /
4-bit-value configuration, a 3 to 4 times reduction. `KVVectorBytes`
(`asymmetric_kv.go:74-100`) is the arithmetic that produces those numbers.

**The sensitivity ordering.** `internal/model/v41_quant_sensitivity.go:59`
states plainly that its numbers are a deterministic ordering key, **not** a
measured compression ratio. That sentence is the model of what this rung is
asking for: the file tells you what its own number is and is not.

The teaching point is the same in all four cases. A modeled number is honest
when it names its inputs and refuses to claim a wall clock. All four do.

## What is designed but not wired?

This is the class that misleads readers most, because the code is real, the
reasoning is sound, and the accounting is correct. The only thing missing is the
call.

**The central case: the asymmetric cache codec.** `internal/model/kvquant.go`
lines 33 to 37 say, in substance, that the file is a codec plus accounting,
that nothing in the decode path calls it yet, that the cache still stores 32-bit
rows, and that wiring it into the cache and attention is the promotion step.
That step needs decode-quality evidence, perplexity or logit drift over a real
long context, which this package cannot produce on its own.

The corroboration is everywhere once you look. `internal/model/kvcache.go:104-106`
returns the 32-bit precision for a nil cache, and `quantized()` is true only for
the 8-bit tier (`kvcache.go:113-115`). `internal/model/kvcache_q8.go:677` says
the 16-bit and 4-bit tiers are declared but not realized, and that leaving the
cache unchanged is fail-closed and honest. `internal/model/device_kv_precision.go:12`
defines `ErrDeviceKVPrecisionUnrealized` for exactly this refusal.

```text
  the asymmetric cache codec:

    design + reasoning + accounting   ->  present   (asymmetric_kv.go, kvquant.go)
    a codec that encodes and decodes  ->  present   (kvquant.go)
    the decode path calling it        ->  ABSENT    (kvquant.go:33-37)
    the live cache stores             ->  32-bit rows (kvcache.go:104-106)
```

**The 8-bit float tensor path.** `internal/compute/fp8.go` lines 16 to 18 say
per-tensor and per-block scale application and the GGUF tensor-type wiring stay
the device and loader gated follow-on, and that the file is only the scalar
element decode. So the 8-bit float **decoder** is real and witnessed, and the
8-bit float **tensor path** is not wired. Two different claims, one file.

**The NVFP4 loader refusal.** `internal/model/modelopt_nvfp4_checkpoint.go`
lines 23 to 25 say fak detects and validates the declaration but mixed-precision
execution is not wired yet, so the loader **refuses** with a typed error before
any tensor is quantized or resident. The same statement appears at
`internal/model/modelopt_nvfp4_load.go:41`. The refusal is the honest move: a
loader that silently ran the checkpoint at the wrong precision would be worse
than one that stops.

**The out-of-axis dtypes.** `internal/covmatrix/precision.go` lines 14 to 16
note that the wider dtype set, f16, bf16, i8, i4 and fp8, and the cache-precision
tier are out of the cross-support axis by design. The support tensor is honest
about its own scope. A reader must not read "not in the tensor" as "not
supported anywhere." Those are different sentences.

## What is wired but uncalibrated?

There is a third thing again, and it is subtler than the other two. The
mechanism is real and it runs. The tuning is a per-model question that has not
been answered.

That is the dynamic precision policy. `internal/model/dynamic_precision.go`
lines 33 to 35 record that with both thresholds at zero the policy accepts every
quantized result, which makes it behave like a fixed choice with extra
statistics. There is no default constructor that picks a sensible threshold.

```text
  the dynamic policy:

    the mechanism runs per token        ->  wired
    the thresholds default to zero      ->  accepts everything
    a calibrated default exists         ->  no
    so today it is a fixed choice       ->  with extra statistics
```

So the claim "the dynamic policy adapts per token" is true of the code and
false of the default configuration. The mechanism is wired. The calibration is
open. That is a distinct class from "designed but not wired," and it deserves
its own line in any honest ledger.

## How does the ladder grade out?

Here are the load-bearing claims from rungs 1 to 4, with their class. This
table is the deliverable of the whole ladder.

| Claim from rungs 1 to 4 | Class | Why |
|---|---|---|
| "Fewer bits means less memory" | **Modeled** | Arithmetic on a known parameter count. |
| "Fewer bits means faster decode" | **Modeled**, conditionally measured | The roofline is a derivation; the 537 MB and 150 MB figures are arithmetic on weight sizes, not a wall-clock reading. |
| "A near-tie can be flipped by rounding" | **Modeled** | A worked softmax example, not a hardware run. |
| "Keys need more precision than Values" | **Modeled** reasoning | The accounting is real; the wiring is absent. |
| "The cache is compressed" | **Designed, not wired** | The codec exists; the live cache stores 32-bit rows. |
| "fak supports 8-bit on Vulkan" | **Measured about code** | The dispatch table is read from source and cross-checked by a test. It is a statement about code, not about achieved speed. |
| "The contract adjudicates precision maps" | **Designed, not wired** | `contract_test.go` proves the typed outcomes, and no production path calls `Evaluate`. The only non-test use of the package is the `Assignment` type, consumed by the schedule adapter. |
| "The cross-support tensor tells the runtime what a backend can run" | **Designed, not wired** | It is a table that reporting and scorecard tooling folds. No code in `internal/model` or `internal/compute` imports it. |
| "The schedule picks a level per tensor, layer, phase" | **Designed, not wired** | `RunScheduledPass` is an opt-in API. Its only callers are its own tests; the decode path never invokes it. |
| "The dynamic policy adapts per token" | **Wired, uncalibrated** | The mechanism runs; the default thresholds accept everything. |

Read the fifth row again. "The cache is compressed" is the claim a reader is
most likely to get wrong, and the page must say so without hedging. The cache is
**not** compressed today. The codec that would compress it is written, reasoned
and accounted. The live path stores 32-bit rows.

Read the sixth row carefully too. "fak supports 8-bit on Vulkan" is a statement
about a dispatch table in `internal/covmatrix/precision.go`, cross-checked by
`internal/covmatrix/precision_test.go`. It is true. It is also not a claim about
speed, and a reader who hears "supports" as "runs fast" has crossed the evidence
boundary without noticing.

## What habit is this rung teaching?

Three questions, asked of any precision claim, anywhere.

```text
  1. Was this number produced by running code on hardware,
     or derived by arithmetic?

  2. If it was derived, from what inputs, and does the claim admit that?

  3. Does the live execution path actually call this code,
     or does the code merely exist?
```

Ask them in order. The first separates measured from modeled. The second tests
whether a modeled number is honest. The third catches the class that fools
people, the code that exists and is never called.

And the design principle that makes the answers safe: **a fence that refuses is
safe; a silent fallback to a different precision is not.** fak fences absent
support in `internal/covmatrix/precision.go` lines 18 to 23, where every backend
ends its weight-dtype switch in an honest panic rather than a wrong result. It
records refused points rather than widening them in
`internal/model/dynamic_precision.go` lines 131 to 134, where a refused point is
recorded as a refusal and skipped instead of being quietly rerun at the widest
level. And it states its own scope in `internal/mixedprecision/README.md` lines
45 to 48.

A stack that refuses loudly, records its refusals, and marks its own unwired
code is one you can trust with the claims it does make. That is the habit. Ask
which class a number is in, and prefer the stack that answers.

## Check yourself

1. What are the three evidence classes, and what distinguishes measured from
   modeled?
2. What does the observed slot in the contract contain today, and why is an
   empty slot better than no slot?
3. Name two modeled results in fak and say what makes each one honest.
4. Is the KV cache compressed in fak's live decode path today? What is the
   evidence?
5. What are the three questions to ask of any precision claim?

<details>
<summary>Answers</summary>

1. Measured, modeled, and designed-but-not-wired. Measured means code ran on
   hardware with the envelope and a non-zero sample count recorded; modeled
   means the number was derived by arithmetic from known inputs with no wall
   clock. The contract enforces the split: `validateEvidence` in
   `internal/mixedprecision/contract.go` requires an observed record to carry
   all four hardware fields and a non-zero sample count, and forbids a modeled
   record from carrying either.
2. Nothing. The slot is real and empty: `EvidenceObserved` exists and is
   validated, and no measurement has been filed through it. An empty slot is a
   seam, not a gap, because a future lab run can attach observed evidence
   without reclassifying a model estimate as a measurement.
3. Any two of: the bytes-per-parameter figures in `internal/model/quant.go` and
   `internal/model/quant_q4k.go`; the decode roofline in
   `internal/model/decode_roofline.go` and `internal/model/batch.go`; the cache
   accounting in `internal/model/asymmetric_kv.go`; the sensitivity ordering in
   `internal/model/v41_quant_sensitivity.go`. Each is honest because it names
   the inputs it derived from and does not claim a wall clock it never took.
4. No. The asymmetric codec in `internal/model/kvquant.go` is a codec plus
   accounting, and its own comment says nothing in the decode path calls it yet.
   The live cache stores 32-bit rows, `kvcache.go:104-106` returns 32-bit for a
   nil cache, and `quantized()` is true only for the 8-bit tier.
5. Was this number produced by running code on hardware or derived by
   arithmetic? If derived, from what inputs, and does the claim admit that? Does
   the live execution path actually call this code, or does the code merely
   exist?

</details>

## See also

- [Level 4 — The precision machinery in fak](level-4-the-precision-machinery.md) — the machinery this rung grades.
- [Level 3 — What to quantize, and what to leave alone](level-3-what-to-quantize.md) — the sensitivity argument, graded above.
- [The fak mixed-precision ladder](README.md) — all five rungs.
- [Level 5 of the inference ladder](../inference/level-5-at-scale.md) — the same boundary, on the inference axis: its closing section names the honest open edges the same way this rung grades the evidence.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · prev ← [Level 4 — The precision machinery in fak](level-4-the-precision-machinery.md)*

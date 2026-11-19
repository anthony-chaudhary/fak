---
title: "The fak mixed-precision ladder: five guides, five audiences"
description: "One subject, how a model stores its numbers, explained five times. From what a bit is, to the evidence boundary where measured claims end and modeled ones begin."
slug: mixed-precision-ladder
keywords:
  - mixed precision explained
  - quantization guide
  - what is fp8
  - int8 vs fp16
  - KV cache quantization
  - asymmetric quantization
  - precision for beginners
  - advanced quantization
  - fak mixed precision ladder
date: 2026-11-18
---

# The fak mixed-precision ladder

> **In one breath:** A model is made of numbers. Each number takes some bits.
> Fewer bits means less memory and faster answers. This ladder explains that
> trade five times.

**One line:** Mixed precision is the practice of storing different parts of a model in different numeric formats, chosen so that the parts which tolerate rounding are compressed and the parts which do not are left alone; this ladder explains the same subject at five depths.

The same subject, written **five times**, once for each kind of reader. Each rung
is pitched at both a human audience *and* a model/agent capability tier: a
newcomer is served at Level 1; a frontier-tier reader is served at Level 5.

This is the deep dive that the [inference ladder](../inference/README.md) hands
off to at its Level 4. If you have not read that rung, read it first: it
establishes why bytes are time.

## Why does this subject deserve its own ladder?

Because it is where the interesting engineering is, and because it is where
overclaiming is easiest.

Every inference stack quantizes something. Very few say clearly which of their
quantization claims have been measured on hardware, which are arithmetic
derivations, and which are designed but not yet connected. That distinction is
the spine of this ladder, and Level 5 is entirely about it.

## The five rungs

| Level | Who it's for | What you'll get |
|---|---|---|
| **[1 · What is a bit, and why fewer?](level-1-what-is-a-bit-and-why-fewer.md)** | A complete beginner. No binary, no floating point. | What a number's width means, why it costs memory, and why it costs time. |
| **[2 · Number formats, from fp32 to 4-bit](level-2-number-formats.md)** | A student who knows what a float is. | The format zoo, what each one can and cannot represent, and how a 4-bit number is unpacked. |
| **[3 · What to quantize, and what to leave alone](level-3-what-to-quantize.md)** | A practitioner who has quantized something and lost quality. | Sensitivity, the margin argument, and why Keys and Values want different treatment. |
| **[4 · The precision machinery in fak](level-4-the-precision-machinery.md)** | A systems engineer wiring precision into a runtime. | The contract, the support tensor, the dynamic policy, the schedule, and the fences. |
| **[5 · The evidence boundary](level-5-the-evidence-boundary.md)** | A researcher who wants to know what is actually proven. | Measured versus modeled versus designed-not-wired, applied to every claim above. |

## The whole ladder in one picture

```text
   L1  What is a bit?              ── width costs memory, and bytes are time
        │
        ▼
   L2  What formats exist?         ── fp32, fp16, bf16, fp8, int8, 4-bit
        │
        ▼
   L3  What can I compress?        ── sensitivity is about margin, not size
        │
        ▼
   L4  How does fak do it?         ── contract, support tensor, policy, schedule
        │
        ▼
   L5  What is actually proven?    ── measured vs modeled vs not-yet-wired
```

## How to read it

- **New to the idea?** Read 1, then 2. You will understand why anyone bothers.
- **Quantized something and it got worse?** Start at 3. It explains which parts
  are sensitive and why.
- **Wiring precision into a runtime?** Start at 4. It is the machinery.
- **Evaluating a claim?** Read 5 first. It grades everything above it.

## The one idea, if you read nothing else

Rounding a number is only dangerous when the model was **close to a decision**.

A confident prediction survives aggressive rounding. A near-tie does not. So the
right question is never "how many bits can I afford" but "which numbers are
close to a decision, and which are not." Every scheme in this ladder is an
answer to that question.

## Where the code is

| Concept | File |
|---|---|
| The neutral precision contract | `internal/mixedprecision/contract.go` |
| The format list | `internal/compute/compute.go` |
| 8-bit float decode | `internal/compute/fp8.go`, `internal/mathx/fp8.go` |
| Adaptive precision policy | `internal/model/dynamic_precision.go` |
| Precision scheduling | `internal/model/precision_schedule.go` |
| Asymmetric KV design | `internal/model/asymmetric_kv.go` |
| The KV wiring caveat | `internal/model/kvquant.go` |
| 4-bit activation-aware format | `internal/model/awq.go` |

## See also

- [The fak inference ladder](../inference/README.md) — the parent ladder; its Level 4 hands off here.
- [AWQ quantization](../awq-quantization.md) — one concrete 4-bit format, in full detail.
- [SOTA optimizations](../sota-optimizations.md) — where fak stands on the tuned-optimization list.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · next → [Level 1 — What is a bit, and why fewer?](level-1-what-is-a-bit-and-why-fewer.md)*

---
title: "Why shape and dtype parity cannot establish loader correctness"
description: "A GGUF tensor and fak's canonical tensor can agree on every dimension, every byte and the dtype while carrying different mathematical meanings. That is what broke Qwen3-Next decoding in #4273: blk.*.ssm_a stores -exp(A_log), fak's canonical tensor is A_log, and the forward applied exp twice. Shape checks, dtype checks and byte-count checks all passed. This page explains the defect class, why the obvious checks are structurally blind to it, and the semantic transform contract fak now declares for every non-identity external-to-canonical mapping."
slug: loader-semantic-transform-contracts
keywords:
  - GGUF loader
  - tensor mapping
  - semantic transform contract
  - loader correctness
  - shape manifest
  - model conversion
  - Qwen3-Next
  - gated delta net
  - silent numerical defect
date: 2026-07-26
---

# Why shape and dtype parity cannot establish loader correctness

*For anyone writing or reviewing a model loader — the code that reads someone else's
checkpoint format and hands tensors to your forward pass. No GPU or model download
needed to follow along; the worked example is a four-element vector. By the end you'll
know why a loader can pass every structural check and still be wrong, and what a
semantic contract adds that a shape manifest cannot.*

A loader bug usually announces itself. You get a panic, a dimension mismatch, a NaN
storm, or output that degenerates into punctuation. Those are the cheap bugs: the
failure is loud, local, and lands on the first token.

Then there is the other kind. In [#4273](https://github.com/anthony-chaudhary/fak/issues/4273),
fak loaded Qwen3-Next and produced text that read fine for a sentence or two and then
quietly lost the thread. Every structural check passed. The tensor had the right name,
the right rank, the right dimensions, the right dtype, and the right byte count. The
bytes that came out of the file were, element for element, the bytes that went into
the forward.

The loader was still wrong, because the bytes did not *mean* what the forward thought
they meant.

## The defect

Qwen3-Next's gated delta net decays state by a learned per-head coefficient. The
canonical parameter is `A_log`, and the forward wants

```text
decay = exp( -exp(A_log) * softplus(dt) )
```

Note the two exponentials. The inner `-exp(A_log)` turns an unconstrained real into a
strictly negative rate; the outer `exp` turns that rate into a multiplier in `(0, 1)`.

The GGUF exporter — `convert_hf_to_gguf.py` — does not store `A_log`. It stores the
inner transform already applied:

```python
A = -A_log.float().exp()   # what lands in blk.*.ssm_a
```

fak's canonical tensor is `A_log`, the *pre-transform* value, because every runtime
path (CPU, CUDA, the quantized paths) recomputes the decay from it. So the loader was
handed `-exp(A_log)`, labelled it `A_log`, and the forward applied `exp` a second
time. The model computed `exp(-exp(-exp(A_log)) * softplus(dt))`.

## Why every obvious check passes

Walk the checks a careful loader review would run:

| Check | Result on the #4273 defect |
|---|---|
| Tensor name maps to a known canonical name | passes — `blk.0.ssm_a` → `linear_attn.A_log` |
| Rank and dimensions match the config | passes — both are `[num_value_heads]` |
| dtype matches | passes — F32 either way |
| Byte count matches | passes — identical |
| Values are finite | passes — `-exp(x)` is finite for finite `x` |
| No NaN or Inf reaches the forward | passes |
| Output is fluent English | passes, for short prompts |

Every one of these is a check on the *container*. `A_log` and `-exp(A_log)` live in the
same container: same shape, same dtype, same element count, both finite. A structural
check cannot separate them because there is no structural difference to find. The
difference is a claim about what the numbers denote, and denotation is not a property
the file format records.

The output check fails for a subtler reason. Because the double `exp` maps into a
narrow band near 1, the corrupted decay is *nearly* a valid decay. State leaks slowly
rather than exploding. Short prompts stay plausible; the error compounds only over
enough tokens for the recurrent state to matter. So the loudest available signal —
"does it sound right?" — is exactly the signal this defect suppresses.

That is the general shape of the class:

> **A semantic loader defect is one where the external and canonical domains overlap
> numerically.** The wider the overlap, the quieter the failure, and the less any
> structural or eyeball check can say about it.

## What the neighbouring mechanisms do and do not cover

fak has three adjacent correctness mechanisms, and it is worth being precise about the
gap each leaves:

- A **shape-first manifest** ([#3251](https://github.com/anthony-chaudhary/fak/issues/3251))
  catches layout and dimension drift cheaply, from the header, without reading weights.
  It is structural by construction, so it is blind to this class.
- An **independent oracle** ([#442](https://github.com/anthony-chaudhary/fak/issues/442) /
  [#474](https://github.com/anthony-chaudhary/fak/issues/474)) compares fak's output
  against a reference implementation and *would* eventually catch it — but only
  end-to-end, only with a real checkpoint and a long enough prompt, and it reports "the
  logits diverge," not "tensor X is in the wrong domain."
- **Dtype and byte-count validation** catches truncation and quantization mistakes,
  which are a different class entirely.

What is missing between them is a statement of *meaning* that can be checked cheaply
and localized precisely. That is the semantic transform contract.

## The contract

`internal/ggufload/gguf_transform_contract.go` declares, for every non-identity
external→canonical mapping, a `TensorTransformContract`: the external and canonical
names, the source and destination **semantic domains** with their validity ranges, a
**named transform identifier**, **provenance** (which exporter convention produced the
source domain and what the forward consumes), and whether the transform is lossless
and invertible.

The `ssm_a` entry is the one that would have caught #4273:

```go
{
    External: "ssm_a", Canonical: "linear_attn.A_log",
    Transform: TransformValueHeadDeinterleave + "+" + TransformInvertNegExpDecay,
    SourceDomain: "negated exponential decay coefficient -exp(A_log): " +
        "finite and strictly negative",
    CanonicalDomain: "raw gated-delta-net decay parameter A_log (finite real)",
    // ... provenance naming convert_hf_to_gguf.py and the #4273 fix ...
    Lossless: false, Invertible: true,
    RejectsCanonicalDomain: true,
    HasValueSample: true,
    SampleSource: -1.6487213, // -exp(0.5)
    SampleCanonical: 0.5,
}
```

Three fields carry the enforcement weight:

**`SourceDomain` becomes a runtime check.** "Finite and strictly negative" is not prose
— the loader validates it, so a fixture or checkpoint carrying raw `A_log` values (which
are routinely non-negative) is *refused* rather than silently misread. `RejectsCanonicalDomain`
records that the two domains are separable this way. Most transforms cannot make that
claim: a full RMSNorm gain and a residual gain `g-1` are both unconstrained reals, and
no range check separates them.

**The value witness kills the identity mutation.** `SampleSource` and `SampleCanonical`
differ by construction, so a tensor filled with `-exp(0.5)` must come out as `0.5`
everywhere. Replace the inverse transform with identity — the exact mutation that caused
#4273 — and the witness fails at `go test` time, before any model generation. This is
what the issue's witness row asks for, and it is much cheaper than an oracle run: no
checkpoint, no GPU, no prompt.

**The transform identifier is header-derivable.** `TransformIDForGGUFTensor` resolves
from the tensor *name* plus `general.architecture` and touches no weight payload, so
`File.TensorTransformID` / `TensorTransformIDs` and the `Transform` field on the
metadata export answer for a multi-hundred-gigabyte checkpoint at the cost of a header
parse. The test proves this non-forgeably: the fixture GGUF ends where the tensor data
blob would begin, so anything that reached for a payload would read past EOF.

## Why the registry is keyed by architecture

The same external name means different things in different families. `attn_q.weight` is:

- **stacked-q-rotary-unpermute** under qwen35 (gated attention stacks a sigmoid gate
  under each head's query rows; only the query half is rotary-permuted),
- **rotary-unpermute** under the llama-family NORM-rope architectures,
- **identity** under the NEOX-layout architectures (qwen3, gemma3, phi3, …), which are
  exported already in the layout fak consumes.

A name-keyed registry would have to pick one and be wrong twice. Architecture-keying is
not incidental — it *is* the semantic distinction, and it is precisely the distinction a
shape manifest cannot express: the qwen35 and llama fixtures in the test are byte-identical
apart from the architecture string, and resolve to different transforms.

## The lint

A contract that only covers today's mappings decays into documentation. So the registry
is enforced in both directions by `TestNonIdentityMappingsDeclareTransformContracts`,
which probes every tensor of each family through the **live** loader path
(`normalizeCanonicalTensorData`) and fails when:

- the loader transforms a tensor that has **no declared contract** — a new transform-bearing
  mapping cannot land silently;
- a contract exists for a mapping the loader now maps **identically** — a stale contract
  cannot linger;
- a contract omits its transform name, provenance, or either domain;
- a contract names an external tensor with **no behavioral probe**, so a contract cannot
  be declared without also wiring the probe that checks it.

The audit is complete rather than sampled: `normalizeCanonicalTensorData` has exactly two
transform sources — the qwen35 hybrid path and the q/k rotary unpermute — which is why
covering qwen35 (15 contracts), the llama-family rotary pair (2), and the NEOX arches
(none, asserted) covers every non-identity mapping in the loader.

## What this does not do

Worth stating plainly, because a correctness mechanism that oversells itself is its own
hazard:

- It checks that the loader applies the transform it **declares**. If a contract declares
  the wrong transform — because the exporter's convention was misread in the first place —
  the contract and the loader agree with each other and both are wrong. Provenance exists
  to make that reviewable by a human, not to make it machine-checkable.
- It says nothing about tensors the loader maps identically and *should not*. A missing
  transform on a mapping nobody realized was non-identity is invisible here; only an
  independent oracle finds that.
- It is per-tensor. Composition errors across tensors, and everything downstream of the
  loader, are out of scope.

One known limit is recorded in the registry itself: `normalizeQwen35LinearTensor` returns
early when `LinearNumValueHeads == LinearNumKeyHeads`, skipping both the deinterleave and
the `ssm_a` inversion. Every shipped Qwen3.5 / Qwen3-Next checkpoint declares more value
heads than key heads, so the contracts hold for every real artifact — but an equal-heads
export would need that early return revisited before they extend to it.

## The takeaway

Shape, dtype, and byte-count checks establish that a loader read the file it was given.
They cannot establish that it understood it. The gap between those two statements is
where the expensive, quiet, hard-to-attribute loader bugs live, and closing it takes an
explicit claim about meaning — source domain, destination domain, named transform,
provenance — that a machine can check and a human can review.

**Related:** [`internal/ggufload/gguf_transform_contract.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/ggufload/gguf_transform_contract.go)
(the registry) ·
[#4744](https://github.com/anthony-chaudhary/fak/issues/4744) (this contract) ·
[#4273](https://github.com/anthony-chaudhary/fak/issues/4273) (the root incident) ·
[#3251](https://github.com/anthony-chaudhary/fak/issues/3251) (shape-first manifest) ·
[#442](https://github.com/anthony-chaudhary/fak/issues/442) /
[#474](https://github.com/anthony-chaudhary/fak/issues/474) (independent oracles)

---
title: "max|Δ| = 0: the bit-exact eviction result as a claim you can check"
description: "fak's addressable KV cache can remove a span from a live attention cache and leave the result bit-identical to a run that never saw the span — max|Δ| = 0, every logit matching to the last bit, with a non-vacuous poison control at max|Δ| ≈ 0.326. This story tells what bit-exact eviction means, states the exact falsifiable assertion, and gives the one command that checks it with no key, model download, GPU, or network. Every figure is witnessed against CLAIMS.md and the in-repo example."
slug: bit-exact
keywords:
  - bit-exact eviction
  - addressable KV cache
  - provable deletion
  - KV cache eviction
  - max delta zero
  - falsifiable claim
  - attention cache
  - quarantine eviction
date: 2026-07-17
---

# max|Δ| = 0

**Short answer.** fak can take a span out of a live, kernel-owned attention cache —
evict it — and the next-token distribution comes out **bit-identical** to a session
that *never saw that span*. Not "close." Not "within tolerance." Every logit matches
to the last bit: **`max|Δ| = 0`**. The claim is falsifiable, and there is one command
that checks it in a few seconds with no key, no model download, no GPU, and no network.

*For anyone who has heard "the cache remembers everything" and wondered whether a KV
cache can actually *forget*. By the end you will know exactly what bit-exact eviction
asserts, why the number is `0` and not `4e-5`, what the honest fences are, and how to
run the check yourself. Every figure here is witnessed against
[`CLAIMS.md`](../../../CLAIMS.md) and the runnable
[`examples/addressable-evict/`](../../../examples/addressable-evict/README.md).*

## What "bit-exact eviction" claims

Most caches are speed structures: they make a repeated computation cheaper. A KV cache
is one of those — it holds the per-token key/value tensors so attention need not
re-prefill the prompt every turn. The trouble is that a plain cache can only *grow* or
be *dropped wholesale*. You cannot reach in, pull one span out — a poisoned tool
result, an expired turn — and trust that what is left is still correct.

fak's KV cache is a **kernel-owned Go structure**, span-addressed, so it can. The
assertion is precise:

> Build a context out of segments, then **evict** one of them. The resulting
> next-token distribution is **bit-identical** to a context that only ever prefilled
> the segments that survived — `max|Δ| = 0`, with the same argmax and the same
> tie-break.

The mechanism is the same `Clone()` + `Evict()` the radix tree uses for its edge
splits: evicting a span re-RoPEs and renumbers the segments after it, so a later token
attends to exactly the positions it would have if the evicted span had never existed
([`docs/explainers/addressable-kv-cache.md`](../../explainers/addressable-kv-cache.md)
walks the offset renumbering segment by segment).

## Why the number is 0, and not 4e-5

There are *two* different "exactness" numbers in this concept and it is worth keeping
them apart, because leading with the wrong one is how you lose a careful reader.

| comparison | what it measures | value |
|---|---|---|
| **forward pass vs HuggingFace oracle** | does fak's pure-Go SmolLM2-135M math track a reference implementation? | `max|Δ| ≈ 4.4e-5` (argmax-exact) |
| **evict-vs-never** | does removing a span match never-having-seen it? | **`max|Δ| = 0`** (bit-identical) |

The `4.4e-5` is float noise between two *independent* implementations of the same
matmuls — real, honest, and argmax-exact, but not zero. The eviction number is `0`
because it is not comparing two implementations at all: it is the *same* fak forward
pass run two ways, and the claim is that the two ways produce the identical bits. A
non-zero eviction delta would mean a stale position offset had leaked; `0` is the
proof it did not. (Both are witnessed in
[`CLAIMS.md`](../../../CLAIMS.md) — the oracle at `go test ./internal/model`, the
eviction bridges at `go test ./internal/kvmmu`.)

## The control that makes it non-vacuous

`max|Δ| = 0` on its own could be a bug that changes nothing — a no-op that "passes" by
never doing the work. So the check ships with a deliberate **poison control**: instead
of evicting the span, *keep* it (a poisoned tool result) and measure the same delta.

- **evict-vs-never: `max|Δ| = 0`** — bit-identical. Removing the span loses nothing.
- **poison-vs-never: `max|Δ| ≈ 0.326`** — clearly non-zero. Keeping the span *does*
  perturb the distribution, so the eviction path is measuring something real.

One number near zero next to a control that is decidedly not zero is the shape of an
honest measurement.

## Run it yourself

The check is a single script — no key, no model download, no GPU, no network:

```
./examples/addressable-evict/run.sh
```

It builds a context, routes a poisoned span through the real quarantine gate, evicts
it write-time, and prints:

```
max|Δ| evict-vs-never = 0.000e+00 (want 0) ; poison-vs-never = 3.257e-01 (want >0)
```

To see the same property from the kernel's own tests:

```
go test ./internal/kvmmu        # quarantine-evict + ApplyPlan: bit-identical (max|Δ|=0) vs a resident-only reference
go test ./internal/model        # HF oracle: forward pass argmax-exact, KV-decode/evict token-for-token identical
go run ./cmd/deletioncert -selfcheck   # signs & re-verifies the deletion certificate that binds max|Δ|=0
```

## The honest fences

Bit-exact eviction is a shipped, tested property — but be precise about its edges:

- **The `max|Δ| = 0` witness uses a synthetic model.** The bit-exactness of the
  *eviction* (offset renumbering, re-RoPE) is what that test proves. The realism of
  the *numerics* is proven separately by the HuggingFace oracle on a real SmolLM2-135M
  forward pass — two claims, each with its own witness, not one claim doing double
  duty.
- **The deletion certificate is a self-signed v1 receipt.** It attests the integrity
  of the recorded facts (evicted count, the bound `max|Δ| = 0`, a hash-chained
  anchor), and fails closed on any forged field or non-zero drift — but the bound
  delta is checked as a *signed string*, not re-measured inside the certificate. The
  re-measurement is the `internal/kvmmu` witness above.
- **It is a coherent compaction, not amnesia mid-attention.** Bit-identical-to-never
  holds for spans no later resident token attended. Evicting a span some later token
  *did* attend to is a different, deliberate operation (a page-back-in handle keeps it
  recoverable), not a silent zero.

## Why it matters

A cache that can only remember is a speed structure. A cache that can **provably
forget a specific span, bit-for-bit,** is a correctness and trust structure: it is what
lets a poisoned tool result be quarantined at the attention layer instead of just the
byte layer, and what lets an expired turn actually leave. `max|Δ| = 0` is the number
that turns "we deleted it" from a promise into a claim you can check.

---

**Related:** [`docs/explainers/addressable-kv-cache.md`](../../explainers/addressable-kv-cache.md)
(the full mechanism, segment by segment) ·
[`examples/addressable-evict/`](../../../examples/addressable-evict/README.md) (the runnable check) ·
[`CLAIMS.md`](../../../CLAIMS.md) (the central claim ledger and every witness).

_Dimension H (Benchmark-as-story) of the
[concept-popularization epic](../../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)._

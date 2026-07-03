---
title: "What fak Is Not — the Honest Boundary"
description: "fak does not replace vLLM, SGLang, or llama.cpp for raw tokens/sec — it fronts them for the agent boundary. The limits, volunteered plainly, are the point."
slug: what-fak-is-not
keywords:
  - fak is not a serving engine
  - vLLM
  - SGLang
  - llama.cpp
  - agent boundary
  - prior-art audit
  - honest scope
  - capability floor
  - gateway tax
  - assembly not invention
date: 2026-07-03
---

# What fak Is Not — the Honest Boundary

> **TL;DR:** `fak` is **not** a serving engine. It does not compete with
> vLLM / SGLang / llama.cpp on raw tokens per second — it **fronts** them for the
> agent boundary: the tool call. None of its primitives is novel (a 0/29 prior-art
> audit says so); the contribution is the **assembly**. This page volunteers the
> limits first, because the limits are the honest part of the pitch.

**Concept served:** one static Go binary that is a *boundary*, not an engine
(one of the [popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md) concepts).

Most projects sell you the ceiling. This page starts with the floor plan of what
`fak` deliberately does **not** do — because a tool that names its own limits is
easier to trust than one that doesn't, and it pre-empts the "they're overclaiming"
takedown before anyone has to write it.

## 1. It is not a faster serving engine — it is the gate in front of one

`fak` does not decode tokens faster than the engine underneath it. Run as a
gateway fronting SGLang, it *trails* raw SGLang **0.75× at peak** — 1085.6 vs
1451.6 tok/s (Qwen3.6-27B, 64-concurrent) — the price of putting an adjudication
decision on the call path. That gateway/adjudication tax converges toward **~3% at
saturation**. [WITNESSED]

That is not a bug to hide; it is the shape of the product. `fak`'s field is
**governance and containment at the agent boundary**, not throughput. It sits in
front of vLLM, SGLang, or llama.cpp and adjudicates every tool call — deny by
structure, repair a malformed call, quarantine a poisoned result — while the engine
behind it does what engines are good at: emit tokens. If you need raw tokens/sec,
the engine is the answer. If you need the *tool call* to pass through a boundary the
model cannot talk past, that is the part `fak` adds.

The in-kernel model path exists, but it is a **bit-exact correctness reference**,
not a tuned server: no continuous batching, no paged attention, no multi-tenant
scheduler. For real serving, you run `fak serve` in front of vLLM / SGLang /
llama.cpp — not instead of them.

What `fak` *does* collapse is the **operational surface** of governing that
boundary. The usual governed-serving stack is four cooperating processes — a
reverse proxy, a policy layer, a result-quarantine service, an audit sidecar —
each its own deploy, port, and config. `fak` carries the same four
responsibilities as in-process stages of one static Go binary: you add flags, not
components.

![Two columns with the same four responsibilities labeled identically on both sides — reverse proxy / gateway, policy / capability floor, result quarantine, audit journal. Left, the usual governed-serving stack runs them as four separate processes; right, one static fak binary holds the same four as in-process stages. A blue arrow reads "four processes → one": you add flags, not components](../adoption/diagrams/single-binary.svg)

That is the honest boundary of the drop-in claim: `fak` does not make your tokens
faster, but it does replace a multi-process governance stack with one process that
does the same job.

## 2. None of its primitives is novel — and that is the point

A 29-claim prior-art audit scored **0/29 novel**
([`CLAIMS.md`](../../CLAIMS.md), catalogued in the
[innovations index](../INNOVATIONS-INDEX.md)). Default-deny capability gates,
KV-cache reuse, prefix caching, audit journals — every part is old and has a
literature.

The contribution is the **assembly**: one fused, fail-closed, witness-gated kernel
where the tool call is promoted to an in-process syscall, co-resident with the KV
cache. The parts are borrowed; the *wiring* — and the honesty ledger that keeps it
honest — is the thing a competitor structurally cannot copy by copying a primitive.
Framing the assembly (not a claimed invention) is the honest posture, and it is the
more durable one.

## 3. The detector is not the guarantee — the floor is

`fak` ships a result-quarantine detector that flags suspicious tool output. It is
**best-effort and ~100% evadable by design** — it is explicitly *not* the
load-bearing defense. Any framing that sells the detector as the guarantee
overstates the product.

The load-bearing defense is the **default-deny capability floor**: an action that
was never allow-listed cannot run, no matter what the model was talked into. The
detector is a bonus on top of a floor that holds without it. Detectors are evadable;
the floor is the cap lock. (Longer treatment:
[Why default-deny beats a classifier](default-deny-vs-classifier.md).)

## The numbers, fenced

When `fak` *does* quote a win, it quotes the witnessed one against the real
alternative, never the strawman:

- **~4.1× vs a tuned warm-cache baseline** on a 50-turn × 5-agent run
  (Qwen2.5-1.5B, Apple M3 Pro, commit `2bbda6f`) is the headline that survives
  procurement. The **60.3×** figure is *only* vs a naive re-send-everything loop
  nobody runs in production — it is never the standalone competitive number.
  [WITNESSED on one machine + a deterministic model.]
- The cache-reuse win applies only to a **self-hosted** model whose KV cache `fak`
  owns. Front a frontier API and you get the safety floor but none of the reuse
  savings — the provider owns prefix caching upstream.
- The WebVoyager **8.8×–9.7×** figure is a **modeled** prefill-work floor over the
  real 643-task set, not wall-clock. Labeled simulated where it appears. [SIMULATED]

## The one line to keep

**`fak` is not an engine — it is the boundary in front of one.** It does not make
your tokens faster; it makes the tool call *governed*. The parts are old, the
detector is evadable, and the throughput tax is real — and saying so is the feature.

## See also

- [The tool call is a syscall](tool-call-is-a-syscall.md) — the keystone mental model.
- [Why default-deny beats a classifier](default-deny-vs-classifier.md) — why the floor, not the detector, is the guarantee.
- [The addressable KV cache in 5 minutes](addressable-kv-cache-in-5-min.md) — the reuse lever, fenced.
- [Innovations index](../INNOVATIONS-INDEX.md) — the 0/29 prior-art audit in full.
- [FAQ](../FAQ.md) — including "is the in-kernel model engine ready to serve production traffic?"

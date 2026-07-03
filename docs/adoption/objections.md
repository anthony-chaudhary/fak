---
title: "Objections & one-line answers: a pocket card for fak advocates"
description: "The common objections to fak — classifier, slowdown, why not vLLM, evadable detector, another gateway — each with a crisp, honest one-to-two-line answer and a deeper link. For defending the concept in a thread."
slug: objections-and-answers
keywords:
  - fak objections
  - agent kernel FAQ
  - default-deny capability gate
  - prompt injection
  - vLLM comparison
  - advocate talking points
  - honest rebuttals
  - is it a classifier
date: 2026-07-02
---

# Objections & one-line answers: a pocket card for fak advocates

> **TL;DR:** advocates win or lose a concept in comment threads. This is the
> pocket set of tight, correct rebuttals to the objections fak actually draws —
> each a one-to-two-line answer plus a link to the deeper page, so you can defend
> the idea in real time without overstating it. Every answer is consistent with
> the explainers it links; where fak's honest scope concedes a point, the card
> concedes it too.

This is dimension **I — Memorable framing & naming** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
It serves the *default-deny capability gate + quarantine*, *one static Go binary*,
and *drop-in* concepts. The rule for this card: **an honest answer beats a winning
one.** If an objection is partly right, say so — the concession is what makes the
rest credible.

## The card

### 1. "Isn't this just another prompt-injection classifier?"

No — the load-bearing floor is a **capability lock**, not a recognizer. A tool
that is off the allow-list cannot be called no matter what the model was told;
refusing does not depend on *catching* an attack. The result detector is a bonus
and is ~100% evadable by design.
→ [Policy in the kernel](../explainers/policy-in-the-kernel.md)

### 2. "Doesn't putting a gate in front slow the agent down?"

The allow/deny decision runs in-process on the same call path — no network hop.
The measured guard tax is **~362 ns per call** (Apple M3 Pro). On long sessions it
is net *faster*, because it sheds re-sent history while keeping the provider's
cache discount.
→ [Observer-effect standard](../standards/observer-effect.md)

### 3. "Why not just use vLLM or SGLang?"

Do — for raw tokens/sec. fak is not a token engine; it **fronts** one. It owns the
agent boundary those engines leave open: capability, quarantine, audit, routing,
and reuse legality. Put `fak serve` in front of vLLM, not instead of it.
→ [fak vs vLLM, SGLang & provider KV caching](../fak-vs-alternatives-comparison.md)

### 4. "If the detector is ~100% evadable, is the security actually real?"

Yes, because the detector is not the floor. The floor is the **capability lock**
(the dangerous lever was never wired up) plus **containment** (poisoned results
held out of context by structure). Both fail *closed*. The evadable detector is an
explicitly-labeled bonus.
→ [Security policy](../../SECURITY.md) · [tool call is a syscall](../explainers/tool-call-is-a-syscall.md)

### 5. "Another gateway? I already have a router / proxy."

fak *complements* request-level routers (OpenRouter, Portkey, LiteLLM): it governs
the **tool-call** boundary and routes per *aspect*, where they route whole
requests. And it is one static binary, not another multi-component service to
operate.
→ [Routers & gateways](../integrations/routers.md)

### 6. "Is this novel, or a repackaging of known ideas?"

Repackaging — honestly so. A 29-claim prior-art audit scored **0/29 novel**; every
primitive is established. The contribution is the **assembly** into one in-process
gate where the tool call is the checkpoint. We claim the assembly, not an
invention.
→ [Claims ledger](../../CLAIMS.md)

### 7. "How is this different from a firewall or a guardrails library?"

The boundary is the **tool call** (a syscall), not a network packet or a text
pattern. "Firewall" is an analogy for the default-deny posture. Guardrails try to
recognize bad text; fak makes the dangerous action structurally impossible.
→ [FAQ](../FAQ.md)

### 8. "Do I have to rewrite my agent to adopt it?"

No. You repoint **one base URL** at `fak serve` (or run `fak guard -- claude`).
Your model, IDE, and keys stay as they are; 41 of 47 surveyed harnesses drop in
with one base-URL change.
→ [One binary is the whole surface](../explainers/one-binary-one-surface.md)

### 9. "The cache/reuse numbers sound too good — 60×?"

The honest bar is the **tuned ~4.1×** on a 50-turn × 5-agent run, versus a tuned
warm-cache stack. The ~60× is versus a naive re-send loop and is not the figure we
lead with. Power/energy numbers elsewhere are simulated and labeled so.
→ [Benchmark authority](../../BENCHMARK-AUTHORITY.md)

### 10. "Can I trust the KV-cache eviction is really lossless?"

The addressable KV cache evicts one span mid-run and leaves the cache
bit-for-bit identical to a run that never saw it — verified at **`max|Δ| = 0`** by
a green test, not asserted.
→ [Addressable KV cache](../explainers/addressable-kv-cache.md)

## Honest scope

No market-adoption claim is made on this card. Where an objection lands, the answer
concedes it (the detector is evadable; the novelty is assembly; use a real engine
for throughput). Numbers are witnessed and traced to
[CLAIMS.md](../../CLAIMS.md) / [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md).

## Verify

```
test -f docs/adoption/objections.md                  # this artifact exists
python tools/seo_aeo_scorecard.py                    # new doc does not red the SEO scorecard
```

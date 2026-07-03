---
title: "Social storyboard: fak's five concepts as a shareable thread"
description: "A drop-in storyboard for a short social thread or slide carousel: one card per fak core concept, each with a one-line hook, the diagram that carries it, and a link. The reusable, honest spine for any social push."
slug: social-storyboard
keywords:
  - fak social thread
  - carousel storyboard
  - agent kernel explainer
  - treat the tool call like a syscall
  - default-deny capability gate
  - addressable KV cache
  - developer advocacy
  - shareable card deck
date: 2026-07-02
---

# Social storyboard: fak's five concepts as a shareable thread

> **TL;DR:** a card-per-concept storyboard an advocate can drop into a social
> thread or a slide carousel. Each card is a one-line hook, a sentence of body,
> the diagram that carries it, and a link to go deeper. Concepts spread in cards,
> not paragraphs — this is the reusable spine so you never rebuild the deck.

This is dimension **J — Distribution & channels** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
It serves all five core concepts by giving them a single shareable shape. Every
number here is witnessed and traced to
[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md); nothing claims market
adoption, and simulated work is labeled simulated.

## How to use this

- **As a thread:** post the cards in order, card 0 as the hook, cards 1–5 one per
  reply, card 6 as the call to action. Trim to whichever concepts fit the venue.
- **As a carousel:** one slide per card. The **Diagram** row names the visual to
  place on the slide; where a bespoke card image is not drawn yet it is tagged
  `[B-pending]` and points at the visual-assets dimension of the epic so the slide
  has a stand-in until the art lands.
- **Honesty fence:** keep the witnessed bar. The reuse figure is the **tuned
  ~4.1×**, not the naive re-send ratio. If you cite a number, cite the one on the
  card — each is the honest one.

## The storyboard

### Card 0 — Hook

- **Hook:** "Your AI agent runs every tool call on trust. What if one binary made
  it earn each one instead?"
- **Body:** fak is a fused agent kernel — one static Go binary you put in front of
  the agent you already run. It checks every tool call, reuses the stable work in
  long sessions, and writes a verdict for each decision.
- **Diagram:** the hero cost-curve still — [`visuals/hero-video.mp4`](../../visuals/hero-video.mp4)
  (stills + source data in [BENCHMARK-GALLERY.md](../../BENCHMARK-GALLERY.md)).
- **Link:** [README](../../README.md)

### Card 1 — Treat the tool call like a syscall

- **Hook:** "The model proposes. The kernel disposes."
- **Body:** An OS kernel never trusts a program's word — every dangerous action
  crosses a boundary the program does not control. fak applies the same boundary to
  the LLM's tool calls: the call is a syscall, and the kernel decides.
- **Diagram:** the agent-kernel boundary still — [`visuals/agent-kernel-video.mp4`](../../visuals/agent-kernel-video.mp4).
- **Link:** [The tool call is a syscall](../explainers/tool-call-is-a-syscall.md)

### Card 2 — Verify, don't trust (DOS)

- **Hook:** "A false 'done' gets refused — from git evidence, not the agent's word."
- **Body:** DOS is the trust substrate under a fleet of agents. A claim is verified
  from evidence, refusals carry a structured reason from a closed vocabulary, and
  recalled memory is re-checked at read time. The kernel is the part that doesn't
  believe the agents.
- **Diagram:** verify-don't-trust flow — `[B-pending]` see the visual-assets
  dimension of the [popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md);
  stand-in: the honesty-split figure in [adoption visuals](../adoption-visuals.md).
- **Link:** [Agent grammar standard](../standards/agent-grammar.md)

### Card 3 — The addressable, bit-exact KV cache

- **Hook:** "Reach into the middle of a running model, evict one span, leave the
  cache bit-for-bit identical (`max|Δ| = 0`)."
- **Body:** Long sessions stop getting expensive because the cached prefix
  survives. fak sheds the old middle turns by splicing on the original bytes, so the
  provider's prompt-cache discount holds instead of breaking.
- **Diagram:** the prefill-reuse diagrams — [prefill visuals](../prefill-visuals.md).
- **Link:** [Addressable KV cache](../explainers/addressable-kv-cache.md)

### Card 4 — Default-deny capability gate + quarantine

- **Hook:** "Refusing an irreversible action doesn't depend on catching an attack.
  The lever was never wired up."
- **Body:** The permission policy runs inside the kernel, on the same call path as
  the tool call — it fails closed. A separate quarantine holds suspicious tool
  *results* out of the model's context by structure, not by a classifier the model
  can argue past.
- **Diagram:** the two-gate capability figure — `[B-pending]` see the visual-assets
  dimension of the [popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md);
  stand-in: [adoption visuals](../adoption-visuals.md).
- **Link:** [Policy in the kernel](../explainers/policy-in-the-kernel.md)

### Card 5 — One static Go binary, drop-in

- **Hook:** "Repoint one base URL. The laptop binary is the fleet binary — you add
  flags, not components."
- **Body:** Gateway, capability gate, result quarantine, and audit surface in a
  single static Go binary with no external dependencies. `fak guard -- claude`
  wraps the agent you already run; 41 of 47 surveyed harnesses repoint with one base
  URL.
- **Diagram:** the where-the-binary-sits figure — [adoption visuals](../adoption-visuals.md).
- **Link:** [One binary is the whole surface](../explainers/one-binary-one-surface.md)

### Card 6 — Call to action

- **Hook:** "See a real verdict in 60 seconds — no key, no model, no GPU."
- **Body:** `fak preflight --tool refund_payment --args "{}"` prints a fail-closed
  `DENY (DEFAULT_DENY)` from the bare binary. Then wrap your own agent with
  `fak guard -- claude`.
- **Diagram:** none — this is the terminal card; link out.
- **Link:** [Getting started](../../GETTING-STARTED.md)

## Honest scope (read before you post)

- **Assembly, not novelty.** A 29-claim prior-art audit scored **0/29 novel** —
  every primitive is established; fak's contribution is the *assembly* into one
  in-process gate. Don't claim an invention the audit refutes.
- **Not a token engine.** fak fronts vLLM / SGLang / llama.cpp / a hosted provider;
  it does not replace them. Say "fronts tuned engines," not "beats them."
- **Numbers are witnessed.** The ~4.1× reuse figure is the tuned warm-cache bar;
  the ~362 ns guard tax is measured on an Apple M3 Pro; the KV-eviction result is
  `max|Δ| = 0`. Power/energy numbers elsewhere in the repo are simulated and
  labeled so.

## Verify

```
test -f docs/adoption/social-storyboard.md          # this artifact exists
python tools/seo_aeo_scorecard.py                    # new doc does not red the SEO scorecard
```

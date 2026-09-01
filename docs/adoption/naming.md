---
title: "How to find and name fak: agent runtime first, kernel architecture second"
description: "Call fak an agent runtime in public category messaging; use 'Fused Agent Kernel' and 'agent kernel' for its technical architecture, history, and disambiguation."
slug: naming-and-search
keywords:
  - fak agent runtime
  - fak agent kernel
  - fused agent kernel
  - long-session prompt cache
  - model routing for agents
  - treat the tool call like a syscall
  - vendor-neutral inference backend
  - how to search for fak
  - disambiguation
  - naming
  - discovery
date: 2026-07-02
---

# How to find and name fak: agent runtime first, kernel architecture second

> **TL;DR:** **fak is an agent runtime: the operator-controlled boundary for
> cache and context, model routing, tool authority, memory, observability, and
> native inference.** Lead with **agent runtime** as the simple public category.
> Use **Fused Agent Kernel** and **agent kernel** for the technical architecture,
> project history, and search disambiguation.

This is dimension **I — Memorable framing & naming** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
It serves three core concepts — *treat the tool call like a syscall*, *one static
Go binary*, and *drop-in* — by making the language used to reach them consistent.
The disambiguated terms below are the same ones the machine-facing answer-engine
surface uses (see [`llms.txt`](../../llms.txt)), so human word-of-mouth and machine
discovery point at the same names.

## The problem: the bare word does not resolve

`fak` is short, spoken, and collides. A raw search is drowned by:

- the **homophone** (an expletive and its many spellings),
- the **F.A.K.** acronym (first-aid kit, and dozens of unrelated initialisms),
- generic typo-noise.

So the rule is simple: **never ship the bare word alone.** In public category
messaging, pair it with **agent runtime**. In technical or historical material,
pair it with one of the architecture terms below.

## The terms to use (pair `fak` with one of these)

These are the canonical search + reference terms, drawn from
[`llms.txt`](../../llms.txt):

| Term | Use it when you mean |
|---|---|
| `fak agent runtime` | the public product category and primary first-contact handle |
| `fak agent kernel` | the technical architecture or a disambiguated historical search |
| `fused agent kernel` | the expanded technical name when `fak` alone is too ambiguous |
| `treat the tool call like a syscall` | the one-line mental model / slogan |
| `fak serve` | the gateway verb (the runnable surface) |
| `fak manage` | the one-command wrapper for an agent you already run |
| `long-session prompt cache` | the cost/reuse framing |
| `model routing for agents` | the per-call routing framing |
| `MCP tool-call boundary` | the MCP/adjudication framing |
| `vendor-neutral inference backend` | the interop/serving-boundary framing |
| `bring-your-accelerator agent serving` | the hardware-portability framing |
| `neo-silicon agent kernel` | the chip-vendor / backend-author angle |
| `fak backend conformance`, `fak-certified backend` | the conformance mark |

**The one category to lead with:** *fak, the agent runtime.*

**The technical name and architecture:** *Fused Agent Kernel*, shortened to
**agent kernel** where the architecture or syscall analogy is the subject.

## What category is this? (so people search the right shelf)

Put fak on the **agent runtime** shelf first. The rest are capability facets or
technical search handles, not competing top-level categories:

- **agent runtime** — the simple public category
- **agent kernel / Fused Agent Kernel** — the technical architecture and name
- **cache-efficient agent serving**
- **per-call model routing**
- **result quarantine for agent tools**
- **addressable KV cache**

## What fak is *not* (disambiguation from adjacent things)

Advocates lose the concept when a listener rounds it to the nearest familiar thing.
Draw these lines:

- **Not a network firewall or a WAF.** The boundary is the tool call (a syscall),
  not a network packet. "Firewall" is only an analogy for one policy posture, not
  the product category to lead with.
- **Not a prompt-injection classifier / guardrails library.** The load-bearing
  floor is the **capability lock** (a tool off the allow-list cannot be called),
  not a recognizer that tries to spot bad text. The result detector is ~100%
  evadable *by design* — a bonus, never the floor.
- **Not a token-serving engine.** fak does not replace vLLM, SGLang, or llama.cpp;
  it **fronts** one, owning the agent boundary (capability, quarantine, audit,
  routing, reuse legality). Use a tuned engine for raw tokens/sec.
- **Not a request-level model router (OpenRouter / Portkey / LiteLLM).** fak
  complements them: it governs the tool-call boundary and routes per *aspect*.
- **Not a hosted SaaS.** It is one static Go binary you run yourself.

## Give advocates the language

When you talk about fak, hand people a term they can search back to:

- Say: "It's an **agent runtime** — one operator-controlled boundary for the
  agent's context, models, tools, memory, evidence, and local inference."
- For the technical follow-up: "Its **Fused Agent Kernel** architecture treats
  every tool call like a syscall."
- Not: "It's called fak" (they will never find it).
- In writing, link the category: [fak, the agent runtime](../index.md).

## Honest scope

No market-adoption claim is made here. fak's contribution is the **assembly** of
established primitives into one in-process gate — a 29-claim prior-art audit scored
**0/29 novel**, so this note claims a clear *name*, not a novel invention.

Category breadth is not feature status: **agent runtime** names the boundary fak
intends operators to control. It does not claim every possible orchestration or
runtime feature is shipped, nor that every operating mode activates every listed
capability. [`CLAIMS.md`](../../CLAIMS.md) and the selected mode's documentation
remain authoritative.

## Verify

```
test -f docs/adoption/naming.md                      # this artifact exists
fak score seo                                        # new doc does not red the SEO scorecard
```

---
title: "Memory Engineering"
description: "Memory engineering, defined: what an AI agent remembers, how recall is verified, and when it is forgotten — decided by mechanism, not model judgment."
slug: memory-engineering
keywords:
  - memory engineering
  - memory engineering for AI agents
  - agent memory
  - agent memory engineering
  - context engineering
  - memory poisoning
  - verified recall
  - memory promotion
  - truth duration
  - write-time admission gate
  - forgetting
date: 2026-07-03
---

# Memory Engineering — the discipline after context engineering

**Memory engineering** is the engineering discipline of deciding what an AI agent
remembers, where that memory lives and who may share it, how a memory is
re-verified before it is trusted again, and when — provably — it is forgotten.
Its defining property is that each of those decisions is made by an inspectable
mechanism (a gate, a record, a check, an eviction) rather than by a model's
in-context judgment. Prompt engineering shaped one completion; context
engineering shapes one session's window; **memory engineering governs everything
an agent carries across sessions** — which is where long-running agents now fail
worst, and where "seems useful, keep it" stops being an acceptable policy.

## Why this needs a name — and why now

Each time the industry named a discipline, the unit of engineering grew. **Prompt
engineering** (2022–23) shaped a single completion. **Context engineering**
(2024–26) shapes a single session: what enters the window, in what order, at what
cost. Both are now taught, tooled, and hiring keywords. But the layer *above* the
session — the facts, preferences, skills, and scars an agent carries from one
session into the next — has accumulated plenty of *features* (vector stores,
memory SDKs, "long-term memory" checkboxes) and almost no *discipline*: no agreed
statement of what a memory system must be able to prove.

The phrase is already in the air. Database vendors position "memory engineering"
as the step after prompt and context engineering; one enterprise blog concedes it
is "still an emerging discipline"; the academic literature meanwhile says "agent
memory" and catalogs taxonomies of episodic/semantic/procedural tiers. What none
of that supplies is a definition with teeth — one that tells you whether a given
system is *doing* memory engineering or merely *storing things*. This page pins
the term to four checkable questions. A memory system either answers them with a
mechanism, or it answers them with vibes.

## The four questions of memory engineering

Every agent memory system answers these four questions. The only choice is
whether it answers them deliberately.

### 1. Admission — what earns promotion to memory?

The default failure is the salience trap: the write trigger is "the model found
this notable," so the system remembers *"it's 3pm"* (loudly present, false within
the hour) and forgets *"I prefer afternoon meetings"* (quietly said, true for
years). Context and memory are not separated by size, recency, or where the bytes
sit — they are separated by **truth-duration**, and that classification is
decidable at exactly one place: the moment a value would cross into durable
store. The engineered answer is a **write-time admission gate** that defaults
ephemeral facts to *expire, not persist*, and a structured **promotion record**
so "why is this fact in memory?" is answered from a ledger captured at write
time, never from a model's retrospective story (`fak memory explain-promotion`).
Full argument: [Context is not memory](../CONTEXT-IS-NOT-MEMORY.md); product
contract: [the managed-context glossary](../managed-context-glossary.md).

### 2. Placement — where does a memory live, and who may share it?

"Agent memory" compresses four different problems into one word: **routing**
(where a cell physically lives), **addressing** (the stable name two readers
share), **fusion** (whether bytes share one arena), and **semantics** (whether a
cell can be coherently mutated, isolated, attributed, and capability-gated across
a trust boundary — and proven). The serving world has largely solved the first
three; the discipline's open frontier is the fourth, because that is where
memory stops being a performance object and becomes a *trust* object. The
layer-by-layer map: [The four layers of agent
memory](../MEMORY-LAYERS-EXPLAINER.md).

### 3. Integrity — is this memory still true, and still safe, at recall?

A stored memory is a frozen self-report from a past session, and it ages three
ways: the world changes under it (staleness), an adversary writes through it
(**memory poisoning**, named in the OWASP Agentic and MCP Top-10 lists), or the
store itself rots (a flipped quarantine flag, a repointed digest). The
engineered answers carry the check *on the cell*: quarantine and taint metadata
that survive persistence, an ECC-style **syndrome** computed at page-in plus an
off-path patrol scrub over the persisted core image
([ECC-style memory integrity](../MEMORY-ECC-INTEGRITY.md)), and **verified
recall** — `fak memory recall` re-checks a note's concrete claims against ground
truth at read time and *withholds* what fails, rather than injecting a stale
memory wearing the authority of a fact.

### 4. Forgetting — can you remove a memory, and prove it is gone?

In most stacks deletion is an aspiration: the row is dropped but the embedding
remains; the "forgotten" fact still sits in a cached prefix. Memory engineering
treats forgetting as a first-class, *witnessable* operation: durability classes
that expire by default (forgetting the timestamp is the memory system
*working*), and at the hot layer, **addressable KV eviction** — remove one span
mid-run and leave the KV cache bit-for-bit identical to a run that never saw it,
verified at `max|Δ| = 0` ([Addressable KV cache](./addressable-kv-cache.md)).

## What memory engineering is not

- **Not RAG or a vector database.** Those are retrieval *mechanisms* a memory
  engineer may select. Buying one answers none of the four questions — a vector
  store admits everything, verifies nothing, and forgets approximately.
- **Not context engineering.** That discipline ends at the session window. A
  perfectly engineered window can still promote a poisoned fact to disk, where it
  outlives the session that would have caught it.
- **Not the KV cache alone.** The cache is the substrate of the placement
  question — one question of four.
- **Not memory management.** The OS term (paging, allocation) and the
  materials-science term (shape-memory alloys) share the words, not the field.
  For search, pair the phrase with agents: *memory engineering for AI agents*.

## The one-line test

> If your memory system cannot answer **"why is this fact in memory?"**, **"is it
> still true right now?"**, and **"can you prove it is gone?"** — you have memory
> *features*, not memory *engineering*.

## Where fak stands (honest fences)

fak's stake in the term is the definition above plus one working assembly that
answers all four questions at a single boundary — the same in-process,
default-deny kernel that adjudicates tool calls ([the tool call is a
syscall](./tool-call-is-a-syscall.md)). Per the project's honesty discipline
([CLAIMS.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)): the
primitives are not novel — admission gates, integrity codes, and eviction are
decades old in other memories; the truth-duration axis was named by cognitive
science and bitemporal databases long before agents. The shipped substance today
is the promotion ledger and its `explain-promotion` query, verified recall over
the notes backend, the syndrome/scrub integrity rungs over persisted session
images, and bit-exact span eviction; the write-time durability gate is specified
in [Context is not memory](../CONTEXT-IS-NOT-MEMORY.md) with its enforcement
rungs tracked there. Where a rung is proposed rather than shipped, that page
says so — this page names the discipline, not a finished product.

## Related reading

- [Context is not memory](../CONTEXT-IS-NOT-MEMORY.md) — the truth-duration
  axis and the write-time gate (question 1).
- [The four layers of agent memory](../MEMORY-LAYERS-EXPLAINER.md) — routing,
  addressing, fusion, semantics (question 2).
- [ECC-style memory integrity](../MEMORY-ECC-INTEGRITY.md) — syndrome, scrub,
  and the quarantine metadata a memory cell must carry (question 3).
- [Addressable KV cache](./addressable-kv-cache.md) — bit-exact mid-run
  eviction, the forgetting witness (question 4).
- [The managed-context glossary](../managed-context-glossary.md) — the
  user-facing vocabulary (memory promotion, budget envelope, reset transaction).
- External context for the emerging term: MongoDB's [agent memory
  guide](https://www.mongodb.com/resources/basics/artificial-intelligence/agent-memory),
  Oracle's [AI agent memory
  post](https://blogs.oracle.com/developers/oracle-ai-agent-memory-a-governed-unified-memory-core-for-enterprise-ai-agents),
  and the arXiv survey [Memory in the Age of AI
  Agents](https://arxiv.org/pdf/2512.13564) (which frames the same territory as
  "agent memory").

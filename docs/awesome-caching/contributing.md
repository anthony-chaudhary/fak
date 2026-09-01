---
title: "Contributing to Awesome Caching"
description: "How to add a caching concept to the in-repo awesome-caching list: one line, link to the concept's own self-description, pick the right section, tag fak status honestly."
---

# Contributing to Awesome Caching

This is an [awesome](https://awesome.re) list, scoped to this repository. Its one rule:

> **Each entry links to the concept's own self-description — it does not restate it.**

An entry that copies a paragraph of explanation into [`README.md`](README.md) creates a
second source of truth that drifts. Instead, point at where the concept already describes
itself, and keep this page a thin index.

## What counts as a "caching concept"

Anything that reuses previously-computed state to avoid recomputation or re-payment in an
LLM / agent workload: a KV-cache mechanism, a prefix/radix reuse scheme, a provider
prompt-cache lever, a tiering/eviction/residency policy, an expert-weight or recurrent
-state cache, an external engine's cache surface fak observes, or a proof/explainer about
any of these.

If it's a *token-efficiency* method that is **not** caching (compression, quantization,
tokenizer choice, batching), it belongs in
[`../awesome-token-efficiency.md`](../awesome-token-efficiency.md), not here.

## Adding an entry

1. **Find the self-description.** Every entry must link to the concept's own words:
   - an in-repo doc's frontmatter `title:` / `description:` (`docs/explainers/…`, `docs/serving/…`, `docs/proofs/…`, `docs/notes/…`);
   - a Go package's doc-comment (`internal/<pkg>` — link the directory);
   - for an external system, the **upstream project** *plus* fak's in-repo analysis of it.
   If no self-description exists yet, write one there first, then link it. **Do not invent
   a description here** — an entry with no real source is worse than no entry.

2. **Pick the section** by what the concept *is*, not who made it:
   - owns the cache object → **Kernel-owned cache primitives**
   - steers a provider's prompt cache → **Managed cache & session economics**
   - a cache built over providers we don't control → **vCache**
   - a rung of the ranked KV ladder → **KV-cache concept ladder**
   - state produced by a model family → **Model-family caching**
   - tiering / transport / disaggregation → **Multi-tier serving & KV transport**
   - a plain-English writeup → **Explainers**
   - a runnable correctness proof → **Proofs**
   - an external engine/system → **External engines & systems**
   - dated triage/study → **Research frontier notes**

3. **Write one line**, this shape:

   ```md
   - **[Name](https://example.com/concept)** — one-sentence self-description, in the concept's own words. TAG
   ```

4. **Tag fak status honestly** (the load-bearing part):

   | Tag | Meaning | Bar |
   |---|---|---|
   | ✅ | shipped | in code, witnessed by a test/proof |
   | 🟡 | partial / seam present | some code, not on the live path |
   | 🔭 | plan or design doc only | a doc exists, no shipped code |
   | ⭐ | a lead | fak does something no shipped engine does |
   | 👁 | observe-only | fak watches an external engine's cache, does not control it |

   Prefer under-claiming. A 🔭 that turns out to be ✅ is a happy surprise; a ✅ that is
   really 🔭 is the exact drift this repo's `dos`/witness discipline exists to kill.

## Ordering

Within a section, order by fak's depth of ownership (owned → observed) or, for
model-family / ladder sections, by the concept's natural order (M1 → M12; dense → hybrid).
No strict alphabetization — readability wins.

## Keeping it honest

- If a linked doc is deleted or renamed, fix or drop the entry — a dead link is a broken
  index.
- If an external engine's fak verdict changes (an observation adapter lands, `unknown` →
  `passive observe`), update the entry *and* the
  [capability inventory](../cache-frontier/external-engine-cache-capability-inventory.md)
  in lockstep; they must not drift.
- This list makes **no** claim an entry's linked source doesn't already back. That is what
  lets an answer engine cite it (see [`outreach.md`](outreach.md)).

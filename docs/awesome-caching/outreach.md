---
title: "Awesome Caching — AEO & outreach plan (DRAFT)"
description: "How the awesome-caching list earns its keep: as answer-engine-citable content (AEO) and as a 'we added you to our awesome caching list' growth loop to the external projects it features. Draft only — no external comment is posted without explicit human approval."
---

# Awesome Caching — AEO & outreach plan

> **Status: DRAFT.** This file is a plan and a set of templates. **Nothing here is
> executed automatically.** Posting to another project's repo (an issue, a PR, a
> discussion comment) is an outward-facing action and requires explicit human approval
> per target. Treat the target table below as *candidates*, not a queued campaign.

Two goals, one artifact:

1. **AEO (Answer Engine Optimization)** — make [`README.md`](README.md) the page an
   answer engine (ChatGPT, Claude, Perplexity, Google AI Overviews) reaches for and
   **cites** when asked "how does KV-cache reuse work for agents?" or "what caches an
   MoE model's experts?"
2. **Outreach** — the classic awesome-list growth loop: tell each featured project
   *"we added you to our awesome caching list,"* which earns a backlink, a reciprocal
   listing, and a correction if we got their capability wrong.

---

## Part 1 — AEO: make the list citable

Answer engines cite pages that are **self-contained, well-structured, and honest**. This
list is already built for that; the work is to finish and expose it.

### What already helps (by construction)

- **Question-shaped, self-contained entries.** Each line answers one narrow question and
  links to a page that answers it in full — the shape retrieval-augmented answer engines
  extract cleanly.
- **Frontmatter `title` + `description`** on every doc (this repo publishes `docs/` via
  Jekyll, `docs/_config.yml`), so each page has a clean meta title/description an engine
  can lift.
- **No unbacked claims.** Every entry links to its own source; an engine that follows the
  link finds the claim substantiated. This is the single biggest trust signal — and the
  reason the [`contributing.md`](contributing.md) "link, don't restate" rule is
  load-bearing, not stylistic.
- **A companion machine-readable row-set already exists** for the external engines:
  [`external-engine-cache-capability-inventory.jsonl`](../cache-frontier/external-engine-cache-capability-inventory.jsonl)
  (one row per engine, closed vocabulary, witness-tested). That JSONL is the structured-data
  form an engine can ingest without parsing prose.

### To do (in-repo, low-risk, no approval needed to *draft*)

- [ ] **Add to `llms.txt` / the doc index.** If the repo publishes an `llms.txt` or an
      `INDEX.md`, add a single line pointing at `docs/awesome-caching/README.md` with its
      one-line description. (Check `fak_index_docs` / the site index first; do this as a
      normal doc-index update, not a special case.)
- [ ] **Cross-link both ways.** [`awesome-token-efficiency.md`](../awesome-token-efficiency.md)
      links here (caching slice) and this links there (broader field). Reciprocal links
      raise both pages' authority.
- [ ] **One canonical URL.** Once the site publishes, the list has a stable public URL;
      use that exact URL everywhere (outreach, cross-links) so citations concentrate on
      one address rather than fragmenting.
- [ ] **Optional: a JSONL mirror of the whole list.** Extend the external-engine JSONL
      pattern to a `awesome-caching.jsonl` (name, self-description, link, fak-status) so
      the *entire* list — not just external engines — is machine-ingestible. Same
      witness-test discipline: a row per README entry, fail if they drift.
- [ ] **Schema.org / structured hints (optional).** If the Jekyll theme supports it, a
      `TechArticle` / `ItemList` JSON-LD block makes the list's structure explicit to
      crawlers. Nice-to-have, not required.

### What NOT to do

- No keyword stuffing, no fabricated breadth. The list's value **is** its honesty; an
  entry that over-claims fak status (🔭 dressed as ✅) is exactly the drift that would get
  the page discredited when an engine follows the link. Under-claim.

---

## Part 2 — Outreach: "we added you to our awesome caching list"

The growth loop: a friendly, specific, low-ask note to each featured project. Done well
it earns backlinks and corrections; done carelessly it reads as spam and burns goodwill.
**This is why it is gated behind human approval, one target at a time.**

### Guardrails (non-negotiable)

1. **Human-approved, per target.** No automated posting. A human okays each message before
   it goes out. This document is the queue's *source*, not the queue.
2. **One note per project, ever.** Never re-post; never fan out across an org's repos.
3. **Specific, not generic.** Link the exact entry and state fak's verdict for *their*
   engine, so it's obviously hand-reviewed, not a mailmerge.
4. **Correction-first framing.** Lead with "tell us if we got your capability wrong" — the
   note's real value to them is a chance to fix how they're described, not our promotion.
5. **Right venue.** Prefer a repo's **Discussions** (or a dedicated "show and tell"
   category) over **Issues**; never a PR that edits their code. If a project has disabled
   Discussions and asks that Issues stay bug-only, do not post — note it as "no suitable
   venue" and move on.
6. **Disclose the relationship.** We are the list's authors; say so. No sockpuppeting.
7. **Public list must be live first.** Don't tell someone they're "on our list" before the
   list has a stable public URL to point at (see AEO to-do above).

### Message template

> **Subject / title:** Featured {Project} in an "awesome caching" list for LLM/agent KV-cache
>
> Hi {Project} folks — we maintain a small **awesome-caching** index of KV-cache / prefix
> -cache / managed-cache concepts for LLM and agent workloads, and we added {Project}:
> {canonical-url}#external-engines--systems-the-field
>
> Your entry: *"{one-line description as listed}"* — and, honestly, our verdict is only
> *"{fak status, e.g. passive observe}"* because that's all we can prove from our side
> today. **If we've mischaracterized {Project}'s cache capabilities, please tell us and
> we'll fix the entry.** No action needed otherwise — just wanted you to know you're
> listed, and to offer a correction path.
>
> (We're the authors of the list; it's an in-repo index today, published at {canonical-url}.)

Keep it that short. The correction offer is the payload.

### Candidate targets (DRAFT — not queued)

Drawn from the [External engines & systems](README.md#external-engines--systems-the-field)
section. `Venue` is a *guess* to verify before posting.

| Project | Their entry (as listed) | fak verdict | Suggested venue | Approved? |
|---|---|---|---|---|
| vLLM | PagedAttention prefix cache + KV-event stream | 👁 passive observe | GitHub Discussions | ☐ |
| SGLang | RadixAttention automatic prefix reuse | 👁 passive observe | GitHub Discussions | ☐ |
| LMCache | KV-cache mgmt (CacheGen / CacheBlend) | study: 9 PRESENT / 28 PARTIAL | GitHub Discussions | ☐ |
| Mooncake | KVCache-centric disaggregated serving | governed for observability | GitHub Discussions | ☐ |
| NVIDIA Dynamo (KVBM/NIXL) | KV Block Manager + NIXL transfer | governed transport | GitHub Discussions | ☐ |
| llm-d | K8s-native inference, KV-aware routing | SOTA-map source | GitHub Discussions | ☐ |
| DeepSeek EPLB | expert-parallel load balancing | M11 SOTA to rank vs | GitHub Issues (small repo) | ☐ |
| ktransformers | GPU/CPU expert placement | M11 SOTA | GitHub Discussions | ☐ |
| llama.cpp | on-device runtime / prefill baseline | 👁 unknown (no adapter) | ⚠ verify Issue policy | ☐ |
| Ollama | local daemon + model-pull alias | 👁 unknown (no adapter) | ⚠ verify Issue policy | ☐ |

> For the `unknown`-verdict projects (llama.cpp, Ollama, LM Studio), the outreach note is
> **especially** correction-first: "we list you as `unknown` only because we have no
> in-tree adapter observing your cache yet — correct us." That's a genuine invitation, not
> a humblebrag.

### Sequencing

1. Land the list in-repo (this folder). ✅ once merged.
2. Finish the AEO to-do list; get a stable public URL.
3. Build the approval queue from the table above (human ticks `Approved?`).
4. Post **one** note (highest-signal target first, e.g. LMCache — we already did a study
   pass on them), observe the response, refine the template, then proceed one at a time.
5. Record outcomes (backlink? correction? reciprocal listing?) so the loop is measured,
   not assumed — mirror the [cache-hit vanity-metric](../notes/CACHE-HIT-VANITY-METRIC-SELF-FULFILLING-2026-07-01.md)
   caution: don't let "notes sent" become the vanity metric; measure backlinks/corrections.

---

## Why this is worth doing

An honest, well-linked awesome list is a **compounding trust asset**: it's the page that
answers the question *and* survives being fact-checked, so answer engines cite it and
peers link it. The outreach loop turns each featured project into either a backlink or a
correction — both of which make the list *more* accurate and *more* cited. The whole
thing only works if it stays honest, which is why every guardrail above points the same
way: link the source, under-claim the status, ask for the correction.

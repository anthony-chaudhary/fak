---
title: "field-borrow, worked once: the read/query axis of the durable-session field scan → three witnessed self-query-quality gaps (2026-07-08)"
description: "A worked instance of the field-borrow pass. The durable-session/shared-memory research note mapped the WRITE/durability axis (P1-P6); this converts its READ/query axis into grounded backlog by dogfooding fak's own self-query surface (fak_feature_query over internal/selfquery) against three field-standard retrieval capabilities — GraphRAG edges (Cognee/Zep), retrieval-accuracy eval (LongMemEval/LOCOMO), and freshness/supersession (Zep bi-temporal + Mem0). Each gap was witnessed present/partial/absent, grounded at selfquery.go:566 (the flat lexical ranker), and filed as #3161/#3162/#3163 under epic #1494. The reusable method is the /field-borrow skill."
---

# field-borrow, worked once: from the durable-session field scan to three witnessed query-quality gaps

> Date: 2026-07-08. Status: instance + method note — the three issues are filed; the reusable
> pass is the [`/field-borrow`](../../.claude/skills/field-borrow/SKILL.md) skill. Companions:
> [RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY (source scan)](RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md),
> [CONCEPT-MEMORY-IN-THE-LOOP](CONCEPT-MEMORY-IN-THE-LOOP-2026-07-02.md),
> [CONCEPT-MEMORY-VALUE-UNBOUNDED-SCORE](CONCEPT-MEMORY-VALUE-UNBOUNDED-SCORE-2026-07-03.md).
> Parent epic: [#1494](https://github.com/anthony-chaudhary/fak/issues/1494) (self-use of the
> query + memory surface).

## 0. Why this note exists

The [durable-session field scan](RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md)
mapped one axis of applying external memory systems to fak: the **write / durability** axis —
its ladder P1-P6 (turn-boundary journal, residual meter, restart-parity probe, witnessed
promotion gate, arbitrated memory lane, healed store). It did **not** cover the complementary
axis: once state is *in* the store, **how good is the query that reads it back?**

This note is the worked output of running the [`/field-borrow`](../../.claude/skills/field-borrow/SKILL.md)
pass over that read/query axis. The point is not the three issues alone — it is that each was
**witnessed against fak's own self-query surface before filing**, so none assumes a gap fak
already closed, and each is grounded at the exact seam.

## 1. The seam under test

fak's self-query surface is `internal/selfquery` (`fak_feature_query`, `fak_capabilities`).
Its ranker is `rankCards` → `score()` at `internal/selfquery/selfquery.go:540,566`: pure
`strings.Contains` lexical scoring over each `FeatureCard`'s (`:37`) name/tags/ref/summary.
A card carries a content-address `Witness` (`:46`) and **nothing else** — no edges to other
cards, no freshness verdict, and no measurement of whether the ranking is any good. Those
three absences are exactly the three field-standard retrieval capabilities below.

## 2. The witness table (borrow → dogfood → verdict → seam → issue)

| Borrowed capability | Named source (dated) | Dogfood query (`fak_feature_query`) | Verdict | Seam | Filed |
|---|---|---|---|---|---|
| **Edge-aware / GraphRAG retrieval** — traverse relationship edges to answer "why X over Y" | Cognee (GraphRAG), Zep/Graphiti (temporal knowledge graph), 2026 | *"graph edges relationships between notes trace why decision was made across documents"* → returned 6 **unlinked** decision-note cards | **ABSENT** (edges thrown away) | `selfquery.go:37` (`FeatureCard` has no `Related` field) | [#3161](https://github.com/anthony-chaudhary/fak/issues/3161) |
| **Retrieval-accuracy eval** — recall@K / MRR over held-out intents | LongMemEval (500-q long-term-memory bench), LOCOMO, 2024-2026 | *"retrieval quality eval harness recall@k MRR benchmark for the feature query ranking"* → returned `fak_feature_query` itself + **agent** benchmarks; **no self-retrieval eval** | **ABSENT** (ranker ungraded) | `selfquery.go:540,566` (`rankCards`/`score`) | [#3162](https://github.com/anthony-chaudhary/fak/issues/3162) |
| **Freshness / supersession** — bi-temporal validity; newer fact invalidates older | Zep/Graphiti (bi-temporal edges), Mem0 (ADD/UPDATE/DELETE conflict-resolution), 2026 | *"freshness staleness supersession temporal validity of a query result card superseded by newer note"* → returned `fak_index_freshness` (**wrong axis**: index↔tree drift) + the DEDUP-EARLIER note | **PARTIAL** (drift ≠ supersession) | `selfquery.go:46` (`Witness`, no `Freshness` rung); reuses `dos_recall` | [#3163](https://github.com/anthony-chaudhary/fak/issues/3163) |

All three filed under epic **#1494** as its *quality* axis (C1-C4 are the *access* axis —
wiring the tools onto the hot path; these improve what the tools return). A registration
comment lists them on #1494. Shared design fences: deterministic (no LLM on the query path),
advisory (never silently drop a card), read-only; #3161 and #3163 share one edge/front-matter
extractor.

## 3. The method, generalized

The pass that produced the table is now reusable as [`/field-borrow`](../../.claude/skills/field-borrow/SKILL.md):
**name a capability from a named source → dogfood `fak_feature_query` to witness
present/partial/absent → ground the gap at a file:line seam → file epic-anchored with the
source + witness + seam + first checkable step → register the trail here + in `INDEX.md`.**
It is the human-curated, witness-first counterpart to the automated `idea-scout`
(`tools/idea_scout.py`), which dedups a candidate only against already-filed issues (a
seen-cache, an issue-body stamp, and a title-Jaccard index — `is_duplicate`, `idea_scout.py:336`)
and never asks fak what it already has in the codebase; and the product/agent-capability
counterpart of `sota-check` (kernels) and `industry-score` (the competitive map).

## 4. Honest fences

- The dogfood verdicts rest on a **lexical** ranker (`selfquery.go:566`); a false-ABSENT is
  possible from a poorly-phrased query. Each verdict above was cross-checked by varying the
  phrasing and reading the top cards — but the ranker being ungraded is itself #3162, so treat
  these as witnessed-at-2026-07-08, re-checkable, not eternal.
- Nothing here ships code. Each issue names its first checkable step; the seams are located,
  not yet cut.
- The borrow claims name real systems; they are summaries of those systems' documented
  behavior, not quotes — verify against each system's own docs before relying on a detail.

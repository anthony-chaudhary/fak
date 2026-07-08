---
name: field-borrow
description: One repeatable pass that turns an outward field idea into grounded backlog WITHOUT guessing whether fak already has it — the dogfood-witnessed capability-import loop. Before filing "fak should add X" from a named external system (Letta/MemGPT, Mem0, Zep/Graphiti, Cognee, vLLM/SGLang, a dated paper) or a row from the RESEARCH-*/industry field scans, it DOGFOODS fak's own self-query surface (`fak_feature_query` / `fak capabilities` / `fak index docs|leaves|verbs|claims`) to witness PRESENT / PARTIAL / ABSENT, grounds each real gap in a file:line seam, and files epic-anchored issues carrying the named source + the dogfood witness + the seam + a first checkable step. The human-curated, witness-first counterpart to the automated `idea-scout` (which dedups a candidate only against already-filed issues — a seen-cache, issue-body stamp, and title-Jaccard index — and never asks fak what it already has in the codebase); the product/agent-capability counterpart of `sota-check` (kernels) and `industry-score` (the competitive map). Use when reading a field/SOTA/research note and wondering "do we already have this?", before filing a capability-gap issue, after a competitor or paper ships a capability, or on a /loop cadence over the docs/notes RESEARCH-* corpus to convert inert research into witnessed, grounded backlog.
---

# field-borrow — borrow from the field, but witness the gap against yourself first

## Why this skill exists

fak researches the outward field well — the `docs/notes/RESEARCH-*` corpus, `industry-score`,
`idea-scout`, `sota-check` all point *outward*. But applying a field idea inward has a silent
failure mode: an agent reads "Cognee does GraphRAG" or "Zep has bi-temporal memory" and files
"fak should add graph memory" **without first asking fak whether it already has it** — so the
tracker accretes gaps fak already closed, or gaps described one abstraction away from the real
seam. The research was outward-complete but the *witness step* was manual and skipped.

The law, in one line: **the default answer to "does fak already have this field-standard
capability?" is _ask fak, don't assume_ — dogfood the self-query surface to witness
present / partial / absent, then file only the witnessed gaps, each grounded in a seam.**

This is the loop that produced #3161 / #3162 / #3163 (edge-aware query, retrieval-eval,
freshness rung) under epic #1494: three retrieval-*quality* gaps borrowed from Cognee/Zep,
LongMemEval/LOCOMO, and Zep/Mem0 — each witnessed by querying `fak_feature_query` against
fak's own `internal/selfquery` before filing, each grounded at `internal/selfquery/selfquery.go`
(the flat lexical ranker at `:566`). See
[`docs/notes/CONCEPT-FIELD-BORROW-QUERY-QUALITY-2026-07-08.md`](../../../docs/notes/CONCEPT-FIELD-BORROW-QUERY-QUALITY-2026-07-08.md)
for that worked instance.

### How this differs from its siblings (do not duplicate them)

| Skill / tool | Direction | Mechanism | Files issues? | Dogfoods the self-query witness? |
|---|---|---|---|---|
| `sota-check` | inward, **kernels only** | `fak sota <op>` prior-art matrix | no | no |
| `industry-score` | outward | scorecard: coverage + parity-debt over a taxonomy | no | no |
| `idea-scout` (`tools/idea_scout.py`) | outward, **automated** | arXiv/GitHub scan, dedup vs **existing issues** (seen-cache · body-stamp · title-Jaccard) | yes (triage queue) | **no** (issue-vs-issue only, never capability-vs-codebase) |
| **field-borrow** (this) | outward→inward | **dogfood `fak_feature_query` to witness the gap**, then file epic-anchored | yes (grounded, under the right epic) | **yes — this is the point** |

field-borrow composes with all three: it can be the human-in-the-loop pass over an
`idea-scout` hit, it feeds witnessed coverage rows to `industry-score`, and once a spine
ships it hands off to `spine-fanout`.

## The pass

### 1 — Name the source and the capability axis (dated)

Pick ONE capability from a **named** external system or a dated source: an agent-memory
product (Letta/MemGPT, Mem0, Zep/Graphiti, Cognee, Supermemory), a serving system
(vLLM/SGLang/TensorRT-LLM), a competitor feature, a benchmark (LongMemEval, LOCOMO), or a
row from the `RESEARCH-*` / industry field scans. Write it down with the source, so the
borrow is falsifiable: *"Zep/Graphiti — bi-temporal edges: facts carry validity windows;
a newer fact invalidates the old one."* One capability per pass; do not batch.

### 2 — Dogfood the self-query WITNESS (the heart of this skill)

Ask fak, three ways, whether it already has the capability — do **not** reason about it
from memory:

- **`fak_feature_query`** (MCP) / `fak capabilities <intent>` — intent → ranked capability
  cards (name, summary, `witness` content-address, the call to make).
- **`fak index docs|leaves|verbs|claims <terms>`** — the dev self-index over docs, packages,
  CLI verbs, and the CLAIMS ledger.
- Also a plain `Grep` for the concept's obvious symbol, to catch a capability the lexical
  ranker missed (see Honest limits).

Classify the result, and **paste the top cards as the witness**:

- **PRESENT** — a card / verb / leaf directly does it. *Read it and confirm* it truly does
  (not just a name collision). PRESENT is a valid, complete, **good** outcome — "fak already
  has this, here is the card" ends the pass; do not file.
- **PARTIAL** — an adjacent surface exists but not this capability (e.g. `fak_index_freshness`
  covers index↔tree drift but not semantic supersession).
- **ABSENT** — the query surfaces nothing on-point. An ABSENT claim **must** show the query
  that returned nothing on-point — that null result is the falsifiable witness.

### 3 — Ground the PARTIAL/ABSENT in a seam (file:line)

The capability cards point at the code. Find the **exact file:line** the capability would
touch and quote **one** dogfood result as the in-issue evidence — e.g. the flat lexical
ranker `score()` at `internal/selfquery/selfquery.go:566`, or the `FeatureCard` struct at
`:37` that carries no freshness/edge field. **No file:line seam, no issue** — a gap you
cannot locate is not ready to file; keep witnessing until you can point at the code.

### 4 — Dedup, then file grounded (under the right epic)

Search existing issues **and** the natural parent epic before filing (avoid re-filing what
`idea-scout` or a human already opened). File under the epic with, in order:

1. the pain, in fak's terms, quoting the seam from step 3;
2. the **named external inspiration** (dated — a fabricated "system X does Y" is the exact
   citation failure `dos_citation_resolve` exists to catch);
3. the **dogfood witness** from step 2 (the query + what it returned);
4. a **concrete first checkable step** (the smallest thing that proves the gap is real and
   the fix is buildable);
5. **honest fences** (deterministic / advisory / read-only where they apply).

Set milestone + labels at creation. On this host, run `gh` via PowerShell or the Bash tool
deliberately (quote bodies through `--body-file`, not inline heredocs that the shell mangles).

### 5 — Register the trail (or it lives only in a transcript)

Record the instance in a dated `docs/notes/CONCEPT-<topic>-<date>.md`: source → witness →
seam → filed issues, with a `Companions:` block. Add its **`INDEX.md`** line in the
*Notes & research* section (an unindexed dated note is an orphan `fak index freshness`
flags). Link the children from the parent epic. **This registration — not the issues alone —
is what makes the research path durable.**

### 6 — (optional) Feed the map

A witnessed PRESENT/PARTIAL/ABSENT is a coverage row `industry-score` can absorb; a genuinely
novel, spine-shaped gap can seed `spine-fanout` once a working spine ships.

## What "done" proves

- Every filed gap carries a **dogfood witness** (the query + its top cards), a **file:line
  seam**, a **named dated external source**, and a **first checkable step**.
- No gap was filed that fak already ships (step 2 caught the PRESENT cases).
- The instance is registered: a dated `docs/notes` entry, its `INDEX.md` line, and the epic
  links its children. `python tools/... ` freshness / `fak index freshness` reads clean for
  the new note.

## The anti-gaming rule (specific to this surface)

**Never file a capability gap you did not dogfood-witness, and never mark ABSENT without the
null query.** The witness is the falsifiable record — the way `sota-check`'s `Prior-art:`
trailer is. Two failure modes this rule kills:

- **Assumed-absent.** Filing "fak lacks X" from memory when `fak_feature_query` would have
  returned the card. The fix is the query, pasted.
- **Fabricated borrow.** "Zep does bi-temporal invalidation" when it does not, or citing a
  system/paper you did not read. Name the source, dated; if you could find no prior art, say
  *that* so the next person can disprove it.

PRESENT is not a failed pass — a witnessed "we already have this" is a real result.

## Where this misleads / honest limits

- **The self-query score is lexical.** `rankCards`/`score()`
  (`internal/selfquery/selfquery.go:540,566`) is `strings.Contains` substring matching over
  name/tags/ref/summary — a poorly-phrased query can miss a card fak *does* have, producing a
  **false-ABSENT**. Read the top cards, vary the phrasing, and cross-check with a raw `Grep`
  before trusting an ABSENT. (This limit is itself issue #3162 — the ranker is ungraded.)
- **ABSENT means "fak's own index doesn't surface it", not "it doesn't exist."** The index can
  be stale or the capability can live under a name you didn't guess.
- **The witness is a snapshot.** A card added tomorrow changes the verdict (the freshness
  problem is issue #3163). Re-witness before acting on an old pass.
- **Composes with, does not replace:** `idea-scout` (automated outward feeder), `industry-score`
  (the competitive coverage map), `spine-fanout` (post-spine fan-out), `sota-check` (kernel
  prior-art). If the capability is a **compute kernel**, use `sota-check` instead — this pass
  is for product / agent / memory / serving capabilities.

## When to run this

- Reading a `RESEARCH-*` / SOTA / competitor note and wondering *"do we already have this?"*
- Before filing any "fak should add X" capability-gap issue.
- After a competitor or a paper ships a capability worth borrowing.
- On a `/loop` cadence over the `docs/notes/RESEARCH-*` corpus, to convert inert research
  into witnessed, grounded, epic-anchored backlog — the inward, human-curated complement to
  the daily `idea-scout`.

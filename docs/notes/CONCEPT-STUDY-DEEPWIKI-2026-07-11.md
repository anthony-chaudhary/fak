# Concept study — DeepWiki (auto repo-wiki generation) → a fak-native, witness-verified wiki track

- **Date:** 2026-07-11
- **Sources read (pinned):**
  - `AsyncFuncAI/deepwiki-open` @ `16f35a0fc0284e99b7963bbf4e8585e9957e2fe1` (MIT — borrows are **inspire**), the open reimplementation of Cognition's hosted DeepWiki. Load-bearing files read: `src/app/[owner]/[repo]/page.tsx` (the two generation prompts), `api/data_pipeline.py` (repo ingest/chunk), `api/rag.py` (FAISS retrieval), `api/prompts.py`.
  - SOTA scan: DeepWiki (Cognition, launched 2025-04-25; MCP server 2025-05-22 — 3 read tools: read-structure / read-page / ask); **CodeWiki** (FSoft-AI4Code, arXiv:2510.24428 — 3-phase hierarchical decomposition + recursive agentic gen + `CodeWikiBench` eval over 21 repos); `gold24park/open-deepwiki`.
- **Method:** read the two generation prompts + the RAG ingest at the pinned SHA; dogfood fak's self-index (`fak_feature_query`, `fak_index_verbs|docs|claims`) + raw `Grep` of the fak seam; ablate each borrow to its axis; witness PRESENT/PARTIAL/ABSENT against fak before filing.

## How DeepWiki actually works (grounded)

1. **Two-phase generation.** (a) *Structure*: one LLM call emits a `<wiki_structure>` XML tree — sections → pages, each page carrying `<relevant_files>`; 8–12 pages for the comprehensive view, over a fixed section taxonomy (Overview / System Architecture / Core Features / Data Flow / …). `src/app/[owner]/[repo]/page.tsx:746-795,829`. (b) *Per-page content*: an LLM writes each page grounded **only** in its `RELEVANT_SOURCE_FILES`, forced to open with a `<details>` source-file list (≥5 files), cite every claim, and end sections with `Sources: [file.ext:start-end]()`. `page.tsx:420-526`.
2. **RAG grounding.** Repo is cloned, filtered (`DEFAULT_EXCLUDED_DIRS/FILES`), token-chunked (`TextSplitter`/`ToEmbeddings`, `api/data_pipeline.py`), embedded into FAISS; per-page `relevant_files` are retrieved by embedding similarity (`api/rag.py:41` `FAISSRetriever`).
3. **Mermaid diagrams**, extensive, forced vertical orientation, 8 arrow types. `page.tsx:450-484`.
4. **Snapshot cache.** `wiki_structure` + `generated_pages` cached and regenerated on a schedule. `page.tsx:1709,1905`.

## The two universal weaknesses (every tool in the field shares them) = fak's opening

- **Citations are LLM-produced and unverified.** The prompt *asks* for `file:line` cites (`page.tsx:497-502`) but nothing resolves them against the tree. Every 2025–26 source says the same: citations "reduce but do not eliminate" hallucination; "human verification of load-bearing claims remains necessary." fak's entire thesis is machine-verifying citations (`dos_citation_resolve`, `dos_commit_audit`, `dos_recall`).
- **Snapshot staleness.** Regenerated on a schedule, "may lag main by hours to days." fak has `internal/devindex/freshness.go` (resolves local doc links against disk *now*) and `dos_recall` (re-verifies a claim against git *now*).

fak also already holds, as **deterministic ground truth**, the structure DeepWiki spends an LLM+FAISS pass to *re-infer*: `internal/devindex` (leaves, lanes, verbs, docs, claims, refs). So a fak wiki seeds its structure from the index, and verifies every citation — the two things the field cannot do.

## Candidate table

| Borrow | Source `path:line@16f35a0` | Axis | fak seam | Witness | Verdict | Filed |
|---|---|---|---|---|---|---|
| Deterministic wiki **structure from the self-index** (not LLM-inferred) | `page.tsx:746-795,829` | section/page tree derivation | `internal/devindex/*` (leaves/lanes/docs/claims) | index is PRESENT; wiki emitter ABSENT | inspire | L1 |
| Per-page content **grounded only in relevant_files**, cite-every-claim | `page.tsx:420-526` | source-grounded prose gen | none (no doc generator) | ABSENT | inspire | L2 |
| **Witnessed** `Sources:[path:line]` — resolve every code cite vs the tree | `page.tsx:497-502` | code-line citation *integrity* | `devindex/freshness.go` resolves *doc* links only; `dos_citation_resolve` is external-legal only; #4079 is doc source-ids | ABSENT on the code-line axis | inspire | L3 |
| **Freshness-gated pages** — pin SHA + cited-file set, flag stale when cited code moves | `page.tsx:1709,1905` (snapshot cache = the anti-pattern) | page↔code freshness | `devindex/freshness.go:74` `CheckFreshness` | PARTIAL (extend freshness) | inspire | L4 |
| **Mermaid** architecture diagrams from the *real* dep graph (not LLM-inferred edges) | `page.tsx:450-484` | diagram fidelity | `devindex/refs.go`, `internal/blastradius` have the graph; render is the gap | PARTIAL | inspire | L5 |
| **Claims-aware honesty** — mark stub/simulated vs shipped on the page | (fak-native; DeepWiki has no claim ledger) | doc honesty | `CLAIMS.md` + `fak index claims` | PARTIAL | inspire | L6 |
| **Quality/coverage score gate** (CodeWikiBench-style) — %citations-resolve, %leaves-covered | arXiv:2510.24428 (CodeWiki) | measurable wiki quality | none | ABSENT | inspire | L7 |

## Ablation vs the existing KernelWiki index-integrity track (do-not-dupe)

The KernelWiki track (`#3946 → #3948 → #4077/#4079/#4078/#4080/#4081/#4082`) hardens the *internal fact-index*: frontmatter-derived pivot regeneration and internal `[[id]]`/source-id citation resolution. This DeepWiki track generates the *human/LLM-facing narrative wiki*. They are siblings: **#4079** resolves *doc source-ids*; **L3** resolves `path:line` *code* citations in *generated* pages — an axis #4079 explicitly excludes. **#4077** regenerates *pivot indices*; **L4** freshness-gates *narrative pages* on cited-code drift. Both new leaves cross-link their KernelWiki sibling.

## Companions

- Feeds/uses [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) witness discipline; produced by [`study-repo`](../../.claude/skills/study-repo/SKILL.md).
- Builds on epic **#1287** (queryable self-index); sits beside epic **#3948** (KernelWiki index integrity).
- Epic + leaves filed: see the epic linked from this note's `INDEX.md` entry.

---
title: "Source study: querying across the local tree and sibling repos"
description: "A study-repo pass over six mature code-query engines identifying five clean-room borrows to let fak query across the local tree and sibling repos."
---

# Source study: querying across the local tree and sibling repos (2026-07-09)

A `study-repo` pass over six mature code-query engines, asking one question: what would
let fak query **across the local tree AND sibling local repos** — not just rank its own
feature cards over a single root? Every borrow below is grounded at a real
`path:line@sha`, license-checked, witnessed against fak (`fak_feature_query` + raw grep),
and filed. Filed track: **epic #3434** with child leaves **#3435–#3439**.

## The fak baseline (what is already PRESENT)

fak's self-query is **single-root and lexical**, but it already ships several of the hard
primitives the borrows need:

- **Single-root view.** `internal/devindex` binds to the one `dos.toml` that `FindRoot`
  walks up to (`devindex.go:99,123-148`) — a VIEW over fak's own `dos.toml`/`INDEX.md`/`CLAIMS.md`.
- **Hybrid BM25 ranker.** `internal/selfquery/ranker.go:224` `rankHybrid` (BM25 + stemming
  over in-memory `FeatureCard`s) — shipped in #3235, so "better lexical ranking" is *done*,
  not a gap.
- **Pure-Go inverted index.** `internal/sessionsearch` — a TF-IDF `Index`, *deliberately*
  no SQLite/FTS5 C dependency; plus IDF in `internal/ctxplan/index.go`,
  `internal/recall/journal_index.go`. All journal/session-scoped, not over code.
- **Vector-similarity index.** `internal/simhash` — `Embed`/`Cosine`/`Index.TopK`,
  model-agnostic (swap in real `[]float32` through the same interface).
- **Structural token engine.** `internal/clonescan` + `go_tokens` — token-window clone
  detection (no query surface).

So the gap is not "an index" or "a ranker" — it is **cross-repo scope** and **retrieval
modalities beyond lexical**: structural shape-query, trigram/regex-at-scale, and code edges.

## The six repos (all permissive; fak is Apache-2.0, compatible)

| Repo | SHA | License | What it is |
|---|---|---|---|
| AmrDeveloper/GQL | `3a76cfe` | MIT | SQL-like query over N local git repos, evaluate-per-repo-then-merge |
| flupkede/codesearch | `19a36f3` | Apache-2.0 | Multi-repo semantic MCP: tree-sitter chunks + BM25 + vectors, RRF-fused |
| sourcegraph/zoekt | `33f1f18` | Apache-2.0 | Trigram-indexed regex/substring search over posting lists |
| ast-grep/ast-grep | `fc26e86` | MIT | Metavariable AST structural search/rewrite via tree-sitter |
| DeusData/codebase-memory-mcp | `20b1153` | MIT (C) | Code knowledge-graph in SQLite; Cypher-like traversal via recursive CTE |
| mergestat/mergestat-lite | `5858dca` | MIT | SQL over local git via SQLite virtual tables (Go, cgo) |

## The five filed borrows (all INSPIRE — clean-room Go, no vendoring)

- **#3435 · L1 — multi-root fan-out + provenance** (GQL). The fan-out lives *below* the
  engine: `GitQLDataProvider.provide` loops every repo and appends rows, tagging each with
  a `repo` column (`src/gitql/gitql_data_provider.rs:30-40,194-197`); the engine depends
  only on `trait DataProvider` (`crates/gitql-engine/src/data_provider.rs:6-8`). → give
  `FeatureCard` a `Root`, load-and-merge N roots below the ranker. **ABSENT** (anchor).
- **#3436 · L2 — RRF fusion over BM25 + simhash** (codesearch). `rrf_fusion` sums
  `1/(k+rank+1)` per list, no score normalization (`src/rerank/mod.rs:48-105`); k is
  query-adaptive (`src/search/mod.rs:390-406`). fak has *both arms already* (ranker.go +
  simhash) and only lacks the ~20-line merge. **ABSENT (plumbing) — highest ROI.**
- **#3437 · L3 — trigram/regex index over tree + sibling repos** (zoekt). Selective-ngram
  pre-filter probes the two rarest trigrams, short-circuits on freq 0
  (`index/indexdata.go:337-383,416-499`); regex→trigram decomposition
  (`index/eval.go:610-693`). **PARTIAL** — extends sessionsearch off the journal onto code.
- **#3438 · L4 — AST shape-query with metavariables over `go/ast`** (ast-grep).
  Pattern→`PatternNode`→recursive walk binding a `MetaVarEnv` (`crates/core/src/matcher/pattern.rs:223`,
  `match_tree/match_node.rs:8`); composable rule algebra (`ops.rs`, `relational_rule.rs`).
  fak is single-language Go, so `go/ast` replaces tree-sitter entirely. **ABSENT** (query surface).
- **#3439 · L5 — code knowledge-graph + recursive-BFS traversal** (codebase-memory-mcp).
  node/edge SQLite schema (`src/store/store.c:234-265`), multi-pass edge extraction
  (`pass_calls.c:218,352`), `cbm_store_bfs` recursive-CTE with inbound/outbound flip
  (`store.c:3095-3139`). Extends #3161's *note* edges to *code* edges (#1494). **ABSENT** (code edges).

## Considered and rejected

- **SQL-over-git virtual tables** (mergestat-lite). The vtab-in-Go shape is clean —
  `Connect`(schema)+`BestIndex`+cursor over an iterator, registered via `go.riyazali.net/sqlite`
  (`extensions/internal/git/git.go:27-52`, `log.go:26-341`) — but it drags a **cgo + 8 MB
  bundled `sqlite3.c` amalgamation** into a pure-Go monorepo, the exact C dependency
  `internal/sessionsearch` was written to avoid. The one transferable idea (fan a query
  across many local sources, tag each row with its origin) is already **L1** without the C
  toolchain. Not filed.

## Cross-links

L2 and L5 also lift **#1494** (self-query quality); L2 is measurable under **#3162**'s
recall@K harness; L5 continues **#3161** (edge-aware query) from note edges to code edges.

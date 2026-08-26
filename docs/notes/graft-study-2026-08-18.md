---
title: "Graft study: local structural context for agent kernels"
description: "A pinned study of Graft's repository graph, retrieval, refresh, and attribution loop, with scoped lessons for fak managed context."
---

# Graft study — local structural context as an agent-kernel input

> Studied 2026-08-18. Upstream: [NanoNets/Graft](https://github.com/NanoNets/Graft), pinned at [`d0ba1e4f0577b17583c82ab57c9eb155627f7867`](https://github.com/NanoNets/Graft/commit/d0ba1e4f0577b17583c82ab57c9eb155627f7867) (2026-08-18). License: MIT. No upstream source is copied here.

## Verdict

Graft is a young but unusually complete **local repository-context product**: Tree-sitter and optional LSP extraction produce a regenerable graph plus terse Markdown cards; lexical search, graph traversal, and optional LLM synthesis answer agent questions; per-query refresh keeps the cache aligned with the working tree; native host installers expose the surface through instructions, hooks, status, and MCP.

The best transfer to fak is not “add another graph database.” It is to connect fak's existing narrow Go call graph and managed-context seam into a **measured, policy-gated structural retrieval arm**. Graft demonstrates the product loop fak lacks: build → refresh → retrieve → attribute exact spans → expose to the agent → measure the context saved. Fak already has stronger adjudication, cache accounting, and context-lifecycle primitives; Graft has the stronger repository orientation and retrieval UX.

Do not adopt Graft's headline savings as evidence for fak, its checked-in-card default, its broad language promise, or its optional query-time LLM as a default. Those require independent same-workload measurement, provenance and staleness controls, and a supportable language envelope.

## Feynman-simple value frame

- **For:** coding agents operating through fak on large or unfamiliar repositories.
- **Problem:** exact search and whole-file reads make agents pay repeatedly to discover symbols, call paths, and the few source spans relevant to the task.
- **Today:** fak has `internal/codegraph` for syntactic Go calls and broad search/context machinery, but no first-class, continuously refreshed structural-retrieval path exposed at the kernel seam.
- **Better because:** a bounded local index can return smaller source-anchored packets before expensive model context is assembled, while fak measures the real net effect and applies the same capability floor as every other tool result.
- **Witness:** on frozen repository tasks, compare exact/whole-source and tuned retrieval arms against `fak + structural retrieval` for task correctness, attributed bytes/tokens admitted, build/refresh/query latency, storage/RSS, staleness, operator time, and total cost.

**Problem centrality:** Enabling. It improves P1 managed context and P2 net-true efficiency directly; bounded refresh budgets address P3; kernel/MCP/host exposure addresses P4. It does not outrank fak's performance-gate spine and should remain a modular input to it.

## What Graft actually ships

### End-to-end mechanism

1. **Extraction.** `src/graph/extract.ts` and language queries build symbols and relations with Tree-sitter for TypeScript/JavaScript, Python, Go, Java, and PHP. `src/graph/enrich.ts` can add type-aware LSP relations. Generic fallback extraction preserves coverage for unsupported file types.
2. **Local representation.** `src/graph/write.ts` emits a graph under `graft/.graph/wiring.json`, an index, and one Markdown card per source file. Cards contain symbol names, kinds, line spans, signatures, and prose; edges remain in the graph rather than being duplicated into cards.
3. **Incrementality.** `src/graph/fingerprint.ts`, `extract-cache.ts`, and `refresh.ts` compare cached fingerprints with the current tree, replay unchanged extraction, remove deleted nodes, and atomically rewrite graph state. The default fingerprint shortcut is size plus mtime; `GRAFT_REFRESH=hash` selects content hashing.
4. **Retrieval.** `graft grep` and `map` provide deterministic orientation. `skeleton`, `callers`, and graph traversal return bounded structural context. `src/ask/graphrank.ts` propagates lexical seed scores over typed edges; `src/ask/fuse.ts` combines retrieval channels and applies test/generated-file penalties and scope gates.
5. **Optional synthesis.** `graft ask` can use configured Anthropic, OpenAI, or OpenRouter adapters after local retrieval. Deterministic graph commands remain available without an API key.
6. **Agent adoption.** `graft init` detects supported hosts, writes bounded native instruction sections, configures MCP, and adds Claude Code status/hooks. Queries refresh by default when the tree moved; `--no-refresh` and `GRAFT_NO_REFRESH=1` are explicit escape hatches.
7. **Accounting.** `src/context/savings.ts` estimates source bytes/tokens versus context-card bytes/tokens and tracks checkpoint history. This is a useful observability shape, but it is an estimate, not proof of end-task cost or correctness.

### Concrete repository evidence

- The pinned tree has 367 commits between 2026-07-03 and 2026-08-18, four Git tags (`v0.7.1` through `v0.9.0`), no GitHub Releases, and `package.json` at unreleased `0.11.0`; npm `latest` was `0.10.1` on 2026-08-18. Package metadata still points to the former `context-graph-engine` repository, a small sign of rapid product churn.
- GitHub API observation on 2026-08-18: 3,538 stars, 308 forks, and 55 open issue/PR items. One GitHub Discussion exists and is a Discord announcement, so architectural rationale lives primarily in code, tests, issues, and PRs rather than discussions.
- `npm test` at the pinned revision completed **713 tests: 709 pass, 0 fail, 4 skipped** in 15.96 seconds on this Windows host. Coverage includes graph extraction/invariants, incremental refresh, workspace federation, ranking gates, MCP, host installers, UTF-16 spans, visualization, and savings accounting.
- A live build against fak's peer-dirty tree completed far enough to emit 10,821 file cards, 114,620 nodes, and 265,978 edges. The generated cache occupied about 513 MB across 10,825 files. That run is **not a performance benchmark**: the command exceeded the 120-second shell deadline, included peer WIP and ignored/generated trees, and Graft's process continued to completion. It is evidence that the path runs and evidence that unbounded root discovery/storage needs admission controls.
- A bounded build against the four committed files in `internal/codegraph` emitted 45 nodes, 89 edges, and four cards. Its card for `codegraph.go` located `BuildCallGraphFiles` and every graph/traversal API at exact line spans. This is a functional witness, not a comparative accuracy claim.

## Architecture and design judgment

### What is strong

- **A complete agent-facing loop, not a library fragment.** Graft couples extraction, refresh, retrieval, host setup, and tests. The useful innovation is the closed loop and default adoption path, not any individual parser.
- **Local-first and inspectable.** Graph state and Markdown cards are ordinary files. Exact spans let an agent open only the edit surface and let a human audit retrieval output.
- **Freshness on the query path.** Refresh before `ask`, `grep`, `map`, `skeleton`, and callers makes staleness visible rather than leaving index maintenance to memory. Hash mode is available when metadata trust is insufficient.
- **Hybrid retrieval.** Lexical relevance seeds graph propagation instead of expecting graph topology alone to understand natural-language intent. Scope gates and penalties reduce common multi-repo and test-file noise.
- **Host-specific installation.** The tool writes bounded managed sections and merges existing config instead of replacing it. This is the operational step many retrieval projects omit.
- **Wide tests for operational seams.** The suite tests stale worktrees, workspace federation, settings merging, MCP, UTF-16 positions, atomic graph writes, and language-specific extraction—not only happy-path parsing.

### Where fak should be more rigorous

- **Headline provenance.** README claims “up to 4× cheaper and 3× faster” and presents a 30-task SWE-bench Verified slice. The methodology is described, but raw per-task traces and a rerunnable benchmark harness are not in the pinned repository. Treat the numbers as upstream-authored claims, not independently reproduced facts.
- **Savings semantics.** Card bytes versus source bytes can show compression, but not net-true token savings, provider-cache effects, additional query/model cost, latency, or task quality. Fak's cachevalue and ctx-query work should count the full pipeline.
- **Default fingerprint trust.** Size+mtime is fast but can miss adversarial or coarse-timestamp changes. Fak should use content identity for correctness-critical admission, with metadata shortcuts only as explicitly measured optimizations.
- **Resource admission.** The fak-root probe generated roughly 513 MB and more than 10,000 files. A kernel-integrated variant needs ignore-policy visibility, max file/node/byte/time budgets, cancellation, partial-state semantics, and no writes into a peer-dirty source tree.
- **Language confidence.** “Supported” currently spans parsers with different resolution quality. Tree-sitter relations, LSP enrichment, and generic fallback must be labeled separately; a generic card is not equivalent to a type-resolved call graph.
- **Security and privacy.** Optional cloud synthesis, generated graph content, and MCP exposure create exfiltration and prompt-injection surfaces. Fak can improve this by adjudicating index build inputs, retrieval results, and any provider-bound synthesis independently.
- **Generated-card lifecycle.** Graft recommends a local regenerable cache and mutates ignore files automatically. Fak's shared checkout rules require scratch allocation, explicit ownership, and no surprise edits; persistent per-file Markdown is not a safe default here.

## Current-fak witness

The cross-check used the current working tree and GitHub issue state on 2026-08-18, not README inference.

| Graft capability | Current fak evidence | State |
|---|---|---|
| Structural call graph | `internal/codegraph/codegraph.go` parses multi-file Go packages and exposes forward/reverse BFS with shortest paths; tests cover functions, methods, and multi-file calls. | **PARTIAL** — Go-only, syntactic/name-based, and no general repository index. |
| Operator-facing structural query | No `fak` verb or MCP tool invokes `internal/codegraph`; current command references are tests/bench comparison only. Closed #3439 shipped the library, while open #6205 records incomplete alternative benchmarking. | **ABSENT** as a usable end-to-end surface. |
| Multi-language extraction/LSP enrichment | No equivalent repository-wide parser/enricher was found. | **ABSENT**. |
| Query-time incremental refresh | Fak has many cache/freshness mechanisms, but none maintain a code-symbol graph on repository queries. | **ABSENT** for this surface. |
| Local graph cards / exact-span skeleton | Fak has broad docs, capability indexes, and code search, but no generated per-source structural cards or skeleton command. | **ABSENT**. |
| Kernel/MCP exposure | `fak guard` registers capability, feature-query, memory, and tool-search MCP tools, but not structural code retrieval. | **PARTIAL** — the safe exposure seam exists. |
| Context/cost accounting | Fak has cachevalue, managed-context, and context-query benchmark work; open #6526 explicitly compares exact aggregation with whole-source and tuned retrieval arms. | **PRESENT** as measurement infrastructure, not as a Graft integration. |
| Host adoption | Fak has guard/harness integrations and skills, but no detected-host installation of a repository-structure query surface. | **PARTIAL**. |

`fak self-query` from the study skill is not a current CLI verb. The equivalent witness therefore came from source search, `fak help` behavior, live GitHub issue read-back, and the existing `internal/codegraph` tests/benchmark packet. This limitation should not be hidden by pretending a dogfood command ran.

## Transfer candidates

Disposition uses `DEFAULT`, `OPTIONAL-MODULE`, `RECIPE`, `WATCH`, and `EXCLUDE`. Priority is relative to this study, not the global backlog.

| Priority | Candidate | Disposition | Why / required witness |
|---:|---|---|---|
| 1 | Expose a minimal structural-retrieval spine over fak's existing Go graph: build a bounded index, query symbol/callers/callees/skeleton with exact spans, and expose it through one `fak` verb plus the existing MCP adjudication seam. | **OPTIONAL-MODULE** | Highest-value missing loop; must prove a real agent can use it end to end before widening languages. Compare against whole-source and tuned retrieval under #6526's full net-cost contract. |
| 2 | Add query-time freshness with content identities, atomic replacement, deletes/renames, cancellation, and explicit stale/partial statuses. | **DEFAULT** inside the module | Retrieval without trustworthy freshness is worse than exact search. Witness changed/deleted/renamed files, interrupted builds, concurrent readers, and shared-tree isolation. |
| 3 | Fuse exact/lexical seeds with typed graph expansion and bounded ranking budgets. | **OPTIONAL-MODULE** | Graft's GraphRank/fusion is more useful than topology-only traversal, but weights and scope gates are workload-sensitive. Tune only from frozen task outcomes, not upstream constants. |
| 4 | Attribute every returned packet to source path, line span, relation kind, index revision, and freshness mode. | **DEFAULT** | This aligns with fak's provenance doctrine and makes retrieval independently auditable. |
| 5 | Add index admission controls: include/exclude explanation, max files/bytes/nodes/time/storage, workspace partitioning, and scratch-only output. | **DEFAULT** | The 513 MB root probe makes this a spine requirement, not later hardening. |
| 6 | Join structural retrieval to fak's context/cachevalue ledger: source bytes avoided, packet bytes admitted, build/query overhead, provider-cache interaction, correctness, and end-task cost. | **DEFAULT** | Replaces “card compression = savings” with net-true evidence. |
| 7 | Add language adapters after the Go spine, with capability labels (`syntactic`, `resolved`, `generic`) and per-language fixtures. | **OPTIONAL-MODULE** | Bounded-superset coverage is useful, but broad nominal support must not weaken the best default or imply equal precision. |
| 8 | Host-native instructions/status/hooks that tell an agent when and how to query structure. | **RECIPE** until the query surface proves value | Installation can drive adoption, but premature instruction injection adds context and operational burden. |
| 9 | Checked-in or always-materialized Markdown cards. | **EXCLUDE** as default; **RECIPE** for audit/export | Human-readable export is useful, but thousands of generated files and ignore mutations conflict with fak's shared-tree and token-economy defaults. Prefer compact indexed state plus on-demand rendering. |
| 10 | Optional LLM answer synthesis over retrieved packets. | **WATCH** | Fak already routes models; first prove deterministic retrieval. If added, adjudicate provider-bound data and report synthesis cost separately. |
| 11 | Adopt Graft's 4×/3× or SWE-bench result as a fak claim. | **EXCLUDE** | No same-workload independent witness and no raw benchmark packet in the pinned tree. |

### Best-default frontier

The likely best fak default is **exact search plus a bounded, automatically refreshed Go structural index selected only when the query asks for code relationships**, with deterministic source-attributed output and hard resource budgets. It should not replace exact search, whole-file reads, or provider prompt caching. A router may choose it when predicted admitted context and task success beat those alternatives, and the ledger must retain counterfactual costs.

### Bounded-superset opportunities

- Type-aware Go via gopls/SCIP can sit behind the same query interface once #6205 supplies real measurements.
- Tree-sitter adapters can add TypeScript/JavaScript, Python, Java, and PHP without changing the Go default, provided each result labels resolution confidence.
- Human-readable cards and visualization can be export modules rather than core storage.
- Workspace federation can be optional for declared repo sets; never infer sibling-repository scope from a broad directory.
- Cloud synthesis can remain an explicit provider-backed recipe with policy and cost visibility.

## Shipped mechanisms versus incomplete ideas

Graft's extraction, graph persistence, refresh, deterministic CLI/MCP queries, host installation, and tests are shipped mechanisms at the pinned revision. The repository's benchmark claims, broad quality equivalence across languages, and savings interpretation are promising but incompletely reproducible from the tree. Open issues and PRs show active work around additional languages, Windows and host setup, ranking quality, monorepos, and extraction accuracy; those are roadmap signals, not current guarantees.

For fak, `internal/codegraph` is shipped as a tested library, but the operator-facing code-context loop is incomplete. #6205 remains open because five external correctness/resource arms are placeholders. #6526 is the right existing measurement contract for comparing a structural arm to whole-source and tuned retrieval. Neither issue by itself supplies the missing end-to-end product spine.

## Recommended next move

Filed [#7338](https://github.com/anthony-chaudhary/fak/issues/7338) for one issue-sized spine: **`fak code context` for Go**, backed by current `internal/codegraph`, content-addressed scratch state, automatic bounded refresh, exact-span `symbol|callers|callees|skeleton` output, and MCP registration through the existing guarded surface. The acceptance witness should drive the real command against a frozen mini-repository, mutate and delete a file, prove refresh, capture deterministic JSON/text output, and then run the same task through #6526's benchmark harness. Fan out type-aware alternatives, additional languages, ranking, visualization, and host installers only after that runnable path exists.

## Sources

All observations below were read on 2026-08-18.

### Upstream primary sources

- [Pinned repository tree](https://github.com/NanoNets/Graft/tree/d0ba1e4f0577b17583c82ab57c9eb155627f7867)
- [README at pin](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/README.md)
- [MIT license at pin](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/LICENSE)
- [Package manifest at pin](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/package.json)
- [Graph extraction](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/graph/extract.ts), [refresh](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/graph/refresh.ts), [fingerprints](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/graph/fingerprint.ts), and [extract cache](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/graph/extract-cache.ts)
- [Traversal](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/graph/traverse.ts), [GraphRank](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/ask/graphrank.ts), and [retrieval fusion](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/ask/fuse.ts)
- [Savings accounting](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/context/savings.ts), [host instructions](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/hosts/instructions.ts), and [Codex hooks](https://github.com/NanoNets/Graft/blob/d0ba1e4f0577b17583c82ab57c9eb155627f7867/src/hosts/codex-hooks.ts)
- [GitHub issues](https://github.com/NanoNets/Graft/issues?q=is%3Aissue), [pull requests](https://github.com/NanoNets/Graft/pulls?q=is%3Apr), [discussion #77](https://github.com/NanoNets/Graft/discussions/77), [tags](https://github.com/NanoNets/Graft/tags), and [npm package](https://www.npmjs.com/package/@nanonets/graft)

### Fak comparison sources

- [`internal/codegraph/codegraph.go`](../../internal/codegraph/codegraph.go) and [`codegraph_test.go`](../../internal/codegraph/codegraph_test.go)
- [`docs/notes/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md`](GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md)
- [Issue #3439](https://github.com/anthony-chaudhary/fak/issues/3439), [open issue #7338](https://github.com/anthony-chaudhary/fak/issues/7338), [open issue #6205](https://github.com/anthony-chaudhary/fak/issues/6205), and [open issue #6526](https://github.com/anthony-chaudhary/fak/issues/6526)
- `cmd/fak/guard_mcp.go` and `cmd/fak/guard.go` for the current guarded MCP registration seam.

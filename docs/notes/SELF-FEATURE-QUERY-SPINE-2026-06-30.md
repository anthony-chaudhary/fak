---
title: "Self-feature query spine: fak asks fak what it can do"
description: "Planning spine for making fak's own dev facts, live tool surface, and memory strategies queryable to agents. Covers both the dev loop (`fak index`) and the live hot path (`fak serve`/MCP), so an agent can discover memory tools, learn how to call them, and request guarded tool calls without a full menu dump."
date: 2026-06-30
---

# Self-feature query spine

Status: planning + GitHub issue spine. Parent epic: #1484. This note does not claim
new runtime behavior; it binds the existing query surfaces into one end state and
names the missing issues.

## Goal

An agent should be able to ask **fak itself** what fak can do, instead of re-reading
prose or carrying every tool schema in prompt:

- **Dev path:** "Which package owns this file?", "Which fak verb handles memory?",
  "Where is the doc for capability X?", "What is shipped vs simulated?"
- **Live hot path:** "What memory operations do I have?", "How do I recall or clean
  memory?", "Which tool schema do I need?", then request the tool call through the
  same adjudicated syscall boundary.

The end state is query, then request:

1. Query returns small cards only.
2. The agent selects a card and asks for details or an explain plan.
3. Any effectful action goes through `fak_adjudicate` / `fak_syscall` or the memory
   apply gate; discovery never bypasses the capability floor.

## What already exists

| Surface | Current state |
|---|---|
| `fak index` / `internal/devindex` | Queryable dev facts: lane/path ownership, leaves, docs, claims, verbs. Epic #1287; C1/C2/C3/C6 are shipped, C4/C5 remain open. |
| MCP dev-index tools | `fak_index_lane`, `fak_index_leaves`, `fak_index_docs`, `fak_index_claims`, `fak_index_verbs` are already in the gateway tool list. |
| Memory algebra | `fak memory drivers|explain|run` and `internal/memq` expose memory strategies as composable queries. Built-ins include `recall`, `render`, `clean`, `compact`, `dream`, plus `codex-recall`. |
| MCP memory tools | `fak_memory_drivers`, `fak_memory_explain`, and `fak_memory_run` already expose the memory surface to MCP clients. |
| Tool search | `fak_tools_search` is the shipped lazy/on-demand tool-schema surface from issue #1100. |
| Capability substrate | `internal/capindex`, `contextq.QueryCapabilities`, and the queried overlay work under #1258 / #1261 are the protocol-blind card/query machinery. |

The gap is not "invent query." The gap is one **self-feature query** layer that joins
these surfaces and makes the live request flow explicit.

## The card

Every queryable fak feature should lower to a small `FeatureCard`:

```
FeatureCard {
  kind          // dev-fact | cli-verb | mcp-tool | memory-driver | memory-query | capability
  name
  summary
  tags
  detail_ref    // command, MCP tool name, doc path, or capindex ref
  effect        // read | propose | mutate
  requires_cap  // empty for read-only discovery; set for apply/mutation
  source        // devindex | memq | gateway tools/list | capindex | docs
  witness       // file digest, issue number, tool schema digest, or catalog digest
}
```

Cards are pointers, not bodies. A detail request faults the schema, doc snippet, or
memory explain plan on demand. That keeps the dev path cheap and the live path safe
under large tool/memory catalogs.

## Rungs

1. **#1486 — Unify the catalog.** Add a pure self-feature catalog that emits `FeatureCard`s
   from `devindex`, `memq.Drivers()`, gateway `toolDescriptors()`, and `capindex`
   cards. It is a view, not a new source of truth.
2. **#1488 — Add the dev CLI.** `fak feature query <intent> [--json]
   [--plane dev|live|all]` answers from the unified catalog and can fault details
   for a selected card.
3. **#1489 — Expose the live MCP tool.** Add `fak_feature_query` and extend
   `fak://server/capabilities` so a live agent can discover memory tools, tool schemas,
   and fak query surfaces without a full menu dump.
4. **#1487 — Wire request discipline.** Define the flow from `FeatureCard` to action:
   query -> detail/explain -> adjudicate -> run. Memory mutations stay proposed unless
   `apply` is explicitly authorized; ordinary tools still cross `fak_adjudicate` or
   `fak_syscall`.
5. **#1485 — Witness freshness and hot-path safety.** Tests prove the catalog is
   derived from live sources, the "what memory tools do I have?" query returns the
   memory cards with call shapes, and lazy schema/detail faults do not bypass policy
   or inflate the resident prompt.

## Acceptance

- `fak feature query memory --json` returns the memory driver/tool cards, including how
  to call `fak_memory_drivers`, `fak_memory_explain`, and `fak_memory_run`.
- The same query through MCP returns the same card set, from the gateway's live tool
  descriptors rather than a handwritten duplicate.
- A selected memory strategy can be explained before execution, and effectful memory
  operations are proposal-only unless the caller supplies the apply capability.
- A selected ordinary tool still goes through `fak_adjudicate` / `fak_syscall`; the
  discovery layer cannot execute a tool directly.
- A freshness test fails if a new gateway tool, memory driver, or indexed fak verb is
  missing from the self-feature catalog.

## Relationship to existing work

- #1287 remains the **dev-facts** epic. This spine consumes it.
- #1258 remains the **base-context / queried overlay** epic. This spine gives the
  overlay a concrete fak feature catalog to query.
- #1100 remains the **tool schema lazy-load** increment. This spine generalizes the
  pattern from tools to memory and fak's own feature surface.
- #1431 remains the **Codex memory backend** increment. This spine makes that backend
  discoverable through the same memory query cards, not a special case.

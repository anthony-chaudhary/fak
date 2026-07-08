---
title: "Token-footprint audit — where every token Claude Code sends actually goes"
description: "A structural audit of the input tokens an Anthropic Messages client (Claude Code) sends on each turn: the system / tools / history / tail split, the fixed per-call floor that is the clean minimal baseline, and the reduction lever plus tracking issue for each slice. Grounded in the RequestFootprint primitive and fak's real default-on savers."
---

# Token-footprint audit

> A request answers "how big is this turn?" with one number. This audit answers the
> more useful question: **which slice is the bloat?** — so a reduction effort aims at
> the biggest, cheapest-to-cut slice instead of guessing.

fak sits in front of the model, so it sees every token Claude Code sends. On the
flagship Claude Code route fak forwards those bytes **byte-identically** to preserve
the provider's prompt-cache prefix (that passthrough is the core thesis). Byte-faithful
passthrough is a constraint as much as a feature: fak cannot rewrite Claude Code's
system prompt or reorder its tools without breaking the cache hit. So the audit is what
is actionable — you cannot cut what you have not counted, and you should not cut what
would cost more (a broken cache) than it saves.

## The measurement: RequestFootprint

`agent.RequestFootprint(req)` (internal/agent/anthropic_footprint.go) decomposes one
inbound Messages request into a labeled partition. It is the bucketed twin of
`EstimateAnthropicTokens`: the same ~4-char/token walk, so
`Total.Tokens == EstimateAnthropicTokens(req)` by construction. Provenance is
**ESTIMATED** — the house heuristic, not the provider's billed count. It composes with
the managed-context arm (`internal/gateway/ctxvalue.go`), which reports the provider's
**OBSERVED** resident tokens as one number: OBSERVED says how *full* the window is;
ESTIMATED says *where the bytes went*.

| Bucket | What it holds | Grows with |
|---|---|---|
| `system` | the system prompt: harness spine + any injected memory / CLAUDE.md | roughly fixed per session |
| `tools` | every tool definition (name + description + JSON-Schema parameters) | tool count |
| `history` | every message except the most recent | conversation length |
| `tail` | the most recent message (the volatile suffix that breaks the cache prefix) | current turn |

Two derived roll-ups:

- **`floor` = system + tools.** The fixed per-call tax paid on *every* turn regardless
  of how long the conversation has grown. **The floor is the clean minimal baseline** —
  the irreducible cost of a request with an empty history. A distillation of the system
  prompt or the tool schemas pays back here once per turn, forever.
- **`total` = system + tools + history + tail**, equal to `EstimateAnthropicTokens`.

Per-tool cost is broken out (`PerTool`), which is the exact primitive
[#2924](https://github.com/anthony-chaudhary/fak/issues/2924) needs to gate growth of
the tool floor.

## The floor is the thing to watch

The floor is where fak has the most leverage because it is paid every turn. Two grounded
data points on the tool half of the floor:

- fak's own MCP server registers **22 tools** (`fak_syscall`, `fak_adjudicate`,
  `fak_context_value`, the `fak_index_*` family, the `fak_memory_*` family, …). Every
  one of those schemas is injected on every call that carries them.
- Their schemas are not free: `fak_memory_run`'s input schema alone is ~1.7 KB
  (≈ 430 estimated tokens); the `fak_index_*` schemas run 300–730 bytes each. Measured
  from source, not projected.

This is exactly why `fak_tools_search` (lazy MCP schema loading) exists: it presents a
search interface and faults in a tool's full schema only when the model picks it, so the
per-call tool tax is a floor you can shrink rather than a constant you pay in full.

## Reduction lever per slice

The honest split from the [awesome-token-efficiency catalog](../awesome-token-efficiency.md):
the biggest *mechanical, lossless* wins change cost without changing what the model sees,
so they are default-on; *lossy* wins (compression, summarization) trade fidelity for size
and stay behind a flag with a witness.

| Slice | Lever | Loss | fak status | Tracking |
|---|---|---|---|---|
| `system` | byte-faithful passthrough keeps the cached prefix alive; author-side distillation is the only way to shrink it without a cache break | lossless (cache) / author-side (distill) | ✅ passthrough; ➖ distillation is Claude Code's to do | [#1258](https://github.com/anthony-chaudhary/fak/issues/1258) syspromptmmu (fak's own future spine, not the Claude route) |
| `tools` | tool-floor pruning drops provably-unreachable defs; `fak_tools_search` lazy-loads schemas | lossless | ✅ both shipped | [#2924](https://github.com/anthony-chaudhary/fak/issues/2924) per-tool footprint budget + floor-growth gate |
| `history` | compact-history sheds the un-cacheable middle past the 48k budget by splicing original bytes; ctxplan O(1) view re-materializes under a budget | bounded | ✅ both on by default | [#3028](https://github.com/anthony-chaudhary/fak/issues/3028) measure compaction cache-hit impact |
| `history` (tool results) | oversized-result elision shrinks a scrolled-past `tool_result` to head+tail at 16 KB | bounded | ✅ on by default | — |
| `tail` | inbound content exact-dedup (same file/output sent twice in one request) | lossless | ✅ shipped | [#1101](https://github.com/anthony-chaudhary/fak/issues/1101) (closed) |

The six default-on savers are audited and locked by
[`fak token-defaults-scorecard`](token-defaults-scorecard.md) (grade A, 6/6). This audit
is the map of *which slice each one attacks*, so a new saver can be aimed at the slice
that is actually largest for a given workload.

## What is measured now, and what is next

- **Now (shipped):** the decomposition itself — `RequestFootprint` plus a test that
  proves it is a faithful partition of `EstimateAnthropicTokens` and that per-tool costs
  reconstruct the tools bucket. This is a library primitive; computing a footprint is
  audit-only and changes nothing on the wire.
- **Offline scorecard (shipped, #3230):** `fak footprint` prices fak's own MCP tool
  registry floor deterministically through the same estimator (see
  `docs/context-budget/mcp-tool-floor.md`).
- **Live gateway footprint (shipped, #3233):** the footprint is now wired into the
  managed-context report (`fak_context_value`) as an ESTIMATED `footprint` block beside
  the OBSERVED resident-token number. It is captured on every inbound Anthropic
  passthrough at the pre-transform anchor (`internal/gateway/ctxfootprint.go`
  `observeCtxFootprint`, called from `handleAnthropicMessages` before
  `maybeCompactInboundTools`), so it reports the harness's AS-SENT floor. It reconstructs
  the built-in (no `mcp__` prefix) vs MCP (`mcp__<server>__*`) split and corrects the
  folded-system double-count (`deFoldSystemReq`). Provenance stays ESTIMATED, never
  conflated with the OBSERVED counters (Law A2).

See also: [awesome-token-efficiency](../awesome-token-efficiency.md) (the full field of
methods), [token-defaults-scorecard](token-defaults-scorecard.md) (what is on by
default), and [CONTEXT-IS-NOT-MEMORY](../CONTEXT-IS-NOT-MEMORY.md) (why shedding history
is safe when the durable state lives elsewhere).

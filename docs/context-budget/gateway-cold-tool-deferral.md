---
title: "Gateway cold-tool deferral - the 10x floor lever"
description: "The gateway cold-tool-deferral lever (#3232) under epic #3229: deferring cold tool schemas to cut the always-sent context budget's biggest floor."
---

# Gateway cold-tool deferral — the 10× floor lever (#3232)

Part of epic **#3229**. This is the reduction lever for the epic's biggest,
systemic slice: on a fresh guarded session the tool schemas alone are ~35.8k of
the ~41k floor (built-in system tools ~25.9k + MCP tools ~9.9k). To fak's gateway
those are all just `req.Tools` on the outbound Anthropic request — the **one seam**
where the systemic built-in slice is reachable.

## What it does

When enabled, `maybeDeferColdTools` (`internal/gateway/messages_tooldefer.go`)
rewrites the outbound `/v1/messages` body:

- every allowed-but-**cold** custom tool is marked `defer_loading: true`;
- the **hot** core stays eager — the guard floor's built-ins
  (Read/Edit/Write/Bash/Grep/Glob/Task/TodoWrite + web) plus the search tool;
- one standard `tool_search_tool` is injected as the new tail (carrying the
  `cache_control` anchor when the client was caching tools).

The provider then loads only the non-deferred defs into context and faults a cold
schema in when the model searches for it. **Load-bearing nuance:** `defer_loading`
does *not* shrink the request bytes — the reduction is provider-side and shows up
in the **OBSERVED** usage relay (`input_tokens` / `cache_read`), never in the
ESTIMATED byte footprint (`fak footprint`, #3230). The measurement that proves the
*after* is `fak_context_value`'s live footprint (#3233) + the provider counters.

## Safety properties (witnessed)

`internal/gateway/messages_tooldefer_test.go` proves:

- cold defs get `defer_loading:true`, the hot core stays eager, one
  `tool_search_tool` is injected;
- the non-tools body bytes (system, messages) are **byte-identical** after the
  splice, and the result **re-decodes** as a valid Anthropic request;
- the transform is **deterministic** — identical input tools[] → byte-identical
  output — so the `cache_control` prefix is stable turn-over-turn and the session
  cache survives (a turn-0-only rewrite would mismatch the client's non-deferred
  turn-1 body and bust the cache every turn);
- an **already-deferred** body (the Claude Code `ENABLE_TOOL_SEARCH` case) is a
  no-op — fak never double-applies;
- **identity on ambiguity** (non-JSON, no tools, only hot tools) — fail-safe.

Metrics: `observeToolDefer` accumulates `deferFiredTurns` / `deferColdCount`
(WITNESSED), the twin of `observeInboundToolPrune`. These fold into the
`AdjudicationSummary` (`DeferColdTurns` / `DeferColdCount`) and surface on two
operator faces (#3531):

- the `/metrics` scrape — `fak_gateway_tool_defer_cold_total` and
  `fak_gateway_tool_defer_turns_total`, the OUTBOUND twin of the inbound
  `fak_gateway_inbound_tools_pruned_total` family;
- the `fak manage` exit summary — a **cold-tool deferral** section printed only
  when the lever fired (a default-off or all-hot session stays quiet), naming the
  cold-def count × turns and flagging that the token drop is **OBSERVED** on
  `/metrics` (provider-side), never a request-byte shrink like the prune lever.

## Default OFF — the validation gates before flipping it on

`--defer-cold-tools` / `Config.DeferColdTools` (also `FAK_DEFER_COLD_TOOLS=1`);
ablate an A/B arm with `FAK_ABLATE_DEFER_TOOLS=1`. It is **off by default** because
this is the epic's highest-risk lever. Before the default flips on:

1. **Wire constants** — `toolSearchToolType` (`tool_search_tool_20250917`) and
   `toolSearchBeta` (`tool-search-2025-09-17`) must match the account's enabled
   Tool Search revision. A mismatch is a 400 upstream, which the fail-open path
   turns into a silent identity (never a broken session) — but also never a
   reduction. Confirm against a live account and update the two named constants.
2. **A/B** — arm-on vs `FAK_ABLATE_DEFER_TOOLS`: the OBSERVED provider input
   tokens for the tool slice must drop materially (target: approach Anthropic's
   ~85% tool-def cut) with **held task-completion accuracy** and **no rise in
   poison/quarantine rate**.
3. **Pin/quarantine (#3200)** — the deferred cold tools are exactly what gets
   faulted in later; that fault-in must clear the first-seen quarantine + pin-by-
   hash guard.

## Follow-ups

- The A/B scorecard entry (token-delta × held-accuracy × poison-rate) extending
  #3230's footprint scorecard, once a live run exists (#3532).

## Cross-links

- **#3229** — epic. **#3230** — the offline MCP floor scorecard. **#3233** — the
  live `fak_context_value` footprint that measures the after. **#3231** — deferral
  of fak's OWN MCP tools (the same idea at the MCP `tools/list` seam). **#3235** —
  the hybrid ranker so a deferred tool stays re-findable. **#3200** — the
  pin/quarantine guard for the fault-in.

---
title: "The Footprint Ladder — every core tool ships on every API call, priced in KV cache"
description: "fak's doctrine for adding a capability at the highest (least-footprint) rung that works. Formalizes Hermes' Footprint Ladder as a fak doctrine backed by a computed number: a new core tool's marginal prefix-token cost x its call frequency is its footprint bill, measured by `fak footprint` and refused by the floor-growth gate unless the bill is justified or the capability drops to a lower rung."
---

# The Footprint Ladder

> To add a capability, choose the **highest (least-footprint) rung that works** —
> because *every core tool ships on every API call*. In most agent stacks that rule
> is enforced by reviewer taste. In fak it is enforced by a **number the kernel
> computes**: a tool's marginal prefix-token cost lives in the KV-cache prefix fak
> reuses on every turn, so the "footprint" the rule reasons about is a value fak can
> *measure and refuse*, not just argue about.

This is a fak doctrine adapted from the "Footprint Ladder" idea in the Hermes agent's
`AGENTS.md` (see [`docs/integrations/hermes.md`](integrations/hermes.md)). Hermes states
the ladder and leaves it to reviewer judgment. fak keeps the ladder and binds it to the
already-shipped tool-footprint measurement so the rung choice has a witness.

## Why footprint is a real cost here, not a metaphor

fak sits in front of the model and forwards the request's bytes **byte-identically** to
preserve the provider's prompt-cache prefix (the core thesis; see
[`docs/serving/token-footprint-audit.md`](serving/token-footprint-audit.md)). Every tool
definition — name + description + JSON-Schema parameters — is part of the `tools` slice
of that prefix, and the prefix is re-sent on **every** turn regardless of whether the tool
is ever called. So a new core tool is not a one-time cost: it is a **per-turn tax paid
forever**, on every request in every session that carries it.

That tax is measured, not estimated by feel:

- **`agent.RequestFootprint`** (`internal/agent/anthropic_footprint.go`) decomposes one
  inbound request into `system | tools | history | tail`, with per-tool costs broken out
  (`PerTool`). It is the bucketed twin of `EstimateAnthropicTokens`, so the numbers never
  drift from the estimator on the wire.
- **`fak footprint`** (`cmd/fak/footprint.go`, #3230) prices fak's own always-sent MCP
  tool-schema floor offline and deterministically, largest-first. Regenerate any time;
  the committed baseline is [`docs/context-budget/mcp-tool-floor.md`](context-budget/mcp-tool-floor.md).

## The footprint bill: marginal prefix tokens × call frequency

The ladder's decision variable is a **bill**, not a raw token count:

```
footprint_bill(tool) = marginal_prefix_tokens(tool) × call_frequency(tool)
```

- **`marginal_prefix_tokens`** — how many estimated tokens the tool's schema adds to the
  always-sent prefix. Computed statically from the tool definition by `fak footprint` /
  `mcpfootprint.Price`; it is the additive quantity, denominated in ESTIMATED tokens
  (never a provider-billed count — Law A2: every value carries its provenance).
- **`call_frequency`** — how often the tool is actually selected, drawn from usage
  telemetry (the gateway usage logs / nightrun feed), normalized so an *always-resident*
  tool has frequency **1.0** (it ships on every turn) and a *deferred* tool trends toward
  **0** (it costs the prefix nothing until faulted in).

The two axes are separable levers:

- A tool that is **cheap and hot** is fine at any rung.
- A tool that is **expensive but hot** must justify its bill in the budget (below) or be
  trimmed.
- A tool that is **expensive but cold** should not pay the always-sent floor at all — it
  drops a rung via **cold-schema deferral** (`fak_tools_search` lazy schema loading;
  #3231/#3232). Deferral is exactly the mechanism that multiplies `call_frequency` toward
  0: the schema is faulted in only on the turn the model picks the tool, so a rarely-used
  capability stops taxing every other turn.

### Frequency fallback for a brand-new tool

A tool that has never shipped has no telemetry, and sparse early data is noisy. The
documented fallback is **conservative**: treat `call_frequency = 1.0` (assume
always-sent, worst case) until real usage exists. This is the same assumption the
floor-growth gate already bakes in — it prices the whole registry as always-resident — so
a new tool is scored at its maximum possible bill first and only earns a lower effective
frequency once the gateway usage logs prove it is cold.

## The gate: the number can refuse

Measuring the floor does not keep the core narrow — a number that cannot refuse a change
is still just taste. `internal/mcpfootprint.CheckFloor` (#2924) gates the measured
always-sent floor against a committed ceiling (`FloorBudgetTokens`) as a **one-way
ratchet**:

| Direction | Reason | Meaning |
|---|---|---|
| measured **>** budget | `FLOOR_BUDGET_EXCEEDED` | a new tool, or a fattened description/JSON-Schema, grew the per-call tax |
| measured **<** budget − slack | `FLOOR_BUDGET_STALE` | a deferral won headroom that was never banked into the constant |

The only way through `FLOOR_BUDGET_EXCEEDED` is to **raise `FloorBudgetTokens` in the same
commit** and re-pin the baseline table — which is precisely the justification the doctrine
demands: the new per-turn tax becomes a diff line a reviewer sees, bound to its cause,
instead of being discovered a quarter later. A sibling gate, `CheckDescriptions`
(`DESC_BUDGET_EXCEEDED` / `DESC_BUDGET_STALE`, #3608), holds the description-prose slice —
the one slice with no machine consumer — to the same ratchet. The gate fails **closed**:
an empty registry prices as 0 tokens and refuses as `FLOOR_BUDGET_STALE` rather than
greening on a measurement of nothing.

## The rungs (highest / least-footprint first)

Choose the highest rung that actually delivers the capability. Each rung below adds strictly
more always-sent footprint than the one above it; rung 6 is the only one that grows the
floor the gate polices.

| Rung | Add the capability as… | fak surface | Always-sent footprint |
|---:|---|---|---|
| 1 | **Extend existing code / a leaf** | edit an existing leaf, or `fak new-leaf <name> --tier <tier>`; "add a feature as a leaf, not a core edit" (AGENTS.md) | **none** — no new tool schema on the wire |
| 2 | **CLI command + skill** | a `fak` subcommand (`cmd/fak/<name>.go` + pure logic in `internal/<name>/`) plus a `.claude/skills/<name>/SKILL.md` the agent invokes | **none** on the tool floor (the skill is invoked, not a resident schema) |
| 3 | **Service-gated / deferred tool** | a tool registered conditionally or lazily — cold-schema deferral + `fak_tools_search`, so the schema is faulted in only when selected | **near-zero** until faulted in (frequency → 0) |
| 4 | **Plugin** | a packaged plugin surface (`plugin.json` / the marketplace manifest) enabled per install, not resident in the core registry | **none** unless the plugin is enabled |
| 5 | **MCP server in the catalog** | an external MCP server the operator adds to their catalog, kept out of fak's own always-sent `fak_*` floor | **paid by the operator's chosen server**, not fak's core floor |
| 6 | **New core tool** *(last resort)* | a new entry in `gateway.MCPFloorToolDefs()` — shipped on every API call fak's MCP server serves | **full marginal bill, every turn, forever** — priced by `fak footprint`, gated by `CheckFloor` |

The mapping from Hermes' generic rung names to fak's concrete surfaces is deliberate: the
doctrine is only enforceable if each rung points at a real mechanism a reviewer can check.

## Which rung? — the checklist for a new capability

Run this before adding a capability. Stop at the first rung that satisfies the need.

1. **Can an existing leaf or command do it with an edit?** → Rung 1. No new footprint. Prefer this.
2. **Is it a workflow the agent can invoke on demand?** → Rung 2 (CLI + skill). The agent
   calls it when needed; it is not a resident schema.
3. **Does it truly need to be a model-selectable tool, but only sometimes?** → Rung 3.
   Register it deferred (`fak_tools_search` / cold-schema defer) so it costs the prefix
   nothing until the model picks it. **Score it:** run `fak footprint` and confirm the
   deferred path keeps it off the always-sent floor.
4. **Is it optional / install-scoped?** → Rung 4 (plugin) or Rung 5 (MCP server in the
   catalog). It rides the operator's surface, not fak's core floor.
5. **Does it genuinely have to ship on every API call?** → Rung 6, and only then. Before
   committing:
   - Run `fak footprint --top 8` to read the current floor and where the new tool would rank.
   - Compute its **footprint bill** (marginal tokens × frequency; use the `1.0` fallback if
     it is brand-new).
   - If the tool pushes the floor past `FloorBudgetTokens`, the gate refuses
     `FLOOR_BUDGET_EXCEEDED`. **Justify the bill** by raising `FloorBudgetTokens` in the
     same commit and re-pinning [`docs/context-budget/mcp-tool-floor.md`](context-budget/mcp-tool-floor.md)
     — so the tax is a reviewed diff line — **or drop to a lower rung**.
   - Keep the description lean; `CheckDescriptions` gates the prose slice separately.

The rule of thumb: **a new core tool must earn its permanent per-turn tax, or it does not
belong on the core floor.** The ladder stops being taste the moment `fak footprint`
prints the number and `CheckFloor` can refuse the diff.

## See also

- [Token-footprint audit](serving/token-footprint-audit.md) — where every token a request
  sends actually goes (the `system | tools | history | tail` split and the reduction lever
  per slice).
- [MCP tool-schema floor — committed baseline](context-budget/mcp-tool-floor.md) — the
  measured floor `CheckFloor` ratchets down, and how to justify growth.
- [Hermes Agent integration](integrations/hermes.md) — the upstream source of the ladder idea.
- [Add a feature as a leaf, not a core edit](../AGENTS.md) — the rung-1 default the kernel
  already enforces via `internal/architest`.

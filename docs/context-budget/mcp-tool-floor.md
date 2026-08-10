---
title: "MCP tool-schema floor - committed baseline"
description: "The committed baseline (#3230) for fak's MCP tool-schema floor, part of epic #3229's work to shrink the always-sent context budget."
---

# MCP tool-schema floor — committed baseline (#3230)

Part of epic **#3229** (shrink the always-sent context budget). This is the
measured baseline the reduction levers ratchet down.

## What this number is

Every tool fak's MCP server advertises ships its full JSON schema in every
`tools/list`, and that set is re-sent on **every** turn — the model pays for it
whether or not the tool is ever called. That fixed per-turn tax is the *tool
floor*. `fak footprint` prices it offline and deterministically, reusing the
same estimator the agent request footprint uses (`agent.RequestFootprint`, via
`internal/mcpfootprint`), so the number can never drift from
`EstimateAnthropicTokens`.

Regenerate at any time:

```
fak footprint            # human table, largest-first
fak footprint --json     # schema fak-mcp-footprint/1
fak footprint --top 8    # just the heaviest N
```

## Baseline (measured)

```
mcp-footprint: 19 tools · floor 4507 est. tokens (20283 bytes, ESTIMATED)
```

Re-pinned for #6011: #6022 retired the repository index MCP tools, taking the registry
from 26 tools to 19 and the measured whole-schema floor from 5464 to 4507 estimated
tokens (957 banked). The reduction was won by removing tools, not by relaxing the
ratchet — the slack it opened is banked into the constant rather than left as headroom.

Heaviest contributors (the cold-schema deferral targets for #3231/#3232):

| rank | est. tokens | bytes | tool |
|-----:|------------:|------:|------|
| 1 | 503 | 2264 | fak_trajquery |
| 2 | 496 | 2234 | fak_memory_run |
| 3 | 451 | 2033 | fak_memory_explain |
| 4 | 348 | 1566 | fak_admit |
| 5 | 271 | 1222 | fak_context_change |
| 6 | 261 | 1175 | fak_read |
| 7 | 244 | 1101 | fak_feature_query |
| 8 | 222 | 999 | fak_adjudicate |

The full 19-tool breakdown is what `fak footprint` prints; only the head is
pinned here so a drift is legible in review.

## The gate (#2924)

Measuring the floor does not keep the core narrow — a number that cannot refuse a
change is still just taste. `internal/mcpfootprint.CheckFloor` gates the measured
floor against a committed ceiling, `FloorBudgetTokens` (currently **4507**), as a
one-way ratchet:

| Direction | Reason | What it means |
|---|---|---|
| measured **>** budget | `FLOOR_BUDGET_EXCEEDED` | a new tool, or a fattened description/JSON-Schema, grew the per-call tax |
| measured **<** budget − 250 | `FLOOR_BUDGET_STALE` | a deferral won headroom that was never banked into the constant |

**How to justify growth.** Raise `FloorBudgetTokens` in the *same commit* as the tool
that grew it, and re-pin the baseline table above. That is the whole mechanism: the
new per-turn tax becomes a diff line a reviewer sees, bound to its cause, instead of
being discovered a quarter later. Prefer deferring a cold schema (#3231/#3232) over
paying the floor.

The 250-token slack absorbs incidental churn (a reworded description) while still
forcing a real reduction to be banked — the same discipline `internal/pythongate`
applies to the `tools/*.py` baseline: the ratchet only ever tightens. The gate fails
closed: an empty registry prices as 0 tokens and refuses as `FLOOR_BUDGET_STALE`
rather than greening on a measurement of nothing.

## The description budget (#3608)

The floor gate above prices the **whole** schema (name + description + JSON-Schema
parameters). But the description is the one slice with no machine consumer — the
parameters are validated, the name is dispatched on, but the description is pure
prompt-prefix prose the model reads, so it is the slice most prone to silently
fattening into a per-turn tax. #3231 defers the *cold* schemas; this keeps the *hot*
(always-sent) descriptions lean, and the two compose.

`internal/mcpfootprint.CheckDescriptions` gates the summed always-sent `fak_*`
description tokens against `DescriptionBudgetTokens` — the same one-way ratchet, priced
through the same estimator (a description-only `ToolDef` carries no name or parameter
bytes, so the number never drifts from `EstimateAnthropicTokens`).

```
always-sent fak_* description floor: 1552 est. tokens across 19 tools
```

Re-pinned for #6011 alongside the whole-schema floor: retiring the repository index
tools (#6022) took the description slice from 1966 to 1552 estimated tokens across
19 rather than 26 tools. The values are estimator-derived, not provider-billed token
counts.

Heaviest description bodies (the trim targets — `fak footprint` ranks the full schema;
`PerToolDescription` ranks the prose slice):

| rank | est. tokens | tool |
|-----:|------------:|------|
| 1 | 156 | fak_admit |
| 2 | 115 | fak_memory_run |
| 3 | 101 | fak_read |
| 4 | 94 | fak_tools_search |
| 5 | 92 | fak_session_reset |
| 6 | 85 | fak_context_change |

| Direction | Reason | What it means |
|---|---|---|
| measured **>** budget | `DESC_BUDGET_EXCEEDED` | a fattened (or newly-added) description grew the per-call prose tax |
| measured **<** budget − 200 | `DESC_BUDGET_STALE` | a trim won headroom that was never banked into the constant |

Justify growth the same way as the floor gate: raise `DescriptionBudgetTokens` in the
*same commit* as the description that grew it, and re-pin the number above — the new
prose tax becomes a diff line a reviewer sees, bound to its cause. The 200-token slack
absorbs a reworded sentence while still forcing a real trim to be banked. Trimming the
hot descriptions to fit a *lower* budget is the standing follow-on this gate makes safe
and measurable (tool-search still resolves a tool from a leaner description).

## Witness

- `internal/mcpfootprint.TestRealFakMCPFloor` prices the **real** registry and
  asserts the floor is a faithful partition (floor bytes == sum of per-tool
  bytes) and non-trivial — the number above is reproducible, not hand-typed.
- `internal/mcpfootprint.TestPricePartitionsExactly` /
  `TestPerToolSortedDescending` lock the estimator invariants.
- `cmd/fak.TestMCPFootprintVerbJSON` witnesses the `fak footprint --json` shape.
- `internal/mcpfootprint.TestCommittedBudgetMatchesMeasuredFloor` proves the ceiling
  above is a measurement, not a hand-typed number that drifted from the registry;
  `TestFloorGateRefusesGrowth` witnesses the refusal firing on a registry grown by one
  tool, and `TestFloorGateDemandsBankedWin` witnesses the ratchet refusing an unbanked
  reduction. `TestFloorGateBandBoundaries` pins the exact admit/refuse edges.
- `internal/mcpfootprint.TestDescriptionBudgetPassesAtHEAD` prices the real registry's
  description slice and asserts it is under `DescriptionBudgetTokens`;
  `TestDescriptionBudgetRefusesGrowth` witnesses the refusal firing on a synthetic
  over-budget description edit, `TestDescriptionBudgetDemandsBankedWin` the unbanked-trim
  refusal, and `TestDescriptionIsDescriptionOnly` proves the number is the prose slice
  alone (strictly less than the full schema floor).

## Cross-links

- **#3229** — epic: shrink the always-sent context budget (this baseline is its
  measurement floor).
- **#3233** — the *live* gateway footprint (same estimator, measured on a real
  request rather than the static registry).
- **#3231** — defer cold `fak_*` MCP schemas (drives the number above down).
- **#3232** — gateway tool-floor deferral (the 10× lever on this same floor).
- **#3234** — the userland analogue: `fak skill footprint` for the resident
  `.claude/skills` description floor.
- **#5444** — that userland floor's own one-way ratchet, built to this page's
  pattern: [resident skill-description floor](skill-description-floor.md)
  (`SKILL_DESC_BUDGET_EXCEEDED` / `SKILL_DESC_BUDGET_STALE`).

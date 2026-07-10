---
title: "Tier to account routing (dispatch marathons)"
description: "How fak fleet dispatch maps a work tier to the account and model that serves it, and how that routing holds up across long dispatch marathons."
---

# Tier ↔ account routing (dispatch marathons)

How fleet dispatch maps work *tier* to the account/model that serves it, how an
issue *declares* its tier, how to *observe* the routing decision, and what is
enforced vs. merely documented today.

## Task tiers

From `internal/modelroute/tierpolicy.go` (`WorkTier`, opposite to capability —
`T0 < T1 < T2`, so all compares go through `MeetsRequirement`, never a raw `<`):

| Tier | Work class | Repo surface | Serve on |
|---|---|---|---|
| **T0** | engineering (frontier) | `cmd/**`, `internal/**` | Opus seat `july6-netra` |
| **T1** | infra | `tools/**`, `scripts/**` | mid seat |
| **T2** | gardening / docs | `docs/**`, `README`, `INDEX.md`, `llms*.txt` | `day26NEW-netra` |

## Account decision (2026-07-06)

Frontier/T0 work runs on **`july6-netra`** — the only ready account preflight
tags `model=opus`, tier 1. Lower-tier T2 work goes to **`day26NEW-netra`**.
There is **no `july7-netra`** account in the fleet; the frontier seat is
`july6-netra`.

## Label → tier bridge (C4) and the readout (C8)

An issue declares its tier with two namespaced GitHub labels — `tier/T<N>-required`
(the risk floor) and `tier/T<N>-optimal` (the ideal target). The grammar is owned by
`internal/issuecontract`; `dispatchtick.IssueTierFromLabels` parses that pair into the
typed `IssueTier` the chooser consumes. A tier is trusted **only** when both labels are
present, valid, and self-consistent (optimal at least as demanding as required);
anything missing, invalid, conflicting, or contradictory degrades to the conservative
frontier floor, carrying a closed-vocabulary flag that names why
(`model_tier_*_missing|invalid|conflict`, `model_tier_contradiction`). A stray
`priority/P1` is never read as tier T1 — the disambiguation is baked into the grammar.

## Observability: `fak dispatch tier-status`

`fak dispatch tier-status` is the offline, launches-nothing readout for the chain.
Given issue rows (each carrying its tier labels and an account pool) it folds them
through the pure chooser and prints, per issue, which seat it would route to, over-tier
waste and under-tier refusals side by side, a MODELED cost delta, and the `tags=[...]`
flags for any issue held at the conservative floor by a bad tag.

```
fak dispatch tier-status --demo             # a runnable five-issue fixture
fak dispatch tier-status --in issues.json   # a JSON array (or {"issues":[...]})
fak dispatch tier-status --in issues.json --json
```

It is pure and deterministic (`dispatchtick.BuildTierStatusReport`) — no clock, no
network, no launch — the same posture as `fak dispatch order` and `fak tier-calibrate`.
The cost delta is MODELED cost points, not billed dollars, and is not netted against
quality.

## Enforced vs. documented

This routing is **documented and now observable, but not yet enforced**.
`RouteAccountForTier` (#3042) is a pure chooser and `fak dispatch tier-status` (#3045)
renders its decisions from an issue's real labels, but neither is wired into the live
dispatcher, so `FleetIssueDispatch` still selects accounts with its existing switcher.
Until the chooser is wired:

- Report each worked leaf with its tier label.
- Flag the misroute where the Opus/`july6-netra` frontier seat is pointed at the
  **T1 `tools` lane** (also the zero-yield sink #2062) instead of T0 engineering.

# Tier ↔ account routing (dispatch marathons)

How fleet dispatch maps work *tier* to the account/model that serves it, and
what is enforced vs. merely documented today.

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

## Enforced vs. documented

This routing is **documented, not enforced**. `RouteAccountForTier` (#3042) is
shipped as a pure chooser but **not wired** into the live dispatcher (see the
`modeltier-chain` note), so `FleetIssueDispatch` still selects accounts with its
existing switcher. Until the chooser is wired:

- Report each worked leaf with its tier label.
- Flag the misroute where the Opus/`july6-netra` frontier seat is pointed at the
  **T1 `tools` lane** (also the zero-yield sink #2062) instead of T0 engineering.

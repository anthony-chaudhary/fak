---
title: "dispatch loop - dispatch:loop identity"
description: "A durable recurring dispatch state machine identified by loop ID and measured through admitted, refused, started, ended, and witnessed runs. Scope: dispatch:loop. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# dispatch loop

**Meaning:** A durable recurring dispatch state machine identified by loop ID and measured through admitted, refused, started, ended, and witnessed runs.

## Do not conflate with

- **fleet supervisor:** A loop owns recurring execution state for one cadence; a supervisor observes multiple witnessed surfaces and decides interventions.

## Query this identity

```console
$ fak disambiguation query "dispatch loop" --scope-kind "dispatch" --scope-value "loop" --json
```

## Identity

- **Scope:** `dispatch:loop`
- **Aliases:** loop
- **Owner:** `loopmgr / loop`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `fleet-source/1`

## Source witnesses

- `internal/loopmgr/loopmgr.go` (go-source, revision `fleet-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-fleet-source`)

[Back to the disambiguation index](../INDEX.md)

---
title: "compute fleet - dispatch:fleet identity"
description: "A transport-agnostic roster of uniquely identified controllable machines whose live reports are folded by the public fleet core. Scope: dispatch:fleet. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# compute fleet

**Meaning:** A transport-agnostic roster of uniquely identified controllable machines whose live reports are folded by the public fleet core.

## Do not conflate with

- **dispatch wave:** A fleet is the available machine roster; a wave is one concurrency-safe batch of work selected for launch.

## Query this identity

```console
$ fak disambiguation query "compute fleet" --scope-kind "dispatch" --scope-value "fleet" --json
```

## Identity

- **Scope:** `dispatch:fleet`
- **Aliases:** fleet
- **Owner:** `fleet / fleet`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `fleet-source/1`

## Source witnesses

- `internal/fleet/roster.go` (go-source, revision `fleet-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-fleet-source`)

[Back to the disambiguation index](../INDEX.md)

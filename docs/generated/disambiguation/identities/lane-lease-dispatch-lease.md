---
title: "lane lease - dispatch:lease identity"
description: "A live ownership claim carrying lease ID, lane or tree, holder identity, and read-only posture for collision admission. Scope: dispatch:lease. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# lane lease

**Meaning:** A live ownership claim carrying lease ID, lane or tree, holder identity, and read-only posture for collision admission.

## Do not conflate with

- **dispatch lane:** A lease is an active holder claim; the lane is the durable taxonomy partition it may claim.

## Query this identity

```console
$ fak disambiguation query "lane lease" --scope-kind "dispatch" --scope-value "lease" --json
```

## Identity

- **Scope:** `dispatch:lease`
- **Aliases:** lease
- **Owner:** `laneadmit / dos`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `fleet-source/1`

## Source witnesses

- `internal/laneadmit/laneadmit.go` (go-source, revision `fleet-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-fleet-source`)

[Back to the disambiguation index](../INDEX.md)

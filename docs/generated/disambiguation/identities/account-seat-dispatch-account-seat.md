---
title: "account seat - dispatch:account-seat identity"
description: "A provider account-capacity slot with availability, session cap, leased slots, free slots, and bound worker IDs. Scope: dispatch:account-seat. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# account seat

**Meaning:** A provider account-capacity slot with availability, session cap, leased slots, free slots, and bound worker IDs.

## Do not conflate with

- **dispatch worker:** A seat supplies bounded account capacity; it is not the worker process consuming one slot.

## Query this identity

```console
$ fak disambiguation query "account seat" --scope-kind "dispatch" --scope-value "account-seat" --json
```

## Identity

- **Scope:** `dispatch:account-seat`
- **Aliases:** seat
- **Owner:** `fleetaccounts / accounts`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `fleet-source/1`

## Source witnesses

- `internal/fleetaccounts/resolve.go` (go-source, revision `fleet-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-fleet-source`)

[Back to the disambiguation index](../INDEX.md)

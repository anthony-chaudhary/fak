---
title: "leaf identity - ownership:leaf identity"
description: "The semantic package or command unit that owns a change and supplies the attribution token in a valid fak commit stamp. Scope: ownership:leaf. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# leaf identity

**Meaning:** The semantic package or command unit that owns a change and supplies the attribution token in a valid fak commit stamp.

## Do not conflate with

- **dispatch lane:** A leaf identifies what semantic unit changed; a lane is the concurrency ownership region that admits work.

## Query this identity

```console
$ fak disambiguation query "leaf identity" --scope-kind "ownership" --scope-value "leaf" --json
```

## Identity

- **Scope:** `ownership:leaf`
- **Aliases:** leaf
- **Owner:** `disambiguation / disambiguation`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `ownership-source/1`

## Source witnesses

- `internal/disambiguation/ownership.go` (go-source, revision `ownership-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-ownership-source`)

[Back to the disambiguation index](../INDEX.md)

---
title: "dispatch lane - dispatch:lane identity"
description: "A named taxonomy partition that maps a work request to a canonical file-tree region and concurrency policy. Scope: dispatch:lane. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# dispatch lane

**Meaning:** A named taxonomy partition that maps a work request to a canonical file-tree region and concurrency policy.

## Do not conflate with

- **lane lease:** The lane names the work partition; a lease is the time-bounded ownership claim that currently holds it or its tree.

## Query this identity

```console
$ fak disambiguation query "dispatch lane" --scope-kind "dispatch" --scope-value "lane" --json
```

## Identity

- **Scope:** `dispatch:lane`
- **Aliases:** lane
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

---
title: "fak commit stamp - ownership:commit-stamp identity"
description: "The validated (fak <leaf>) commit-subject token binding one commit to its semantic leaf; it is not a lane lease or module version. Scope: ownership:commit-stamp. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# fak commit stamp

**Meaning:** The validated (fak <leaf>) commit-subject token binding one commit to its semantic leaf; it is not a lane lease or module version.

## Do not conflate with

- **module revision identity:** A stamp attributes a commit to a leaf; module@rev identifies the history-derived version after commits land.

## Query this identity

```console
$ fak disambiguation query "fak commit stamp" --scope-kind "ownership" --scope-value "commit-stamp" --json
```

## Identity

- **Scope:** `ownership:commit-stamp`
- **Aliases:** commit stamp
- **Owner:** `fak / cmd`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `ownership-source/1`

## Source witnesses

- `internal/hooks/commitstamp.go` (go-source, revision `ownership-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-ownership-source`)

[Back to the disambiguation index](../INDEX.md)

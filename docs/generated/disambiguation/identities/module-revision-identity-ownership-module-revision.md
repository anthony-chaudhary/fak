---
title: "module revision identity - ownership:module-revision identity"
description: "A history-derived identity rendered as module@r<touch-count>+g<commit>, naming which module moved and at what revision. Scope: ownership:module-revision. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# module revision identity

**Meaning:** A history-derived identity rendered as module@r<touch-count>+g<commit>, naming which module moved and at what revision.

## Do not conflate with

- **leaf identity:** A module is the versioned path surface; a leaf is the semantic owner used for attribution and stamps.

## Query this identity

```console
$ fak disambiguation query "module revision identity" --scope-kind "ownership" --scope-value "module-revision" --json
```

## Identity

- **Scope:** `ownership:module-revision`
- **Aliases:** module@rev
- **Owner:** `fak / cmd`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `ownership-source/1`

## Source witnesses

- `internal/modver/modver.go` (go-source, revision `ownership-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-ownership-source`)

[Back to the disambiguation index](../INDEX.md)

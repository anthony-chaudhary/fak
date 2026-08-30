---
title: "dispatch ownership lane - ownership:lane identity"
description: "A declared file-tree region used to arbitrate concurrent work; it may own several leaves and must not be inferred from a similar name. Scope: ownership:lane. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# dispatch ownership lane

**Meaning:** A declared file-tree region used to arbitrate concurrent work; it may own several leaves and must not be inferred from a similar name.

## Do not conflate with

- **leaf identity:** A lane controls collision-safe admission; a leaf provides semantic attribution and may map into that lane.

## Query this identity

```console
$ fak disambiguation query "dispatch ownership lane" --scope-kind "ownership" --scope-value "lane" --json
```

## Identity

- **Scope:** `ownership:lane`
- **Aliases:** ownership lane
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

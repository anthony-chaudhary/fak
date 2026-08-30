---
title: "dispatch wave - dispatch:wave identity"
description: "An indexed, bounded batch of dispatch members with a shared step budget and explicit lease regions or whole-lane claims. Scope: dispatch:wave. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# dispatch wave

**Meaning:** An indexed, bounded batch of dispatch members with a shared step budget and explicit lease regions or whole-lane claims.

## Do not conflate with

- **compute fleet:** A wave is a selected launch batch; the fleet is the machine population on which batches may execute.

## Query this identity

```console
$ fak disambiguation query "dispatch wave" --scope-kind "dispatch" --scope-value "wave" --json
```

## Identity

- **Scope:** `dispatch:wave`
- **Aliases:** wave
- **Owner:** `issuecohort / dispatch`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `fleet-source/1`

## Source witnesses

- `internal/issuecohort/issuecohort.go` (go-source, revision `fleet-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-fleet-source`)

[Back to the disambiguation index](../INDEX.md)

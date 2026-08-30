---
title: "capability floor - policy:authority identity"
description: "The minimum authority boundary represented by negotiated capability tokens and policy constraints; it limits what may proceed but does not describe the outcome of a particular call. Scope: policy:authority. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# capability floor

**Meaning:** The minimum authority boundary represented by negotiated capability tokens and policy constraints; it limits what may proceed but does not describe the outcome of a particular call.

## Do not conflate with

- **policy declaration:** The floor is the effective authority boundary; a declaration is one configured input used to establish it.
- **adjudication verdict:** A capability is authority advertised or granted ahead of a call; a verdict is the call-specific result after checks run.

## Query this identity

```console
$ fak disambiguation query "capability floor" --scope-kind "policy" --scope-value "authority" --json
```

## Identity

- **Scope:** `policy:authority`
- **Aliases:** capability
- **Owner:** `abi / abi`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `policy-source/1`

## Source witnesses

- `internal/abi/types.go` (go-source, revision `policy-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-policy-source`)

[Back to the disambiguation index](../INDEX.md)

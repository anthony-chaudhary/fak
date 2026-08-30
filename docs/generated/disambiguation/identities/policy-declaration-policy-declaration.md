---
title: "policy declaration - policy:declaration identity"
description: "A declarative set of tool, argument, network, and resource rules loaded by the adjudicator; it states configured constraints but is not itself a decision for one call. Scope: policy:declaration. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# policy declaration

**Meaning:** A declarative set of tool, argument, network, and resource rules loaded by the adjudicator; it states configured constraints but is not itself a decision for one call.

## Do not conflate with

- **capability floor:** A policy declaration supplies configured rules; the capability floor is the minimum authority those rules allow a caller to exercise.
- **adjudication verdict:** A declaration is input to adjudication; a verdict is the typed result for one tool call.

## Query this identity

```console
$ fak disambiguation query "policy declaration" --scope-kind "policy" --scope-value "declaration" --json
```

## Identity

- **Scope:** `policy:declaration`
- **Aliases:** policy manifest
- **Owner:** `adjudicator / adjudicator`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `policy-source/1`

## Source witnesses

- `internal/adjudicator/decide.go` (go-source, revision `policy-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-policy-source`)

[Back to the disambiguation index](../INDEX.md)

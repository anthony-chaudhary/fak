---
title: "structural preflight - policy:structural-preflight identity"
description: "The local pre-dispatch fold over grammar and adjudicator rungs for one tool call, producing a verdict without executing the tool or asking a model to interpret intent. Scope: policy:structural-preflight. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# structural preflight

**Meaning:** The local pre-dispatch fold over grammar and adjudicator rungs for one tool call, producing a verdict without executing the tool or asking a model to interpret intent.

## Do not conflate with

- **model-mediated check:** Structural preflight applies deterministic local rules before dispatch; a model-mediated check asks a model to classify meaning or intent.
- **adjudication verdict:** Preflight is the checking operation; the adjudication verdict is its typed output.

## Query this identity

```console
$ fak disambiguation query "structural preflight" --scope-kind "policy" --scope-value "structural-preflight" --json
```

## Identity

- **Scope:** `policy:structural-preflight`
- **Aliases:** preflight
- **Owner:** `fak / cmd`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `policy-source/1`

## Source witnesses

- `cmd/fak/main.go` (go-source, revision `policy-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-policy-source`)

[Back to the disambiguation index](../INDEX.md)

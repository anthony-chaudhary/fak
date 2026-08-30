---
title: "model-mediated check - policy:model-mediated identity"
description: "A semantic assessment whose result depends on a model interpreting content or intent; it is distinct from fak's deterministic structural preflight and is not part of the preflight command's local fold. Scope: policy:model-mediated. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# model-mediated check

**Meaning:** A semantic assessment whose result depends on a model interpreting content or intent; it is distinct from fak's deterministic structural preflight and is not part of the preflight command's local fold.

## Do not conflate with

- **structural preflight:** A model-mediated check depends on model inference; structural preflight reaches its verdict from local grammar and adjudicator rules with no model in the loop.

## Query this identity

```console
$ fak disambiguation query "model-mediated check" --scope-kind "policy" --scope-value "model-mediated" --json
```

## Identity

- **Scope:** `policy:model-mediated`
- **Aliases:** model check
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

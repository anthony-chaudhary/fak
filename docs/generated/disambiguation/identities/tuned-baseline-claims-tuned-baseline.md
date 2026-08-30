---
title: "tuned baseline - claims:tuned-baseline identity"
description: "The best-practice alternative an operator would actually run, required as the decision-grade comparator for a performance headline. Scope: claims:tuned-baseline. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# tuned baseline

**Meaning:** The best-practice alternative an operator would actually run, required as the decision-grade comparator for a performance headline.

## Do not conflate with

- **naive baseline:** The tuned baseline reflects credible optimization; the naive baseline is only an untuned floor.
- **fak measurement arm:** The tuned baseline is the next-best alternative; the fak arm is the treatment being evaluated against it.

## Query this identity

```console
$ fak disambiguation query "tuned baseline" --scope-kind "claims" --scope-value "tuned-baseline" --json
```

## Identity

- **Scope:** `claims:tuned-baseline`
- **Aliases:** real baseline
- **Owner:** `claimcheck / claims`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `claims-source/1`

## Source witnesses

- `internal/claimcheck/claimcheck.go` (go-source, revision `claims-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-claims-source`)

[Back to the disambiguation index](../INDEX.md)

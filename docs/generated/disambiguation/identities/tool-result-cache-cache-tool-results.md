---
title: "tool-result cache - cache:tool-results identity"
description: "A fak-owned cache of completed tool-call results keyed by tool, argument hash, principal when isolated, and world-version epochs. Scope: cache:tool-results. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# tool-result cache

**Meaning:** A fak-owned cache of completed tool-call results keyed by tool, argument hash, principal when isolated, and world-version epochs.

## Do not conflate with

- **model KV cache:** The tool-result cache stores tool outputs and invalidates on modeled world changes; the KV cache stores per-token attention tensors.
- **radix prefix cache:** The tool-result cache looks up effect-safe tool calls; the radix cache longest-prefix-matches token sequences to reusable model snapshots.
- **provider prompt cache:** The tool-result cache is kernel-owned and directly invalidated by fak epochs; provider prompt-cache entries are externally owned and observed through usage accounting.

## Query this identity

```console
$ fak disambiguation query "tool-result cache" --scope-kind "cache" --scope-value "tool-results" --json
```

## Identity

- **Scope:** `cache:tool-results`
- **Aliases:** vDSO cache
- **Owner:** `vdso / vdso`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `cache-source/1`

## Source witnesses

- `internal/vdso/vdso.go` (go-source, revision `cache-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-cache-source`)

[Back to the disambiguation index](../INDEX.md)

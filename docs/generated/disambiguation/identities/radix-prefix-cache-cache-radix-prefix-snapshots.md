---
title: "radix prefix cache - cache:radix-prefix-snapshots identity"
description: "A fak-owned radix tree that longest-prefix-matches namespaced token sequences to reusable KV snapshots under token and byte budgets. Scope: cache:radix-prefix-snapshots. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# radix prefix cache

**Meaning:** A fak-owned radix tree that longest-prefix-matches namespaced token sequences to reusable KV snapshots under token and byte budgets.

## Do not conflate with

- **tool-result cache:** The radix cache keys token-prefix paths and snapshot residency; the tool-result cache keys tool calls plus effect epochs.
- **model KV cache:** The radix cache is a multi-prefix lookup and residency index; a model KV cache is the live tensor state for one sequence.
- **provider prompt cache:** The radix cache is namespace-scoped, budgeted, and evicted by fak; provider prompt caching is controlled upstream and reported through cache read/write usage.

## Query this identity

```console
$ fak disambiguation query "radix prefix cache" --scope-kind "cache" --scope-value "radix-prefix-snapshots" --json
```

## Identity

- **Scope:** `cache:radix-prefix-snapshots`
- **Aliases:** radix cache, prefix cache
- **Owner:** `radixkv / radixkv`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `cache-source/1`

## Source witnesses

- `internal/radixkv/radixkv.go` (go-source, revision `cache-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-cache-source`)

[Back to the disambiguation index](../INDEX.md)

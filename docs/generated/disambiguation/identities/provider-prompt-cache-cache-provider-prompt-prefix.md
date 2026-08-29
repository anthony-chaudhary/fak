---
title: "provider prompt cache - cache:provider-prompt-prefix identity"
description: "An upstream provider-owned prompt-prefix reuse service observed as cache-read and cache-creation token accounting with provider TTL and pricing rules. Scope: cache:provider-prompt-prefix. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# provider prompt cache

**Meaning:** An upstream provider-owned prompt-prefix reuse service observed as cache-read and cache-creation token accounting with provider TTL and pricing rules.

## Do not conflate with

- **tool-result cache:** Provider prompt caching reuses model input prefixes outside fak; the tool-result cache locally serves completed tool outputs under effect-aware invalidation.
- **model KV cache:** Provider prompt cache state is not directly addressable tensor memory in fak and cannot be edited like the kernel-owned KV cache.
- **radix prefix cache:** Provider cache lifetime and identity are upstream contracts; the radix prefix cache is fak-owned, namespace-keyed, and explicitly budgeted and evicted.

## Query this identity

```console
$ fak disambiguation query "provider prompt cache" --scope-kind "cache" --scope-value "provider-prompt-prefix" --json
```

## Identity

- **Scope:** `cache:provider-prompt-prefix`
- **Aliases:** provider cache, prompt cache
- **Owner:** `gateway / gateway`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `cache-source/1`

## Source witnesses

- `internal/gateway/cache_pricing.go` (go-source, revision `cache-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-cache-source`)

[Back to the disambiguation index](../INDEX.md)

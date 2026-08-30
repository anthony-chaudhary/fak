---
title: "model KV cache - cache:model-attention identity"
description: "Kernel-owned per-layer attention key/value tensors indexed by token position and invalidated or rewritten when the model sequence changes. Scope: cache:model-attention. Includes its nearest contrasts, owner, lifecycle, freshness, and exact fak query command."
---

# model KV cache

**Meaning:** Kernel-owned per-layer attention key/value tensors indexed by token position and invalidated or rewritten when the model sequence changes.

## Do not conflate with

- **tool-result cache:** The KV cache contains model attention state, not completed tool outputs or world-versioned effects.
- **radix prefix cache:** A KV cache is one sequence's live attention state; the radix cache indexes token prefixes and references reusable snapshots across requests.
- **provider prompt cache:** The model KV cache is directly owned and mutable inside fak; provider prompt caching is an upstream reuse service exposed as billed token axes.

## Query this identity

```console
$ fak disambiguation query "model KV cache" --scope-kind "cache" --scope-value "model-attention" --json
```

## Identity

- **Scope:** `cache:model-attention`
- **Aliases:** KV cache, KVCache
- **Owner:** `model / model`
- **Lifecycle:** `current`
- **Rollout:** `on`

## Freshness

- **Verdict:** `fresh`
- **Reason:** `SOURCE_CURRENT`
- **Checked at:** `2026-08-17T00:00:00Z`
- **Probe:** `cache-source/1`

## Source witnesses

- `internal/model/kvcache.go` (go-source, revision `cache-source/1`, checked `2026-08-17T00:00:00Z`; probe `fak-disambiguation-cache-source`)

[Back to the disambiguation index](../INDEX.md)

---
title: "cache names"
description: "This map positions the current cache coverage backlog. Each entry names the exact repository symbol;"
---
# cache names

This map positions the current `cache` coverage backlog. Each entry names the exact repository symbol; the family label remains the broader domain and is not a substitute for the symbol.

- **`hicache`** — the exact `cache` symbol `hicache`; use this spelling for that operation rather than the undifferentiated family name.
- **`cachedcontent`** — the exact `cache` symbol `cachedcontent`; use this spelling for that operation rather than the undifferentiated family name.
- **`cachedlogits`** — the exact `cache` symbol `cachedlogits`; use this spelling for that operation rather than the undifferentiated family name.
- **`cachewrite1hmultiplier`** — the exact `cache` symbol `cachewrite1hmultiplier`; use this spelling for that operation rather than the undifferentiated family name.
- **`managedcacheactive`** — the exact `cache` symbol `managedcacheactive`; use this spelling for that operation rather than the undifferentiated family name.
- **`reasonunsupportedactivecachecapability`** — the exact `cache` symbol `reasonunsupportedactivecachecapability`; use this spelling for that operation rather than the undifferentiated family name.
- **`seencache`** — the exact `cache` symbol `seencache`; use this spelling for that operation rather than the undifferentiated family name.
- **`armedcache`** — the exact `cache` symbol `armedcache`; use this spelling for that operation rather than the undifferentiated family name.
- **`bookeduncachedtokens`** — the exact `cache` symbol `bookeduncachedtokens`; use this spelling for that operation rather than the undifferentiated family name.
- **`cachedfile`** — the exact `cache` symbol `cachedfile`; use this spelling for that operation rather than the undifferentiated family name.
- **`cachedinput`** — the exact `cache` symbol `cachedinput`; use this spelling for that operation rather than the undifferentiated family name.
- **`cacheroot`** — the exact `cache` symbol `cacheroot`; use this spelling for that operation rather than the undifferentiated family name.
- **`cachetier`** — the exact `cache` symbol `cachetier`; use this spelling for that operation rather than the undifferentiated family name.
- **`cachevaluesavingspricing`** — the exact `cache` symbol `cachevaluesavingspricing`; use this spelling for that operation rather than the undifferentiated family name.
- **`guardinfocacheattribution`** — the exact `cache` symbol `guardinfocacheattribution`; use this spelling for that operation rather than the undifferentiated family name.
- **`guardmanagedcacheinputs`** — the exact `cache` symbol `guardmanagedcacheinputs`; use this spelling for that operation rather than the undifferentiated family name.
- **`inducedcachecreationtokens`** — the exact `cache` symbol `inducedcachecreationtokens`; use this spelling for that operation rather than the undifferentiated family name.
- **`minimaxkvcache`** — the exact `cache` symbol `minimaxkvcache`; use this spelling for that operation rather than the undifferentiated family name.
- **`readvcachetelemetry`** — the exact `cache` symbol `readvcachetelemetry`; use this spelling for that operation rather than the undifferentiated family name.
- **`treecache`** — the exact `cache` symbol `treecache`; use this spelling for that operation rather than the undifferentiated family name.
- **`microcachedemo`** — the exact `cache` symbol `microcachedemo`; use this spelling for that operation rather than the undifferentiated family name.


### Go build-cache lifecycle symbols

| Symbol | Canonical meaning | Distinct from |
|---|---|---|
| **`GoCacheRootFromEnv`** | Resolves the ambient Go build-cache root from `GOCACHE`, falling back to the platform user cache directory plus `go-build`; `GOCACHE=off` yields no managed root. | It selects the cache location only; it does not scan, classify, or remove entries. |
| **`SweepGoCache`** | The `internal/treedoctor` owner operation that scans the resolved ambient Go build cache, classifies only owner-approved stale entries as reclaimable, and mutates them only when its explicit `apply` argument is true. Callers such as storage-pressure use `apply=false` for a read-only report. | It is not a second retention policy and is not a generic Go-cache cleaner; active, held, recent, unknown-size, scan-error, incomplete, and otherwise ineligible entries remain non-reclaimable. |
| **`GoCacheReport`** | The provenance-bearing result from `SweepGoCache`, separating observed bytes from owner-approved reclaimable bytes and carrying incomplete, truncation, timeout, error, hold, active, and apply-state evidence. | Observed bytes are measured storage, not eligibility; reclaimable totals are complete only when the report says they are complete. |


### fak_gateway_inference_cached_prompt_hit_ratio (provider-cache turn ratio)

fak_gateway_inference_cached_prompt_hit_ratio is the gateway Prometheus gauge dividing provider prompt-cache-hit turns by all served model turns, reported as zero before the first turn.

**Distinct from:** It measures the fraction of turns with any provider cache hit; it is not fak_gateway_inference_cached_prompt_tokens_total, which accumulates the number of cached prompt tokens.

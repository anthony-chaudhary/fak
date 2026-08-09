---
title: "External-engine cache capability inventory (SGLang, vLLM, llama.cpp, Ollama, LM Studio)"
description: "Completion witness for DEFAULT-ENABLEMENT-NEXT-50 item 31 — one row per external engine, each verdict drawn from the closed cache-capability vocabulary, grounded in what the tree already proves (or marked unknown by honest absence of evidence)."
---

# External-engine cache capability inventory

Backlog row (see [`DEFAULT-ENABLEMENT-NEXT-50.md`](DEFAULT-ENABLEMENT-NEXT-50.md), line 138):

> **External engines — Build a cache capability inventory for SGLang, vLLM,
> llama.cpp, Ollama, and LM Studio.** Default/evidence target: *each adapter row
> says passive observe, active warm, exact evict, prefix clone, paged KV, or
> unknown.*

Tracked as issue **#1549** (parent epic **#1490**). This is the first row of the
"External engines" lane and the foundation the rest of the lane reads from (the
`engine.CacheCapability` contract #1550, the observation adapters #1551–#1553).

The machine-read row-set lives beside this note at
[`external-engine-cache-capability-inventory.jsonl`](external-engine-cache-capability-inventory.jsonl);
the focused witness test in `internal/enginecache/capability_inventory_test.go`
loads that JSONL and fails if any engine is missing a row or any verdict falls
outside the closed vocabulary.

## Closed capability vocabulary

Every verdict is exactly one of: `passive observe`, `active warm`, `exact evict`,
`prefix clone`, `paged KV`, `unknown`. No free-text verdict is allowed (the test
enforces membership). A verdict states **what fak can prove it does with that
engine's cache today** — not the engine's internal mechanism, and not what a
future adapter might add.

## Inventory

| Engine | Verdict | Why (fak-proven capability) | Evidence anchor |
|---|---|---|---|
| **SGLang** | `passive observe` | fak observes RadixAttention prefix reuse; its only cache-control primitive is a whole-radix reset, not exact-span eviction. | `internal/engine/sglang.go` (`meta_info.cached_tokens` + radix-residency poll → `PrefixResidencyIndex`); `internal/enginecache/enginecache.go` (`flush_cache` whole-radix reset, `SupportsExactSpan(EngineSGLang)=false`). |
| **vLLM** | `passive observe` | fak observes paged-block residency via the KV-event stream; its only cache-control primitive is a whole-prefix reset, not exact-span eviction. | `internal/engine/vllm.go` (KV-event subscription: `BlockStored`/`BlockRemoved`/`AllBlocksCleared`); `internal/enginecache/enginecache.go` (`reset_prefix_cache` whole-prefix reset, `SupportsExactSpan(EngineVLLM)=false`). |
| **llama.cpp / llama-server** | `unknown` | No in-tree cache observation adapter. It appears only as a pluggable on-device runtime and as a prefill *baseline*, neither of which observes or controls its cache. | `internal/engine/on_device.go` (named as an `OnDeviceRuntime`, no cache surface); `internal/benchscore` (prefill throughput baseline only). Observation adapter is item 35 / #1553. |
| **Ollama** | `unknown` | No in-tree cache observation adapter. Named only as a pluggable local daemon and as an "Ollama-style" model-pull alias. | `internal/engine/on_device.go` (local daemon runtime, no cache surface); `internal/devindex/verbs.go` (`fak model pull` alias only). |
| **LM Studio** | `unknown` | No in-tree adapter, observation lane, or cache-control surface at all. | Named only by `DEFAULT-ENABLEMENT-NEXT-50` row 31. Unknown by honest absence of evidence, not a guessed capability. |
| **LMCache** | `unknown` | LMCache persists and reuses KV across CPU RAM, local storage, and remote backends, but fak has no adapter, observation feed, or control surface. | Upstream `LMCache/LMCache@4521c3f9f1b8` README documents the external tier and vLLM integration; in-tree absence keeps the verdict unknown rather than inferring support. |

## Provenance separation

These are all **engine-kernel** (external-engine KV) facts. They are not provider
dollar-cache rebates, not context-window mechanisms, and not forecast planes — the
JSONL carries `"provenance":"engine-kernel"` on every row so a report never blends
an observed engine-KV signal into the provider or forecast lane. vLLM's internal
PagedAttention (paged KV) is recorded as an **engine mechanism**, deliberately kept
separate from fak's proven verdict (`passive observe`): fak fronting an engine that
uses paged KV is not the same as fak controlling paged KV.

## Cold-path correctness

This inventory arms **no** active cache behavior. Every verdict is `passive
observe` or `unknown`; no `active warm`, `exact evict`, or `prefix clone` is
claimed for any engine. The whole-prefix/whole-radix reset that SGLang and vLLM do
expose is the safe over-invalidation superset (`enginecache` degrades to it and
records `SupportsExactSpan=false`), so a cache miss always re-sends full required
context. Cold-path correctness is therefore trivially preserved: there is no hit
this inventory lets fak depend on.

## Generation classification

Horizon: **`gen/second-next`** (`Generation G2 - Second Next Gen`), per the
committed program map
[`GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md`](../notes/GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md)
which lists **#1549–#1558** ("Cross-engine and architecture options") in the
`gen/second-next` band and names *"capability inventory rows"* as the exact
promotion evidence for that band. This is a citation of the repo's own
classification, not a guess.

- **Promotion evidence** (`gen/second-next` → `gen/next`): this inventory *is* the
  "cache capability row" the map requires as the minimum evidence to split the
  External-engines lane into runnable `gen/next` adapter work; paired with a
  per-engine cold-path correctness witness (item 38 / #1538), the lane can promote.
- **Demotion / retirement evidence**: an `unknown` row with no cheap re-witness
  path (LM Studio has zero in-tree evidence) stays `unknown`; if a downstream
  adapter (#1551–#1553) proves a capability and this inventory is not updated in
  lockstep, the stale row is demotion evidence against trusting the inventory.
- **Invalidating assumption**: the verdict is fak's *proven-today* capability, not
  the engine's internal mechanism. If the #1551–#1553 observation adapters land,
  SGLang/vLLM may move beyond `passive observe` and llama.cpp may leave `unknown`
  — this row-set must be re-witnessed then, or it will misreport capability as the
  lane advances. The vocabulary is also assumed closed at exactly the six listed
  terms (#1550's `engine.CacheCapability` contract must keep the same terms so the
  two do not drift).

## Witness

`internal/enginecache/capability_inventory_test.go`
(`TestExternalEngineCacheCapabilityInventory`) loads the JSONL and asserts (a) a
row exists for each of SGLang, vLLM, llama.cpp, Ollama, LM Studio, and LMCache, (b) every
verdict is a member of the closed vocabulary, and (c) the SGLang/vLLM rows do not
claim `exact evict`, tied to the live `enginecache.SupportsExactSpan(...) == false`
fact so the inventory cannot drift from the code it summarizes. Captured `go test`
output is the repo witness.

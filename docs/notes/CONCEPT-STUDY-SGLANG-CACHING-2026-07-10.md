---
title: "study-repo: SGLang (RadixAttention + HiCache) → fak (2026-07-10)"
description: "A study-repo pass over SGLang's KV-cache subsystem — RadixAttention prefix tree, HiCache L1/L2/L3 tiering, SWA/Mamba hybrid eviction, paged allocator, cache-aware scheduling — extracting candidate cache borrows and witnessing each against fak's KV-cache program (epic #2236). Three genuinely-actionable gaps filed (#3889 namespace isolation, #3890 eviction-strategy seam, #3891 galloping match); the rest already owned, a fak differentiator, deferred-until-batching, or GPU/SWA-inapplicable."
---

# study-repo: SGLang → fak (2026-07-10)

A `study-repo` pass over **[sgl-project/sglang](https://github.com/sgl-project/sglang)**, pinned at
**`7090a49198ae74575cbcadc6ca67fab6e7b897f0`**. Clone arrived shallow (depth-1); the completeness
reader ran `git fetch --unshallow` (15,065 commits) so the recent-commit signal below is real
history, not inference. Scope: the **KV-cache** surface — `python/sglang/srt/mem_cache/**` (radix
tree, HiCache host/storage tiers, SWA/Mamba hybrid, paged allocator) and the cache-aware parts of
`managers/schedule_policy.py` — the part that maps onto fak's own KV-cache value program.

## Why SGLang is a direct reference

`internal/radixkv` is *literally* "SGLang's RadixAttention rebuilt over fak's kernel-owned KV cache"
(package doc, `internal/radixkv/radixkv.go:1-47`). So this is the closest possible reference: the
same data structure, measured head-to-head on cache-hit-rate. The structural mapping used
throughout: **SGLang radix node = fak KV prefix span; SGLang HiCache L1/L2/L3 = fak's
`internal/cachemeta` + `internal/l3region` residency ladder; SGLang `extra_key`/`session_ids` =
the tenant/adapter identity fak milestone #2236 wants owned.** The milestone this feeds —
*"The KV cache value is owned, observed & 2x"* (epic **#2236**: #2239 cost-aware evict [shipped],
#2666 wiring [shipped], #2674 tenant fairness [open]).

## License gate

SGLang is **Apache-2.0**; fak is **Apache-2.0**. Every candidate is Python → Go, so all are
**`inspire`** (clean-room reimplementation, source cited at `path:line@7090a49`). No bytes vendored.

## What was read (code, not the pitch)

Five parallel readers, each grounding candidates at a real `path:line`, then a per-candidate
witness against fak (raw grep + direct source read of `internal/radixkv`, `internal/compute`,
`internal/cachemeta`, `internal/l3region`, `internal/model`, `internal/agent`, `cmd/fak/dispatch_*`):

- **Tiering / host pool / L3 storage** — `mem_cache/pool_host/{base,common,mha}.py`,
  `hicache_storage.py`, `storage/backend_factory.py`, `cache_controller.py`.
- **Cache-aware scheduling** — `managers/schedule_policy.py`, `schedule_batch.py`, `base_prefix_cache.py`.
- **Paged allocator / memory pool** — `mem_cache/allocator/{base,token,paged,swa}.py`, `utils.py`,
  `evict_policy.py`, `radix_cache.py`, `cpp_utils/hash_binding.cpp`.
- **SWA / Mamba hybrid eviction** — `swa_radix_cache.py`, `mamba_radix_cache.py`, `swa_memory_pool.py`,
  `pure_swa_radix_cache.py`, `common.py`.
- **Tests + completeness sweep + git history** — the `test/registered/**/mem_cache` suite,
  `events.py`, `session_radix_cache.py`, `registry.py`, `unified_radix_cache.py`, `cpp_radix_tree/**`.

## Witness table

Verdict legend: **PRESENT** (fak already has it) · **PARTIAL** (seam exists, gap is real) ·
**ABSENT** (missing) · **N/A** (GPU/SWA/distributed-specific, or architecturally inapplicable to
fak's full-prefix-KV radix design).

### Cluster A — radix tree identity & matching (`internal/radixkv`)

| # | Candidate (sglang source) | Verdict | fak evidence |
|---|---|---|---|
| A1 | **Namespace/holder-scoped identity** — `RadixKey`=(tokens,`extra_key`); `session_ids` holder set, free on last-holder-drop + bounded late-arrival tombstone (`radix_cache.py`, `session_radix_cache.py:37-121`) | **ABSENT → filed #3889** | nodes keyed by token only (`radixkv.go:60-63`); `Lookup/Insert/MatchLen` take only `tokens []int`; one shared tree (`inkernel_planner.go:175`); single-owner `refs int` (`:77`). Unblocks #2674. |
| A2 | **Eviction as a `get_priority` strategy registry** (lru/lfu/fifo/mru/filo/priority/slru), fail-closed string knob; SLRU 2-segment scan resistance (`evict_policy.py:10-65`, `utils.py:55`) | **PARTIAL → filed #3890** | closed 2-value enum in a switch (`radixkv.go:82-90,140-147,362-368`); cost-aware IS freq-aware (`:401-433`) but no open seam, no protected-segment guarantee. |
| A3 | **Gallop+binary-search edge compare** — O(log n) not O(n) per run (`radix_cache.py:162`) | **PARTIAL → filed #3891** | token-by-token run compare (`radixkv.go:164-166`); one long shared edge = O(n) int compares — the exact big-preamble agent workload. |
| A4 | Page-aligned floor-to-page matching (`radix_cache.py:136,162`) | **N/A** | radixkv stores the **full-prefix** `model.KVCache` per node (doc `:39-46`), token-granular via proven Clone/Evict — not paged slabs. fak's paging is a separate layer (`internal/model/pagedkv.go`). |
| A5 | Content-addressed per-page **hash chain** — SHA(prior‖page), Merkle prefix identity (`hash_binding.cpp:62`, `utils.py:106`) | **ABSENT (deferred)** | radixkv identity is structural (in-memory tree), not a rolling prefix hash. Enables L3 persist / cross-process prefix identity — a bigger feature; note, not filed. |
| A6 | Cost-aware eviction (hits+recency+lease) (`evict_policy.py`) | **PRESENT** | `costAwareLeaf` → `compute.PickEvictionVictim(spans)` with Hits/LastUsed/Leased (`radixkv.go:401-443`). Shipped #2239/#2666. |
| A7 | — | **PRESENT (fak differentiator)** | `EvictNode` = policy eviction of a *named span* regardless of LRU (poison/quarantine, `radixkv.go` doc `:32-37`). An LRU radix tree structurally cannot do this. |

### Cluster B — cache-aware scheduling / admission

| # | Candidate (sglang source) | Verdict | Note |
|---|---|---|---|
| B1 | LPM reuse-depth sort `-num_matched_prefix_tokens`, `inf` deprioritize sentinel (`schedule_policy.py:316`) | **PARTIAL (deferred)** | fak dispatch doesn't order by warm residency; maps to fleet worker-affinity — larger surface than radixkv. |
| B2 | **In-batch shared-prefix dedup / thundering-herd** — admit one of N siblings sharing an uncached prefix, defer rest one tick (`schedule_policy.py:271-309`) | **ABSENT (deferred)** | genuinely absent, but **premature**: fak batch-decode is box-gated at B==1; the "cross-request coalescer" is an unbuilt NOTE (`inkernel_planner.go:1232`) and is continuous-batch decode, not prefix dedup. Revisit when B>1 lands. |
| B3 | Admission eviction-guard: ref-count pin + RAII release + `protected_size` vs `evictable_size` (`schedule_policy.py:849`, `base_prefix_cache.py`) | **PARTIAL** | fak has the pin (`Lookup` refs++/`Done`; leaf refs>0 never evicted, `radixkv.go:347`) but no protected/evictable accounting split, no post-lock budget recheck. |
| B4 | In-cache-token accounting — charge only the uncached suffix (`schedule_policy.py:1017`) | **PARTIAL (deferred)** | `internal/compute/kvadmission` is value-aware (W-TinyLFU-ish) but not a prefill-suffix cost discount; different serving model. |
| B5 | Cost-capped FCFS fallback when queue>128 (`schedule_policy.py:238`) | **N/A** | no LPM queue sort to degrade. |
| B6 | Routing-key locality from the *running* batch, `(bucket,-count,key)` (`schedule_policy.py:399`); cache-aware DFS ordering (`:339`) | **PARTIAL (deferred)** | co-schedule same-context work → fleet dispatch affinity; larger surface. |

### Cluster C — tiering / host pool / L3 (`internal/cachemeta`, `internal/l3region`)

| # | Candidate (sglang source) | Verdict | Note |
|---|---|---|---|
| C1 | 3-tier L1/L2/L3 (HBM/DRAM/CXL), demote-not-evict, TTL (`pool_host/*`, `cache_controller.py`) | **PRESENT** | cachemeta ladder; #2236 family (#2671 tier-action ladder, #2673 TTL-pins). |
| C2 | Lazy backend registry — `register_backend(name,module,class)` loader closure, optional deps (`storage/backend_factory.py`) | **PARTIAL (deferred)** | pluggable L3 storage-backend registry with lazy load; ties to `agentl3` label. Note. |
| C3 | `is_stride_page_aligned` gates O_DIRECT / zero-copy ptr-meta vs flat-page fallback (`pool_host/mha.py`) | **PRESENT-ish** | cachemeta `ShareKind` (copy/mmap/CXL/RDMA/dma-buf) is the policy-level analog. |
| C4/C5 | Layout-as-abstraction (4 host layouts × kernel/direct IO); PP-synced sizing; `cudaHostUnregister` teardown | **N/A** | GPU-kernel / distributed-specific. |
| C6 | Layer-granular transfer overlap — `LayerDoneCounter`/`LayerLoadingEvent` (`cache_controller.py`) | **Deferred** | overlap L1↔L2↔L3 with compute per layer; bigger. |
| C7 | `MetadataCache` client-side file-existence stat cache, #29716 (`hicache_storage.py:320`) | **Deferred** | cut L3 `stat`/`exists` round-trips. Note. |
| C8 | KV-event stream — `BlockStored/Removed`, parent-hash chain, tier tag, `take_events` drain (`events.py:34-128`) | **Deferred** | for external KV-aware routers; `agentic-serving`/`agentl3` scope; bigger. |

### Cluster D — SWA / Mamba hybrid — **N/A for fak now** (fak serves no sliding-window/SSM model, does no windowed suffix eviction)

Recorded as design observations, not gaps: **dual-tier tombstone eviction** (two LRU lists + a
`tombstone` flag → free a cheap short-lived tier without dropping the lookup path,
`swa_radix_cache.py:584`); **tombstone-leaf invariant** (only internal nodes may be tombstoned,
`:1310`); **window-anchored matching** as a *validity gate on a partial hit* — reject reuse that
needs already-reclaimed state instead of faulting later (`:887`) — the one genuinely
model-agnostic idea here; **heterogeneous per-entry residency** with a lock-ordering invariant
(`full_lock_ref >= mamba_lock_ref`, coarse tier can't outlive the fine tier it depends on,
`mamba_radix_cache.py:78`); **two-ended `UnifiedKVPool`** with virtual-id indirection so
compaction never rewrites references (`unified_memory_pool.py`). If fak ever adds a
cheap-window/expensive-summary two-lifetime context entry, this cluster is the reference.

### Cluster E — allocator / memory pool (`internal/model/pagedkv.go`)

| # | Candidate | Verdict | Note |
|---|---|---|---|
| E1 | Paged free-list alloc by slice, reserved sentinel slot (`allocator/paged.py:149`) | **PRESENT/PARTIAL** | fak `pagedkv.go`, `paged_evict.go` (#277/#34). |
| E2 | Batched/deferred free — free-group + lazy merge-sort on pressure (`allocator/base.py:69`) | **PARTIAL** | fak already batch-frees at the Vulkan buffer layer (`vulkan_shim.cpp` `g_batchFreed`); radix lazy-sort refinement is marginal. |
| E3/E4 | Two-ended arena, multi-ended eager compaction | **N/A** | hybrid Mamba/SWA co-residency. |

## Robustness invariants worth stealing (from the test suite)

- **Post-lock budget recheck** — `pop_preallocated` re-checks the token budget *after* taking the
  lock and `dec_lock_ref` on failure without `_pre_alloc` (`test_decode_radix_lock_ref.py`). A
  classic admission-race fix; fak's `Lookup`→prefill lease path could adopt the same discipline.
- **inc/dec lock_ref balance** across success/failure transfer paths; root lock is a no-op pinned
  at 1. Mirrors fak's `TestLookupLeaseMustBeReleased`.
- **Force-miss slices, not allocates** — `zero_match_result` preserves dtype/device, zeroes length.

## Filed this pass

- **#3889** `feat(radixkv)`: namespace/holder-scoped prefix isolation (ABSENT; unblocks #2674).
- **#3890** `feat(radixkv)`: open the closed eviction enum into a `get_priority` strategy seam +
  register scan-resistant SLRU (PARTIAL).
- **#3891** `perf(radixkv)`: gallop+binary-search the edge run-compare (PARTIAL, perf-only).

## Honest limits of this pass

- radixkv deliberately stores the **full-prefix KVCache per node** (memory for a proven-primitive
  guarantee, doc `:39-46`), so every *paged-pool* borrow (A4, E-cluster) is architecturally N/A —
  not a gap. Correctly excluded rather than filed.
- The SWA/Mamba cluster is the sglang caching **frontier** (recent commits: unified FULL+SWA+Mamba
  tree over one byte pool #29678, HiCache write-back/metadata-cache/hugepage-SSD hardening), but
  none of it lands for fak until fak serves such a model. Deferred honestly, not forced.
- B2 (shared-prefix thundering-herd) is the most tempting agent-fleet borrow but is blocked on
  real B>1 batching; filing now would be premature. Left as the top revisit candidate.

_Uncommitted study artifact — not indexed, not committed._

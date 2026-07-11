---
title: "study-repo: Mooncake (Moonshot/Kimi) → fak (2026-07-10)"
description: "A study-repo pass over Moonshot AI's Mooncake KVCache-centric disaggregated serving store, extracting 15 candidate cache borrows and witnessing each against fak's KV-cache program (epic #2236). One genuinely-absent leaf (a count-min-sketch frequency admission doorkeeper); the rest already owned, superseded, or Go-GC-inapplicable."
---

# study-repo: Mooncake → fak (2026-07-10)

A `study-repo` pass over **[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)**, the
KVCache-centric disaggregated LLM-serving store behind Moonshot AI's Kimi. Full clone (whole
repo, complete history), pinned at **`4188e3ae9923b93a74bf82d67be6a56d2ffb20e9`** (HEAD
2026-07-10). Scope of this pass: the **`mooncake-store`** hot-cache / eviction / admission /
tenant-quota surface — the part that maps onto fak's own KV-cache value program.

## Why Mooncake is a direct reference

Mooncake is a **distributed KVCache pool** with prefix reuse, tiered residency (DRAM ↔ local
disk), demand-driven promotion, and multi-tenant admission — exactly the surface fak's milestone
*"The KV cache value is owned, observed & 2x"* (epic **#2236**, the KVBM family: #2239 cost-aware
evict, #2666 wiring, #2668 GDSF aging, #2669 hazard-rate, #2670 fanout, #2671 tier-action ladder,
#2672 value-aware admission, #2673 TTL-pins, #2674 tenant fairness) is building. The structural
mapping used throughout: **Mooncake object = fak KV prefix span; Mooncake DRAM/disk tiers = fak
L1/L2/L3 residency ladder; Mooncake `LocalHotCache` = a client-side prompt/prefix cache.** The
GPU/RDMA/NIXL transfer-engine code is not in scope — fak has no use for it.

## License gate

Mooncake is **Apache-2.0** (`LICENSE-APACHE`); fak is **Apache-2.0**. Integrate would be
license-legal, but every candidate is C++ → Go, so **all are `inspire`** (clean-room
reimplementation, source cited). No bytes vendored.

## What was read (code, not the pitch)

Three parallel readers over the load-bearing `mooncake-store` modules, each grounding candidates
at a real `path:line`, then a per-candidate witness against fak (`fak_feature_query` / raw grep /
direct source read of `internal/compute`, `internal/radixkv`, `internal/l3region`):

- **Admission / promotion / offload wiring** — `include/count_min_sketch.h`,
  `include/client_service.h`, `src/client_service.cpp`, `src/master_service.cpp`
  (`TryPushPromotionQueue`, `try_evict_or_offload`), `src/local_hot_cache.cpp`.
- **Eviction strategy + tenant quota** — `include/eviction_strategy.h`, `src/master_service.cpp`
  (`EvictionThreadFunc`, `BatchEvict`, `EvictTenantMemoryForQuota`), `src/tenant_quota.cpp`,
  `include/allocation_strategy.h`.
- **Technique sweep** — `include/utils/zstd_util.h`, `include/offset_allocator/offset_allocator.h`,
  `include/deadline_scheduler.h`, `include/pinned_buffer_pool.h`, `include/mmap_arena.h`.

**15 candidates, all grounded at a verified `path:line@4188e3ae`; witness: 1 PRESENT, 9 PARTIAL,
5 N/A (Go-GC-inapplicable or architecture-mismatch).** The three headline citations were
spot-read to confirm the code implements the technique (not a README claim): the
`admission_sketch_->increment(key) >= admission_threshold_` gate (`client_service.h:647`), the
token-checked publish under a single lock (`local_hot_cache.cpp:82-93`), and the CMS
increment/halving-decay (`count_min_sketch.h:25-79`) all verified verbatim.

## The decisive finding — fak already owns almost all of it

The witness stage checked *code*, and fak's KVBM family (epic #2236) has already landed the
eviction/tiering/pin/admission clusters as **closed** issues (#2239, #2666, #2671, #2672, #2673).
So Mooncake's eviction library, three-level pins, lease refresh, and tier-action ladder witnessed
**PRESENT or superseded**, and the allocator/arena/pool/scheduler techniques are **Go-GC-moot**
(fak has no hand-rolled memory allocator). What survives as a *genuine* gap is a single, precise,
ship-alone primitive fak does **not** have anywhere in `internal/compute`:

> **A count-min-sketch frequency *doorkeeper* that admits on repeated MISSES.** fak's shipped
> admission gate `DecideKVAdmission` (#2672, `internal/compute/kvadmission.go`) compares a
> candidate's value against the victim it would displace using **resident `Hits`** — the frequency
> of a span *already in the cache*. It has no memory of a key that is **not yet resident or was
> already evicted**: on a first miss a span has `Hits = 0`, and once `removeLeaf` nils a node its
> frequency is gone. Mooncake's `CountMinSketch` (`count_min_sketch.h`) is the complementary half:
> a sublinear-memory sketch that counts *requests for keys the cache does not hold*, so a key is
> admitted only after it has been **missed ≥ `admission_threshold_` (default 2)** times
> (`client_service.h:640-648`), with an 8-bit saturating counter and automatic halving-decay
> (`count_min_sketch.h:35-37,72-79`) so "frequent" means *recently* frequent. This is the classic
> TinyLFU doorkeeper the #2672 title gestures at ("W-TinyLFU-style") but did not build — #2672
> shipped the *victim-comparison rule*, not the *frequency sketch that survives eviction*.

That one is the only genuinely new leaf. Everything else is recorded below with an honest
PRESENT / superseded / N-A disposition.

## Filing plan (proposed — see "Status" below)

- **New leaf** under epic **#2236** (extends closed #2672): *count-min-sketch frequency-doorkeeper
  admission* — a pure `internal/compute/countminsketch.go` primitive (saturating counters +
  halving decay, zero deps, following the `kvcost_aging.go` extension discipline) plus a
  `DecideKVAdmissionDoorkeeper` arm that requires ≥K cross-eviction misses before a cold newcomer
  may displace a warm resident. Ship-alone, host-free replay-witnessable (the #2244 ledger corpus
  already drives `ReplayKVCache`).
- **Enrich #2674** (OPEN, tenant fairness): Mooncake's per-tenant quota is a concrete impl of the
  fairness lever — a **proportional largest-remainder effective quota** under oversubscription
  (`tenant_quota.cpp:42-117`) with a hard `used+reserved+new ≤ effective` admission gate
  (`:182-208`), and, on breach, **evict the offending tenant's *own* lease-expired footprint**
  before rejecting (`master_service.cpp:6297-6529`) — noisy-neighbor containment without touching
  quiet tenants.
- **Enrich #809** (OPEN, speculative next-turn warm): two mechanisms this epic's own design says the
  riskier speculative-prewarm siblings will need. (1) The **`HotCachePutToken{cache_epoch,
  key_generation}`** generation/epoch guard (`local_hot_cache.cpp:82-93,335-358`) — the
  "effect-witnessed invalidation" `internal/radixkv/prewarm.go` explicitly defers: an async fill
  that lands after its key was evicted/overwritten is dropped under the publish lock instead of
  resurrecting stale bytes. (2) The **4-gate promotion-on-hit** admission (frequency + DRAM
  high-watermark + in-flight dedup + global cap, `master_service.cpp:5248-5350`) as a concrete
  demand-tiering reference, contrasted with fak's *durability-classed* L3 promotion
  (`internal/l3region/promotion.go`), which deliberately rejects frequency as the L3 signal.

## Full candidate table

| # | subsystem | borrow (technique) | source `path:line@4188e3ae` | witness | route | disposition |
|---|---|---|---|---|---|---|
| 1 | hot-cache-admission | **CMS frequency doorkeeper** — admit after ≥K misses; sublinear cross-eviction frequency memory w/ halving decay | `client_service.h:640-648` + `count_min_sketch.h:14-86` | PARTIAL | inspire | **FILE new leaf** (extends #2672) |
| 2 | master-promotion | Second independent CMS gating LOCAL_DISK→DRAM promotion-on-hit | `master_service.cpp:5261` | PARTIAL | inspire | enrich #809 (same primitive as #1) |
| 3 | local-hot-cache | **`HotCachePutToken{epoch,generation}`** — drop stale async fill under the publish lock | `local_hot_cache.cpp:82-93,335-358` | PARTIAL (deferred) | inspire | enrich #809 |
| 4 | local-hot-cache | Deferred LRU touch — atomic `accessed` under shared_lock, drain under exclusive | `local_hot_cache.cpp:419-442,149` | N/A (radixkv single-threaded, no mutex) | inspire | DROP (arch mismatch) |
| 5 | master-eviction | Lease-on-read min-residency TTL — grant on write+read, refresh past half-life | `master_service.h:1079-1114` + `.cpp:2725` | PARTIAL | inspire | note (superseded by #2673 TTL-pins) |
| 6 | master-eviction | Two-pass near-LRU + **watermark hysteresis** — high-water trigger → low-water floor, `nth_element` cutoff | `master_service.cpp:6823-6899` + `master_config.h:43-45` | PARTIAL | inspire | note (candidate leaf; extends closed #2666) |
| 7 | master-eviction | Three-level pin: hard / soft(VIP,TTL) / normal | `master_service.h:914-917,1116-1127` | PRESENT | inspire | DROP (PRESENT — #2673) |
| 8 | tenant-quota | **Proportional largest-remainder effective quota** + evict-own-footprint-before-reject | `tenant_quota.cpp:42-208` + `master_service.cpp:6297-6529` | PARTIAL | inspire | **enrich #2674** |
| 9 | master-tiering | Promotion-on-hit + lazy offload-on-evict (4-gate; spill only under pressure) | `master_service.cpp:2727-2745,5248-5350,6345-6389` | PARTIAL | inspire | enrich #809 / note l3region |
| 10 | local-hot-cache | Single-flight dedup of concurrent fills + in-flight cap | `master_service.cpp:5287-5307` | PARTIAL | inspire | note (≈ #2037 affinity single-flight) |
| 11 | store-safety | Compression-before-store + decompressed-size zip-bomb guard | `zstd_util.h:9-27,180-200` | PARTIAL | inspire | note (blobfs safety, out of KV scope) |
| 12 | allocator | Offset allocator — O(1) bin free-list, serializable | `offset_allocator.h:23-27,259-332` | N/A (Go GC) | — | DROP |
| 13 | infra | Min-heap deadline scheduler for TTL sweeps | `deadline_scheduler.h:26-129` | N/A | — | DROP |
| 14 | infra | Bounded slab/buffer pool (`sync.Pool` + hard cap) | `pinned_buffer_pool.h:20-103` | N/A (low) | — | DROP |
| 15 | allocator | Lock-free bump arena w/ generational reset | `mmap_arena.h:14-101` | N/A (Go GC) | — | DROP |

## Wiring-honesty flags found in the source (recorded so a later re-witness trusts the right lines)

- Mooncake's own `eviction_strategy.h` LRU/FIFO classes are **defined + unit-tested but NOT wired**
  into the master — forward-decl (`master_service.h:54`) + comment only. Production eviction is the
  lease-timeout near-LRU `BatchEvict` scan. (So candidate #4/#7 map to the *real* path, not the
  unused library.)
- `AllocationStrategyType::LOCAL_FIRST` falls back to `RandomAllocationStrategy`
  (`allocation_strategy.h:785-786`); `CxlAllocationStrategy::AllocateFrom` is a stub
  (`:764-768`). The allocation-placement surface is not a clean borrow and is out of scope here.
- Tenant-quota logic is inert unless `enable_multi_tenants_` (`master_service.cpp:6301`).

## Honest limits

- The witness is **lexical + a snapshot**: fak-side grep/`fak_feature_query` is substring ranking
  (guarded here with direct source reads against a false-ABSENT), true only as of 2026-07-10.
  Re-witness before acting on an old row.
- **Candidate #1's value is a claim about a gap, not a measured win.** That fak's admission has no
  cross-eviction frequency memory is witnessed in the code (`kvadmission.go` reads resident `Hits`;
  no CMS primitive exists under `internal/compute` — grep-confirmed). Whether the doorkeeper
  *improves* the good-decision-ratio is a replay experiment the leaf must run, not a result of this
  study.
- Five candidates are dropped as **Go-GC-moot** (offset allocator, bump arena, buffer pool,
  deadline scheduler): they solve C++ manual-memory problems Go's runtime already owns. Recorded
  for completeness, not as backlog.
- License reading is a good-faith Apache-2.0-vs-Apache-2.0 compatibility check, not legal advice.

## Companions

- KVBM family (witness targets): epic **#2236**; closed **#2239 / #2666 / #2671 / #2672 / #2673**;
  open **#2674** (tenant fairness), **#809** (speculative next-turn warm).
- Design dossiers cross-read: [hazard-rate reuse #2669](RESEARCH-kv-hazard-rate-reuse-2669.md),
  [tenant fairness floors #2674](RESEARCH-kv-tenant-fairness-floors-2674.md),
  [min-cost tier action #2671](RESEARCH-kv-min-cost-tier-action-2671.md).
- Skills: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) (this pass),
  [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) (the witness + file back-half).
- Sibling study notes: [NVIDIA dynamo](CONCEPT-STUDY-DYNAMO-2026-07-08.md),
  [LMCache](CONCEPT-STUDY-LMCACHE-2026-07-08.md),
  [MinIO MemKV](CONCEPT-STUDY-MINIO-MEMKV-2026-07-08.md),
  [field-borrow landscape](CONCEPT-FIELD-BORROW-LANDSCAPE-2026-07-08.md).
</content>
</invoke>

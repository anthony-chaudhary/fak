---
title: "study-repo: LMCache → fak (2026-07-08)"
description: "A study-repo pass mapping LMCache's KV-cache-management patterns onto fak: 9 PRESENT, 28 PARTIAL, 0 ABSENT, filing epic #3366 with 23 borrow leaves."
---

# study-repo: LMCache → fak (2026-07-08)

A `study-repo` pass over **[lmcache/lmcache](https://github.com/lmcache/lmcache)**, the KV-cache
management layer for scalable LLM inference (the vLLM/Dynamo KV offload+reuse layer). Full clone
(whole repo, complete history — per the operator's "copying whole repo"), pinned at
**`aaf7c0d305eab4551ae5a81711d0ed75bb2f5c80`** (HEAD 2026-07-08, "fix macOS FFmpeg" #4047).

## Why LMCache is a direct reference

LMCache is a *KV-cache-management* layer; fak is a *prompt / tool-page / session / cache-value*
layer. fak does not run inference or touch GPU KV tensors, so the transferable surface is the
**cache-management pattern one level up**: chunk/prefix keying, tiered fall-through + promotion,
admission/eviction/pinning policy, backend health & routing, cache observability (token-level hit
rate, realized lifecycle), and offline savings projection — **not** the CUDA/NIXL/tensor code. The
structural mapping used throughout: KV chunk = prompt/tool-page chunk, GPU/CPU/disk/remote tier =
hot-cache/spill/blob/remote tier, cache entry = cached prompt-prefix or session artifact. Maps onto
fak milestones **M2** *KV cache owned/observed/2x*, **M3** *Disaggregated serving*, **M1** *Durable
sessions*, and the `agentic-serving` area; feeds the **#2236** superset concept-by-concept ranking.

## License gate

LMCache is **Apache-2.0**; fak is **Apache-2.0**. Integrate would be license-legal, but every
candidate is Python/Rust/C++ → Go, so **all are `inspire`** (clean-room reimplementation, source
cited). **No bytes vendored.**

## What was read (code, not the pitch)

Nine parallel readers over the load-bearing subsystems, each grounding candidates at a real
`path:line`, then a per-candidate witness against fak (`fak_feature_query` + `fak index` + raw grep
+ `gh` backlog dedup):

- **cache-engine core** — `lmcache/v1/cache_engine.py` (84 KB), `cache_interface.py`, `manager.py`, `ec_engine.py`, `event_manager.py`.
- **token-db / chunking / hashing** — `lmcache/v1/token_database.py`.
- **storage tiers & backends** — `lmcache/v1/storage_backend/` (`abstract_backend`, `storage_manager`, `local_cpu_backend`, `local_disk_backend`, `remote_backend`, `path_sharder`).
- **eviction / admission / pin** — `cache_policy/` (`base_policy`, `lru`, `lfu`, `fifo`, `mru`), `pin_monitor.py`, `memory_management.py`, `memory_allocators/`.
- **audit / trust / health** — `storage_backend/audit_backend.py`, `connector/audit_connector.py`, `health_monitor/`.
- **observability metrics** — `lmcache/observability.py` (74 KB), `v1/mp_observability/`.
- **CacheBlend / serde / compression** — `v1/compute/blend/`, `v1/kv_codec/`, `storage_backend/naive_serde/`.
- **controller / coordinator / lookup** — `v1/cache_controller/`, `v1/mp_coordinator/`, `v1/lookup_client/`.
- **config / simulator / decoupled daemon** — `v1/config.py`+`config_base.py`, `tools/cache_simulator/`, `v1/standalone/`+`offload_server/`.

**~56 grounded candidates → deduped to 37 → witnessed against fak: 9 PRESENT, 28 PARTIAL, 0 ABSENT.**
(0 fully absent is the honest headline: fak's cache surface — `cachevalue`/`vcache`/`enginecache`/
`cachemeta`/`radixkv`/`ctxmmu`/guard `managed_cache` — is already deep.)

## The decisive finding — fak already owns the fundamentals

Nine candidates witnessed **PRESENT** and were dropped (fak already ships the mechanism): partial-
prefix suffix keying (`radixkv` Lookup/Insert), cross-process stable content keys (`crypto/sha256`,
fixed-arity NUL-separated tuples), non-content key-dim collapse (`ScopeFleet`, topology excluded),
pluggable eviction (`radixkv.EvictionPolicy`), alloc-deadlock discipline (preemptor + bounded
OOM-retry), read-back integrity (`blobfs`/`blobhttp` verify-on-read, fail-closed), interval-delta
health (`serving_metrics` goodput), cross-config poisoning gate (`cachemeta.MaterializationKey`),
and fleet warm-prefix routing (`gateway/residency_router.go`).

## Filed this pass — epic #3366 + 23 leaves (all PARTIAL borrows)

Every leaf carries a source `path:line@aaf7c0d3`, a fak seam `path:line`, the witnessed "what fak does
today", and a first checkable step. Grouped:

- **Keying & tiering:** #3378 prefix-hash-chain keying · #3379 cross-tier prefix assembly · #3380 non-blocking store · #3381 backpressure partial store · #3382 multi-root spill sharder.
- **Eviction, pinning & reservation:** #3384 two-axis evictability · #3385 pin-timeout reaper · #3386 atomic batch reserve · #3387 watermark ratio-eviction.
- **Trust / integrity:** #3388 audit-wrapper backend · #3395 versioned CRC blob envelope.
- **Cache observability:** #3390 cacheability-vs-realized · #3367 hit-rate histogram · #3391 token-rate counter-ratio · #3392 shadow-map lifecycle · #3393 evict→reuse-gap · #3394 pipeline self-meter · #3396 bloom reuse-potential · #3397 offline projection · #3398 savings-vs-budget curve.
- **Fleet health & config:** #3383 fleet degraded-mode · #3389 canary recovery · #3399 three-verdict config gate.

## Merged into existing issues (not re-filed)

Code-grounded enrichment comments, not new tickets (anti-re-file discipline):

- **#3143** LMCache CacheBlend non-prefix reuse ← `K2` segment-index + rolling any-offset match.
- **#3144** LMCache CacheGen SERDE/tiered ← `S3` per-tier fidelity/TTL budget table.
- **#1469** restore-on-access ← `T2` read-through cold→hot promotion.
- **#2853** cache-warm resume ← `D4` decoupled survivable cache daemon (no fate-sharing).
- **#2211** session pin/unpin ← `E6` operator pin API (local key resolution, refcount predicate).

## Full candidate table

| # | subsystem | borrow (technique) | source `path:line@sha` | witness | route | disposition |
|---|---|---|---|---|---|---|
| 1 | keying | Rolling prefix-hash chain — chunk key = hash(prev_prefix, chunk); exact prefix-hit length | `lmcache/v1/token_database.py:358-365@aaf7c0d3` | PARTIAL | inspire | filed #3378 |
| 2 | keying | Non-prefix reuse — segment index + rolling any-offset match, returns (stored,query) offset | `lmcache/v1/token_database.py:470-491 + multiprocess/modules/blend_v3.py:212-277@aaf7c0d3` | PARTIAL | inspire | enriched #3143 |
| 3 | keying | Partial-prefix skip — key only the new suffix, chunk-size alignment invariant | `lmcache/v1/token_database.py:404-431@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 4 | keying | Canonical fixed-arity inputs + seed-stable hash for cross-process key agreement | `lmcache/v1/token_database.py:250-267@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 5 | keying | Collapse non-content key dims (world_size->1) to widen fleet sharing | `lmcache/v1/token_database.py:234-248@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 6 | tiering | Cross-tier contiguous-prefix assembly, per-range tier ownership, break at first gap | `lmcache/v1/storage_backend/storage_manager.py:970 + cache_engine.py:1749@aaf7c0d3` | PARTIAL | inspire | filed #3379 |
| 7 | tiering | Read-through promotion — cold-tier hit async-copied to hot tier (source excluded) | `lmcache/v1/storage_backend/storage_manager.py:489@aaf7c0d3` | PARTIAL | inspire | enriched #1469 |
| 8 | tiering | Non-blocking store — owned-buffer copy + background writer + in-flight dedup | `lmcache/v1/storage_backend/storage_manager.py:384@aaf7c0d3` | PARTIAL | inspire | filed #3380 |
| 9 | tiering | Backpressure partial store — admit prefix that fits, drop tail, wait-vs-drop knob | `lmcache/v1/cache_engine.py:505@aaf7c0d3` | PARTIAL | inspire | filed #3381 |
| 10 | tiering | Deterministic shard select (worker_id % roots) + eager create-all-dirs | `lmcache/v1/storage_backend/path_sharder.py:80-94@aaf7c0d3` | PARTIAL | inspire | filed #3382 |
| 11 | routing | Single active-backends filter + freeze/bypass + refcounted degrade-restore | `lmcache/v1/storage_backend/storage_manager.py:1165-1193 + health_monitor/base.py:413-486@aaf7c0d3` | PARTIAL | inspire | filed #3383 |
| 12 | eviction | Pluggable eviction policy (LRU/LFU/FIFO/MRU), skip-un-evictable, return fewer | `lmcache/v1/storage_backend/cache_policy/base_policy.py:70-87@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 13 | eviction | Two-axis evictability — reader ref_count vs persistent pin_count | `lmcache/v1/memory_management.py:937-943@aaf7c0d3` | PARTIAL | inspire | filed #3384 |
| 14 | eviction | Pin-timeout reaper — force-unpin leaked pins, reset-on-repin keepalive | `lmcache/v1/pin_monitor.py:91-107@aaf7c0d3` | PARTIAL | inspire | filed #3385 |
| 15 | eviction | Alloc-under-pressure asymmetry — readers busy-loop, headroom-producers never wait | `lmcache/v1/storage_backend/local_cpu_backend.py:640-725@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 16 | eviction | Atomic batch reserve — stage-then-commit-or-noop, no rollback path | `lmcache/v1/memory_management.py:1482-1571@aaf7c0d3` | PARTIAL | inspire | filed #3386 |
| 17 | eviction | Operator pin/unpin API — high-level id resolved to keys locally, refcount predicate | `lmcache/v1/mp_coordinator/cache_control/eviction_manager.py:78-124@aaf7c0d3` | PARTIAL | inspire | enriched #2211 |
| 18 | eviction | Watermark-triggered ratio-bounded eviction + apply-on-confirm via event stream | `lmcache/v1/mp_coordinator/cache_control/eviction_manager.py:100-193@aaf7c0d3` | PARTIAL | inspire | filed #3387 |
| 19 | trust | Read-back checksum witness — corrupt hit downgraded to MISS, not served | `lmcache/v1/storage_backend/connector/audit_connector.py:254-305@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 20 | trust | Opt-in interface-identical audit-wrapper backend (per-op latency/outcome/bytes) | `lmcache/v1/storage_backend/audit_backend.py:30-91@aaf7c0d3` | PARTIAL | inspire | filed #3388 |
| 21 | trust | Cross-config poisoning gate — identity hashes, non-empty mismatch refuses, empty=wildcard | `lmcache/v1/kv_codec/encoded_kv.py:130-154@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 22 | trust | Self-describing versioned CRC32 cache-blob envelope (magic+version+len+CRC) | `lmcache/v1/kv_codec/encoded_kv.py:26-403@aaf7c0d3` | PARTIAL | inspire | filed #3395 |
| 23 | serde | Per-tier fidelity/TTL budget table vs single global compaction rate | `lmcache/v1/storage_backend/naive_serde/cachegen_basics.py:36-46@aaf7c0d3` | PARTIAL | inspire | enriched #3144 |
| 24 | health | Interval-delta failure counting (delta of monotonic counter, not lifetime) | `lmcache/v1/health_monitor/checks/remote_backend_check.py:205-221@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 25 | health | Canary round-trip recovery — put+get+compare behind a cooldown window | `lmcache/v1/health_monitor/checks/remote_backend_check.py:178-202@aaf7c0d3` | PARTIAL | inspire | filed #3389 |
| 26 | observability | Split cacheability (lookup) vs realized (retrieve) token-weighted hit-rate | `lmcache/v1/observability.py:673-701@aaf7c0d3` | PARTIAL | inspire | filed #3390 |
| 27 | observability | Partial-hit histogram (0.1..1.0) + separate 0-hit/cold counter | `lmcache/v1/observability.py:762-766@aaf7c0d3` | PARTIAL | inspire | filed #3367 |
| 28 | observability | Token hit-rate as counter-ratio, eligibility-filtered denominator, model+tenant | `lmcache/v1/mp_observability/subscribers/metrics/lookup.py:90-93@aaf7c0d3` | PARTIAL | inspire | filed #3391 |
| 29 | observability | Shadow-map realized lifecycle (lifetime/idle-before-evict/reuse-gap; share vs reuse) | `lmcache/v1/mp_observability/subscribers/metrics/l0_lifecycle.py:225-285@aaf7c0d3` | PARTIAL | inspire | filed #3392 |
| 30 | observability | Evict->reuse-gap thrash detector, bounded TTL-swept side-map | `lmcache/v1/mp_observability/subscribers/metrics/l1_lifecycle.py:116-166@aaf7c0d3` | PARTIAL | inspire | filed #3393 |
| 31 | observability | Self-meter the metrics/accounting pipeline (drain-lag=staleness, drops, errors) | `lmcache/v1/mp_observability/event_bus.py:124-141@aaf7c0d3` | PARTIAL | inspire | filed #3394 |
| 32 | coordinator | Fleet warm-prefix lookup — which peer holds longest warm prefix + chunk count | `lmcache/v1/cache_controller/controllers/kv_controller.py:388-439@aaf7c0d3` | PRESENT | inspire | DROP (present) |
| 33 | coordinator | Bloom-filter reuse-POTENTIAL estimator + async bounded-queue recorder | `lmcache/v1/lookup_client/record_strategies/memory_bloom_filter.py:52-64@aaf7c0d3` | PARTIAL | inspire | filed #3396 |
| 34 | simulator | Offline cache-value projection — replay logged key stream through simulated cache | `lmcache/tools/cache_simulator/simulator.py:176-316@aaf7c0d3` | PARTIAL | inspire | filed #3397 |
| 35 | simulator | Savings-vs-budget curve over one key-log; size at diminishing-returns knee | `lmcache/tools/cache_simulator/plot_hit_rate.py:35-182@aaf7c0d3` | PARTIAL | inspire | filed #3398 |
| 36 | config | Three-verdict config validate() — refuse / auto-correct+warn / refuse-until-CI | `lmcache/v1/config.py:687-871@aaf7c0d3` | PARTIAL | inspire | filed #3399 |
| 37 | daemon | Decoupled survivable cache daemon (no fate-sharing) + bounded-join shutdown | `lmcache/v1/standalone/__main__.py + offload_server/zmq_server.py@aaf7c0d3` | PARTIAL | inspire | enriched #1469/2853 |

## Honest limits

- The witness is **lexical + a snapshot**: `fak_feature_query`/`fak index` is substring ranking
  (guarded here with raw grep + `gh` search against a false-ABSENT), true only as of 2026-07-08.
  Re-witness before acting on an old row.
- **0 ABSENT / 9 PRESENT / 28 PARTIAL**: no borrow is a from-scratch gap; every PARTIAL is a
  *specific delta* against an existing fak mechanism (the seam + "what fak does today" per issue).
- The full clone dropped nothing (whole history), but the shallow-read is one pass over the
  load-bearing modules, not every file — a reader may have missed a subsystem.
- License reading is a good-faith Apache-2.0-vs-Apache-2.0 compatibility check, not legal advice.
- `class:*` work-class labels on the new issues are backfilled by the nightly
  `tools/issue_lane_router.py` (its snapshot predated these just-created issues).

## Companions

- Filed: epic **#3366** → leaves **#3379, #3378, #3380–#3399** (23 total, M2/M1/M11).
- Enriched: **#3143**, **#3144**, **#1469**, **#2853**, **#2211**.
- Feeds: **#2236** (superset concept-by-concept ranking); sibling of **#3236** (ZML), **#2908/#2834** (Hermes).
- Skills: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) (this pass),
  [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) (the witness+file back-half).
- Sibling study notes: [Dynamo](CONCEPT-STUDY-DYNAMO-2026-07-08.md),
  [MinIO MemKV](CONCEPT-STUDY-MINIO-MEMKV-2026-07-08.md),
  [headroom](CONCEPT-STUDY-HEADROOM-2026-07-08.md).

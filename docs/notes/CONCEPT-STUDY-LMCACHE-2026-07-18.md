---
title: "study-repo: LMCache → fak (2026-07-18, delta pass)"
description: "A fresh DEEP study-repo pass over LMCache at HEAD e38ee415 — the 10-day delta since the 2026-07-08 pass (aaf7c0d3) plus the subsystems that pass excluded as tensor/transport code. 5 novel on-axis borrows filed (#5263, #5264, #5265, #5267, #5269); the rest dedupe against epic #3366 or resolve DIVERGENT (fak is a management layer, not a GPU-tensor layer)."
---

# study-repo: LMCache → fak (2026-07-18, delta pass)

A fresh, **deep** `study-repo` pass over **[LMCache](https://github.com/LMCache/LMCache.git)** — the
KV-cache management layer for scalable LLM inference — pinned at HEAD
**`e38ee4157a11703b07845f45fd98e714b25c13cd`** (`e38ee415`, "[Docs] Add SGLang-to-vLLM KV cache
sharing example", 2026-07-18). License **Apache-2.0** (both repos) → integrate is legal, but every
candidate is Python/CUDA → Go, so **all routes are `inspire`** (clean-room, source cited, no bytes
vendored). All citations `path:line@e38ee415`.

## Relationship to the prior pass — this is a DELTA pass

A very thorough pass ran **2026-07-08 at `aaf7c0d3`**
([`CONCEPT-STUDY-LMCACHE-2026-07-08.md`](CONCEPT-STUDY-LMCACHE-2026-07-08.md)): 37 candidates, **9
PRESENT / 28 PARTIAL / 0 ABSENT**, filing epic **#3366** + 23 leaves (#3378–#3399) and enriching
#3143/#3144/#1469/#2853/#2211. That pass already mapped the *storage / eviction / allocator /
keying / cache-observability core*. To avoid re-filing it, this pass deliberately targets **(a) the
42-commit delta `aaf7c0d3..e38ee415`** and **(b) the subsystems the prior pass excluded as "the
CUDA/NIXL/tensor code"** — `distributed/` (l2 adapters, prefetch controller, bitmap fold/unfold,
quota), `gpu_connector/`, `transfer_channel/`, and the compression codecs. The dedup against the
prior 37 is applied per row below.

Evidence the prior pass already borrowed from LMCache: `internal/cacheobs/bloom.go` is a landed
clean-room borrow of LMCache's memory bloom-filter reuse estimator (prior #3396); `cacheprice/breakeven.go`
already implements the disaggregation break-even-length gate.

## Fan-out coverage (6 parallel deep sub-readers) + completeness-critic

| Subsystem | Load-bearing files read | Verdict cluster |
|---|---|---|
| KV compression / codec | `kv_codec/{asym_k16_v8,encoded_kv}.py`, legacy `serde/cachegen_*`, `csrc/ac_{enc,dec}.cu`, `distributed/serde/turboquant/*` (~5k L) | mostly DIVERGENT (kernels); 2 policy borrows |
| CacheBlend / non-prefix reuse | `compute/blend/blender.py`, `compute/attention/*`, `positional_encoding.py` (~2.2k L) | DIVERGENT (kernels) + 1 worldview finding |
| Storage / eviction / allocator | `storage_backend/{storage_manager,local_cpu,local_disk,gds}.py`, `cache_policy/*`, `memory_allocators/*` (~7k L) | ~all dedupe vs prior 37 |
| Controller / lookup / distributed | `cache_controller/*`, `lookup_client/*`, `distributed/{prefetch_controller,eviction_controller,bitmap_ops/fold,l2_adapters/{base,p2p},api}.py` (~10k L) | **richest**: 3 filed + several note |
| gpu_connector / PD / transfer | `gpu_connector/gpu_connectors.py`, `transfer_channel/*`, `storage_backend/pd_backend_async.py`, `offload_server/*` (~4.9k L) | DIVERGENT (transport) |
| Observability / telemetry / health | `mp_observability/*`, `usage_telemetry/*`, `health_monitor/*`, `internal_api_server/*` (~4k L) | 1 filed (MetricSpec); rest dedupe vs prior #3390–#3396 |

**Completeness-critic residue (justified skips):** individual L2 adapter bodies (`s3` 1296, `dax`
1581, `bigtable` 1224, `valkey` 1117 — one read of the shared `l2_adapters/base.py` interface
suffices; they differ only in wire glue); `store_controller.py` (write/replication path — mirror of
the prefetch read path) and `full_sync_tracker.py` (the 0.8-threshold reconciliation math) — flagged
for a follow-up pass; the CUDA/SYCL kernel bodies in `csrc/` and the NIXL/RDMA transport internals
(`nixl_storage_backend.py`, `transfer_channel/impl/nixl_impl.py`) — **DIVERGENT**: fak owns no GPU
kernels and no tensor transport. Nothing material to fak's management altitude was left unopened.

## Worldview (reconstructed from defaults, non-goals, benchmarks)

LMCache serves **production LLM serving that wants cross-request / cross-instance KV reuse + tiered
offload**, optimizing TTFT via cache hits. Grounding: `remote_serde` defaults to `"naive"`
(compression **off** — `config.py:119`); `enable_controller`/`enable_p2p`/`enable_blending` all
default **False** — the distributed/blend machinery is opt-in layered on a single-node cache. The
tier order is hard-coded CPU→disk→remote (`storage_backend/__init__.py:145`). Two disjoint telemetry
systems (operator OTel/Prometheus + anonymous phone-home) with independent opt-outs. The recurring
thesis in the code: **"node up" is not the metric — token-level cache-hit ratio, sliced per model and
per tenant (`cache_salt`), and its GB/s + latency payoff, is.** CacheBlend exists because in RAG the
reusable text is **not a prefix** (docs concatenated in varying orders); the P2P/controller plane
exists because at fleet scale the KV lives on many nodes and the job is "who holds this prefix and how
do I get it there before decode needs it."

**The load-bearing divergence (why most of LMCache is not a fak borrow):** fak is a *management
layer one level up* — it does not run inference, own GPU KV tensors, or move bytes on the KV data
plane (`gateway/residency_router.go:31`: *"control-plane signal only: no KV bytes move here"*). So
LMCache's marquee mechanisms — the fused multi-layer CUDA transfer kernel, RDMA-WRITE PD handoff,
NIXL descriptor pipelining, CacheGen arithmetic coder, TurboQuant WHT+Lloyd-Max, block-sparse
recompute — are **DIVERGENT by construction**: fak deliberately doesn't own that layer. The
transferable surface is the *policy / accounting / control-plane* altitude, which is exactly where
the 5 filed borrows sit.

## Candidate table (deduped, on-axis, ablated)

| # | borrow (technique) | source `path:line@e38ee415` | the AXIS | witness (fak seam) | route | disposition |
|---|---|---|---|---|---|---|
| 1 | Servable-prefix intersection across heterogeneous attention windows (fold/unfold) | `distributed/bitmap_ops/fold.py:67,127` | bound a model-wide prefix hit by the most restrictive per-layer group (SWA window), not by raw length | ABSENT — `deepseekv4kv.go:112` accounts per-kind *storage* but not the cross-kind servable-prefix *intersection* | inspire | **filed #5263** |
| 2 | Layer-group write-sharding of a large KV object by backend byte-cap | `distributed/l2_adapters/bigtable_l2_adapter.py:692-739`, `bigtable_key_encoder.py:16-40` | shard one oversized KV object into N-layer groups (key `@lg{N}`) each ≤ backend cap, parallel + partial | PARTIAL — `stripeload.go:69` fans a *read* across mirrors; no write-side object-cap shard | inspire | **filed #5264** |
| 3 | Fail-open-until-synced eviction default for a per-tenant byte quota | `distributed/quota_manager.py:42-128`, `eviction_controller.py:347-384` | a not-yet-synced control plane resolves unknown tenants to *exempt*; deny is armed only post-sync | PARTIAL — fak has fail-open *philosophy* (`cachemeta/prefix_coherence.go:71`, guard landlock) but no cache-eviction default posture | inspire | **filed #5265** |
| 4 | Declarative map-reduce MetricSpec registry proving reporter parity | `usage_telemetry/metric_specs.py:26-78`, `mp_continuous.py:99-112` | a metric = `{event, extract, reduce, field}` data row; two reporters provably at-parity from one spec list | ABSENT — `cacheobs/cacheobs.go:73` Observer is hand-written Go counters | inspire | **filed #5267** |
| 5 | Peer instance's KV modeled as a read-only fetch tier (data-plane of cache-aware routing) | `distributed/l2_adapters/p2p_l2_adapter.py:128,213-358` | pull a peer's KV over the same prefetch/lock/trim path as disk/S3 — bytes move, not just the request | PARTIAL — `gateway/residency_router.go:31` is control-plane only ("no KV bytes move here") | inspire | **filed #5269** |
| 6 | Asymmetric K/V precision — K lossless, quantize only V (FP8) | `kv_codec/asym_k16_v8.py:300-306` | spend bits where attention scores rotate (K); V is ~½ the KV volume at near-zero quality loss | PARTIAL — `deepseekv4kv` models a heterogeneous KV plane in normalized units but not K-vs-V precision asymmetry | inspire | note (fold into deepseekv4kv accounting) |
| 7 | Split-tier sub-object placement — keep K in hot L1, demote only V to L2 | `distributed/serde/asym_k16_v8.py:14-33,285-392` | tier *within* a KV object (K hot, V cold) not whole-object, cutting L2 bytes to ~1/3 | PARTIAL — `l3kv`/`stripeload` tier whole objects | inspire | note |
| 8 | Non-blocking L2 adapter contract (submit / query-once / eventfd + per-key bitmap) | `distributed/l2_adapters/base.py:78,145-277` | one async contract lets any remote store (S3/Redis/Bigtable/peer) plug into one poll loop | PARTIAL — `abi.KVBackend` (`l3kv/audit.go`) is synchronous get/put | inspire | note |
| 9 | Two-phase lock-then-load with prefix-trim | `distributed/storage_controllers/prefetch_controller.py:76-121,850` | lock located chunks so they can't be evicted mid-transfer; load only the contiguous prefix, release the rest | PARTIAL — `radixkv` prefix + cachemeta, no lock-during-transfer trim | inspire | note |
| 10 | First-adapter-wins dedupe across redundant tiers | `distributed/storage_controllers/prefetch_policy.py:129-170` | if a prefix lives in 3 tiers, fetch each chunk from exactly one (opposite of stripeload's parallel-mirror read) | DIVERGENT-ish — `stripeload` intentionally fans one read across mirrors for bandwidth | inspire | note |
| 11 | Controller-orchestrated push-based cross-node KV move | `cache_controller/executor.py:281-305`, `worker.py:558-569` | a control primitive to relocate a hot prefix onto an idle node (fleet rebalance) via registry peer URLs | PARTIAL — companion to #5269; #4296 KV-mobility | inspire | note (companion of #5269) |
| 12 | Event-sourced cache index with per-key seq-gap drift detection | `cache_controller/message.py:130-139`, `utils.py:82-160` | keep a remote index fresh with a compact ordered op-log + monotonic seq to *detect* index drift | PARTIAL — `cachemeta/shadowmap.go` tracks lifecycle, not seq-gap drift | inspire | note |
| 13 | Piggyback typed commands on the heartbeat response | `cache_controller/message.py:290`, `registration_controller.py:229` | steer workers with a tagged-union `commands[]` on the beat they already send — no new socket | PARTIAL — fak dispatch worker heartbeat exists | inspire | note |
| 14 | Latency-bounded lookup → recompute fallback | `lookup_client/lmcache_async_lookup_client.py:159,176-188` | a slow cache resolution returns "0 hit tokens" so the engine recomputes — never stalls the request | PARTIAL — `cacheprice/breakeven` is a length gate, not a latency gate | inspire | note |
| 15 | CacheBlend reverse-RoPE: cross-position KV reuse is *re-alignable*, not poison | `compute/positional_encoding.py:20-77`, `blend/blender.py:89-119` | undo RoPE and re-apply at a new offset → reuse a chunk at a different position soundly | WORLDVIEW-FINDING — fak `radixkv/binding.go` treats RoPE-position mismatch as *poison to refuse*; opposite stance | inspire | note-only (consideration vs #3319 regenerable-KV) |
| 16 | Prefix-aware eviction ordering (reverse-insert LRU / suffix-first) | `distributed/eviction_policy/lru.py:83-92`, `local_cpu_backend.py:137-143` | evict the tail of a prefix chain before the head so a surviving suffix is never orphaned | **PRESENT** — fak `radixkv` evicts *leaves* (tails) structurally; the tree gives this for free, no reverse-insert hack | inspire | DROP (present, divergence noted) |
| 17 | Attach-to-live-PID multi-mode flame profiler under real load | `cli/commands/tool/flamegraph.py:1-70`, `cli/profiling.py` | profile a running server in-place (gil/wall/on-cpu/off-cpu/wakeup) under real traffic, not synthetic | PARTIAL — `cmd/fak/profile.go` wraps `go tool pprof` for offline *benchmarks*, no live-attach | inspire | note (off core KV axis; #4254) |
| 18 | Fused CUDA multi-layer transfer / RDMA-WRITE PD handoff / NIXL descriptor pipeline / CacheGen arithmetic coder / TurboQuant WHT+Lloyd-Max / block-sparse recompute | `gpu_connectors.py:255-330`, `pd_backend_async.py:916-925`, `nixl_channel.py:419-460`, `csrc/ac_enc.cu:147`, `turboquant.py:265-367`, `block_sparse_attention.py:22-187` | GPU-tensor transport + KV entropy/vector compression kernels | **DIVERGENT** — fak owns no GPU kernels / tensor transport by design (management layer) | — | DROP (divergent, worldview stated) |

## Honest limits

- The witness is **lexical + a snapshot** (raw `Grep` + `gh` search, guarded by phrasing variation +
  a twice-witness on high-value ABSENTs). `mcp__fak__fak_feature_query` returned ~800 KB per query
  (unusable inline), so witnessing leaned on raw grep of the Go tree — the guard-approved path. True
  only as of 2026-07-18; re-witness before acting.
- **`inspire` for all**: Python/CUDA → Go, clean-room; the Apache-2.0-vs-Apache-2.0 check is
  good-faith, not legal advice.
- Rows 6–14, 17 are real PARTIAL borrows held in the note (respecting "~2–6 issues; rest in note")
  — a follow-up may promote any to a leaf under #3366/#4296.
- The DELTA framing means storage/eviction/allocator borrows are mostly credited to the prior pass;
  a reader who skipped the 2026-07-08 note would over-file there.

## Companions

- **Filed (5 leaves):** #5263 (fold/unfold servable-prefix → `deepseekv4kv`, M2), #5264 (layer-group
  write-shard → `l3kv`, M3), #5265 (fail-open-until-synced eviction default → `cachemeta`, M2), #5267
  (declarative MetricSpec parity → `cacheobs`, M8), #5269 (peer-KV-as-fetch-tier → `gateway`, M3).
- **Parent epics:** #3366 (lmcache-study — the prior pass's umbrella; #5263/#5265/#5267 extend it),
  #4296 (distributed-compute / KV mobility — #5264/#5269), #3259 (disaggregation), #3320 (cachemeta
  plane), #4254 (observability hot-path).
- **Field-borrow companions (note-only, un-filed):** asymmetric K/V precision + split-tier sub-object
  placement (rows 6–7 → `deepseekv4kv`); L2 async adapter contract (row 8 → `abi.KVBackend`);
  reverse-RoPE cross-position reuse (row 15 → consideration vs #3319 regenerable-KV).
- **Prior pass:** [`CONCEPT-STUDY-LMCACHE-2026-07-08.md`](CONCEPT-STUDY-LMCACHE-2026-07-08.md)
  (epic #3366 + #3378–#3399). Skills: [`study-repo`](../../.claude/skills/study-repo/SKILL.md),
  [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md).

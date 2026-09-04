# GPU Cache Pressure & L3 Throughput Bottlenecks

> How GPU KV-cache pressure affects CAMA backup/prefetch throughput, what the TuningAdvisor warning means, and how to tune for it.

---

## The Three-Tier Cache Hierarchy

SGLang's HiCache system organizes KV cache into three tiers:

```
Tier   Location        Typical Size     Contents
────   ────────        ────────────     ────────
L1     GPU HBM         ~80 GB (H100)    Hot KV pages, locked by active requests
L2     Host DRAM       L1 × hicache_ratio (default 2×)    Warm pages, evicted from GPU
L3     CAMA (PrisKV)   Server memory    Cold pages, shared across instances over RDMA
```

Pages flow downward on eviction (L1 → L2 → L3 backup) and upward on miss (L3 prefetch → L2 → L1). CAMA occupies the L3 slot.

---

## The TuningAdvisor Warning

```
[TuningAdvisor] WARN conn=65: GPU cache under pressure
  (91% used, 0% evictable). Prefetch/backup throughput may be critical.
```

**Source:** `cama-server/internal/metrics/tuning_advisor.go:86-92`

This fires when **both** conditions are true simultaneously:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `token_usage` | > 90% | Fraction of GPU KV-cache slots occupied |
| `evictable_ratio` | < 10% | Fraction of pages with `lock_ref == 0` (not held by an active request) |

At 91% used / 0% evictable, **every cached page is pinned by an in-flight request**. The GPU cannot free space for new prefixes until a request completes and releases its lock.

### Where the metrics come from

```
SGLang scheduler
  ├─ token_usage:     num_used_tokens / max_total_num_tokens
  └─ evictable_ratio: evictable_size_ / max_total_num_tokens
        ↓
CamaStorage.get_stats()          (connector, per-rank)
        ↓
report_stats() → OP_REPORT_STATS → server
        ↓
ClientStatsRegistry → TuningAdvisor.Evaluate()
```

### Metric reliability caveats

From `tuning_advisor.go` comments:

- **`token_usage`** can spike to ~1.0 during large batches even when healthy — a single high reading is not necessarily pressure. Sustained high usage with low evictable_ratio is the real signal.
- **`evictable_ratio`** of 0 doesn't always mean trouble — it can mean all cached pages are actively referenced (`lock_ref > 0`), which is normal under heavy load. The warning is about sustained combined pressure.
- Both are point-in-time snapshots reported per `report_stats()` interval, not moving averages.

---

## How GPU Pressure Bottlenecks L3 Writes (Backup)

The backup path: **GPU evicts page → backup queue → BackupCoalescer → batch_set → mset to CAMA**.

Under pressure (91% used, 0% evictable):

```
1. Eviction stalls
   └─ No pages have lock_ref == 0, so nothing can be evicted
   └─ New requests queue — GPU cannot accept new prefixes

2. Burst release
   └─ When requests complete, lock_ref drops → burst of pages become evictable
   └─ Burst floods the backup queue

3. Coalescer drains
   └─ BackupCoalescer merges burst into large batches
   └─ Hits storage_batch_size ceiling (2048 pages) or coalesce_deadline_ms (20ms)

4. Large mset hits CAMA server
   └─ Slab allocation + key insertion for 2048 pages
   └─ Server dispatch queue occupied → blocks other operations

5. Backup latency rises
   └─ Queue grows deeper → GPU can't drain fast enough
   └─ Next eviction burst stacks behind pending writes
```

**The feedback loop:** GPU full → must evict → eviction blocked by slow CAMA writes → GPU stays full → request latency increases → throughput drops.

---

## How GPU Pressure Bottlenecks L3 Reads (Prefetch)

The prefetch path: **cache miss → prefetch queue → mexists (hit-check) → mget_rdma → restore to L1**.

Under pressure:

```
1. More misses
   └─ L1 is saturated → incoming requests increasingly miss their prefix

2. Prefetch queue grows
   └─ More mget_rdma requests submitted to CAMA

3. Bandwidth contention
   └─ Prefetch reads and backup writes share the same RDMA NICs
   └─ A large mset blocks the server dispatch queue, delaying mget_rdma responses

4. Restored pages can't land
   └─ Even if CAMA returns data quickly, L1 has no room
   └─ Must wait for eviction, which is itself stalled

5. Request execution delays
   └─ Request waits for prefetch data → GPU sits idle → throughput drops
```

---

## The Vicious Cycle

Backup and prefetch are **mutually antagonistic** under pressure — they compete for the same CAMA server dispatch queue and RDMA network bandwidth.

```
GPU 91% full, 0% evictable
        │
        ├──► Backup: burst eviction → large mset → server write latency ↑
        │         └──► backup queue depth ↑ → eviction backpressure → GPU stays full
        │
        └──► Prefetch: more misses → mget_rdma competes with mset for dispatch
                  └──► prefetch latency ↑ → request waits → GPU idle → throughput ↓

Meanwhile: more backup traffic delays prefetch, and delayed prefetch causes more
cold misses, which eventually cause more evictions and more backup traffic.
```

"Prefetch/backup throughput may be critical" means: at this pressure level, model throughput depends entirely on how fast CAMA can absorb writes and serve reads. If CAMA throughput is insufficient, the GPU starves.

---

## What to Monitor

### Backup thread health log (every 60s)

```
[backup_thread] health: ops_ok=45 ops_fail=0 inflight=2
  avg_lat=87.4ms batch_size=2048 coalesce_avg=128.3
  jitter_total=234ms gap_avg=12.3ms queue_depth=5
```

| Field | Under pressure looks like | Healthy |
|-------|--------------------------|---------|
| `avg_lat` | > 200ms, rising | < 100ms, stable |
| `queue_depth` | Growing between reports | 0-5 |
| `batch_size` | Oscillating (auto-tune fighting) | Stable |
| `coalesce_avg` | High (many ops merged = bursty) | 1-50 |
| `inflight` | At `backup_io_workers` limit | 0-1 |

### CamaStorage I/O stats (every ~10s)

```
[MHA] CAMA I/O stats: 45 calls, avg_batch=256.0, avg=87.4ms,
  max=156.2ms, 25.3 GB/s | inflight: get=0 set=5
  phases: pre=2.1ms xfer=84.2ms post=1.1ms
```

| Field | Under pressure | Healthy |
|-------|---------------|---------|
| `inflight_sets` | High (5+), growing | 0-2 |
| `xfer` (transfer time) | Dominates total latency | < 50ms |
| `max` latency | Spikes (200ms+) | Close to avg |

### Prometheus metrics

| Metric | What it shows |
|--------|--------------|
| `cama_client_backup_queue_depth` | Pending backup ops (growing = falling behind) |
| `cama_client_backup_avg_latency_ms` | Write latency trend |
| `cama_client_backup_batch_size` | Current auto-tuned batch size |
| `cama_client_backup_coalesce_avg_ops` | Burst size (high = bursty eviction) |
| `cama_client_host_alloc_drops` | Pages dropped because host pool is full |

---

## Tuning Guide

### Understanding the knobs

There are three independent controls that interact:

```
BackupCoalescer
  ├─ coalesce_deadline_ms (default 20ms)     — how long to wait for more pages
  ├─ storage_batch_size   (default 2048)     — max pages per coalesced batch
  └─ batch_size_auto      (default true)     — auto-adjust storage_batch_size
       └─ batch_size_latency_target_ms (200ms)  — target for auto-adjustment
```

**`coalesce_deadline_ms`** is usually the binding constraint, not `storage_batch_size`. Under moderate load, the queue fills at ~50-200 pages/20ms — the deadline fires before the 2048 cap is reached. The cap only kicks in during extreme burst eviction.

### Decision tree

```
Is backup_avg_latency_ms consistently > 200ms?
├─ YES → Is batch_size_auto=true?
│    ├─ YES → It should be self-correcting (halving batch size).
│    │        Check if it's oscillating (2048→1024→2048→1024).
│    │        If oscillating: lower batch_size_latency_target_ms to 150ms
│    │        If stuck high: the server is slow — see "Server-side tuning" below
│    └─ NO → Enable batch_size_auto=true, or manually lower storage_batch_size
│
├─ NO, but prefetch latency is high →
│    Lower coalesce_deadline_ms to 10ms (smaller msets, less dispatch blocking)
│
└─ NO, everything is healthy → Current settings are fine
```

### Scenario-based recommendations

| Scenario | `storage_batch_size` | `coalesce_deadline_ms` | `batch_size_auto` | Rationale |
|----------|---------------------|----------------------|-------------------|-----------|
| **Default (most setups)** | 2048 | 20 | true | Single-roundtrip for typical contexts, auto-corrects |
| **GPU pressure + prefetch starvation** | 512 | 10 | true | Smaller msets interleave with reads, less dispatch blocking |
| **High throughput, no pressure** | 2048-4096 | 50 | true | Maximize amortization, let auto-tune grow |
| **Cross-rack RDMA (high latency)** | 2048+ | 50-100 | true | Amortize high RTT cost, batch as much as possible |
| **Single rank, low volume** | 256 | 5 | false | Low latency per-batch, few roundtrips anyway |
| **Benchmarking** | (fixed) | 20 | false | Remove auto-tune variable for clean measurements |

### Server-side tuning (when CAMA is the bottleneck)

If backup latency remains high despite smaller batches, the server is the bottleneck:

1. **Check per-shard latency** — `cama_shard_op_latency_p99_ms` shows if one shard is slow (hot key or allocator pressure)
2. **Increase dispatch workers** — `rdma_dispatch_workers` (default varies by version) adds parallelism to the CQ poller
3. **Enable multi-NIC** — stripes mget_rdma and mset across server NICs for N× bandwidth
4. **Check slab utilization** — if slot utilization is high, the allocator is searching for free slots (slow alloc → slow mset)
5. **Increase server memory** — more slab capacity → less eviction pressure → less backup traffic from connector

### Connector-side tuning (when the network is fine but queues are deep)

1. **Increase `backup_io_workers`** from 2 to 3-4 — more parallel mset submissions drain the queue faster
2. **Increase `prefetch_io_workers`** from 2 to 4 — more overlap between hit-queries and transfers
3. **Lower `backup_jitter_ms`** — if you have few TP ranks, reduce cross-rank jitter to speed up drain
4. **Disable dedup** (`dedup_mode: "never"`) — under pressure, hit rate is typically low (most pages are new), so the exists check is pure overhead

### Model/scheduler-side tuning

1. **Increase `max_total_num_tokens`** — expand L1 capacity → fewer evictions → less backup traffic
2. **Increase `hicache_ratio`** — larger L2 host cache → pages stay local longer before hitting L3
3. **Reduce concurrent batch size** — fewer tokens per step → lower peak GPU cache occupancy
4. **Adjust `chunked_prefill_size`** — smaller prefill chunks lock fewer pages simultaneously → higher evictable_ratio

---

## Worked Example

**Setup:** 8× H100 TP, DeepSeek-V3, page_size=64, model_page_bytes=320KB

**Observation:**
```
[TuningAdvisor] WARN conn=65: GPU cache under pressure (91% used, 0% evictable)
[backup_thread] health: avg_lat=340ms batch_size=2048 queue_depth=47
[MHA] CAMA I/O stats: inflight: get=0 set=4, phases: xfer=320ms
```

**Diagnosis:**
- Backup latency 340ms > 200ms target → auto-tune should halve batch_size to 1024
- Queue depth 47 and growing → backup thread can't keep up
- `inflight_sets=4` → all backup workers saturated
- `inflight_gets=0` → no prefetch contention (yet), but if misses increase it will compete

**Actions:**
1. Wait one 60s window — check if auto-tune halved batch_size
2. If still high after auto-tune: lower `coalesce_deadline_ms` from 20 to 10ms
3. If still high: increase `backup_io_workers` from 2 to 4
4. If server-side: check `cama_shard_op_latency_p99_ms` — if one shard is hot, enable vacuum rebalancing

**Result after tuning:**
```
[backup_thread] health: avg_lat=95ms batch_size=512 queue_depth=2
[MHA] CAMA I/O stats: inflight: get=1 set=1, phases: xfer=88ms
```

---

## Related Documents

- [03_CONFIGURATION_REFERENCE.md](03_CONFIGURATION_REFERENCE.md) — Full parameter reference (batch size, coalescing, adaptive sizing)
- [02_ARCHITECTURE_DEEP_DIVE.md](02_ARCHITECTURE_DEEP_DIVE.md) — BackupCoalescer internals, threading topology
- [05_TROUBLESHOOTING.md](05_TROUBLESHOOTING.md) — Debugging I/O failures
- [11_THROUGHPUT_VARIABILITY.md](11_THROUGHPUT_VARIABILITY.md) — Why per-rank GB/s fluctuates (warmup phases, adaptive sizing, NIC striping)
- `cama-server/internal/metrics/tuning_advisor.go` — TuningAdvisor source (108 LOC)

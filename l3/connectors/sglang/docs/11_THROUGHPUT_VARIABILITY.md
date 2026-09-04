# Per-Rank Throughput Variability

> Why your per-rank GB/s fluctuates wildly — and why that's intentional.

---

## The Core Tradeoff

The CAMA connector deliberately sacrifices transfer throughput to protect inference latency (TTFT). The system actively controls batch size to prevent storage I/O from starving the GPU:

```
throughput = batch_size × page_bytes / batch_latency
```

`batch_size` is the variable the system actively controls. When backup latency rises, the adaptive sizing engine halves the batch size — cutting throughput in half to keep per-batch latency under the target. This is not a bug. A single latency spike that reaches the GPU scheduler costs 120+ seconds of recovery time; a 2x throughput reduction costs nothing if the backup queue isn't growing.

---

## What You Will Observe

### The COLD Burst (>1 GB/s)

When the first KV pages arrive after model loading, the warmup system enters **COLD phase** with maximum-throughput settings:

| Parameter | COLD value | STEADY value |
|-----------|-----------|--------------|
| `batch_size` | `batch_size_max` (4096) | auto-tuned (typically 512-2048) |
| `coalesce_deadline_ms` | 2 | 20 |
| `jitter_ms` | 0 | configured (10-30) |
| `dedup` | skipped | auto |

This produces a brief burst of >1 GB/s throughput. It lasts ~1 second — until the server confirms all shards have rebuilt their allocators, at which point the system transitions to STEADY.

**Why it's temporary:** COLD phase exists to fill the freshly-built slab allocator as fast as possible. Once the server is warm, the connector switches to latency-protective settings.

See [warmup-design.md](warmup-design.md) for the full COLD/STEADY state machine.

### The STEADY Drop

After COLD → STEADY transition, three things change simultaneously:

1. **Batch size drops** — adaptive sizing takes over, starting from `storage_batch_size` (default 2048) and adjusting based on observed latency
2. **Jitter activates** — `backup_jitter_ms` spreads rank submissions across a time window
3. **Coalescer deadline widens** — from 2ms to 20ms, waiting longer for more pages before submitting

The combined effect is a 2-10x throughput reduction from the COLD burst:

| Phase | Effective batch | Deadline | Jitter | Typical throughput (5MB pages) |
|-------|----------------|----------|--------|-------------------------------|
| COLD | 4096 | 2ms | 0 | 1.0-1.5 GB/s |
| STEADY (healthy) | 1024-2048 | 20ms | 10ms | 300-600 MB/s |
| STEADY (under pressure) | 256-512 | 20ms | 10ms | 80-200 MB/s |

### Adaptive Oscillation

When `batch_size_auto=true` (default), the batch size follows a sawtooth pattern:

```
batch_size
    ^
4096|     *
    |    / \
2048|   /   \         *
    |  /     \       / \
1024| /       \     /   \
    |/         \   /     \
 512|           \ /       \
    |            *         *...
    └──────────────────────────── time (60s windows)
```

Every 60 seconds, the adaptive engine evaluates:
- `avg_latency > target` → halve batch size (floor: 32)
- `avg_latency < target × 0.5` → double batch size (ceiling: 4096)
- otherwise → no change

A single latency spike triggers a halving that takes 2-3 windows (120-180s) to recover from. This is intentional — the system errs on the side of protecting inference latency.

**Metric:** `cama_client_backup_batch_size` tracks the current effective batch size in real time.

---

## Six Factors That Determine Per-Rank GB/s

### 1. Adaptive Batch Sizing

The dominant factor. The adaptive engine adjusts `storage_batch_size` every 60s based on backup latency, producing the throughput oscillation described above.

| Config | Default | Effect |
|--------|---------|--------|
| `batch_size_auto` | `true` | Enable/disable adaptive sizing |
| `storage_batch_size` | `2048` | Initial batch size (pages per mset) |
| `batch_size_latency_target_ms` | `200` | Latency target for halving/doubling |
| `batch_size_max` | `max(configured, 4096)` | Ceiling for auto-tuning |

**Metric:** `cama_client_backup_batch_size`

See [03_CONFIGURATION_REFERENCE.md — Adaptive Batch Sizing](03_CONFIGURATION_REFERENCE.md#adaptive-batch-sizing) for the full algorithm.

### 2. Warmup Phase (COLD vs STEADY)

The warmup system overrides all batch/coalescer/jitter settings during COLD phase to maximize initial fill throughput. The COLD → STEADY transition causes the single largest throughput drop.

| Config | Default | Effect |
|--------|---------|--------|
| `warmup_cold_batch_size` | `batch_size_max` | Batch size during COLD |
| `warmup_cold_deadline_ms` | `2.0` | Coalescer deadline during COLD |
| `warmup_min_batch_size` | `256` | Floor that prevents death-spiral |

See [warmup-design.md](warmup-design.md) for the INIT → COLD → STEADY state machine.

### 3. Coalescer Deadline

The `BackupCoalescer` waits up to `coalesce_deadline_ms` after draining the first operation before submitting the merged batch. A longer deadline accumulates more pages (higher throughput per batch) but adds latency to each submission.

| Phase | Effective deadline | Effect on throughput |
|-------|-------------------|---------------------|
| COLD | 2ms | Minimal wait — submit fast |
| STEADY | 20ms (default) | Wait for more pages — larger batches |
| Under TP scaling | Dynamically adjusted | Wider window for more ranks |

| Config | Default | Effect |
|--------|---------|--------|
| `coalesce_deadline_ms` | `20.0` | Max wait time for additional pages |
| `coalesce_backup_ops` | `true` | Enable/disable coalescing |

**Metric:** `cama_client_backup_coalesce_avg_ops`

See [03_CONFIGURATION_REFERENCE.md — Backup Queue Coalescing](03_CONFIGURATION_REFERENCE.md#backup-queue-coalescing) for details.

### 4. Multi-NIC Striping

The primary mechanism for exceeding single-link bandwidth. With `nic_striping=true` (default), `mget_rdma` stripes RDMA Reads across all server NICs in parallel.

| NICs | TP Ranks | Ranks per NIC | Expected per-rank read BW (100 Gb/s NICs) |
|------|----------|---------------|-------------------------------------------|
| 1 | 8 | 8 | ~1.5 GB/s |
| 2 | 8 | 4 | ~3.0 GB/s |
| 4 | 8 | 2 | ~6.0 GB/s |

Write throughput (`mset`) does **not** benefit from striping in the same way — writes go to the NIC that owns the target shard. Read striping is the primary throughput multiplier.

| Config | Default | Effect |
|--------|---------|--------|
| `nic_striping` | `true` | Enable multi-NIC striped reads |
| `pool_size` | `8` (auto-set to NIC count when striping) | Connections per rank |

**Metric:** `cama_rdma_nic_wire_gb_total` (per-NIC server-side)

### 5. Connection Pool Serialization

Each pool connection has its own lock. When `backup_io_workers` threads submit `mset` calls, they serialize on whichever connection they acquire. With `pool_size=8` and `backup_io_workers=2`, contention is rare — but under burst eviction with high `backup_io_workers`, connections become the bottleneck.

| Config | Default | Effect |
|--------|---------|--------|
| `pool_size` | `8` | Number of connections per rank |
| `backup_io_workers` | `2` | Parallel backup submissions |

### 6. RDMA Link Saturation

The hard ceiling for per-rank throughput is the NIC line rate divided by the number of ranks sharing that NIC. Reads and writes compete for the same link bandwidth.

```
max_per_rank_bw = nic_line_rate / ranks_per_nic

Example: 100 Gb/s NIC, 8 TP ranks, 1 NIC
  → 100 / 8 = 12.5 Gb/s = ~1.5 GB/s per rank (theoretical max)
  → In practice ~1.0-1.2 GB/s due to protocol overhead and read/write contention
```

When reads and writes saturate the link simultaneously, both paths slow down. This is the vicious cycle described in [10_GPU_CACHE_PRESSURE.md](10_GPU_CACHE_PRESSURE.md#the-vicious-cycle).

---

## Worked Example

**Setup:** 8× H100 TP, 2 NICs, `model_page_bytes` = 5 MB, `batch_size_auto=true`

### Window 1: COLD Burst (~first 1s after data flows)

```
Phase: COLD
batch_size: 4096, deadline: 2ms, jitter: 0
```

All 8 ranks fire maximum-size batches simultaneously. Two NICs handle the load. Per-rank throughput peaks at ~1.2 GB/s. The server receives the `model_page_bytes` hint and rebuilds slab classes.

### Window 2: STEADY State (~1-60s)

```
Phase: STEADY
batch_size: 2048, deadline: 20ms, jitter: 10ms
```

Warmup poller confirms all shards rebuilt → COLD exits. Adaptive sizing starts at the configured `storage_batch_size=2048`. Jitter spreads rank submissions. Per-rank throughput drops to ~400 MB/s.

### Window 3: Adaptive Halving (~60-120s)

```
Phase: STEADY
batch_size: 1024 (halved), deadline: 20ms, jitter: 10ms
```

If the first 60s window shows `avg_latency > 200ms` (the default target), the adaptive engine halves batch size from 2048 to 1024. Per-rank throughput drops to ~200 MB/s. The next window evaluates whether 1024 keeps latency under target — if so, it stays; if latency is well below 50% of target, it doubles back.

---

## Metrics Quick Reference

| Metric | What it shows | Which factor |
|--------|--------------|--------------|
| `cama_client_backup_batch_size` | Current effective batch size | Adaptive sizing, warmup |
| `cama_client_backup_avg_latency_ms` | Average backup latency (60s window) | Adaptive sizing trigger |
| `cama_client_backup_queue_depth` | Pending backup operations | Overall health |
| `cama_client_backup_coalesce_avg_ops` | Ops merged per coalesced batch | Coalescer deadline |
| `backup_bandwidth_gbps` | Connector-reported write bandwidth | Overall throughput |
| `cama_rdma_nic_wire_gb_total` | Per-NIC cumulative bytes (server-side) | Multi-NIC striping |

---

## Related Documents

- [03_CONFIGURATION_REFERENCE.md](03_CONFIGURATION_REFERENCE.md) — Full parameter reference (batch sizing, coalescing, jitter)
- [10_GPU_CACHE_PRESSURE.md](10_GPU_CACHE_PRESSURE.md) — GPU cache pressure, read/write contention, tuning decision tree
- [warmup-design.md](warmup-design.md) — COLD/STEADY state machine, server readiness detection
- [02_ARCHITECTURE_DEEP_DIVE.md](02_ARCHITECTURE_DEEP_DIVE.md) — BackupCoalescer internals, threading topology
- [06_DESIGN_DECISIONS.md](06_DESIGN_DECISIONS.md) — Per-rank prewarm, warmup retry, dedup probe design rationale
- [End-to-End Bottleneck Guide](../../docs/end-to-end-bottleneck-guide.md) — Full-stack bottleneck map from SGLang to NIC wire

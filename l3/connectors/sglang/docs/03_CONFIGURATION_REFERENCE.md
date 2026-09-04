# CAMA Configuration Reference

> Complete reference for every configurable parameter affecting the CAMA storage backend.

---

## Configuration Priority

CAMA loads configuration from three sources. The first source found with a `remote_addr` wins — sources are not merged.

```
┌─────────────────────────────────────────────────────────┐
│                  Configuration Resolution               │
│                                                         │
│  1. extra_config dict                                   │
│     (from --hicache-storage-backend-extra-config)       │
│     Has remote_addr? ──── YES ──▶ Use extra_config      │
│           │                                             │
│           NO                                            │
│           ▼                                             │
│  2. SGLANG_CAMA_CONFIG_PATH env var                     │
│     Is set? ──────────── YES ──▶ Load JSON file         │
│           │                                             │
│           NO                                            │
│           ▼                                             │
│  3. Individual SGLANG_CAMA_* env vars                   │
│     Always available ─────────▶ Use env var defaults    │
└─────────────────────────────────────────────────────────┘
```

**Source:** `cama_storage.py` lines 192-201

---

## Core Parameters

| Parameter | Env Variable | JSON Key | Type | Default | Description |
|-----------|-------------|----------|------|---------|-------------|
| `remote_addr` | `SGLANG_CAMA_REMOTE_ADDR` | `remote_addr` | `str` | `"127.0.0.1"` | PrisKV server IP address or hostname. **Required** in JSON/extra_config (omitting raises `ValueError`). |
| `remote_port` | `SGLANG_CAMA_REMOTE_PORT` | `remote_port` | `int` | `6379` | PrisKV server port. Uses Redis-like default. |
| `password` | `SGLANG_CAMA_PASSWORD` | `password` | `str` | `""` (empty) | PrisKV authentication password. Leave empty if PrisKV has no auth configured. |
| `use_mput_mget` | `SGLANG_CAMA_USE_MPUT_MGET` | `use_mput_mget` | `bool` | `true` | Enable native batch wire-protocol ops (`mexists`, `mset`, `mdel`) for `_batch_exist` and `_put_batch_zero_copy`. When `false`, falls back to individual ops via ThreadPoolExecutor. |
| `check_server` | `SGLANG_CAMA_CHECK_SERVER` | `check_server` | `bool` | `false` | If `true`, poll PrisKV at startup (up to 600s / 10 min) until reachable. Use for orchestrated deployments where PrisKV may start after SGLang. When `true`, the preflight check is skipped (it assumes PrisKV will become available). |

**Source:** `cama_storage.py` `CamaConfig` dataclass, `environ.py`

---

## Performance & Reliability Parameters

These parameters control I/O concurrency, timeouts, and thread pool sizing. All are **per-rank** — each GPU rank creates its own CamaStorage instance with independent pools and connections.

| Parameter | Env Variable | JSON Key | Type | Default | Description |
|-----------|-------------|----------|------|---------|-------------|
| `op_timeout_s` | `SGLANG_CAMA_OP_TIMEOUT_S` | `op_timeout_s` | `float` | `10.0` | Per-batch I/O timeout in seconds. Applied to `concurrent.futures.as_completed()` in `_get_batch_zero_copy`, `_put_batch_zero_copy`, and `_batch_exist`. If any key operation in a batch hasn't completed within this window, the batch returns partial results — timed-out keys report failure (-1 for get/set, EXISTS_ERROR for exists). Also passed to `conn.set_timeout()` if the PrisKV client supports it. |
| `io_workers` | `SGLANG_CAMA_IO_WORKERS` | `io_workers` | `int` | `16` | Number of threads in `CamaStorage._io_pool` that execute individual key operations (get/set/exists) concurrently within a single batch call. These threads are distributed across `pool_size` connections. This pool is **shared** between the prefetch and backup paths. **Note:** With `mget_rdma` enabled (default when server supports it), `_io_pool` is bypassed entirely for reads — only writes and the legacy read fallback use it. |
| `pool_size` | `SGLANG_CAMA_POOL_SIZE` | `pool_size` | `int` | `8` | Number of RDMA/TCP connections per rank. Creates a connection pool for N-way parallelism — each connection has its own lock, so `io_workers` threads can execute in parallel across connections instead of serializing on a single lock. RDMA pools share one Protection Domain (PD) with `skip_read_buf` to save ~32 MB per extra connection. Set to `1` for single-connection (backward-compatible) behavior. **With NIC striping enabled** (default), `pool_size` is auto-set to `len(rdma_endpoints)` when the server has multiple NICs, so each connection targets a different NIC for bandwidth saturation. |
| `nic_striping` | `SGLANG_CAMA_NIC_STRIPING` | `nic_striping` | `bool` | `true` | Enable multi-NIC striped RDMA reads. When `true` and multiple RDMA endpoints are discovered, the pool connects to ALL server NICs and stripes `mget_rdma` across them in parallel for N× bandwidth. When `false`, falls back to legacy single-NIC-per-rank assignment (`endpoints[local_rank % len(endpoints)]`). |
| `send_buf_size` | `SGLANG_CAMA_SEND_BUF_SIZE` | `send_buf_size` | `int` | `0` | RDMA Send buffer size in bytes. `0` = use client default (16 MB). When `mset` payloads exceed this, they are automatically chunked into sub-batches (client logs a warning when batching degenerates). Also passed as `recv_buf_size` to the client (they must match). |
| `prefetch_io_workers` | — | `prefetch_io_workers` | `int` | `2` | **(SGLang-side, not CamaConfig)** Number of threads in the prefetch I/O executor inside `cache_controller.py`. Controls how many prefetch requests can have their page transfers running concurrently. Set via `--hicache-storage-backend-extra-config '{"prefetch_io_workers": 2}'`. See [Architecture Deep Dive: Threading Topology](02_ARCHITECTURE_DEEP_DIVE.md#threading-topology-per-rank) for detailed tuning guidance. |
| `backup_io_workers` | — | `backup_io_workers` | `int` | `2` | **(SGLang-side, not CamaConfig)** Number of threads in the backup I/O executor inside `cache_controller.py`. `2` = parallel by default, matching `prefetch_io_workers`. Set to 3–4 to drain the backup queue faster when host pool saturation is observed. Each worker calls `_page_backup()` independently; ack ordering is safe because `drain_storage_control_queues()` uses operation-ID lookup. |
| `storage_batch_size` | — | `storage_batch_size` | `int` | `2048` | **(SGLang-side, not CamaConfig)** Pages per sub-batch in `_page_backup()` and `_page_transfer()`. Lower values (32–64) reduce per-sub-batch latency at the cost of more roundtrips; higher values (1024+) improve throughput. Auto-tuning can grow up to `max(configured, 4096)`. See [Batch Size Tuning](#batch-size-tuning) below. |
| `backup_jitter_ms` | — | `backup_jitter_ms` | `int` | `10` | **(SGLang-side, not CamaConfig)** Maximum per-sub-batch jitter in milliseconds to prevent thundering herd across TP ranks. Each sub-batch (except the first) sleeps a random `[0, backup_jitter_ms]` ms before issuing its `mset`. `0` = disabled (zero overhead). Clamped to [0, 500]. See [Backup Write Jitter](#backup-write-jitter) below. |
| `batch_size_auto` | — | `batch_size_auto` | `bool` | `true` | **(SGLang-side, not CamaConfig)** Enable adaptive batch sizing. When `true`, `storage_batch_size` is automatically adjusted every 60s based on observed backup latency — halved when latency exceeds target, doubled when latency is well below target. See [Adaptive Batch Sizing](#adaptive-batch-sizing) below. |
| `batch_size_latency_target_ms` | — | `batch_size_latency_target_ms` | `float` | `200` | **(SGLang-side, not CamaConfig)** Latency target for adaptive batch sizing. Only used when `batch_size_auto=true`. Batch size is halved when avg latency exceeds this value, and doubled when avg latency is below 50% of this value. |
| `batch_size_max` | — | `batch_size_max` | `int` | `max(storage_batch_size, 4096)` | **(SGLang-side, not CamaConfig)** Ceiling for adaptive batch sizing. Auto-tuning will never grow `storage_batch_size` above this value. Set explicitly to cap growth when using a small `storage_batch_size`. |
| `coalesce_backup_ops` | — | `coalesce_backup_ops` | `bool` | `true` | **(SGLang-side, not CamaConfig)** Merge consecutive backup operations into larger batches before sending to storage. Dramatically improves throughput when SGLang enqueues many single-page operations. See [Backup Queue Coalescing](#backup-queue-coalescing) below. |
| `coalesce_deadline_ms` | — | `coalesce_deadline_ms` | `float` | `20.0` | **(SGLang-side, not CamaConfig)** Max time (ms) to wait for additional operations after draining the first. `0` = drain without waiting (only take what's already queued). Clamped to [0, 500]. |

### Thread Pool & Connection Pool Interaction

The thread pool and connection pool are **nested** — but with `mget_rdma` (default), the read path bypasses the thread pool entirely:

```
prefetch_io_workers (default 2)
  └─ each worker calls batch_get_v1
       └─ _get_batch_zero_copy
            ├─ [primary] conn.mget_rdma(keys, sgls)     ← no thread pool
            │   1 control roundtrip + batch RDMA Read (1 doorbell)
            │   Uses a single pool connection for the entire batch.
            │
            └─ [fallback] io_workers (default 16) threads
                 └─ each thread acquires a connection from the pool (round-robin)
                      └─ pool_size (default 8) connections
```

With `mget_rdma`, read performance is bandwidth-limited and page-size-independent — `io_workers` and `pool_size` have no effect on read throughput. **With NIC striping** (default), `mget_rdma` stripes keys across multiple server NICs in parallel, multiplying available read bandwidth. The thread pool is still used for writes (`_put_batch_zero_copy`) and as a read fallback when `mget_rdma` is unavailable.

**Memory overhead per pool connection:**
- RDMA: ~32 MB (16 MB send + 16 MB recv; 32 MB read buffer skipped via `skip_read_buf`)
- TCP: one socket per connection (negligible)
- Total for default `pool_size=8`: ~224 MB extra RDMA memory (7 extra connections)

| Scenario | `pool_size` | `io_workers` | `prefetch_io_workers` | Rationale |
|----------|------------|-------------|----------------------|-----------|
| Default (most setups) | 8 | 16 | 2 | 2 threads per connection, good parallelism without excess memory |
| High-latency RDMA (cross-rack) | 4–8 | 32 | 2 | More connections + in-flight keys to hide latency |
| Saturated NIC | 2 | 8 | 1 | Reduce contention when link is already at capacity |
| Memory-constrained | 1 | 16 | 2 | Single connection, backward-compatible behavior |
| Large batch sizes (>500 sub-keys) | 4 | 16 | 2–3 | More prefetch overlap, but watch for backup starvation |
| Host pool saturation (alloc drops) | 4 | 16 | 2 | Default `backup_io_workers=2` handles moderate saturation; set to 3–4 for severe cases and optionally lower `storage_batch_size` to 64 |

### Batch Size Tuning

`storage_batch_size` controls how many pages are sent per `mset` wire call in both the backup and prefetch paths. It is **independent** of `page_size`:

```
page_size            = tokens per page      (set by SGLang attention backend, e.g. 1, 16, 64, 128)
storage_batch_size   = pages per mset call  (configurable, default 2048)
tokens per mset      = storage_batch_size × page_size
wire payload per mset ≈ storage_batch_size × page_size × bytes_per_token
```

`page_size` is determined by the model architecture and attention backend (e.g. FlashMLA → 64, FA4 → 128) and is **not** tunable via CAMA config. `storage_batch_size` controls batching at the wire level and is tunable.

> **Automatic slab tuning (v0.37.0):** The connector sends the computed value size
> (`bytes_per_token × page_size`) to the server as a `model_page_bytes` hint during
> `register_mem_pool_host()` — before the first batch write. The server uses this to
> build optimized slab classes immediately, regardless of `page_size`. No manual
> `model_page_bytes` server config is needed.

**Why 2048 default?** A 50k-token context with `page_size=64` is ~781 pages. The old default of 256 split this into 4 sub-batches with jitter between each, adding unnecessary latency. At 2048, most single-context backups fit in one sub-batch — one roundtrip, no jitter overhead. Auto-tuning can grow up to 4096 if latency allows.

| `storage_batch_size` | Sub-batches for 781 pages | Tradeoff |
|---------------------|--------------------------|----------|
| 64 | 13 | Lower per-batch latency, 13 roundtrips + jitter |
| 256 | 4 | Old default — 4 roundtrips + jitter |
| **2048** (default) | **1** | Single roundtrip, best throughput |
| 4096 | 1 | Same as 2048 for typical ops; reachable via auto-tune |

**Tuning guide:**
- If `backup_avg_latency_ms` is high (>200ms) and pages per op are small, try lowering to 256–512
- If you see many sub-batches per op in debug logs, try raising to match your typical page count
- The `backup_avg_gap_ms` metric shows time between sub-batches — high values indicate server contention

### Backup Write Jitter

With multiple TP ranks (e.g. 8-way tensor parallelism), all ranks enqueue backup `mset` operations simultaneously after each forward pass. This thundering herd pattern causes 8 concurrent bursts on the CAMA server, inflating latency from ~15ms to 60-87ms for 1.5MB payloads.

`backup_jitter_ms` adds a random delay between sub-batches (not before the first) to spread rank submissions across a time window:

```
Rank 0:  [batch 0]──sleep(rand(0,25ms))──[batch 1]──sleep(rand(0,25ms))──[batch 2]
Rank 1:  [batch 0]──sleep(rand(0,25ms))──[batch 1]──sleep(rand(0,25ms))──[batch 2]
  ...without jitter, all ranks' batch 0/1/2 arrive simultaneously
  ...with jitter, submissions spread across the 25ms window
```

**Design details:**
- Uses `stop_event.wait(timeout=)` instead of `time.sleep()` — shutdown is not delayed
- First sub-batch is never delayed (no added latency for single-batch operations)
- Zero overhead when disabled: two comparisons, no random/wait calls
- Thread-safe: `random.uniform` under GIL, `stop_event.wait` is reentrant

**Recommended values:**

| Deployment | `backup_jitter_ms` | Rationale |
|------------|-------------------|-----------|
| Single rank | `0` | No cross-rank contention — override default to disable |
| 2–4 TP ranks | `10–15` | Light spreading |
| 8 TP ranks, RDMA | `20–30` | Good spread without adding significant latency |
| 8 TP ranks, TCP | `30–50` | TCP has higher per-call overhead, needs wider window |

**Observability:** After enabling jitter, monitor these metrics in the 60s health log:
- `backup_jitter_total_ms` — total jitter applied (confirms it's active)
- `backup_avg_gap_ms` — time between sub-batches (without jitter this is ~0; with jitter it should spread to ~`jitter_ms / num_ranks`)
- `backup_avg_latency_ms` — should decrease as contention drops

### Adaptive Batch Sizing

When `batch_size_auto=true`, the backup thread automatically adjusts `storage_batch_size` every 60 seconds based on the average backup latency observed in that window:

```
avg_latency > target          → batch_size = batch_size / 2  (floor: 32)
avg_latency < target × 0.5   → batch_size = batch_size × 2  (ceiling: max(configured, 4096))
otherwise                     → no change
```

The algorithm requires at least 3 latency samples per window to act (avoids reacting to startup transients). Changes are logged at INFO level with the old/new values and the triggering latency.

**Example:** `storage_batch_size=2048, batch_size_auto=true, batch_size_latency_target_ms=200`
1. Window 1: avg_lat=350ms → halve to 1024
2. Window 2: avg_lat=150ms → within [100, 200] → no change
3. Window 3: avg_lat=80ms → below 100ms → double to 2048
4. Window 4: avg_lat=190ms → within [100, 200] → steady state

Auto-tuning can grow *above* the initial configured value, up to `max(configured, 4096)`. The floor is 32 pages.

**When to enable:**
- Multi-tenant deployments where server load varies
- When you don't know the optimal batch size for your workload
- During initial tuning — let it find the sweet spot, then hardcode the result

**When NOT to enable:**
- Benchmarking (adds a variable; fix the batch size instead)
- Stable production with known-good batch size

**Prometheus metric:** `cama_client_backup_batch_size` tracks the current effective batch size, making auto-tuning adjustments visible in Grafana.

For how adaptive sizing affects observed per-rank throughput, see [11_THROUGHPUT_VARIABILITY.md](11_THROUGHPUT_VARIABILITY.md).

### Backup Queue Coalescing

SGLang's radix tree evicts nodes one at a time, each calling `write_storage()` with **1 page**. Each becomes a `StorageOperation` with `len(hash_value) = 1`. Without coalescing, the backup thread dequeues one operation at a time and submits it immediately — for MHA models (1 page = 2 sub-keys K+V), this yields `avg_batch=2.0`, producing hundreds of tiny wire roundtrips instead of one large batch.

The existing `storage_batch_size` only controls chunking *within* a single operation. Since each operation has 1 page, it has no effect. Coalescing fixes this by draining and merging multiple operations from the queue before submitting.

When `coalesce_backup_ops=true` (default), a `BackupCoalescer` drains up to `storage_batch_size` pages from the backup queue:

```
Phase 1: Blocking get(timeout=1s) for first op      ← matches original behavior
Phase 2: Timed drain until max_pages reached           ← bounded by coalesce_deadline_ms
         or deadline expires (blocking get with remaining timeout)
Merge:   torch.cat host_indices, concatenate         ← one large StorageOperation
         hash_value + token_ids lists
```

The merged operation is submitted to `_backup_io_task` as a single unit. On completion, each original source operation is individually acked via `ack_backup_queue` so that `drain_storage_control_queues()` sees the correct operation IDs.

**Expected impact:** `avg_batch` jumps from ~2.0 (MHA) to ~2048+ depending on queue depth. This eliminates hundreds of roundtrips per second. The coalescer tracks the live `storage_batch_size` value, so auto-tuning changes are reflected immediately.

| Setting | `coalesce_backup_ops` | `coalesce_deadline_ms` | Behavior |
|---------|----------------------|----------------------|----------|
| Default | `true` | `20.0` | Drain up to `storage_batch_size` pages, wait up to 20ms for more |
| Aggressive | `true` | `0` | Skip phase 2 — returns only the first op (no coalescing wait) |
| Conservative | `true` | `50.0` | Wait longer to accumulate larger batches |
| Disabled | `false` | — | Original behavior (one op per dequeue) |

**Observability:** The `coalesce_avg_ops` field in the 60s health log shows the average number of original operations merged per coalesced batch. Also available as `backup_coalesce_avg_ops` in Prometheus via `report_stats()`.

---

## Compression Parameters

Value-level compression at the connector layer. The server stores opaque bytes — compression is purely connector-side with zero server changes.

| Parameter | Env Variable | JSON Key | Type | Default | Description |
|-----------|-------------|----------|------|---------|-------------|
| `codec` | `SGLANG_CAMA_CODEC` | `codec` | `str` | `""` (disabled) | Compression codec. `""` = disabled (raw bytes, zero-copy RDMA Read). `"int8"` = INT8 symmetric quantization (~2x, lossy). `"shuffle_zstd"` = byte-shuffle + zstd (~1.3x, lossless). `"int8+shuffle_zstd"` = chained (~2.6x). **Changing codec requires FLUSH** — existing values use old encoding. |
| `codec_zstd_level` | `SGLANG_CAMA_CODEC_ZSTD_LEVEL` | `codec_zstd_level` | `int` | `3` | Zstd compression level (1–22). Only used when codec includes `shuffle_zstd`. Higher levels improve ratio but increase CPU cost. |

### Codec Trade-offs

| Codec | Ratio | Lossy | RDMA Read | CPU overhead | Use case |
|-------|-------|-------|-----------|-------------|----------|
| `""` (disabled) | 1.0x | No | Zero-copy batch | None | Default — best latency |
| `"int8"` | ~2.0x | Yes (atol~0.05) | Batch raw + decode | Minimal | Double capacity, near-zero accuracy loss |
| `"shuffle_zstd"` | ~1.3x | No | Batch raw + decode | Moderate | Lossless, good for structured data |
| `"int8+shuffle_zstd"` | ~2.6x | Yes | Batch raw + decode | Moderate | Maximum compression |

**Important:** When a codec is enabled, the GET path bypasses zero-copy `mget_rdma` (which lands data directly in tensor buffers) and instead uses `mget_rdma_raw` — a batched RDMA Read into an internal buffer, followed by per-key decompression and `memmove` into the tensor buffer. The RDMA reads are still fully batched (single roundtrip + single doorbell); the decompression loop is sequential in Python. See [09_CODEC_TRADEOFFS.md](09_CODEC_TRADEOFFS.md) for copy accounting, break-even analysis, and batch decode projections.

**Source:** `cama_storage.py` `CamaConfig` dataclass, `codec.py`

---

## Write Deduplication Parameters

Write deduplication checks key existence before writing to avoid redundant RDMA transfers. This is beneficial when the same KV pages are written repeatedly (e.g. common system prompts), but adds an extra `mexists` roundtrip per batch.

| Parameter | Env Variable | JSON Key | Type | Default | Description |
|-----------|-------------|----------|------|---------|-------------|
| `dedup_mode` | `SGLANG_CAMA_DEDUP_MODE` | `dedup_mode` | `str` | `"auto"` | Dedup strategy. `"auto"` starts with dedup enabled but auto-disables after `dedup_auto_window` consecutive low-hit batches, then periodically probes to re-enable on workload shifts. `"always"` keeps dedup on permanently. `"never"` skips the existence check entirely. |
| `dedup_auto_threshold` | `SGLANG_CAMA_DEDUP_AUTO_THRESHOLD` | `dedup_auto_threshold` | `float` | `0.05` | Hit-rate threshold for auto mode. A batch with fewer than 5% of keys already existing is considered "low-hit". |
| `dedup_auto_window` | `SGLANG_CAMA_DEDUP_AUTO_WINDOW` | `dedup_auto_window` | `int` | `2` | Number of consecutive low-hit batches before auto mode disables dedup. |
| `dedup_cost_ratio_threshold` | `SGLANG_CAMA_DEDUP_COST_RATIO_THRESHOLD` | `dedup_cost_ratio_threshold` | `float` | `0.5` | If `exists_ms / transfer_ms` exceeds this for `dedup_auto_window` consecutive batches, auto-disable regardless of hit rate. |
| `dedup_probe_interval` | `SGLANG_CAMA_DEDUP_PROBE_INTERVAL` | `dedup_probe_interval` | `int` | `20` | When dedup is auto-disabled, run a probe batch with dedup ON every this many batches. Set to 0 to disable probing (legacy permanent disable). |
| `dedup_probe_window` | `SGLANG_CAMA_DEDUP_PROBE_WINDOW` | `dedup_probe_window` | `int` | `2` | Consecutive probe batches above `dedup_auto_threshold` needed to re-enable dedup. |

### Dedup Mode Decision Guide

| Workload | Recommended Mode | Rationale |
|----------|-----------------|-----------|
| **Shared system prompts** (many users, same prefix) | `"always"` | High hit rate — dedup saves significant write bandwidth |
| **Unique prompts** (each request is different) | `"never"` | Near-zero hit rate — the `mexists` roundtrip is pure overhead |
| **Mixed / unknown** | `"auto"` (default) | Starts with dedup, auto-disables if low-hit, probes every 20 batches to re-enable on workload shift |
| **Write-then-read benchmarks** | `"auto"` | Dedup disables during write phase, re-enables when reads start hitting |
| **Benchmarking (throughput only)** | `"never"` | Eliminates dedup overhead for clean throughput measurements |

**Source:** `cama_storage.py` `CamaConfig` dataclass, `batch_set_v1()`, `_batch_set_exists_dedup()`

---

## Reconnection Parameters

Automatic reconnection with exponential backoff for transport failures. The connector explicitly builds a `ReconnectConfig` from these fields and passes it to the client constructor, so the client's own `reconnect=True` default does not apply in SGLang deployments.

| Parameter | Env Variable | JSON Key | Type | Default | Description |
|-----------|-------------|----------|------|---------|-------------|
| `reconnect_enabled` | `SGLANG_CAMA_RECONNECT_ENABLED` | `reconnect_enabled` | `bool` | `true` | Enable automatic reconnection with exponential backoff after transport failures. |
| `reconnect_max_retries` | `SGLANG_CAMA_RECONNECT_MAX_RETRIES` | `reconnect_max_retries` | `int` | `10` | Max attempts before giving up. Total worst-case delay: ~152s (0.5+1+2+4+8+16+30+30+30+30). |
| `reconnect_base_delay_s` | `SGLANG_CAMA_RECONNECT_BASE_DELAY_S` | `reconnect_base_delay_s` | `float` | `0.5` | Base delay for exponential backoff (delay = min(base × 2^attempt, max) ± 10% jitter). |
| `reconnect_max_delay_s` | `SGLANG_CAMA_RECONNECT_MAX_DELAY_S` | `reconnect_max_delay_s` | `float` | `30.0` | Maximum cap for backoff delay. |

> **Note:** Environment variables are strings. `SGLANG_CAMA_RECONNECT_ENABLED=0` may be treated as truthy depending on how the value is coerced. To reliably disable reconnect, use JSON config or `extra_config` with `"reconnect_enabled": false` (proper boolean).

**Source:** `cama_storage.py` `CamaConfig` dataclass, `reconnect.py` `ReconnectConfig`

### Reconnection Behavior

When a transport failure is detected (e.g., `BrokenPipeError`, `WR_FLUSH_ERR`, `ConnectionResetError`):

1. The failed operation raises to the caller (batch returns partial results for that key)
2. The reconnect engine starts exponential backoff: `delay = min(base × 2^attempt, max) ± jitter`
3. On successful reconnect, MRs are re-registered automatically (RDMA pools re-register on the new PD)
4. The `_on_reconnect()` callback fires: refreshes server info, resets dedup state

**Pool-aware reconnection:**
- **Non-owner failure** (conn 1..N-1): PD and MRs survive; only the failed transport is replaced. Sub-second recovery.
- **PD-owner failure** (conn[0]): Full pool rebuild — all connections torn down and rebuilt, all MRs re-registered. Takes 1-3 seconds.

---

## Profiling Parameters

These environment variables control the optional Pyroscope + NVTX profiling subsystem. When `SGLANG_CAMA_PROFILING_ENABLED` is `false` (default), the profiling decorators are zero-cost no-ops.

| Env Variable | Type | Default | Description |
|-------------|------|---------|-------------|
| `SGLANG_CAMA_PROFILING_ENABLED` | `bool` | `false` | Master switch for profiling. When `true`, configures Pyroscope continuous profiler and enables NVTX range markers. Requires `pyroscope` and `nvtx` packages (gracefully degrades if missing). |
| `SGLANG_CAMA_PROFILING_SERVER_ADDRESS` | `str` | `"http://0.0.0.0:4040"` | Pyroscope server URL for sending profiling data. |
| `SGLANG_CAMA_PROFILING_SERVICE_NAME` | `str` | `"cama-connector"` | Application name reported to Pyroscope. Used for filtering/grouping in the Pyroscope UI. |

**Source:** `environ.py` lines 292-295, `profiling.py` lines 42-116

---

## Extra Config Parameters

These parameters are only available through the `--hicache-storage-backend-extra-config` JSON string or the JSON config file. They are not environment variables.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extra_backend_tag` | `str` | None | Key prefix for multi-instance isolation. When set, all keys are prefixed with `{tag}_`. Use when multiple SGLang instances share one PrisKV cluster to prevent key collisions. |

**Source:** `cama_storage.py` lines 294-298

---

## Backup Thread Prometheus Metrics

These metrics are pushed via `update_sglang_metrics()` every 60 seconds and flow through the `report_stats()` → server Prometheus pipeline.

| Metric | Type | Description |
|--------|------|-------------|
| `cama_client_host_alloc_drops` | counter | Pages dropped due to host pool exhaustion |
| `cama_client_backup_queue_depth` | gauge | Pending backup operations (sampled every 60s) |
| `cama_client_backup_ops_completed` | counter | Successful backup operations |
| `cama_client_backup_ops_failed` | counter | Failed backup operations |
| `cama_client_backup_in_flight` | gauge | Currently executing backup tasks |
| `cama_client_backup_avg_latency_ms` | gauge | Average backup latency over last 60s window |
| `cama_client_backup_jitter_total_ms` | gauge | Total jitter sleep applied over last 60s window (0 when disabled) |
| `cama_client_backup_avg_gap_ms` | gauge | Average time between sub-batches within a single backup op (low = thundering herd) |
| `cama_client_backup_jitter_cfg_ms` | gauge | Configured `backup_jitter_ms` value (for dashboard display) |
| `cama_client_backup_batch_size` | gauge | Current effective `storage_batch_size` (changes when `batch_size_auto=true`) |
| `cama_client_backup_coalesce_avg_ops` | gauge | Average source operations merged per coalesced batch over last 60s window (1.0 when coalescing disabled) |

---

## Logging Behavior

CAMA uses two log levels for its operational output:

| Level | What | When |
|-------|------|------|
| **INFO** | Periodic health summaries (60s thread health, ~10s `get_stats()`), startup/shutdown, one-time events (dedup auto-disabled, dedup reset, RDMA buffer registration), error counters (WARNING when non-zero) | Always visible at default log level |
| **DEBUG** | Per-batch details (hit/miss counts, dedup stats, phase timings, batch sizes), per-request prefetch lifecycle (enqueue, pickup, query, dispatch, I/O completion) | Visible only with `SGLANG_LOG_LEVEL=DEBUG` |

To enable per-batch and per-request logging for debugging:
```bash
SGLANG_LOG_LEVEL=DEBUG python -m sglang.launch_server ...
```

---

## HiCache CLI Arguments Affecting CAMA

These are SGLang `launch_server` arguments. They are not CAMA-specific but directly control how CAMA is activated and configured.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-hierarchical-cache` | flag | off | **Required.** Enables the HiCache system. Without this, no L3 storage backend is used. |
| `--hicache-storage-backend` | choice | None | Set to `cama` to activate the CAMA connector. Other options: `mooncake`, `aibrix`, `nixl`, `hf3fs`, `eic`, `file`. |
| `--hicache-write-policy` | choice | `write_through` | When to write KV pages to L3. `write_through` writes immediately on eviction from L1→L2. `write_back` defers writes. `write_through_selective` writes selectively. Use `write_through` for cache warming. |
| `--hicache-ratio` | float | `2.0` | Ratio of L2 host cache size to L1 GPU KV cache size. `2.0` means L2 is 2x the GPU KV cache. |
| `--hicache-size` | int | None | Explicit L2 host cache size in bytes. Overrides `--hicache-ratio` when set. |
| `--hicache-mem-layout` | choice | `page_first` | Host memory layout. CAMA requires `page_first`, `page_first_direct`, or `page_head`. Other layouts will raise an assertion error at RDMA buffer registration. |
| `--hicache-storage-backend-extra-config` | JSON str | None | JSON string passed as `extra_config` dict to the storage backend. Highest-priority configuration source for CAMA. |

---

## Example Configurations

### Minimal JSON Config File

```json
{
    "remote_addr": "10.0.0.1"
}
```

All other parameters use defaults (port 6379, no password, check_server off).

### Full JSON Config File

```json
{
    "remote_addr": "10.0.0.1",
    "remote_port": 18001,
    "password": "",
    "use_mput_mget": true,
    "check_server": false,
    "op_timeout_s": 10.0,
    "io_workers": 16,
    "pool_size": 8,
    "send_buf_size": 0,
    "nic_striping": true,
    "dedup_mode": "auto",
    "dedup_auto_threshold": 0.05,
    "dedup_auto_window": 2,
    "dedup_probe_interval": 20,
    "dedup_probe_window": 2,
    "reconnect_enabled": true,
    "reconnect_max_retries": 10,
    "reconnect_base_delay_s": 0.5,
    "reconnect_max_delay_s": 30.0
}
```

### Environment Variable Equivalent

```bash
export SGLANG_CAMA_REMOTE_ADDR=10.0.0.1
export SGLANG_CAMA_REMOTE_PORT=18001
export SGLANG_CAMA_PASSWORD=""
export SGLANG_CAMA_USE_MPUT_MGET=1
export SGLANG_CAMA_CHECK_SERVER=0
export SGLANG_CAMA_POOL_SIZE=4
export SGLANG_CAMA_SEND_BUF_SIZE=0
export SGLANG_CAMA_NIC_STRIPING=1
export SGLANG_CAMA_DEDUP_MODE=auto
export SGLANG_CAMA_RECONNECT_ENABLED=1
export SGLANG_CAMA_RECONNECT_MAX_RETRIES=5
export SGLANG_CAMA_RECONNECT_BASE_DELAY_S=0.5
export SGLANG_CAMA_RECONNECT_MAX_DELAY_S=30.0
```

### Kubernetes extra_config (Highest Priority)

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{
        "remote_addr": "priskv-service.default.svc.cluster.local",
        "remote_port": 18001,
        "check_server": true,
        "extra_backend_tag": "worker-pod-0",
        "op_timeout_s": 15.0,
        "io_workers": 16,
        "pool_size": 8,
        "dedup_mode": "auto",
        "prefetch_io_workers": 2,
        "backup_io_workers": 2,
        "storage_batch_size": 64
    }'
```

### Multi-Instance with Key Isolation

```bash
# Instance A — keys prefixed with "cluster-a_"
--hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "extra_backend_tag": "cluster-a"}'

# Instance B — keys prefixed with "cluster-b_"
--hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "extra_backend_tag": "cluster-b"}'
```

### Enabling Profiling

```bash
export SGLANG_CAMA_PROFILING_ENABLED=1
export SGLANG_CAMA_PROFILING_SERVER_ADDRESS=http://pyroscope.internal:4040
export SGLANG_CAMA_PROFILING_SERVICE_NAME=sglang.cama.prod
```

---

## Validation Behavior

What happens when configuration is wrong:

| Misconfiguration | Error | When |
|-----------------|-------|------|
| `remote_addr` missing from JSON/extra_config | `ValueError: 'remote_addr' is required in config file` | During `CamaStorage.__init__` |
| `SGLANG_CAMA_CONFIG_PATH` points to invalid file | `RuntimeError: Failed to load config from <path>: ...` | During `CamaStorage.__init__` |
| JSON file has invalid JSON syntax | `RuntimeError: Failed to load config from <path>: Expecting value...` | During `CamaStorage.__init__` |
| `priskv` package not installed | `ImportError: Please install the priskv package` | During `CamaStorage.__init__` or preflight |
| PrisKV server unreachable (check_server=false) | `RuntimeError: Cama preflight check failed: PrisKV server at ... is not reachable` | During `ServerArgs.check_server_args()` (before model load) |
| PrisKV server unreachable (check_server=true) | `RuntimeError: PrisKV server not reachable after 600s` | During `CamaStorage.__init__` (after polling for 10 min) |
| Incompatible memory layout | `AssertionError: Cama storage backend only supports page_first, page_first_direct, or page_head layout` | During `register_mem_pool_host` |
| RDMA buffer registration failure | `RuntimeError: PrisKV reg_memory returned 0 — RDMA buffer registration failed` | During `register_mem_pool_host` |

---

## Related Documents

- [01_OVERVIEW.md](01_OVERVIEW.md) — What CAMA is and why it exists
- [04_DEPLOYMENT_GUIDE.md](04_DEPLOYMENT_GUIDE.md) — Step-by-step deployment using these parameters
- [05_TROUBLESHOOTING.md](05_TROUBLESHOOTING.md) — Debugging configuration errors
- [10_GPU_CACHE_PRESSURE.md](10_GPU_CACHE_PRESSURE.md) — GPU cache pressure, L3 throughput bottlenecks, and batch size tuning under load
- [11_THROUGHPUT_VARIABILITY.md](11_THROUGHPUT_VARIABILITY.md) — Why per-rank GB/s fluctuates and how to interpret it

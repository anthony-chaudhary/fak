# CAMA Design Decisions & Rationale

> Why CAMA works the way it does. For code reviewers, future maintainers, and engineers who want to understand the trade-offs.

---

## Why Direct PrisKV Instead of aibrix_kvcache

The `AibrixKVCacheStorage` connector wraps `aibrix_kvcache.BaseKVCacheManager`, which introduces an unnecessary caching layer between SGLang's HiCache and PrisKV. The original AIBrix connector had 10 specific problems:

| # | Problem | CAMA's Solution |
|---|---------|----------------|
| 1 | No MLA support (raises NotImplementedError) | Auto-detect via `is_mla_model`, adjust key naming and buffer meta |
| 2 | No pipeline parallel (ignores pp_rank) | Include pp_rank in key suffix |
| 3 | No V1 API (batch_get_v1 / batch_set_v1) | Implement V1 API, added to zero-copy backend list |
| 4 | No zero-copy (3x Python memcpy per page) | Direct SGL construction into RDMA-registered host buffer |
| 5 | No write deduplication | exists() check before every write |
| 6 | No bandwidth metrics | `get_stats()` returns `StorageMetrics` with bandwidth data |
| 7 | No meaningful buffer registration | Single `reg_memory` of entire host KV buffer |
| 8 | No health check or warmup | Poll-based health check + 3-phase warmup with RDMA round-trip validation |
| 9 | No config flexibility | Triple-source config (extra_config > file > env) |
| 10 | Double-caching via BaseKVCacheManager | Direct PriskvClient connection — no intermediate cache layers |

**The fundamental issue:** SGLang's HiRadixCache already manages caching policy (L1/L2 eviction, promotion, prefetch scheduling). Wrapping `BaseKVCacheManager` around PrisKV creates a second, uncoordinated cache layer that adds overhead without benefit.

CAMA's approach: talk directly to `PriskvClient`. The only dependency is the `priskv` pip package — no `aibrix_kvcache` library at all.

---

## Single Buffer Registration vs Per-Page

**Decision:** Register the entire host KV buffer as one contiguous RDMA region.

**Alternative:** Register each page individually (as the original `PrisKVConnector` in aibrix does with `register_slabs`).

| Approach | Registration Cost | SGL Construction | Risk |
|---------|------------------|-----------------|------|
| Single buffer (CAMA) | O(1) — one `reg_memory` call | SGL points to sub-region within registered range | What if PrisKV requires exact-match registration? |
| Per-page (aibrix) | O(N) — one call per slab | SGL matches its exact registration | Safe but expensive, requires tracking handle-per-slab dict |

**Why single buffer works:** The RDMA specification allows any SGL whose `iova` (I/O virtual address) falls within a registered memory region's `[base, base+length)` range. PrisKV's `reg_memory` registers a contiguous range, and `SGL(ptr, size, handle)` is valid as long as `ptr` and `ptr+size` are within that range. This is standard RDMA MR (Memory Region) behavior.

**Empirical validation:** The warmup sequence explicitly tests this by registering small numpy buffers and doing an SGL round-trip. If sub-region SGLs were broken, warmup would fail immediately.

**The win:** O(1) registration regardless of how many pages exist. For large models with millions of pages, this avoids a startup bottleneck.

---

## Adopting Mooncake's Patterns

CAMA deliberately borrows 10 specific patterns from `MooncakeStore` (707 lines). These are proven, production-tested patterns — not speculative design.

| # | Pattern | Why Taken |
|---|---------|-----------|
| 1 | MHA + MLA key naming with TP/PP suffixes | Correct data isolation; well-tested across many deployments |
| 2 | V1 API (batch_get_v1 / batch_set_v1) | Required for zero-copy path activation in cache_controller |
| 3 | Single buffer registration for RDMA | O(1) vs O(N), same RDMA spec guarantees |
| 4 | Write deduplication via exists() check | Proven bandwidth savings for prefix-heavy workloads |
| 5 | StorageMetrics bandwidth tracking | Production observability; same format as mooncake |
| 6 | Triple-source config (extra_config > file > env) | Handles Kubernetes, standalone, and dev deployment modes |
| 7 | Health check with timeout + retry | Fail-fast for misconfiguration, graceful for orchestrated startup |
| 8 | MHA K/V pair postprocessing (zip results[::2], results[1::2]) | Correctly maps per-sub-key results to per-page booleans |
| 9 | extra_backend_tag for multi-instance key isolation | Essential for shared PrisKV clusters |
| 10 | Explicit close() with deregistration | Prevents RDMA memory leaks on detach |

### What Was NOT Taken from Mooncake

| Pattern | Why Not |
|---------|---------|
| Multi-replica replication | PrisKV has no replication protocol; would require PrisKV server changes |
| Multi-NIC per-transfer striping | CAMA uses per-rank NIC assignment (`local_rank % len(endpoints)`), not Mooncake's per-transfer striping across all NICs simultaneously |
| Standalone storage mode | Not needed — PrisKV server is always external |
| Master service coordination | Complexity not justified for current deployment scale |
| gRPC metadata service | PrisKV uses Redis-like protocol natively |

---

## PrisKV 0=success Semantics

**Decision:** Use `result == 0` to mean "success" for all operations (get, set, exists).

**Mooncake's convention:** Mixed — `get` returns `bytes > 0` for success, `set` returns `0` for success, `is_exist` returns `1` for exists. This requires different postprocessing logic for each operation type.

**PrisKV's convention:** Uniform — `0 = success/exists` for everything. This is simpler but creates a **copy-paste risk**: if someone copies Mooncake's `_batch_postprocess` verbatim (which checks `result > 0` for get success), reads would be inverted (all hits become misses).

CAMA's `_batch_postprocess` deliberately uses `r == 0` for both get and set paths, with a comment explaining the difference from Mooncake:

```python
# PrisKV: 0 = success for both get and set (unlike mooncake where
# get success = bytes > 0).
```

---

## Write Deduplication Strategy

**Decision:** Before writing, check if each sub-key already exists. Only write keys that don't exist.

**Trade-off:**

| Cost | Benefit |
|------|---------|
| N extra `exists()` round-trips per batch_set_v1 call | Saves N RDMA write transfers for keys that already exist |
| ~microseconds per exists() call | ~milliseconds per avoided RDMA write (page sizes are 10s of KB to MBs) |

**When this wins:** For workloads with high prefix reuse (coding agents with common system prompts, repeated conversation turns), most pages in a batch_set_v1 call already exist. The exists() check costs microseconds; the avoided RDMA write saves milliseconds. For a 10K-token shared prefix, this might save 80%+ of write bandwidth.

**When this loses:** For purely novel workloads where every key is new, the exists() calls are wasted overhead. But even then, the overhead is small relative to RDMA transfer time.

**Mooncake uses the same strategy** at `mooncake_store.py` lines 503-527, confirming this is a proven pattern.

---

## Individual Ops Fallback (pybind11 Bug Workaround)

**Decision:** Use individual `exists()`/`set()`/`get()` calls instead of `mexists()`/`mset()`/`mget()`.

**Root cause:** PrisKV's pybind11 wrappers pass `std::vector<uint32_t>&` as output parameters. pybind11 copies Python lists into C++ vectors — modifications to the C++ vector are never reflected back to Python. Result: status lists are always all-zeros.

**Impact of the bug:**
- `mexists()` returns all-zeros → dedup thinks every key exists → skips all writes
- `mset()` status is unreliable → can't detect failed writes
- `mget()` status is unreliable → can't detect failed reads

**Resolution:** CAMA's Python client now implements native batch wire-protocol ops (`OP_MTEST`, `OP_MSET`, `OP_MDEL`), bypassing PrisKV's pybind11 wrappers entirely. The `use_mput_mget` config flag (default `true`) controls whether the connector uses native batch ops or falls back to individual ops via ThreadPoolExecutor.

---

## Triple-Source Configuration

**Decision:** Support three configuration sources with explicit priority: `extra_config > JSON file > env vars`.

**Why three sources:**

| Source | Deployment Mode | Use Case |
|--------|----------------|----------|
| `extra_config` (highest) | Kubernetes / runtime attach | Config injected via `--hicache-storage-backend-extra-config` at launch. Enables per-pod configuration without modifying images or env vars. |
| JSON config file | Standalone / on-prem | Config stored on disk, referenced by `SGLANG_CAMA_CONFIG_PATH`. Suitable for persistent deployments with config management. |
| Env vars (lowest) | Development / quick testing | Zero-config defaults work for local dev. Override individual params as needed. |

**Why not merge sources:** The first source with a `remote_addr` wins entirely. Merging across sources would create confusing precedence rules (e.g., "port from env but addr from file"). The single-winner model is simpler to reason about and matches Mooncake's approach.

---

## Preflight Check Architecture

**Decision:** Validate PrisKV connectivity in `ServerArgs.check_server_args()` — before model loading begins.

**Why fail-fast matters:** Model loading takes 5-10 minutes for large models. If PrisKV is misconfigured (wrong address, not running, package not installed), discovering this after model loading wastes significant time.

The preflight check runs a lightweight `exists("__cama_preflight__")` probe. If it fails, SGLang raises immediately with an actionable error message.

**Skip when orchestrated:** When `check_server=true`, the user expects PrisKV to start later. The preflight check would always fail in this case, so it's skipped. Instead, the health check in `__init__` handles the polling.

---

## Multi-NIC Design

**Decision:** Automatically discover RDMA NICs via `rdma_endpoints()` and stripe `mget_rdma` across all NICs in parallel (`nic_striping=True`, default).

### NIC Striping (Default)

When `nic_striping=True` (default) and multiple RDMA endpoints are discovered, the connector passes ALL endpoints to `create_pool()`. The pool creates one connection per NIC (`pool_size` auto-set to `len(endpoints)`) and the client's `_mget_rdma_striped()` partitions keys round-robin across connections, submitting parallel `_mget_rdma_on_conn` via a `ThreadPoolExecutor`. This achieves N× read bandwidth by saturating all available NICs simultaneously.

### Legacy Mode (`nic_striping=False`)

When `nic_striping=False`, falls back to the original per-rank NIC assignment via `endpoints[local_rank % len(endpoints)]`. Each rank gets exactly one NIC for the lifetime of the connection.

### Why Graceful Fallback

The discovery mechanism is defensive by design:
- **No `rdma_endpoints` method** (old client library) → `hasattr` guard returns `[]`, no error
- **Empty endpoint list** (TCP mode or old server) → keep original connection, debug log
- **Single endpoint** → no reconnect needed, info log
- **Exception during discovery** → warning logged, original connection preserved

This ensures CAMA never breaks existing deployments. TCP connections, old PrisKV servers, and old client libraries all continue to work exactly as before.

### Independent PD Fallback

When a pool connection targets a NIC on a different client RDMA device, `connect_with_shared_pd()` fails. The pool falls back to an independent PD for that connection, with separate MR registration. This is transparent to the caller — `reg_memory()` and `dereg_memory()` handle both shared and independent PDs.

### Comparison with Mooncake's Multi-NIC

Mooncake's Transfer Engine stripes each individual transfer across all NICs simultaneously at the byte level. CAMA stripes at the key level — each key's `mget_rdma` goes to one NIC, but different keys are spread across all NICs. This achieves similar aggregate bandwidth with simpler implementation:
- No Transfer Engine dependency required
- Each per-NIC `mget_rdma` is a complete control roundtrip + batch RDMA Read + ack
- Per-NIC metrics track read counts and bytes for load balance verification

---

## Prefetch ThreadPoolExecutor (Replacing Daemon Thread)

**Decision:** Replace the fragile daemon-thread + Queue architecture for prefetch I/O with a `ThreadPoolExecutor` managed by `prefetch_thread_func`.

**Previous design:**
```
prefetch_thread_func → prefetch_buffer (Queue) → prefetch_io_aux_func (daemon thread)
```

**Problems with the old design:**
1. The daemon thread had no exception safety — any exception from the storage backend killed it silently
2. No log output, no recovery, no notification to the scheduler
3. Requests waiting on prefetch would block forever with `prefetch_buffer.get()` never completing
4. Liveness monitoring (checking `is_alive()` and restarting) is fragile and adds complexity

**New design:**
```
prefetch_thread_func → executor.submit(_prefetch_io_task) → ThreadPoolExecutor
```

**Why ThreadPoolExecutor wins:**
- `_prefetch_io_task` is a simple per-operation method (14 lines) with `try/except/finally` — exceptions are caught, logged, and the operation is terminated cleanly
- `finally` block guarantees host memory release even on failure
- `executor.shutdown(wait=True)` in the `finally` of `prefetch_thread_func` ensures clean shutdown — no orphaned transfers
- Futures list enables tracking in-flight operations and reporting `io_completed`/`io_failed` at shutdown
- No need for daemon thread liveness monitoring, restart logic, or `prefetch_buffer` Queue

**Why `prefetch_io_workers` defaults to 2:** See [Architecture Deep Dive: Threading Topology](02_ARCHITECTURE_DEEP_DIVE.md#why-prefetch_io_workers-defaults-to-2).

---

## Error Handling: Graceful Degradation Over Silent Failure

**Decision:** All storage backend errors are caught, logged at ERROR level, counted, and treated as "miss" or "failure" — never allowed to kill threads or propagate silently.

**Previous behavior:**
- `_do_get`/`_do_set`/`_do_exists` caught exceptions but logged at WARNING level (invisible at default INFO log level)
- `batch_exists` short-circuit caught exceptions but logged at WARNING
- No error counters — impossible to distinguish "key doesn't exist" from "backend is broken"
- Prefetch threads had no exception safety — any uncaught exception killed the thread

**New behavior:**

| Layer | Error handling | Logging | Counting |
|-------|--------------|---------|----------|
| `_do_get` / `_do_set` / `_do_exists` | Return error sentinel (-1 or EXISTS_ERROR) | `logger.error` | `_get_errors` / `_set_errors` / `_exists_errors` |
| `as_completed()` timeout | Partial results (completed keys return normally) | `logger.error` with key list | `_exists_timeouts` + per-key counters |
| `batch_exists` short-circuit | Return 0 hits (safe fallback) | `logger.error` | `_exists_errors` |
| `_prefetch_io_task` | `mark_terminate()` + release memory | `logger.exception` | — |
| `prefetch_thread_func` | Revoke + release memory, continue loop | `logger.exception` | — |

**Why ERROR instead of WARNING:** The default log level for production is INFO. WARNING is invisible. Backend errors should never be invisible — they indicate a broken or degrading storage backend that needs operator attention.

**Why cumulative counters:** Per-interval counters get cleared and can miss transient spikes. Cumulative counters make trends obvious: if `_get_errors` grows while `_get_successes` stays flat, the backend is broken. If both grow, the backend is partially healthy. This is reported via `get_stats()` alongside the per-interval bandwidth metrics.

**Why EXISTS_ERROR sentinel (-99):** Without it, `conn.exists()` exceptions returned `EXISTS_MISSING`, making network errors look like cache misses. The sentinel preserves the distinction in `_batch_exist` results while downstream code (batch_exists, dedup) treats both as "not found" for safe degradation.

---

## I/O Timeout via `as_completed()` + `op_timeout_s`

**Decision:** Replace `executor.map()` (blocks indefinitely) with `concurrent.futures.as_completed()` + configurable timeout.

**Problem:** If PrisKV hangs on a single key (e.g., RDMA NIC error, server deadlock), `executor.map()` blocks the entire prefetch pipeline indefinitely. There's no way to set a per-batch timeout on `map()`.

**Solution:** `as_completed(futures, timeout=self.config.op_timeout_s)` raises `TimeoutError` if any future hasn't completed within the deadline. Completed futures return their results normally; only timed-out keys get the default error value (-1 or EXISTS_ERROR).

**Trade-off:**

| Aspect | `executor.map()` | `as_completed()` + timeout |
|--------|-----------------|---------------------------|
| Simplicity | Simpler — one line | More code — futures dict, result array, try/except |
| Timeout support | None | Configurable via `op_timeout_s` |
| Partial results | All or nothing | Completed keys return normally |
| Key ordering | Preserved | Requires index tracking via futures dict |

**Why 10s default:** Most RDMA operations complete in microseconds to low milliseconds. A 10-second timeout is very generous — it only fires for genuinely hung operations. For cross-rack RDMA with higher latency, operators can increase it via config.

---

## Per-Rank Prewarm (Multi-GPU)

**Decision:** Signal prewarm config via env var (`_SGLANG_CAMA_PREWARM_SIGNAL`) so each spawned rank process starts its own independent prewarm daemon.

**Problem:** The original prewarm (v1.16.0) ran `start_cama_prewarm()` in the parent process during `check_cama_preflight()`. But SGLang uses `mp.set_start_method("spawn")` for all rank processes — children get fresh Python interpreters and cannot see the parent's `PrewarmRegistry`, daemon threads, or module-level state. The parent's prewarm was always wasted.

Before v0.31.0 bumped RDMA buffers from 8→32 MB (now 16 MB since v1.0.0), the blocking `ibv_reg_mr` page pinning was fast enough (~100-200ms) that nobody noticed. With 16 MB buffers, MR registration takes 250ms+ per connection, visible in every rank's [2/7] startup phase.

**Why env-var signaling:**

| Alternative | Problem |
|-------------|---------|
| Shared memory / `mp.Value` | Requires `mp.set_start_method("fork")` or explicit `Manager()` — too invasive |
| File-based signal | Race conditions, cleanup complexity, cross-platform issues |
| CLI flag propagation | SGLang doesn't expose custom flags to rank processes |
| **Env var** (chosen) | Survives `mp.Process("spawn")`, zero infrastructure, underscore-prefix = internal |

**Timing:** `environ.py` is imported very early in every rank process (via `server_args.py` → `environ.py`), well before `Scheduler.__init__()` and the expensive model loading. This gives the prewarm daemon the maximum window (~minutes for large models) to complete connection setup + MR registration in the background.

```
Parent (preflight)                  Child rank process
─────────────────                   ──────────────────
check_cama_preflight()
  ├─ connectivity probe
  └─ os.environ["_SGLANG_CAMA_     import environ.py (very early)
      PREWARM_SIGNAL"] = {...}  ──>   └─ _maybe_start_rank_prewarm()
                                           └─ start_cama_prewarm() → daemon
mp.Process(...).start()        ──>  Scheduler.__init__()
                                      init_model_worker()       ← SLOW (minutes)
                                      ── prewarm runs in parallel ──
                                      CamaStorage.__init__()
                                        claim_prewarmed_connection()
                                          → finds ready connection!
```

**Edge cases:**

| Case | Behavior |
|------|----------|
| `tp=1` | One child, one prewarm, claim succeeds |
| `tp=8` | 8 children each start independent prewarm daemons |
| `dp>1` | DP controller spawns further children, each inherits env var |
| `check_server=true` | No env var set (preflight returns early), children fall back to blocking connect |
| Standalone mode | `environ.py` is not the patched version, function doesn't exist, no impact |
| Parent process | `environ.py` is imported in parent BEFORE `check_cama_preflight()` sets the env var → parent never fires prewarm from environ.py |

**Env var consumed on read:** `_maybe_start_rank_prewarm()` uses `os.environ.pop()` to remove the env var after reading it, preventing double-fire if the process spawns further children.

---

## Future Considerations

### TurboKV Replacement

[TurboKV](../../turbokv/TURBOKV_DOCUMENTATION.md) is a next-generation KV cache server designed as a drop-in replacement for PrisKV with 10x throughput targets. It eliminates PrisKV's global LRU spinlock bottleneck via a shared-nothing, zero-lock architecture.

When TurboKV is production-ready, CAMA may need to:
- Adapt to TurboKV's wire protocol (if different from PrisKV's Redis-like protocol)
- Support TurboKV's lease/pin system for eviction protection
- Take advantage of TurboKV's multi-NIC striping

### Batch Operations (Completed)

Native batch wire-protocol ops are now implemented in `cama-client` and enabled by default via `use_mput_mget=true`:

- `_batch_exist` → `conn.mexists()` (single `OP_MTEST` roundtrip)
- `_put_batch_zero_copy` → `conn.mset()` (single `OP_MSET`, with send-buffer size guard)
- `_get_batch_zero_copy` → `conn.mget_rdma()` (single `OP_MGET_RDMA` + batch RDMA Read)
- Server dispatches batch ops to shards in parallel via `fanOutShards`

### Batch RDMA GET (OP_MGET_RDMA) — Page-Size Independence

**Decision:** Batch the entire GET control plane into a single roundtrip, post all RDMA Reads with one doorbell, and eliminate the Python ThreadPoolExecutor for reads.

**Problem:** With individual `get()` calls, each KV cache page required its own TCP control roundtrip + RDMA Read + ReadAck. At `page_size=32`, a 100K-token prefill needed ~3,125 roundtrips through a ThreadPoolExecutor, making latency scale linearly with page count (~437ms vs ~126ms at `page_size=128`).

**Solution:** Three new opcodes (`OP_MGET_RDMA` 0x34, `OP_MGET_READ_READY` 0x35, `OP_BATCH_READ_ACK` 0x36):

1. Client sends all keys in one `OP_MGET_RDMA` message
2. Server fans out to shards, collects RDMA coordinates (rkey, remote_addr, length), grants leases in bulk, returns one `OP_MGET_READ_READY`
3. Client builds a linked WR list and posts all RDMA Reads with a single `ibv_post_send` doorbell (chunked at `MAX_SEND_WR=128`)
4. Client sends one `OP_BATCH_READ_ACK` with all WC statuses

**Result:** Performance is bandwidth-limited, not roundtrip-limited. `page_size=32` performs the same as `page_size=128`.

**Capability detection:** Server advertises `"mget_rdma"` in handshake capabilities. Client auto-detects and falls back to sequential `get()` with old servers.

**Migration safety:** If any shard involved in the batch is migrating (non-ODP), the server falls back to inline `RESP_MULTI_VALUE` response. The client detects the response opcode and handles accordingly. This whole-batch fallback is simpler than mixed inline/RDMA responses, and migrations are rare/brief (~10s).

### Connection Pooling

The `_io_pool` ThreadPoolExecutor (16 workers) is no longer needed for the GET path when `mget_rdma` is available — a single call replaces all N concurrent `get()` futures. The pool is still used for SET operations and as a fallback when `mget_rdma` is unavailable.

### Async Operations

Current implementation is synchronous. For higher throughput:
- Investigate PrisKV's async callback API (if available)
- Wrap synchronous calls in an async executor
- Consider overlapping RDMA transfers with GPU compute

---

## Why Load-Back Backpressure (SGLang Patch)

When CAMA's cache hit rate is high, `load_back()` eagerly allocates GPU device
memory for every cache hit. Without backpressure, the `load_queue` grows
unbounded, consuming all free device slots before decode can allocate — causing
GPU OOM at decode time.

**Why SGLang doesn't already handle this:**

| Existing mechanism | What it actually checks | Prevents device OOM? |
|---|---|---|
| `prefetch_rate_limited()` | Host memory (`prefetch_capacity_limit`) | No — host budget only |
| `load_back(mem_quota=)` | Device quota if non-None | No — `mem_quota` was never wired (always `None`) |
| PrefillAdder budget | `available_size() + evictable_size()` | No — doesn't account for load_queue in-flight tokens |
| `alloc()` failure | Allocator exhaustion | Too late — decode is already starved |

**Our patch** (in `cache_controller.py` + `schedule_policy.py`):
1. Reserves 20% of device pool for decode (`load_back_headroom_pct`, configurable 5-50%)
2. Guards `load()` with pending-cap + headroom checks before calling `alloc()`
3. Tracks `load_tokens_pending` across `load()` → `start_loading()` lifecycle
4. Wires `mem_quota=available_size()` into `init_load_back()` (activates existing dead-code guard)
5. Adds device pressure gate to `prefetch_rate_limited()` (stops prefetch at 2x headroom)

Full audit: [`docs/load-back-backpressure-audit.md`](../../docs/load-back-backpressure-audit.md)

---

## Related Documents

- [01_OVERVIEW.md](01_OVERVIEW.md) — What CAMA is and why it exists
- [02_ARCHITECTURE_DEEP_DIVE.md](02_ARCHITECTURE_DEEP_DIVE.md) — The "what" and "how"
- [05_TROUBLESHOOTING.md](05_TROUBLESHOOTING.md) — Practical debugging
- [09_CODEC_TRADEOFFS.md](09_CODEC_TRADEOFFS.md) — Compression vs zero-copy trade-offs, copy accounting, batch decode projections
- [load-back-backpressure-audit.md](../../docs/load-back-backpressure-audit.md) — Device OOM audit, pre-existing mechanism analysis
- `cama-connector-plan.md` (reference archive) — Original design plan
- [aibrix_mooncake_integration_strategy.md](../../aibrix_mooncake_integration_strategy.md) — Hybrid approach analysis
- [TURBOKV_DOCUMENTATION.md](../../turbokv/TURBOKV_DOCUMENTATION.md) — TurboKV as next-gen replacement

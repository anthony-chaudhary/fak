# Warmup Design: Data-Driven Server-Feedback System

## Why the Old Time-Based Warmup Was Removed

The original 3-phase system (AGGRESSIVE 30s -> RAMP 60s -> STEADY) had a
fundamental flaw: **the timer started at connector init, not when data first
flows.**

SGLang model loading takes 10-20+ minutes for large models. By the time KV
cache pages actually arrive, warmup has already expired:

```
t=0s    connector init, warmup timer starts -> AGGRESSIVE
t=30s   AGGRESSIVE -> RAMP (nothing happened, no data)
t=90s   RAMP -> STEADY (nothing happened, no data)
t=600s  SGLang starts inference, first pages flow — warmup already expired
```

Additional issues:
- **No server feedback.** The connector never checked if the server completed
  its slab rebuild — it just waited 90 seconds.
- **Unnecessary RAMP phase.** 60s of linear interpolation was overengineered.
  The server handles slab rebuild complexity; the connector just needs to know
  "ready or not."
- **Not reset on reconnect.** If reconnect happened during warmup, stale phase
  references caused wrong batch sizes.

## How the New System Works

### Two Phases: COLD and STEADY

```
INIT (no phase)
  |
  |  first record_pages() call
  v
COLD -----> STEADY
       ^          ^
       |          |
   server poll    timeout (5 min safety valve)
   confirms
```

- **INIT** — No timer running. Returns steady-state values (no throttling
  while idle). This is the state during model loading.
- **COLD** — Activated when `record_pages()` is first called with count > 0.
  Uses aggressive coalescer params (large batch, tight deadline, no jitter,
  skip dedup). Spawns a background poller thread.
- **STEADY** — Activated when server confirms all shards have rebuilt their
  allocators, OR after a 5-minute safety timeout.

### Server Readiness Detection

The poller thread calls `conn.maintenance_status()` every ~2 seconds (with
jitter). It checks that every shard's `detection.status` is one of:

- `"rebuilt"` — allocator has been rebuilt for the detected page size
- `"detected"` — page size detected but rebuild not needed
- `"disabled"` — auto-detection disabled (manual config)

If any shard reports `"warming_up"`, the poller keeps waiting.

### The model_page_bytes Hint Flow

1. `CamaStorage.register_mem_pool_host()` sends `model_page_bytes` via
   `conn.report_stats({"model_page_bytes": N})`
2. Server's `handle_stats.go` calls `SetModelPageHint(N)`
3. `SetModelPageHint` broadcasts to all shards
4. Each shard with an empty allocator fast-swaps to optimized slab classes
5. Shard status changes from `"warming_up"` to `"rebuilt"`
6. Poller detects all shards ready -> COLD -> STEADY

This means warmup typically completes in ~1 second after first data, since
the hint was already sent during `register_mem_pool_host()`.

## Why RAMP Phase Was Eliminated

The old RAMP phase linearly interpolated between aggressive and steady-state
parameters over 60 seconds. This was unnecessary because:

1. The server's slab rebuild is atomic — once it's done, the allocator is
   optimized. There's no gradual improvement that needs matching.
2. The `min_batch_size` floor (default 256) prevents the adaptive batch
   sizing from death-spiraling, which was the original motivation for
   blocking adaptive sizing during RAMP.

## min_batch_size Floor

The old system blocked adaptive batch sizing entirely during AGGRESSIVE and
RAMP phases. The new system uses a simpler approach: `allow_adaptive_sizing()`
always returns `True`, but the adaptive sizing code enforces a floor:

```python
min_floor = max(self._batch_size_min, self._warmup._config.min_batch_size)
```

This prevents the death spiral without requiring phase-based blocking.

## Config Migration

| Old Key | New Key | Default |
|---------|---------|---------|
| `warmup_aggressive_batch_size` | `warmup_cold_batch_size` | batch_size_max |
| `warmup_aggressive_jitter_ms` | `warmup_cold_jitter_ms` | 0.0 |
| `warmup_aggressive_deadline_ms` | `warmup_cold_deadline_ms` | 2.0 |
| `warmup_min_seconds` | (removed) | — |
| `warmup_max_aggressive_seconds` | (removed) | — |
| `warmup_pages_threshold` | (removed) | — |
| `warmup_ramp_seconds` | (removed) | — |
| — | `warmup_min_batch_size` | 256 |
| — | `warmup_server_poll_interval_s` | 2.0 |
| — | `warmup_server_poll_timeout_s` | 300.0 |

Old `aggressive_*` keys are automatically mapped to `cold_*` with a
deprecation warning if the new keys are not explicitly set.

## Reconnect Behavior

On reconnect, `CamaStorage._on_reconnect()`:
1. Calls `warmup.reset()` — stops poller, returns to INIT
2. Re-sends `model_page_bytes` hint (server may have restarted)
3. Next `record_pages()` call re-enters COLD and spawns a new poller

## Expected Behavior

```
t=0s    connector init, warmup state = INIT (steady-state params)
...     model loading, RDMA setup (no warmup running)
t=600s  SGLang starts inference, first pages arrive
t=600s  record_pages() -> COLD activated, poller spawned
t=601s  poller: maintenance_status() -> all shards "rebuilt" -> STEADY
```

Total warmup: ~1s after first data (vs 90s of wasted time before).

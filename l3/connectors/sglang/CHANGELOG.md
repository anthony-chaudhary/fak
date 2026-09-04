# Changelog

All notable changes to CAMA Standalone will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [1.22.4] - 2026-05-21

### Added

- **Repeatable SGLang upgrade process** (`docs/UPGRADE_SGLANG.md`). The bundled
  SGLang tree can now be rebased onto a new upstream release with one command.
  - `patch_manifest.json` — single source of truth for which SGLang files CAMA
    patches. `deploy.py` reads it instead of a hardcoded 4-file map.
  - `sglang-with-cama-connector/UPSTREAM.txt` — pins the upstream base
    (v0.5.7, commit `232982a`) so it never has to be guessed from file sizes.
  - `upgrade-sglang.py` — clones base + target, 3-way merges every patched file
    (LF-normalized, `--diff3`), writes a conflict report, and on `--apply`
    atomically swaps in the new tree and regenerates `patches/`.
  - `scripts/sync_patches.py` — regenerate `patches/` from the in-tree tree.
  - `scripts/find_cama_patches.py` — verify the manifest is complete vs the
    pinned base.
  - `cama_patchlib.py` — shared helpers with LF-safe file comparison.
  - CI gates: `connector-patch-drift` (patches/ ↔ in-tree) and
    `connector-patch-completeness` (every modified file is in the manifest).

### Fixed

- **Four CAMA patches were invisible to `deploy.py`.** The hardcoded `PATCH_MAP`
  listed only 4 files; ground-truthing against upstream v0.5.7 found the real set
  is 8. `schedule_policy.py` (load-back OOM guard), `scheduler_metrics_mixin.py`
  (scheduler→cache-controller metrics), `hicache_storage.py` (`pp_rank`/`pp_size`
  + `prefix_keys` fix) and `hiradix_cache.py` (reconnect crash fix) were silently
  dropped when deploying onto a fresh SGLang tree. All 8 are now tracked.
- **`patches/` had drifted from the in-tree source of truth** (cache_controller
  off by 76 lines, missing the load-back backpressure hotfix). Regenerated and
  CI-guarded.

### Docs

- `DIFF_STANDALONE_VS_SGLANG.md` corrected: the base is v0.5.7, not v0.5.9 as the
  doc previously claimed. Patch list expanded from 4 to 8 files.

---

## [1.22.3] - 2026-04-16

### Docs

- **SGLang tree folder rename.** `README.md` and `INSTALL.md` updated to reference the new `sglang-with-cama-connector/` directory name (was `sglang-v0.5.7-with-cama-connector/`). Version-agnostic naming decouples folder from SGLang base version for future upgrades.

---

## [1.22.2] - 2026-03-24

### Fixed

- **NIC discovery double-failure safeguard.** Raises `RuntimeError` if both multi-NIC discovery and fallback reconnect fail, preventing use of a broken connection.
- **Partial initialization cleanup.** New `_cleanup_partial_init()` releases stats executor, IO pool, and connection on `__init__` failure.
- **9 stale test fixes.** Patches target `_make_connection` instead of `_PriskvClient`; `io_workers` default 16→8; test renames for clarity.

### Changed

- **`CamaNotReadyError` backpressure handling.** Batch ops (`mset`, `mget_rdma_raw`, `mget_rdma`, `mexists`) catch `CamaNotReadyError` alongside `CamaServerOverloadError`.

---

## [1.22.1] - 2026-03-13

### Fixed

- **Batch operation top-level exception handlers.** `batch_get_v1()`, `batch_set_v1()`, and `_get_batch_zero_copy()` wrapped in catch-all handlers — log errors and return failure arrays instead of propagating exceptions to the scheduler.

### Changed

- **CHANGELOG archive split.** Entries for v1.0.0–v1.10.0 moved to `CHANGELOG-archive.md`.

### Docs

- **Load-back backpressure design decision.** Added "Why Load-Back Backpressure" section to `06_DESIGN_DECISIONS.md` — documents device OOM problem, analysis of insufficient SGLang guards, and CAMA's headroom-based solution.
- **Load-back backpressure audit.** New `docs/load-back-backpressure-audit.md`.

### Tests

- **Backup jitter mock improvement.** Enhanced mock stub with `_warmup` namespace for `effective_batch_size()` / `effective_jitter_ms()`.

### SGLang Tree

- **HiRadix assertion hardening.** 4 hard assertions → graceful warnings with recovery paths; 2 `del` → safe `.pop()`. Prevents cache crashes from internal inconsistencies.

---

## [1.22.0] - 2026-03-13

### Features

- **Server overload backpressure.** `_BackpressureGuard` with exponential backoff (50 ms base → 2 s max, jittered) for `CamaServerOverloadError`. Integrated in mset, mget_rdma_raw, mget_rdma, and mexists I/O paths — one automatic retry before returning failures.
- **Load-back device headroom (SGLang tree).** `load_back_headroom_pct` config (default 0.20) reserves GPU device memory for decode. Backpressure refuses load-back requests when pending tokens exceed capacity or available memory drops below headroom. Prefetch rate-limited under device pressure.

### Fixed

- **Host alloc drop log throttling (SGLang tree).** First 5 drops + every 50th log as WARNING; remainder as DEBUG. Prevents log spam on repeated host allocation failures.
- **Prefetch/backup executor shutdown safety (SGLang tree).** `_flush_pending()`, direct executor submits, and backup thread wrapped in try-except for RuntimeError on executor shutdown. Gracefully drops pending ops instead of crashing.
- **Single-key `exists()` exception handling.** Wrapped in try/except returning `False` on error — prevents unhandled transport exceptions from propagating to the scheduler.

### Changed

- **`io_workers` default 16 → 8.** Reduced default backup I/O workers to match typical deployment thread budgets.
- Startup phase references updated from `[x/6]` → `[x/7]` in prewarm docs, design decisions, and CHANGELOG.
- SGLang tree updated with latest connector code, patches, and schedule_policy quota-aware load-back.

---

## [1.21.0] - 2026-03-13

### Features

- **`model_page_bytes` config.** New `SGLANG_CAMA_MODEL_PAGE_BYTES` env var / config field. When > 0, prewarm sends an early page-size hint to the server via `report_stats()` before model loading completes, allowing slab optimization on `auto`/`sglang` preset servers. Mismatch warning logged if computed page size differs from config value.
- **Preflight model_page_bytes propagation.** Preflight signal includes `model_page_bytes` for child-process prewarm adoption.

### Fixed

- **HiRadix reconnection crash (SGLang tree).** `writing_check()` guarded against `IndexError: pop from empty list` when `ack_write_queue` is cleared by a concurrent `reset()` during CAMA reconnection. `reset()` now releases lock refs for in-flight write/load acks, preventing non-evictable orphan nodes after reconnection.
- **Warmup controller `local_rank` wiring (SGLang tree).** `HiCacheController` now passes `local_rank` to warmup controller constructor.

### Changed

- **Rank-gated logging (SGLang tree).** Backend factory creation and init phase messages logged at INFO for rank 0, DEBUG for others — reduces multi-TP log noise.
- **Init log condensation (SGLang tree).** CamaStorage init phases condensed to single-line timing summaries (7 phases). Verbose per-step messages moved to DEBUG.
- SGLang tree updated with latest connector code and patches.

---

## [1.20.4] - 2026-03-13

### Changed

- **Profiling service name rebrand.** Default Pyroscope service name changed from `"aibrix.kvcache"` to `"cama-connector"` in `profiling.py`, `environ.py` patch, integration patch, and all documentation.
- **Architecture deep-dive clarification.** `gil_only=False` documented as capturing both Python and native C/C++ frames (pybind11 RDMA extension, PrisKV).
- SGLang tree updated with latest connector code.

---

## [1.20.3] - 2026-03-13

### Fixed

- **Dedup counter reset on re-enable.** `_dedup_batches_since_disable` is now reset to 0 when dedup is re-enabled after a probe succeeds. Previously the stale count carried over, skewing the next cost-aware disable window.
- **Dedup cost streak reset on good batch.** `_dedup_cost_streak` is now reset alongside `_dedup_low_hit_streak` when a batch has acceptable hit rate. Previously, cost streak accumulated across unrelated low-hit episodes.
- **Warmup poller stop event race.** `_poll_server_ready` now captures `self._poller_stop` into a local variable before entering the poll loop. If `reset()` replaces the event mid-poll, the old thread correctly observes its original stop signal instead of spinning indefinitely.

### Added

- **Dedup stats export.** `dedup_batches_since_disable` and `dedup_probe_hit_streak` added to storage metrics dict for Prometheus/logging visibility.

### Changed

- **Architecture deep-dive line counts.** Updated `cama_storage.py` line count (1,450→2,500) and module table (added warmup.py, prewarm.py, codec.py; updated preflight.py/profiling.py counts).
- SGLang tree updated with latest connector code.

---

## [1.20.2] - 2026-03-13

### Added

- **Saturation metrics.** `coalesce_fill_pct` (batch size as % of coalesce max) and `transfer_utilization_pct` (transfer phase as % of total I/O latency) added to storage metrics and stats export. Logged as `fill=%.0f%% xfer_util=%.0f%%`.

### Changed

- **`pool_size` default 4→8.** Aligned across preflight, prewarm, environ, and config reference.
- **`reconnect_max_retries` default 5→10.** Extends worst-case reconnection window from ~15.5s to ~152s.
- **Buffer size docs 32→16 MB.** Aligned documentation with current client defaults.
- **Version metadata updated.** FolderOverview, CHANGELIST, DIFF_STANDALONE_VS_SGLANG refreshed.
- SGLang tree updated with latest connector code.

---

## [1.20.1] - 2026-03-13

### Fixed

- **Warmup stuck in INIT on backup failure.** `record_pages()` was inside the `try` block of `_backup_io_task` — if `_page_backup()` raised, execution jumped to `except` and `record_pages()` was never called. This is the sole trigger for the INIT->COLD warmup transition. Moved page counting and `record_pages()` into the `finally` block so partial progress always credits the warmup system.
- **Adaptive batch sizing wrote to dead field.** Adaptive sizing (lines ~1406-1413) updated `self.storage_batch_size`, but the coalescer and `_page_backup` both read `self._warmup.effective_batch_size()`, which returned the warmup controller's `_steady_batch_size` — a value set once at init and never updated. Now calls `update_steady_batch_size()` to propagate changes to the warmup controller.

### Added

- **`WarmupController.update_steady_batch_size()`** — thread-safe setter for runtime adaptive sizing updates.
- **`WarmupController.update_steady_deadline_ms()`** — thread-safe setter for runtime coalesce deadline updates.

---

## [1.20.0] - 2026-03-12

### Features

- **Data-driven warmup system.** Replaced time-based AGGRESSIVE→RAMP→STEADY warmup with data-driven INIT→COLD→STEADY state machine. INIT idles during model loading (no wasted timer). COLD activates on first `record_pages()` call with aggressive coalescer params. STEADY triggered by server shard rebuild confirmation via `maintenance_status()` polling (or timeout fallback after `server_poll_timeout_s`, default 300s).
- **Server readiness poller.** Background thread polls `maintenance_status()` every `server_poll_interval_s` (default 2s) during COLD phase, waiting for all shards to report rebuilt/detected/disabled before transitioning to STEADY.
- **Death-spiral protection.** New `min_batch_size` config (default 256) enforces a floor on adaptive batch sizing at all times, preventing latency-driven halving from collapsing batch size to 1.
- **Warmup reset on reconnect.** When connection reestablishes, warmup resets to INIT (server may have restarted with cold shards). Page size hint is re-sent.

### Changed

- **Warmup config renamed.** `aggressive_batch_size`→`cold_batch_size`, `aggressive_jitter_ms`→`cold_jitter_ms`, `aggressive_deadline_ms`→`cold_deadline_ms`. Old keys auto-mapped with `DeprecationWarning`. Removed: `min_seconds`, `max_aggressive_seconds`, `pages_threshold`, `ramp_seconds`.
- **Rank-gated logging.** `_r0()` helper logs at INFO on rank 0, DEBUG on all others. Applied to init, prewarm, preflight, backup thread health, and batch sizing events. Reduces N×TP duplicate messages.
- **Buffer size docs 32→16 MB.** Comments and docs updated to reflect current client defaults (16 MB send + 16 MB recv per connection, ~32 MB total overhead).
- **Adaptive batch sizing always allowed.** Removed STEADY-phase gate — sizing runs in all phases but respects `min_batch_size` floor.
- **Transient retry jitter.** `_retry_transient()` adds `random.uniform(0.5, 1.5)` multiplier to backoff delay, preventing thundering herd across ranks.
- SGLang tree updated with latest connector code.

---

## [1.19.1] - 2026-03-12

### Fixed

- **Double-pool OOM during NIC discovery.** Both striping and non-striping paths created a NEW pool before closing the old one, pinning 2x RDMA buffers simultaneously (~1+ GB per rank). Now closes old pool BEFORE creating new one.
- **Pool size cap in `_make_connection`.** Belt-and-suspenders cap at 32 connections, preventing unbounded endpoint scaling from reaching the client.
- **Prewarm `pool_size` default 4→8.** Aligned with `CamaConfig` default (was 4 since inception). Mismatch caused fingerprint rejection → prewarm pool leak → fresh pool creation, doubling memory usage during init.

### Changed

- SGLang tree updated with latest connector code.

---

## [1.19.0] - 2026-03-12

### Changed

- **Adaptive dedup probe re-evaluation.** When `dedup_mode="auto"` auto-disables dedup, the connector now periodically runs a probe batch with dedup ON (every `dedup_probe_interval` batches, default 20) to detect workload shifts. If `dedup_probe_window` (default 2) consecutive probes exceed `dedup_auto_threshold`, dedup is automatically re-enabled. This fixes the scenario where a write-heavy warmup phase permanently disables dedup even after the workload shifts to read-heavy (repeated keys). Set `dedup_probe_interval=0` for legacy permanent-disable behavior.
- **New config fields:** `dedup_probe_interval` (env `SGLANG_CAMA_DEDUP_PROBE_INTERVAL`, default 20), `dedup_probe_window` (env `SGLANG_CAMA_DEDUP_PROBE_WINDOW`, default 2).
- **New stats fields:** `dedup_probes_total`, `dedup_reenables_total` exposed in `report_stats()` for Prometheus visibility.

---

## [1.18.9] - 2026-03-12

### Fixed

- **Warmup retry on transient errors.** `_warmup()` is now wrapped in `_retry_transient()` — retries `RuntimeError` (e.g. server "shard dispatch timeout" during hugepage allocation) with exponential backoff (1s, 2s, 4s). `AssertionError` (data integrity) is NOT retried. Prevents a single transient server hiccup from crashing the entire SGLang scheduler during startup.
- **`warmup_retries` config.** New `warmup_retries` field (default 3, env `SGLANG_CAMA_WARMUP_RETRIES`) controls retry attempts.

### Docs

- INSTALL.md: "no git required" notes, RDMA install option in Path A, port 18001 fix for RDMA default.

### Changed

- SGLang tree updated with latest connector code.

---

## [1.18.8] - 2026-03-12

### Docs

- **GPU cache pressure guide.** New `10_GPU_CACHE_PRESSURE.md` — explains TuningAdvisor "GPU cache under pressure" warning, L1/L2/L3 cache hierarchy interaction with CAMA backup/prefetch, the vicious cycle between backup writes and prefetch reads, monitoring guide with Prometheus metrics, and tuning decision tree for batch size, coalescing, server-side, connector-side, and model/scheduler knobs.
- Configuration reference cross-link to GPU cache pressure doc.

---

## [1.18.7] - 2026-03-12

### Fixed

- **Telemetry thread safety.** All histogram recording, batch telemetry (latency/phase measurements in `batch_get_v1`/`batch_set_v1`), and dedup auto-disable logic now wrapped under `_telemetry_lock`/`_counter_lock`. `get_stats()` uses swap-and-operate pattern — snapshots all metrics under lock, computes percentiles/averages outside lock. Prevents TOCTOU races when concurrent batch ops modify lists while stats collection reads/clears them.

### Changed

- SGLang tree updated with latest connector code.

---

## [1.18.6] - 2026-03-12

### Features

- **Eager server-side slab tuning.** Connector sends `model_page_bytes` hint to server via `report_stats()` immediately after RDMA buffer registration, allowing optimized slab classes before the first write batch. Graceful fallback on failure.

### Fixed

- **Thread-safe page counters.** `_interval_pages_get/set` and `_total_pages_get/set` increments in `batch_get_v1`/`batch_set_v1` now wrapped under `_counter_lock`.

### Changed

- **Expanded metrics reporting.** `backup_bandwidth_gbps`, `io_max_latency_ms`, and `model_page_bytes` now included in periodic stats reports.
- Updated architecture deep-dive, configuration reference, and deployment guide with eager slab tuning docs.
- SGLang tree updated with latest connector code.

---

## [1.18.5] - 2026-03-12

### Fixed

- **Parent process prewarm.** Preflight now calls `start_cama_prewarm()` directly in the parent process after setting the env var for children. Previously the parent never prewarmed because `environ.py`'s module-level hook runs before preflight sets the env var. Handles both single-GPU (same-process) and multi-GPU (spawn) modes.

### Changed

- **Error log throttling.** Error counter summary logging throttled to rank 0 only, at most once per 10 seconds. Reports delta errors (last window) + cumulative totals instead of logging on every report_stats call. Suppresses log when all deltas are zero.
- SGLang tree updated with latest connector code.

---

## [1.18.4] - 2026-03-11

### Changed

- **pool_size default 4→8.** CamaConfig default increased from 4 to 8, matching client defaults. With 4 NICs gives 2 connections per NIC for full parallelism.
- **pool_size=max(pool_size,len(endpoints)) override.** Multi-endpoint logic uses `max()` instead of auto-setting to endpoint count, allowing higher parallelism.
- SGLang tree updated with latest connector code.

---

## [1.18.3] - 2026-03-11

### Fixed

- **Per-rank prewarm (multi-GPU).** Prewarm now runs in each spawned rank process instead of the parent. SGLang uses `mp.set_start_method("spawn")` — children get fresh interpreters and could never see the parent's `PrewarmRegistry` or daemon threads. Preflight now serializes config to `_SGLANG_CAMA_PREWARM_SIGNAL` env var; each child reads it at `environ.py` import time and starts an independent prewarm daemon, giving the full model-loading window for connection setup + MR registration.

### Changed

- **Prewarm signal architecture.** `check_cama_preflight()` no longer calls `start_cama_prewarm()` directly. Instead it sets an env var consumed by `_maybe_start_rank_prewarm()` in `environ.py`. Env var is popped after read to prevent double-fire in nested spawns.
- SGLang tree updated with latest connector code and environ.py patch.

### Docs

- **Per-Rank Prewarm design decision.** New section in `06_DESIGN_DECISIONS.md` — timing diagram, alternative analysis (shared memory / file / CLI flags vs env var), edge cases (tp=1/8, dp>1, standalone mode).

### Tests

- `test_prewarm.py`: 5 new tests — env signal roundtrip, absent signal noop, invalid JSON noop, import error noop, env var consumption.

---

## [1.18.2] - 2026-03-11

### Fixed

- **Safe connection replacement.** Multi-NIC discovery and address migration now create the new connection before closing the old one. On failure, the old connection is preserved instead of leaving `self.conn` in a broken state.

### Changed

- **Prewarm wait synchronization.** `claim_prewarmed_connection()` now waits (up to `wait_timeout=10s`) for an in-progress prewarm daemon instead of returning `None` immediately. Prevents duplicate MR registration when model loading finishes before the prewarm thread completes.
- **Buffer size logging.** Connection phase logs now show buffer sizes (e.g. `buf=16 MB (default)`) and MR registration context for easier performance debugging.
- **Codec docs update.** Configuration reference corrected: codec GET path uses "Batch raw + decode" (not "Fallback (threaded)"). New `09_CODEC_TRADEOFFS.md` — copy accounting, break-even analysis, batch decode projections.
- SGLang tree updated with latest connector code.

### Tests

- `test_cama_storage.py`: mset mock returns `list[int]`, warmup batch mset type error regression test.
- `test_prewarm.py`: prewarm wait synchronization tests (slow prewarm, timeout), claim result type fixes, state cleanup.

---

## [1.18.1] - 2026-03-11

### Fixed

- **ChainCodec reconstruction.** Per-worker codec customization creates a new `ChainCodec` instance instead of mutating the shared `_codecs` list — prevents cross-worker codec corruption.
- **Warmup jitter clamp.** `_tp_jitter_scale` computation uses `max(0.0, min(1.0, ...))` to prevent negative jitter when `tp_size=0`.
- **Inflight counter thread safety.** `_inflight_gets` and `_inflight_sets` increments/decrements wrapped in `_counter_lock`.
- **Reconnect callback on prewarm.** Prewarm connection setup wires `set_reconnect_callback` when reconnect is enabled.
- **Dedup skip reason accuracy.** Warmup-aggressive skip now logged as `"warmup_aggressive"` instead of falling through to `"auto_disabled"`.

### Changed

- `_record_get_metrics()` helper extracted — deduplicates GET success/error counting across 3 code paths.
- SGLang tree updated with all fixes.

---

## [1.18.0] - 2026-03-11

### Added

- **Compressed batch RDMA GET.** `_get_batch_zero_copy()` now uses `mget_rdma_raw()` when codec is active, fetching raw bytes via batch RDMA Read and decompressing client-side. Replaces sequential per-key fallback that caused 10-50x throughput loss with compression enabled.

- **Multi-NIC write striping.** `_put_batch_zero_copy()` attempts `mset_striped()` when available, distributing SET operations across all NIC connections. Falls back to regular `mset()` transparently.

- **Cost-aware dedup auto-disable.** New `dedup_cost_ratio_threshold` config (default 0.5). Tracks ratio of exists-check overhead to transfer time; auto-disables dedup when exists cost dominates. `dedup_auto_window` reduced from 5 → 2 batches for faster response.

- **Warmup-aware dedup skip.** `set_warmup_phase_ref()` method wired from cache controller. During AGGRESSIVE warmup phase (empty cache), exists RPCs are skipped entirely since hit rate is ~0%.

- **TP-scaled warmup jitter.** `WarmupConfig` accepts `tp_size` (default 8). Jitter scaled by `min(1.0, (tp_size - 1) / 4.0)` to reduce contention for small tensor-parallel deployments.

### Changed

- Extracted `_record_get_metrics()` helper to eliminate duplicate elapsed/ok/error counting across GET paths (compressed RDMA, standard RDMA, sequential workers).
- SGLang tree updated with latest connector code and patches.

---

## [1.17.0] - 2026-03-11

### Added

- **Warmup phase system.** New `warmup.py` — 3-phase state machine (AGGRESSIVE → RAMP → STEADY) preventing cold-start performance degradation. During AGGRESSIVE phase (first 30s or 5000 pages), uses elevated batch size, reduced jitter, and shorter deadlines. RAMP phase (60s) linearly interpolates to steady-state. Adaptive batch sizing blocked until STEADY phase.

- **Warmup configuration.** 8 new config keys: `warmup_enabled`, `warmup_min_seconds` (15), `warmup_max_aggressive_seconds` (30), `warmup_pages_threshold` (5000), `warmup_ramp_seconds` (60), `warmup_aggressive_batch_size`, `warmup_aggressive_jitter_ms`, `warmup_aggressive_deadline_ms`.

- **BackupCoalescer dynamic deadline.** New `deadline_ref` callable parameter allows warmup-driven deadlines instead of static config.

- **Health log warmup metrics.** 60s health log reports `warmup_phase`, `eff_batch`, `eff_jitter`, `eff_deadline`. Phase and effective batch size forwarded to Prometheus.

### Fixed

- **Timeout partial results warning.** `batch_mset()`/`batch_mget()`/`batch_mexists()` log WARNING with completed/timed-out key counts, clarifying that timed-out keys appear as cache misses.
- **Dedup auto-disable notification.** Imminent auto-disable warns one batch early; final auto-disable upgraded from INFO to WARNING with remediation steps.
- **Codec size mismatch diagnostics.** Warning now includes stale data size and dtype/shape agreement hint.
- **Keepalive failure clarity.** Message now indicates dead connection and suggests checking server availability.

### Changed

- **Buffer size documentation.** CamaConfig docstrings and startup banner updated from 8 MB to 32 MB to match v0.31.0 defaults.
- SGLang tree updated with latest connector code and patches.

---

## [1.16.0] - 2026-03-11

### Added

- **Prewarm full connection setup.** `CamaPrewarmProvider` now performs a 3-phase setup: (1) initial connection, (2) RDMA endpoint discovery, (3) multi-NIC pool recreation with full MR registration. `CamaStorage` skips the expensive [3/7] NIC discovery step when adopting a pre-warmed multi-NIC pool.

- **Prewarm keepalive.** 60-second keepalive pings (`exists("__cama_keepalive__")`) prevent idle-connection timeouts during long model loading. Runs as daemon thread with graceful stop on claim/close.

- **Scheduler metrics forwarding.** New `push_scheduler_metrics()` on `HiCacheController` forwards cache hit rate, token usage, running requests, and generation throughput to storage backend for Prometheus emission. Host KV pool evictable ratio tracked separately.

### Changed

- **Prewarm TTL extended 5 → 15 minutes.** Handles 70B+ models that load longer than the previous 5-minute window. Reaper interval increased from 30s to 60s.
- **Preflight nic_striping propagation.** `run_preflight_checks()` now passes `nic_striping` config to `start_cama_prewarm()`.
- **PrewarmResult enhanced.** New `endpoints` field and `_keepalive_stop` event; `close()` stops keepalive before closing connection.
- SGLang tree updated with latest connector code and patches.

---

## [1.15.0] - 2026-03-11

### Added

- **Live batch size tracking.** `BackupCoalescer` accepts optional `max_pages_ref` callable that returns the current `storage_batch_size`, allowing the coalescer to track auto-tuned batch sizes in real-time instead of using a static value.

### Changed

- **`storage_batch_size` default 256 → 2048.** Modern 50k-token contexts (~781 pages) now fit in a single sub-batch instead of requiring 4 roundtrips. Auto-tune ceiling expanded from configured value to `max(configured, 4096)` to allow growth.
- **`batch_size_latency_target_ms` default 50 → 200.** Aligns with realistic per-batch latency at larger batch sizes.
- **`coalesce_deadline_ms` default 5 → 20 (max 100 → 500).** Allows more time to accumulate larger batches.
- **`send_buf_size` default 8 MB → 32 MB.** Matches server/client buffer increase.

### Tests

- 3 new tests for `max_pages_ref` live tracking: override static, track changes, fallback to static.
- SGLang tree updated with latest connector code and patches.

---

## [1.14.3] - 2026-03-11

### Changed

- **Health summary logging throttled.** Server health `stats()` call and summary logging restricted to rank 0 only, with a minimum 10-second interval between calls. Prevents multi-rank scenarios from hammering the server.
- SGLang tree updated with latest connector code.

---

## [1.14.2] - 2026-03-11

### Fixed

- **ChainCodec intermediate size loss (BUG 1).** `encode()` now records intermediate input sizes in a sub-header (`count(1B) + sizes(4B × count)`), and `decode()` passes the correct per-stage `original_size` to each codec in the reversed chain. Previously all codecs received the final tensor size, breaking chains with expanding intermediates (e.g. zstd decompress received too-small `max_output_size`).
- **Global ShuffleZstdCodec mutation (BUG 2).** Codec setup in `cama_storage.py` now creates a local `ShuffleZstdCodec(level=...)` instance when `codec_zstd_level != 3`, instead of mutating the global `_REGISTRY` singleton. Prevents cross-instance level contamination.
- **Non-deterministic chain codec_id (BUG 3).** Replaced monotonically incrementing `_next_chain_id` counter with `_derive_chain_id()` — a deterministic hash from chain name and constituent codec IDs. `register_chain()` is now idempotent with collision detection.
- **Silent size mismatch in _DecodeSGL (BUG 4).** `from_bytes()` now logs a WARNING when decoded size differs from expected `_original_size`.

### Tests

- 19 new tests across 6 new test classes: chain intermediate sizes, global mutation isolation, deterministic chain IDs, decode size mismatch warning, end-to-end wrap/unwrap chain, and storage-level zstd level isolation.
- SGLang tree updated with latest connector code.

---

## [1.14.1] - 2026-03-11

### Fixed

- **Coalesced prefix_keys correctness.** `_merge_ops()` now sets `prefix_keys=None` for multi-op merges. Previously used first-op-wins, which was incorrect when coalesced ops originate from different radix tree branches with different prefixes.
- **Safe _source_ops access.** `_page_backup()` uses `getattr(operation, '_source_ops', None)` instead of `operation._source_ops`, preventing `AttributeError` on non-coalesced operations that never went through the drain path.
- **Prefix_keys list copy.** `_page_backup()` copies `operation.prefix_keys` to a list before iteration, preventing mutation issues.
- **Smarter multi-NIC reconnect.** `_connect_multi_nic()` no longer unconditionally reconnects — skips reconnect when the pool already has multi-endpoint connections, avoiding unnecessary pool rebuilds on every startup.

### Changed

- **`backup_jitter_ms` default 0 → 10.** Light jitter enabled by default to prevent thundering herd across TP ranks. Override with `"backup_jitter_ms": 0` for single-rank deployments.
- **`batch_size_auto` default false → true.** Adaptive batch sizing enabled by default — auto-tunes `storage_batch_size` based on observed latency.
- **Preflight uses CamaConfig.resolve().** Eliminated duplicated 3-source config extraction in `preflight.py`; now delegates to `CamaConfig.resolve()` with a minimal fallback for standalone mode.
- **Prewarm uses _make_connection().** Eliminated duplicated connection construction in `prewarm.py`; now delegates to `_make_connection()` from `cama_storage.py`.
- **Settings banner.** `CamaStorage` logs a comprehensive startup banner showing all performance-relevant settings (pool_size, io_workers, nic_striping status, codec, dedup_mode, reconnect config). Backup thread logs a similar banner with its own settings.

### Docs

- Updated Multi-NIC docs across 5 files (overview, config reference, deployment guide, troubleshooting, design decisions) to reflect NIC striping as the default mode.

### Tests

- Updated `test_backup_coalescing.py` for prefix_keys=None merge behavior and getattr fallback.
- SGLang tree updated with latest connector code and patches.

---

## [1.14.0] - 2026-03-11

### Added — Backup Queue Coalescing
- **`BackupCoalescer` class.** Drains multiple 1-page backup operations from the queue into large merged batches (up to `storage_batch_size` pages). Two-phase blocking/timed drain with configurable `coalesce_deadline_ms` (default 5ms). Expected impact: avg_batch 2.0 → 256+, reducing wire roundtrips proportionally.
- **`coalesce_backup_ops` config** (default `true`). Disable with `SGLANG_CAMA_COALESCE_BACKUP_OPS=0`.
- **`coalesce_deadline_ms` config** (default 5.0, range 0-100ms). Maximum wait for additional ops before dispatching.
- **`cama_client_backup_coalesce_avg_ops` Prometheus metric.** Source ops merged per batch.

### Added — Backup Write Jitter
- **`backup_jitter_ms` config** (default 0, range 0-500ms). Random `[0, jitter]` delay between sub-batches spreads TP rank submissions to reduce thundering herd. Uses `stop_event.wait()` for clean shutdown.
- **Jitter Prometheus metrics:** `cama_client_backup_jitter_total_ms`, `cama_client_backup_avg_gap_ms`, `cama_client_backup_jitter_cfg_ms`.

### Added — Adaptive Batch Sizing
- **`batch_size_auto` config** (default `false`). Auto-halves batch size when avg latency exceeds `batch_size_latency_target_ms` (default 50ms), doubles when below half target. Respects floor (32) and ceiling (configured `storage_batch_size`).

### Added — Compression Codec Framework
- **`codec.py` module.** Pluggable codec architecture with 8-byte header (magic `\xCA\xCA`, codec ID, original size).
- **Built-in codecs:** `int8` (~2x lossy, symmetric FP16 quantization), `shuffle_zstd` (~1.3x lossless, byte-shuffle + zstd), `none` (identity).
- **Chain codecs:** `int8+shuffle_zstd` (~2.6x, compose left-to-right). Auto-allocated IDs (0x80+).
- **`codec` config** — codec selector string. `codec_zstd_level` (1-22, default 3).
- **SGL wrappers:** `_CompressedSGL` (SET path), `_DecodeSGL` (GET path with auto-decompress).
- **Note:** Compression disables `mget_rdma` (batch RDMA Read) and uses fallback GET path, since compressed values are smaller than tensor buffers.

### Added — Multi-NIC Striping
- **`nic_striping` config** (default `true`). When multiple RDMA endpoints exist, `create_pool()` connects to ALL server NICs for round-robin bandwidth distribution across the full pool.

### Changed
- **`storage_batch_size` default 128 → 256.** Aligns with coalescer's max_pages default.
- **`reconnect_max_retries` default 5 → 10.** Covers ~152s of server downtime.
- **Async `report_stats()`.** Runs in background `_stats_executor` with 3s timeout, preventing scheduler stalls.
- **`update_sglang_metrics()` merges instead of replaces.** Multiple callers (backup, prefetch) can push independent metric keys.
- **Server health summary in `get_stats()`.** Lightweight atomic reads of eviction rate, entries count, memory utilization, inflight ops logged every stats interval.
- **VERSION file deployment.** `deploy.py` copies VERSION alongside module; `__init__.py` checks both deployed and source layouts.

### Tests
- `test_codec.py` — 31 tests covering all codecs, header format, registry, wrap/unwrap, SGL wrappers.
- `test_backup_coalescing.py` — 15 tests for BackupCoalescer (merge, drain, acking, observability).
- `test_backup_jitter.py` — 14 tests for jitter sleep, config clamping, adaptive batch sizing.
- `test_cama_storage.py` — expanded with NIC striping and compression config tests.

---

## [1.13.2] - 2026-03-11

### Fixed

- **Preflight connection leak.** Connectivity probe now closes the test connection in a `finally` block, preventing socket leak on success path.
- **Pre-warm reconnect config.** Adopted pre-warmed connections now have user-configured reconnect settings applied (max_retries, base_delay_s, max_delay_s) instead of using prewarm defaults.
- **Preflight config extraction.** `pool_size` and `send_buf_size` now extracted from all three config sources (extra_config, JSON file, env vars) in preflight, fixing pre-warm with non-default pool/buffer sizes.
- **Import fallback.** `cama_storage.py` and `preflight.py` now try SGLang package import path first, falling back to `cama_module` for standalone mode.
- **Reaper thread hardening.** Exception handler around reaper loop body prevents thread death on unexpected errors (e.g. broken `close()`).
- **Config mismatch log level.** Pre-warm claim with mismatched fingerprint now logs at WARNING (was INFO).
- **Close() debug logging.** `PrewarmResult.close()` logs close exceptions at DEBUG with exc_info for diagnosability.

### Tests

- 3 new hardening tests: config mismatch warning log level, close() debug logging, reaper survival on close exception.

---

## [1.13.1] - 2026-03-11

### Docs
- **Non-editable install workflow.** All documentation updated from `pip install -e` (editable) to regular `pip install`. Explicit `--force-reinstall --no-deps` update commands documented throughout.
- **Path B rewrite.** Pre-built SGLang tree installation guide (`INSTALL.md`) substantially expanded: tree layout diagram, network requirements table, annotated fresh-install steps, dedicated B2 update workflow, and "when to reinstall what" decision table.
- **Install skill updated.** `.claude/skills/install/SKILL.md` aligned with non-editable install and `--find-links wheels/` flag.

---

## [1.13.0] - 2026-03-11

### Added — Phase-Level Latency Observability
- **Per-batch phase timing.** Tracks preprocess, exists-check, transfer, and postprocess durations in `batch_get_v1()` and `batch_set_v1()`, reported as averages in 60s `get_stats()` log and wired to Prometheus via `report_stats()`.
- **IO latency histogram.** 10-bucket histogram (1/5/10/25/50/100/250/500/1000/5000ms) with p50/p99 percentile reporting in health log and Prometheus exposition.
- **In-flight operation gauges.** `_inflight_gets` and `_inflight_sets` counters for saturation detection.
- **Transport stats integration.** `get_transport_stats()` called when available, splitting client latency into control roundtrip vs. RDMA Read.

### Added — Backup Thread Parallelization (cache_controller patch)
- **`backup_io_workers` config** (default 2). Configurable concurrent backup writer threads via `ThreadPoolExecutor`, replacing single-threaded sequential design.
- **`storage_batch_size` config** (default 128). Configurable pages per batch.
- **`_host_alloc_drops` counter.** Tracks host memory pool exhaustion events with cumulative logging.
- **Backup health metrics.** 60s health log with completed/failed ops, in-flight count, average latency; pushes to Prometheus via `update_sglang_metrics()`.
- **Queue depth warning.** WARNING log when backup queue >= 100 (saturation signal).

### Changed
- **Reconnect at constructor time.** Reconnect config passed directly to client constructor instead of deferred `enable_reconnect()` call.
- **Logging level reduction.** Per-batch and per-request operation logs downgraded from INFO to DEBUG; 60s health summaries remain at INFO.
- **`_batch_set_exists_dedup()` returns 4-tuple** with separate exists and transfer timing for phase attribution.

### Removed
- **`deploy.sh`.** Removed legacy shell deployment script (replaced by `deploy.py`).

### Docs
- Configuration reference updated with `backup_io_workers`, `storage_batch_size`, and new metrics.
- Troubleshooting guide expanded with host pool saturation diagnosis and latency profiling.
- `CHANGELIST.md` and `DIFF_STANDALONE_VS_SGLANG.md` updated (deploy.sh → deploy.py).

---

## [1.12.0] - 2026-03-10

### Added — OP_MGET_RDMA: Batch RDMA GET
- **`RDMAClient.mget_rdma()`** and **`RDMAClientPool.mget_rdma()`**: Batch GET
  with single control roundtrip + batch RDMA Reads + single batch ack. Sends all
  keys in one `OP_MGET_RDMA` (0x34) message, receives per-key RDMA coordinates
  via `OP_MGET_READ_READY` (0x35), posts all RDMA Reads with a single
  `ibv_post_send` doorbell (linked WR list, chunked at MAX_SEND_WR=128), sends
  `OP_BATCH_READ_ACK` (0x36). Falls back to sequential `mget()` when server
  doesn't advertise `mget_rdma` capability.
- **`batch_rdma_read_into()` C++ method**: New pybind11-bound method in
  rdma_transport.cpp for batch RDMA Read. GIL released during entire operation.
- **Connector integration**: `_get_batch_zero_copy()` now uses `mget_rdma` as
  primary path, with old ThreadPoolExecutor path extracted to
  `_get_batch_zero_copy_fallback()` as fallback.
- **Wire protocol**: Three new opcodes and encode/decode functions in protocol.py:
  `OP_MGET_RDMA` (0x34), `OP_MGET_READ_READY` (0x35), `OP_BATCH_READ_ACK` (0x36).
- **Result**: GET performance is now bandwidth-limited and page-size-independent.
  `page_size=32` performs the same as `page_size=128`.

### Added — Client Reconnection Integration
- **Reconnect config fields**: `reconnect_enabled` (default True),
  `reconnect_max_retries` (default 5), `reconnect_base_delay_s` (default 0.5),
  `reconnect_max_delay_s` (default 30.0) with `SGLANG_CAMA_RECONNECT_*` env vars
  and all three config loaders (env, JSON, extra_config).
- **Auto-enable**: Calls `conn.enable_reconnect()` after all init phases complete.
  Post-reconnect callback refreshes server info and resets dedup state.
- **Reconnect metrics**: `reconnect_count` tracked and exported in `get_stats()`
  and `get_storage_metrics()`.

### Changed
- **GET path refactored.** Original `_get_batch_zero_copy()` body extracted to
  `_get_batch_zero_copy_fallback()`. Primary path now uses `mget_rdma` when
  available, with automatic fallback.

### Docs
- Updated architecture deep dive with `mget_rdma` as primary GET path
- Added "Batch RDMA GET — Page-Size Independence" design decision
- Updated GET flow diagram with `mget_rdma` decision point

---

## [1.11.0] - 2026-03-10

### Added — Phase 1: mset Sub-Batch Chunking
- **`RDMAClient.mset()`** and **`RDMAClientPool.mset()`**: When the full MSET
  payload exceeds the RDMA send buffer (8 MB default), the batch is now
  automatically partitioned into sub-batches that each fit, instead of falling
  back to sequential `set()`. Single entries that exceed the buffer alone still
  fall back to individual `set()`.
- **`CamaClient.mset()`**: Same chunking logic applied to the TCP client for
  consistency. Chunks at `send_buf_size` (default 8 MB) minus header size.
- Both clients expose `_fits_send_buf()` helper for size checking.

### Added — Phase 2: Adaptive Write Dedup
- **`CamaConfig`**: New fields `dedup_mode` (`"auto"` / `"always"` /
  `"never"`), `dedup_auto_threshold` (default 0.05), `dedup_auto_window`
  (default 5) with `SGLANG_CAMA_DEDUP_*` env vars and all three config loaders.
- **`batch_set_v1()`**: Refactored to use `_batch_set_exists_dedup()` helper.
  In `"auto"` mode, tracks consecutive low-hit batches and auto-disables dedup
  after `dedup_auto_window` batches below `dedup_auto_threshold`.
- **`reset_dedup_state()`**: Resets adaptive dedup counters (e.g. after flush).
- Dedup metrics (`dedup_mode`, `dedup_auto_disabled`, `dedup_low_hit_streak`)
  added to `get_stats()` and `report_stats()`.

### Added — Phase 3: Connection Pooling (C++ Transport)
- **`rdma_transport.cpp`**: Added `owns_pd_` flag, `get_pd_handle()`,
  `get_ctx_handle()`, and `connect_with_shared_pd()` method. Shared-PD
  connections verify same RDMA device, reuse the PD (skip `ibv_alloc_pd`),
  create their own CQ/QP/buffers, and support `skip_read_buf` to save 64 MB
  per pool connection. `cleanup_all()` only frees PD if `owns_pd_`.
- pybind11 bindings added with GIL release for all new methods.

### Added — Phase 4: Connection Pooling (Python)
- **`RDMAClientPool`**: N RDMA connections sharing one PD, round-robin
  dispatch, per-connection locks. `get()` holds one connection for the entire
  GET→RDMA Read→ReadAck flow. `reg_memory()` registers on first transport's PD
  (lkey valid across all). Admin ops route to conn[0].
- **`CamaClientPool`**: N independent TCP clients with round-robin dispatch.
  Each `CamaClient` has its own socket and lock.
- **`create_pool()`** factory in `cama_client/__init__.py`: Auto-selects RDMA
  or TCP pool, falls back to single client when `pool_size <= 1`.

### Added — Phase 5: Connector Integration
- **`CamaConfig`**: New fields `pool_size` (default 4) and `send_buf_size`
  (default 0) with `SGLANG_CAMA_POOL_SIZE` and `SGLANG_CAMA_SEND_BUF_SIZE`
  env vars across all three config loaders.
- **`CamaStorage.__init__`**: Conditionally creates pool via `create_pool()`
  when `pool_size > 1`. Multi-NIC reconnect also recreates pool. Passes
  `send_buf_size` through to client constructor. Pool size logged in stats.

### Changed
- **Default `pool_size` set to 4** (previously 1). With `io_workers=16`, 4
  connections give ~4 threads per connection, avoiding single-lock
  serialization. Shared PD keeps memory overhead low (~16 MB per extra
  connection with `skip_read_buf`). Override with `SGLANG_CAMA_POOL_SIZE=1`
  for backward-compatible single-connection behavior.
- **Batch ops now use native wire-protocol opcodes** (`OP_MSET`, `OP_MTEST`,
  `OP_MDEL`) instead of sequential per-key fallbacks, with sub-batch chunking
  for payloads exceeding the send buffer.

---

For releases v1.0.0 – v1.10.0, see [CHANGELOG-archive.md](CHANGELOG-archive.md).

# Changelog Archive (v1.0.0 – v1.10.0)

Early releases of CAMA Connector. For current releases see [CHANGELOG.md](CHANGELOG.md).

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [1.10.0] - 2026-03-05

### Changed
- Renamed `_USING_CAMA` module-level flag to `_USING_CAMA_GO` for clarity
  (distinguishes CAMA-Go backend from PrisKV backend)

### Docs
- Updated version references, line counts, and file descriptions across
  `CHANGELIST.md`, `DIFF_STANDALONE_VS_SGLANG.md`, `FolderOverview.md`, and
  `docs/01_OVERVIEW.md` to reflect v1.9.0 changes
- Added `metrics/collector.py` as 5th integration point in overview docs
- Updated comparison table line counts (930 → 1,238)

---

## [1.9.0] - 2026-03-05

### Added
- Phased init logging (`[1/7]`–`[7/7]`) with per-phase timing for startup
  performance visibility
- `_fmt_bytes()` helper for human-readable byte sizes in log output
- `_get_metrics_url()` and `_fetch_server_progress()` for polling server
  `/ready` endpoint during health check, showing phase, percent, memory, and ETA
- Detailed warmup sub-step logging (5 steps with individual timing)
- RDMA buffer registration timing and human-readable size in
  `register_mem_pool_host()`
- `update_sglang_metrics()` method for scheduler-level metrics (cache hit rate,
  token usage, throughput) to be included in `report_stats()` calls
- `prefetch_bandwidth_gbps` derived stat in `get_stats()` output

### Improved
- Health check now shows attempt count, max attempts, elapsed time, and
  server-side progress (phase, percent, ETA) when available
- `MAX_HEALTH_ATTEMPTS` constant derived from `SETUP_TIMEOUT / HEALTH_CHECK_INTERVAL`
- Server info stored as `self._server_info` instance attribute for reuse

### Changed
- Init log messages restructured from flat format to numbered phases with timing
- Removed large block comment in `__init__` describing threading topology
  (covered in docs)

---

## [1.8.0] - 2026-03-03

### Added
- Complete logging coverage for `CamaStorage` and `HiCacheController`: 12
  structured log lines added to every uncovered step identified by auditing the
  GET flow diagram, covering the primary prefetch/load hot path, backup path,
  and legacy APIs

### Improved
- **Observability**: `start_loading()` now logs index count, layer count, and
  producer_id at INFO; DEBUG log emitted after all layers are transferred
- **Observability**: `load()` and `write()` now emit WARNING on alloc failure
  with requested count and node_id
- **Observability**: `start_writing()` logs merged index count and node_ids at
  DEBUG level
- **Observability**: `write_storage()` logs enqueue with op_id, token count,
  and page count at INFO
- **Observability**: `_page_backup()` logs per-batch success at DEBUG and fixes
  f-string to %-style formatting; `backup_thread_func` logs op completion at
  INFO with op_id, tokens, and total_ops
- **Observability**: `reset()` logs start and completion at INFO
- **Observability**: `_storage_hit_query()` logs per-batch hit/total pages at
  DEBUG
- **Observability**: `detach_storage_backend()` logs before joining threads
- **Observability**: Legacy single-key `get()`, `set()`, `exists()` log key
  (truncated to 64 chars) and result at DEBUG
- **Observability**: Legacy `batch_get()` / `batch_set()` log success count,
  total pages, and elapsed ms at DEBUG
- **Observability**: `get_stats()` `report_stats` exception now logged at DEBUG
  instead of silently swallowed
- **Observability**: `register_mem_pool_host()` logs actual layout at DEBUG and
  includes it in the assertion message

### Fixed
- f-string in `_page_backup()` warning replaced with %-style formatting to
  match project logging conventions

---

## [1.7.0] - 2026-03-03

### Improved
- **Thread safety**: All error/success counter increments (`_get_errors`,
  `_set_errors`, `_exists_errors`, etc.) now protected by `threading.Lock`
- **Timeout handling**: Pending futures are now explicitly cancelled on
  `TimeoutError` in `_get_batch_zero_copy`, `_put_batch_zero_copy`, and
  `_batch_exist`
- **Graceful shutdown**: Backup `_page_transfer` loop now checks
  `stop_event` between batches; executor shutdown uses `cancel_futures=True`

### Changed
- `_RC.EXISTS_ERROR` monkey-patch replaced with module-level `_EXISTS_ERROR`
  sentinel constant (safe read-only, no import-time mutation)
- `_prefetch_io_task` now returns `bool` for correct `io_completed`/`io_failed`
  tracking; reaping loop uses `f.result()` instead of `f.exception()`
- Improved log messages: `terminate` uses `dispatched_pages`, `page_get`
  simplified failure log format

### Added
- `conn.report_stats()` call in `close()` to push cumulative error counters
  to the server for Prometheus exposure
- Documentation: `07_HASHING_AND_KEY_HANDLING.md`, `08_E2E_AUDIT_KEY_HANDLING.md`,
  `GET_FLOW_DIAGRAM.md`
- Expanded architecture, configuration, troubleshooting, and design-decisions docs
  with threading topology, `io_workers` tuning, and counter-lock rationale

### Fixed
- Test stub `_StubStorageMetrics` updated with missing counter fields

---

## [1.6.0] - 2026-03-03

### Improved
- **Timeout handling**: Batch `get`/`set`/`exists` operations now use
  `as_completed` with configurable `op_timeout_s` instead of unbounded
  `ThreadPoolExecutor.map`, preventing indefinite hangs on stalled keys
- **Observability**: Cumulative error counters (`get_errors`, `set_errors`,
  `exists_errors`, `exists_timeouts`) tracked on `CamaStorage` and reported
  via `get_storage_metrics()`; warning emitted when any counter is non-zero
- **Logging**: All prefetch log lines now use structured `[prefetch][*]` tags
  with consistent fields (request_id, token counts, elapsed time)

### Changed
- Prefetch IO refactored from single aux thread to a `ThreadPoolExecutor`
  with configurable `prefetch_io_workers` (default 2), eliminating the
  fragile thread-restart logic and enabling concurrent prefetch transfers
- `EXISTS_ERROR` return code added to `_RC` for both CAMA and PrisKV backends;
  failed `exists()` calls now return `EXISTS_ERROR` instead of `EXISTS_MISSING`
- Dedup logic treats `EXISTS_ERROR` same as `EXISTS_MISSING` (writes the key
  rather than assuming it exists), improving correctness under storage errors
- Per-key exception logging in batch ops upgraded from `warning` to `error`

### Fixed
- `_get_batch_zero_copy` success check now uses `_RC.GET_OK` constant instead
  of hardcoded `0`, fixing incorrect success counting on PrisKV backend

---

## [1.5.0] - 2026-03-03

### Improved
- **Observability**: Upgraded prefetch dispatch/revoke and dedup logs from `debug`
  to `info` with structured fields (request_id, token counts, percentages)
- **Observability**: Added batch mismatch warning in `_page_transfer` with
  request_id and expected vs actual token counts
- **Observability**: Added rate-limited (10s) info logging when prefetch hits
  capacity limit
- **Observability**: Enhanced health logs with received/dispatched/revoked counters,
  buffer queue size, aux thread liveness, and write_policy
- **Observability**: `batch_exists` short-circuit exceptions now log at `warning`
  with exception details instead of silent `debug`

### Fixed
- **Resilience**: `prefetch_io_aux_func` now catches fatal unhandled exceptions
  that would silently kill the aux thread, logging them at `error` level
- **Resilience**: `prefetch_thread_func` detects dead aux IO thread and
  automatically restarts it with a logged error
- **Correctness**: `skip_dedup` in backup writes is now conditional on
  `write_policy == "write_back"` instead of always True

### Added
- `detach_storage_backend()` method for clean shutdown of storage threads and
  backend connection during server teardown

---

## [1.4.0] - 2026-03-03

### Fixed
- **Crash on startup when environ.py patch is stale**: `load_from_env()` crashed
  with `AttributeError: 'Envs' object has no attribute 'SGLANG_CAMA_OP_TIMEOUT_S'`
  when the sglang environ.py patch was missing newer CAMA env var definitions
- All `envs.*` attribute accesses in `cama_storage.py`, `preflight.py`, and
  `profiling.py` now use safe accessors (`_env_get` / `_env_default` / `hasattr`
  guards) that fall back to hardcoded defaults when an attribute is missing

### Added
- `SGLANG_CAMA_IO_WORKERS` env var definition in `patches/environ.py`
  (was referenced in code but never declared)
- Resilience tests: `test_config_resilient_to_missing_env_attrs` and
  `test_config_from_extra_config_resilient_to_missing_env_attrs` verify config
  loading succeeds even when `Envs` class lacks CAMA attributes

---

## [1.3.0] - 2026-03-03

### Added
- Multi-NIC RDMA support: automatic per-rank NIC assignment via
  `rdma_endpoints()` discovery with round-robin allocation
  (`local_rank % len(endpoints)`) — no configuration needed
- Concurrent I/O via `ThreadPoolExecutor` in `_put_batch_zero_copy` and
  `_get_batch_zero_copy`; pybind11 releases the GIL so per-key `set()`/`get()`
  calls now execute in parallel across the thread pool
- `io_workers` config parameter (default 16) to control I/O thread pool size;
  available via extra_config, JSON config file, and env var
- `skip_dedup` support in `batch_set_v1` extra_info for caller-controlled
  dedup bypass
- Unit test suite (`tests/test_cama_storage.py`)
- `docs/PYBIND11_BATCH_BUG.md` — detailed write-up of the pybind11 batch
  operations bug and workaround

### Changed
- `__init__` reordered: model config is now set before Multi-NIC discovery
  so `local_rank` is available for NIC assignment; health check and warmup
  moved after reconnect to validate the final connection
- `_put_batch_zero_copy` / `_get_batch_zero_copy` switched from sequential
  `for` loops to `ThreadPoolExecutor.map()` with per-batch timing and
  success/failure logging
- Cache controller patch: `HiCacheStorageExtraInfo` now passes
  `extra_info={"skip_dedup": True}` to allow backends to skip dedup on
  controller-initiated writes

### Docs
- Added Multi-NIC deployment section to deployment guide with architecture
  diagram and expected log output
- Updated architecture deep dive with Multi-NIC endpoint discovery sequence
  diagram and design rationale
- Updated feature comparison table (Multi-NIC: Yes, Batch ops: via thread pool)
- Added Multi-NIC troubleshooting section (wrong NIC, all ranks on same NIC,
  discovery exceptions)
- Added Multi-NIC design decisions (why round-robin, why reconnect, why no
  config, difference from Mooncake)
- Updated all line-number references across docs to match new 930-line module
- Removed "Multi-NIC Support" from future considerations (now implemented)

---

## [1.2.0] - 2026-03-03

### Added
- Configurable operation timeout via `op_timeout_s` config field and
  `SGLANG_CAMA_OP_TIMEOUT_S` env var (default 10 s); calls `set_timeout()` on
  the PrisKV client when available
- Periodic health logging (every 60 s) in prefetch-IO, prefetch, and backup
  threads — reports `ops_completed` and `consecutive_failures`

### Changed
- Moved backend detection (cama-client vs priskv) from per-instance `__init__`
  to module-level import time; backend type is now resolved once and logged at
  startup
- Introduced `_RC` return-code abstraction to handle differing conventions
  between CAMA (`EXISTS_FOUND=1`) and PrisKV (`EXISTS_FOUND=0`); all hardcoded
  return-code comparisons replaced with `_RC` constants
- Warmup assertions now include backend name and expected vs actual return codes
  for easier debugging

### Fixed
- `_put_batch_zero_copy`, `_get_batch_zero_copy`, and `_batch_exist` now catch
  per-key exceptions and log warnings instead of crashing the entire batch
- `prefetch_io_aux_func`, `prefetch_thread_func`, and `backup_thread_func` wrap
  each operation in try/except with consecutive-failure tracking and escalating
  log severity (warn → error at 10 failures); threads no longer crash on
  transient storage errors

---

## [1.1.0] - 2026-03-02

### Added
- `deploy.sh` — one-command deployment script to sync the CAMA module into an SGLang
  checkout; supports `--module`, `--patch`, `--diff`, `--zip`, and `--all` modes

### Changed
- `profiling.py`: set `gil_only=False` so Pyroscope captures native C/C++ frames
  (pybind11, RDMA, PrisKV) that execute outside the Python GIL

### Docs
- Updated `DIFF_STANDALONE_VS_SGLANG.md` to reflect `deploy.sh` sync workflow and
  corrected directory references to `cama-connector` / `sglang-fresh`

---

## [1.0.0] - 2026-03-02

### Added
- Core `CamaStorage` backend with zero-copy RDMA put/get for SGLang HiCache
- Triple-source configuration loading (env vars, JSON config, extra_config)
- MHA and MLA key-naming support with TP/PP rank suffixes
- Batch operations: `batch_get_v1`, `batch_set_v1` with write deduplication
- `_put_batch_zero_copy` / `_get_batch_zero_copy` zero-copy primitives
- Preflight connectivity check (`preflight.py`)
- Pyroscope + NVTX profiling helpers (`profiling.py`)
- Reference SGLang patches for v0.5.8 integration (4 files)
- Comprehensive documentation suite (6 guides in `docs/`)
- Full deployment & troubleshooting guide in README
- Integration changelist and standalone-vs-SGLang diff docs

### Fixed
- Bypassed broken pybind11 `mexists`/`mset`/`mget` batch wrappers with per-key fallbacks

# CAMA Integration Changelist: Standalone vs SGLang

**Base:** SGLang v0.5.9 (commit `bbe9c7e`)
**CAMA version:** v1.20.1
**Date:** 2026-03-13

---

## Summary

| Category | Files | Lines Added | Lines Removed |
|----------|-------|-------------|---------------|
| Modified existing SGLang files | 5 | 226 | 66 |
| New CAMA module files | 4 | ~1,363 | 0 |
| New test files | 2 | ~804 | 0 |
| **Total** | **11** | **~2,393** | **66** |

---

## Modified Existing SGLang Files (in patch)

### 1. `python/sglang/srt/environ.py`
- **Change:** +13 lines after line 298
- **What:** Adds 11 CAMA environment variables to the `Envs` class:
  - `SGLANG_CAMA_CONFIG_PATH` — JSON config file path
  - `SGLANG_CAMA_REMOTE_ADDR` — PrisKV server address (default: 127.0.0.1)
  - `SGLANG_CAMA_REMOTE_PORT` — PrisKV server port (default: 6379)
  - `SGLANG_CAMA_PASSWORD` — PrisKV auth password (default: empty)
  - `SGLANG_CAMA_USE_MPUT_MGET` — Use batch ops (default: true)
  - `SGLANG_CAMA_CHECK_SERVER` — Poll server at startup (default: false)
  - `SGLANG_CAMA_OP_TIMEOUT_S` — Operation timeout in seconds (default: 10.0)
  - `SGLANG_CAMA_IO_WORKERS` — I/O thread pool size (default: 16)
  - `SGLANG_CAMA_PROFILING_ENABLED` — Enable Pyroscope+NVTX (default: false)
  - `SGLANG_CAMA_PROFILING_SERVER_ADDRESS` — Pyroscope endpoint
  - `SGLANG_CAMA_PROFILING_SERVICE_NAME` — Pyroscope service name

### 2. `python/sglang/srt/mem_cache/storage/backend_factory.py`
- **Change:** +9 lines
- **What:**
  - Adds `elif backend_name == "cama"` branch in `_create_builtin_backend()` (line 185)
  - Adds `StorageBackendFactory.register_backend("cama", ...)` at module level (end of file)

### 3. `python/sglang/srt/managers/cache_controller.py`
- **Change:** +186 lines added, -64 lines removed (net +122)
- **What:**
  - Adds `"cama"` to the zero-copy backend list:
    ```python
    # Before:
    if (self.storage_backend_type in ["hf3fs", "mooncake", "eic", "nixl"]) or (
    # After:
    if (self.storage_backend_type in ["hf3fs", "mooncake", "eic", "nixl", "cama"]) or (
    ```
  - Replaces single `prefetch_io_aux_thread` with `ThreadPoolExecutor` (`prefetch_io_workers` threads, default 2) for concurrent prefetch I/O
  - Adds `_prefetch_io_task()` method with proper error handling and host-memory release
  - Refactors `prefetch_thread_func()` with executor-based pipelining, future reaping, and shutdown logging
  - Adds `prefetch_io_workers` configurable parameter from `extra_config`
  - Adds detailed threading topology documentation

### 4. `python/sglang/srt/server_args.py`
- **Change:** +9 lines added, 2 lines modified
- **What:**
  - Adds `"cama"` to `--hicache-storage-backend` choices list (line 4313)
  - Adds `"cama"` to help text for built-in backends
  - Adds preflight check block in `_handle_hicache_args()` (line 2330):
    ```python
    if self.hicache_storage_backend == "cama":
        from sglang.srt.mem_cache.storage.cama.preflight import check_cama_preflight
        check_cama_preflight(self.hicache_storage_backend_extra_config)
    ```

### 5. `python/sglang/srt/metrics/collector.py`
- **Change:** +9 lines
- **What:** Adds debug print for storage metrics when prefetch/backup page counts are non-empty (bandwidth and page count logging)

---

## New CAMA Module Files (not in patch — must be copied)

### 6. `python/sglang/srt/mem_cache/storage/cama/__init__.py`
- **Lines:** 0 (empty package marker)
- **Source (standalone):** `cama_module/__init__.py`

### 7. `python/sglang/srt/mem_cache/storage/cama/cama_storage.py`
- **Lines:** 1,150
- **Source (standalone):** `cama_module/cama_storage.py`
- **What:** Main `CamaStorage(HiCacheStorage)` class with:
  - Module-level backend detection (cama-client vs priskv) with `_RC` return-code abstraction
  - `CamaConfig` dataclass — triple-source config loading (extra_config > file > env)
  - `__init__` — import guard, connect, health check, Multi-NIC RDMA discovery, 3-phase warmup, model config
  - `register_mem_pool_host()` — single O(1) RDMA buffer registration
  - Key naming helpers (MHA/MLA suffixes, TP/PP rank, extra_backend_tag)
  - Concurrent zero-copy primitives via `ThreadPoolExecutor`: `_put_batch_zero_copy`, `_get_batch_zero_copy`, `_batch_exist`
  - V1 API: `batch_get_v1`, `batch_set_v1` (with write deduplication and `skip_dedup` support)
  - Legacy API: `get`, `set`, `batch_get`, `batch_set` (bandwidth metrics)
  - Thread-safe error counters (`_get_errors`, `_set_errors`, `_exists_errors`, `_exists_timeouts`)
  - Configurable `op_timeout_s` with future cancellation
  - Periodic health logging and comprehensive observability (INFO-level throughout)
  - `close()` with `report_stats()`, `clear()`, `get_stats()`

### 8. `python/sglang/srt/mem_cache/storage/cama/preflight.py`
- **Lines:** 89
- **Source (standalone):** `cama_module/preflight.py`
- **What:** `check_cama_preflight()` — early fail-fast connectivity check called from `ServerArgs.check_server_args()`. Validates cama_client/priskv is importable and PrisKV server is reachable before model loading begins. Uses safe env var accessors.

### 9. `python/sglang/srt/mem_cache/storage/cama/profiling.py`
- **Lines:** 124
- **Source (standalone):** `cama_module/profiling.py`
- **What:** Conditional Pyroscope + NVTX profiling. Zero-cost no-ops when `SGLANG_CAMA_PROFILING_ENABLED` is false. Uses `gil_only=False` for capturing native C/C++ frames.

---

## New Test Files (not in patch — must be copied)

### 10. `python/sglang/srt/mem_cache/storage/cama/test_cama_storage.py`
- **Lines:** 631
- **What:** Progressive 7-layer unit test suite (requires live PrisKV):
  - Layer 0: PrisKV server alive (setstr/getstr/exists/delete)
  - Layer 1: RDMA registration + raw byte round-trip (1MB buffer)
  - Layer 2: Batch operations (mset/mget/mexists with 8 keys)
  - Layer 3: CamaStorage config, warmup, key naming (MHA/MLA/PP/tag)
  - Layer 4: Full KV cache page round-trip via MockHostKVCache
  - Layer 5: Write deduplication verification
  - Layer 6: Error handling & resilience
  - Layer 7: Thread safety & concurrent access

### 11. `test/registered/hicache/test_hicache_storage_cama_backend.py`
- **Lines:** 173
- **What:** E2E integration tests using HiCache test framework:
  - `TestCamaBackendPageFirstLayout` — page_first layout, TP=2
  - `TestCamaBackendMLA` — MLA model (DeepSeek), page_first, TP=2
  - `TestCamaBackendAccuracy` — page_first_direct layout, direct IO, eval accuracy
  - Starts/stops PrisKV server automatically via subprocess

---

## Standalone-Only Files (not present in SGLang)

| File | Description |
|------|-------------|
| `patches/environ.py` | Full reference copy of patched `environ.py` |
| `patches/backend_factory.py` | Full reference copy of patched `backend_factory.py` |
| `patches/server_args.py` | Full reference copy of patched `server_args.py` |
| `patches/cache_controller.py` | Full reference copy of patched `cache_controller.py` |
| `deploy.py` | Automated deployment to SGLang checkout |
| `release.sh` | Release automation (version bump, tag, archive) |
| `CHANGELOG.md` | Full version history (v1.0.0–v1.8.0) |
| `FolderOverview.md` | Integration instructions |
| `README.md` | Comprehensive deployment + troubleshooting guide (642 lines) |
| `docs/01_OVERVIEW.md` | Architecture overview |
| `docs/02_ARCHITECTURE_DEEP_DIVE.md` | Detailed architecture |
| `docs/03_CONFIGURATION_REFERENCE.md` | Config reference |
| `docs/04_DEPLOYMENT_GUIDE.md` | Deployment guide |
| `docs/05_TROUBLESHOOTING.md` | Troubleshooting guide |
| `docs/06_DESIGN_DECISIONS.md` | Design rationale |
| `docs/07_HASHING_AND_KEY_HANDLING.md` | Key naming details |
| `docs/08_E2E_AUDIT_KEY_HANDLING.md` | End-to-end key audit |
| `docs/GET_FLOW_DIAGRAM.md` | Data flow diagram |
| `docs/PYBIND11_BATCH_BUG.md` | pybind11 batch ops bug workaround |

---

## File Correspondence Map

| Standalone Path | SGLang Path | Status |
|----------------|-------------|--------|
| `cama_module/cama_storage.py` | `python/sglang/srt/mem_cache/storage/cama/cama_storage.py` | Synced (v1.8.0) |
| `cama_module/preflight.py` | `python/sglang/srt/mem_cache/storage/cama/preflight.py` | Identical |
| `cama_module/profiling.py` | `python/sglang/srt/mem_cache/storage/cama/profiling.py` | Identical |
| `cama_module/__init__.py` | `python/sglang/srt/mem_cache/storage/cama/__init__.py` | Identical (empty) |
| `patches/environ.py` | `python/sglang/srt/environ.py` | +13 lines |
| `patches/backend_factory.py` | `python/sglang/srt/mem_cache/storage/backend_factory.py` | +9 lines |
| `patches/cache_controller.py` | `python/sglang/srt/managers/cache_controller.py` | +186/-64 lines |
| `patches/server_args.py` | `python/sglang/srt/server_args.py` | +9 lines, 2 modified |
| — | `python/sglang/srt/metrics/collector.py` | +9 lines |
| `README.md` | (none in SGLang) | Standalone=full guide |
| — | `python/sglang/srt/mem_cache/storage/cama/test_cama_storage.py` | SGLang only |
| — | `test/registered/hicache/test_hicache_storage_cama_backend.py` | SGLang only |

---

## Key Differences Between Standalone and SGLang Versions

| Aspect | Standalone (`cama_module/`) | SGLang (`storage/cama/`) |
|--------|---------------------------|--------------------------|
| `StorageMetrics` | `from sglang.srt.metrics.collector import StorageMetrics` | Same |
| `profiling.py` Pyroscope | `gil_only=False` | Same |
| `profiling.py` service name | `"cama-connector"` | `"cama-connector"` |
| README | 642-line comprehensive guide (root) | None (SGLang has its own docs) |
| Test files | 631 lines (`tests/`) | Same file deployed to `storage/cama/` |
| Documentation | 10 detailed docs in `docs/` | None (docs in standalone) |
| `cache_controller.py` | Full patched copy (reference) | ThreadPoolExecutor prefetch I/O |

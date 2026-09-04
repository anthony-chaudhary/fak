# CAMA Standalone — Integration Reference

**Version:** 1.20.1 | **Date:** 2026-03-13 | **SGLang base:** v0.5.9

This folder contains the complete CAMA (PrisKV) storage backend for SGLang HiCache,
extracted for clean reintegration into any fresh SGLang release.

## Structure

```
cama-connector/
├── cama_module/              # Drop into: python/sglang/srt/mem_cache/storage/cama/
│   ├── __init__.py           # Empty package marker
│   ├── cama_storage.py       # Main CamaStorage class (1,238 lines)
│   ├── preflight.py          # Fail-fast connectivity check (89 lines)
│   └── profiling.py          # Pyroscope + NVTX profiling helpers (124 lines)
├── patches/                  # Reference copies of patched SGLang files
│   ├── environ.py            # +13 lines: 11 SGLANG_CAMA_* env vars
│   ├── backend_factory.py    # +cama registration block + _create_builtin_backend branch
│   ├── server_args.py        # +"cama" in choices + preflight import/call
│   └── cache_controller.py   # +"cama" in zero-copy list + ThreadPoolExecutor prefetch I/O
├── tests/
│   └── test_cama_storage.py  # 7-layer progressive unit test suite (691 lines)
├── docs/                     # 10 detailed documentation guides
│   ├── 01_OVERVIEW.md
│   ├── 02_ARCHITECTURE_DEEP_DIVE.md
│   ├── 03_CONFIGURATION_REFERENCE.md
│   ├── 04_DEPLOYMENT_GUIDE.md
│   ├── 05_TROUBLESHOOTING.md
│   ├── 06_DESIGN_DECISIONS.md
│   ├── 07_HASHING_AND_KEY_HANDLING.md
│   ├── 08_E2E_AUDIT_KEY_HANDLING.md
│   ├── GET_FLOW_DIAGRAM.md
│   └── PYBIND11_BATCH_BUG.md
├── VERSION                   # 1.20.1
├── CHANGELOG.md              # Full version history (v1.0.0–v1.20.1)
├── CHANGELIST.md             # Integration changelist vs SGLang
├── DIFF_STANDALONE_VS_SGLANG.md  # Exact diff report
├── README.md                 # Comprehensive deployment guide (642 lines)
├── INSTALL.md                # Step-by-step manual install (fresh + pre-packaged SGLang)
├── cama-sglang-integration.patch  # Unified patch for SGLang integration
├── deploy.py                 # Automated deployment to SGLang tree (Python)
└── release.sh                # Release automation (version bump, tag, archive)
```

## Integration Points (5 files to patch)

### 1. `environ.py` — Add CAMA env vars
Location: `python/sglang/srt/environ.py`
Add inside the `Envs` class, after the Mooncake Store block:

```python
    # Cama PrisKV Store
    SGLANG_CAMA_CONFIG_PATH = EnvStr(None)
    SGLANG_CAMA_REMOTE_ADDR = EnvStr("127.0.0.1")
    SGLANG_CAMA_REMOTE_PORT = EnvInt(6379)
    SGLANG_CAMA_PASSWORD = EnvStr("")
    SGLANG_CAMA_USE_MPUT_MGET = EnvBool(True)
    SGLANG_CAMA_CHECK_SERVER = EnvBool(False)
    SGLANG_CAMA_OP_TIMEOUT_S = EnvFloat(10.0)
    SGLANG_CAMA_IO_WORKERS = EnvInt(16)

    # Cama Profiling (Pyroscope + NVTX)
    SGLANG_CAMA_PROFILING_ENABLED = EnvBool(False)
    SGLANG_CAMA_PROFILING_SERVER_ADDRESS = EnvStr("http://0.0.0.0:4040")
    SGLANG_CAMA_PROFILING_SERVICE_NAME = EnvStr("cama-connector")
```

### 2. `server_args.py` — Add "cama" choice + preflight
Location: `python/sglang/srt/server_args.py`
- In `--hicache-storage-backend` choices, add `"cama"`
- In `_handle_hicache_args()`, add cama preflight block

### 3. `backend_factory.py` — Register cama backend
Location: `python/sglang/srt/mem_cache/storage/backend_factory.py`
- Add `elif backend_name == "cama":` in `_create_builtin_backend()`
- Add `StorageBackendFactory.register_backend("cama", ...)` at module level

### 4. `cache_controller.py` — Add cama to zero-copy list + prefetch I/O refactor
Location: `python/sglang/srt/managers/cache_controller.py`
- Add `"cama"` to the backend list that uses `_page_get_zero_copy` / `_page_set_zero_copy`
- Replace `prefetch_io_aux_thread` with `ThreadPoolExecutor` for concurrent prefetch I/O
- Add `prefetch_io_workers` config parameter (default 2)

### 5. `metrics/collector.py` — Storage metrics logging
Location: `python/sglang/srt/metrics/collector.py`
- Add debug print for storage metrics when prefetch/backup data is present

## Deployment

```bash
# Full setup: venv + SGLang install + CAMA deploy + client install
python deploy.py /path/to/sglang --setup

# Deploy module + patches only (no venv/install)
python deploy.py /path/to/sglang             # Deploy module + patches
python deploy.py /path/to/sglang --module    # Module only (cama_module/)
python deploy.py /path/to/sglang --patch     # Integration patches only
python deploy.py /path/to/sglang --diff      # Dry-run (show changes)
python deploy.py /path/to/sglang --zip       # Deploy + create archive
```

For manual step-by-step install (including pre-packaged SGLang trees), see **INSTALL.md**.

## Key Features (v1.10.0)

- Zero-copy RDMA transfers via PrisKV's native SGL API
- Multi-NIC RDMA with automatic per-rank NIC discovery
- MHA + MLA key naming with TP/PP rank suffixes
- Concurrent I/O via ThreadPoolExecutor (configurable `io_workers`)
- Write deduplication with `skip_dedup` support
- Configurable operation timeout (`op_timeout_s`) with future cancellation
- Thread-safe error counters and periodic health logging
- Complete INFO-level observability (12 structured log points)
- Dual backend support (cama-client RDMA + priskv TCP)
- Triple-source configuration (extra_config > JSON file > env vars)
- Safe env var accessors (resilient to missing Envs attributes)
- Pyroscope + NVTX profiling (`gil_only=False`)
- Phased initialization logging with `_fmt_bytes()` helper
- Server progress polling with configurable timeout
- SGLang metrics integration (storage stats via collector.py)

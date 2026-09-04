# Diff Report: cama-connector vs cama-in-sglang

**Updated:** 2026-05-20
**CAMA version:** v1.22.3
**SGLang base:** v0.5.7 (commit `232982a0dee4f0f9545189a7d9b6b9bb802e4910`)

> **Correction (2026-05-20):** earlier revisions of this doc claimed an SGLang
> base of v0.5.9 (commit bbe9c7e). That was aspirational and never re-measured.
> Diffing the in-tree tree against each upstream tag confirmed the real base is
> **v0.5.7**. The base is now pinned machine-readably in
> `sglang-with-cama-connector/UPSTREAM.txt`, and the patch set is enumerated in
> `cama-connector/patch_manifest.json`, so it never has to be guessed again.

This document lists the exact differences between the standalone CAMA module
(`cama-connector/cama_module/`) and the version integrated into SGLang
(`sglang-with-cama-connector/python/sglang/srt/mem_cache/storage/cama/`).

As of this update, both are fully synced via `deploy.py`.

---

## 1. `cama_storage.py` (1,150 lines) — IDENTICAL

Both versions share:
- Module-level backend detection (`cama_client` or `priskv`) with `_RC` abstraction
- Dual import guard with version logging
- Multi-NIC RDMA discovery via `rdma_endpoints()` with round-robin NIC assignment
- 3-phase warmup (string → RDMA SGL → missing-key assertion)
- Concurrent I/O via `ThreadPoolExecutor` (`io_workers` threads, default 16)
- Individual ops workaround for pybind11 batch bug (Section E)
- Configurable `op_timeout_s` with future cancellation on timeout
- Thread-safe error counters (`_get_errors`, `_set_errors`, `_exists_errors`, `_exists_timeouts`)
- `_EXISTS_ERROR` module-level sentinel constant
- `conn.report_stats()` on `close()` for Prometheus exposure
- Safe env var accessors (`_env_get` / `_env_default` / `hasattr` guards)
- Complete INFO-level observability logging (12 structured log lines added in v1.8.0)
- `StorageMetrics` imported from `sglang.srt.metrics.collector`

---

## 2. `preflight.py` (89 lines) — IDENTICAL

Both versions support `cama_client` or `priskv` with version logging.
Uses safe env var accessors for resilience against missing `Envs` attributes.

---

## 3. `profiling.py` (124 lines) — IDENTICAL

Both versions use `gil_only=False` for capturing native C/C++ frames.

---

## 4. `__init__.py` — IDENTICAL

Both are empty package markers.

---

## 5. `README.md` — DIFFERENT (scope, not content)

- **Standalone root `README.md`:** 642 lines — comprehensive deployment guide with full SGLang install instructions, CUDA toolkit setup, troubleshooting (py-spy, JIT hangs, NCCL, tokenizer deadlock), slow-vs-stuck guide
- **SGLang `cama/`:** No README (relies on SGLang's own docs)

Both describe the same system; the standalone version is the complete reference.

---

## 6. Files only in SGLang (not in standalone)

| File | Lines | Description |
|------|-------|-------------|
| `test_cama_storage.py` | 631 | Progressive 7-layer unit test suite |
| `../../test/registered/hicache/test_hicache_storage_cama_backend.py` | 173 | E2E integration tests |

---

## 7. Files only in standalone (not in SGLang)

| File | Description |
|------|-------------|
| `patch_manifest.json` | **Canonical list of patched files** (source of truth for the tooling) |
| `patches/environ.py` | Full patched copy of SGLang's environ.py (+CAMA env vars) |
| `patches/server_args.py` | Full patched copy of SGLang's server_args.py |
| `patches/backend_factory.py` | Full patched copy of SGLang's backend_factory.py |
| `patches/cache_controller.py` | Full patched copy with ThreadPoolExecutor prefetch I/O + load-back backpressure |
| `patches/schedule_policy.py` | Wires `mem_quota` into `init_load_back()` (load-back OOM guard) |
| `patches/scheduler_metrics_mixin.py` | Pushes scheduler metrics to the cache controller |
| `patches/hicache_storage.py` | `pp_rank`/`pp_size` fields + `prefix_keys` default fix |
| `patches/hiradix_cache.py` | HiRadix reconnect crash fix |
| `cama_patchlib.py` | Shared helpers (manifest, upstream pin, LF-safe diff) |
| `deploy.py` | Automated deployment to SGLang checkout |
| `upgrade-sglang.py` | Rebase the bundled tree onto a new upstream (3-way merge) |
| `scripts/sync_patches.py` | Regenerate `patches/` from the in-tree source of truth |
| `scripts/find_cama_patches.py` | Verify the manifest is complete vs the pinned base |
| `release.sh` | Release automation (version bump, tag, archive) |
| `CHANGELOG.md` | Full version history (v1.0.0–v1.8.0) |
| `CHANGELIST.md` | Integration changelist vs SGLang |
| `FolderOverview.md` | Integration instructions |
| `docs/01_OVERVIEW.md` | Architecture overview |
| `docs/02_ARCHITECTURE_DEEP_DIVE.md` | Detailed architecture (27K) |
| `docs/03_CONFIGURATION_REFERENCE.md` | Config reference |
| `docs/04_DEPLOYMENT_GUIDE.md` | Deployment guide |
| `docs/05_TROUBLESHOOTING.md` | Troubleshooting guide (23K) |
| `docs/06_DESIGN_DECISIONS.md` | Design rationale (20K) |
| `docs/07_HASHING_AND_KEY_HANDLING.md` | Key naming details |
| `docs/08_E2E_AUDIT_KEY_HANDLING.md` | End-to-end key audit |
| `docs/GET_FLOW_DIAGRAM.md` | Data flow diagram |
| `docs/PYBIND11_BATCH_BUG.md` | pybind11 batch ops workaround |
| `tests/test_cama_storage.py` | Unit tests (standalone copy, 631 lines) |

---

## 8. Deployment workflow

Use `deploy.py` to keep both in sync:

```bash
# Show what differs (dry-run — no files modified)
python deploy.py /path/to/sglang-fresh --diff

# Deploy module + integration patches
python deploy.py /path/to/sglang-fresh

# Deploy + create zip for remote cluster transfer
python deploy.py /path/to/sglang-fresh --zip
```

`deploy.py`'s patch list comes from `patch_manifest.json` (no longer a hardcoded
4-file map), so it deploys all 8 patched files. To rebase the bundled SGLang tree
onto a newer upstream release, see [`docs/UPGRADE_SGLANG.md`](docs/UPGRADE_SGLANG.md).

## 9. Known intentional differences (none currently)

When the repos are in sync, there are no intentional source differences.
If SGLang-specific adaptations are needed in the future (e.g. different
StorageMetrics import path after an SGLang upgrade), document them here.

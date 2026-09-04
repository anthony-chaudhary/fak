# CAMA Troubleshooting Guide

> Debugging and known issues reference for engineers hitting problems in production or development.

---

## Error Message Reference

Every error and assertion string from `cama_storage.py`, `preflight.py`, and supporting modules:

| # | Error String | Source | Cause | Fix |
|---|-------------|--------|-------|-----|
| 1 | `Please install the priskv package (pip install priskv)` | `cama_storage.py:57` | `priskv` Python package not installed | `pip install priskv` |
| 2 | `Config file path not set. Please set SGLANG_CAMA_CONFIG_PATH` | `cama_storage.py:48` | JSON config source selected but env var not set | Set `SGLANG_CAMA_CONFIG_PATH` to the JSON file path |
| 3 | `Failed to load config from <path>: <error>` | `cama_storage.py:55` | JSON file doesn't exist, isn't valid JSON, or isn't readable | Check file path, fix JSON syntax, check permissions |
| 4 | `'remote_addr' is required in config file` | `cama_storage.py:58` | JSON config or extra_config missing `remote_addr` key | Add `"remote_addr": "X.X.X.X"` to config |
| 5 | `PrisKV server not reachable after 600s at X:Y` | `cama_storage.py:351` | Health check timed out (10 min). PrisKV server not running or not reachable | Start PrisKV server, check network connectivity, verify address/port |
| 6 | `Warmup setstr failed with code N` | `cama_storage.py:360` | PrisKV connected but string write failed | Check PrisKV server logs, verify auth password, check server health |
| 7 | `Warmup: send/recv buffer RDMA registration failed` | `cama_storage.py:374-375` | RDMA `reg_memory` returned 0 for warmup buffer | Check `ulimit -l`, verify RDMA drivers with `ibv_devices`, may need root |
| 8 | `PrisKV reg_memory returned 0 — RDMA buffer registration failed` | `cama_storage.py:427` | RDMA registration of host KV buffer failed | Same as #7; also check total registered memory doesn't exceed NIC limits |
| 9 | `Cama storage backend only supports page_first, page_first_direct, or page_head layout` | `cama_storage.py:416` | Incompatible host memory layout | Set `--hicache-mem-layout page_first` |
| 10 | `Cama preflight check failed: PrisKV server at X:Y is not reachable` | `preflight.py:75` | Pre-model-loading connectivity check failed | Start PrisKV, or set `check_server=true` if PrisKV starts later |
| 11 | `Warmup: exists() returned 0 for non-existent key — PrisKV client bug` | `cama_storage.py:399` | pybind11 vector-copy bug affecting `exists()` | Update priskv package; see "pybind11 batch operations bug" below |
| 12 | `Multi-NIC discovery failed, keeping original connection: ...` | `cama_storage.py:275` | `rdma_endpoints()` raised an exception | Warning only — CAMA falls back to original connection. Check client version supports `rdma_endpoints()` |
| 13 | `Single RDMA endpoint, no multi-NIC reconnect needed.` | `cama_storage.py:271` | Server advertises only one RDMA endpoint | Informational — normal for single-NIC servers |
| 14 | `No RDMA endpoints advertised (TCP or old server).` | `cama_storage.py:273` | Server returned empty endpoint list | Debug-level — expected for TCP connections or older PrisKV versions |
| 15 | `_get_batch_zero_copy: N/M keys timed out after Xs: [...]` | `cama_storage.py` | `as_completed()` exceeded `op_timeout_s` for some keys in a get batch | Increase `op_timeout_s`, check PrisKV load, check RDMA health |
| 16 | `_put_batch_zero_copy: N/M keys timed out after Xs: [...]` | `cama_storage.py` | `as_completed()` exceeded `op_timeout_s` for some keys in a set batch | Same as #15 |
| 17 | `_batch_exist: N/M keys timed out after Xs: [...]` | `cama_storage.py` | `as_completed()` exceeded `op_timeout_s` for some keys in an exists batch | Same as #15 |
| 18 | `batch_exists: backend error on first key (...): ... — treating as 0 hits` | `cama_storage.py` | Short-circuit exists() call threw an exception | Check PrisKV connectivity; error is counted in `_exists_errors` |
| 19 | `CAMA error counters: get_errors=N, get_ok=M, ...` | `cama_storage.py` `get_stats()` | Cumulative error counters are non-zero | See "Error Counter Diagnostics" section above |
| 20 | `_prefetch_io_task: unhandled exception during page transfer for request ...` | `cache_controller.py` | `_page_transfer` threw inside the prefetch I/O executor | Check CAMA backend health; operation is marked terminated and memory is released |
| 21 | `prefetch_thread_func: unhandled exception during hit query for request ...` | `cache_controller.py` | `_storage_hit_query` or `all_reduce` threw in the prefetch thread | Check CAMA backend and NCCL health; operation is revoked and memory released |

---

## The pybind11 Batch Operations Bug

### Root Cause

PrisKV's pybind11 wrappers for `mexists`, `mset`, and `mget` accept `std::vector<uint32_t>&` as output parameters. pybind11 **copies** Python lists into C++ vectors — modifications to the C++ vector are never reflected back to Python. The Python-side status list stays all-zeros regardless of actual server responses.

### Anatomy of the Bug

```
Python side:                    C++ side (pybind11):

status = [0, 0, 0, 0]  ──COPY──▶  std::vector<uint32_t> status
                                   ↓
                                   PrisKV modifies status in-place
                                   status = [0, 1, 0, 1]  (real results)
                                   ↓
Python gets back:               C++ vector is destroyed
status = [0, 0, 0, 0]  (unchanged — the copy was never synced back)
```

### Symptoms

- `_batch_exist` (mexists) always reports all keys as "existing" (0 = exists)
- `batch_set_v1` never writes any real KV data (dedup thinks everything already stored)
- `batch_exists` reports 100% cache hits (triggers useless prefetch reads)
- PrisKV server stats show: `set_ops: 1, keys_inuse: 0, set_bytes: 2` (only warmup data stored)

### Resolution (RESOLVED)

This bug is fully bypassed. CAMA's `cama-client` now implements native batch wire-protocol ops that encode/decode messages directly in Python, bypassing PrisKV's pybind11 wrappers entirely:

- **`mexists`** → `OP_MTEST` (single roundtrip)
- **`mset`** → `OP_MSET` (single roundtrip, with sub-batch chunking)
- **`mget_rdma`** → `OP_MGET_RDMA` + batch RDMA Read + `OP_BATCH_READ_ACK` (single control roundtrip + single doorbell)

All batch ops are enabled by default (`use_mput_mget=true`). The `mget_rdma` path is auto-detected via server capability handshake. To disable and fall back to individual ops, set `use_mput_mget=false`.

### Legacy PrisKV Fix Options (for reference)

If fixing PrisKV's own pybind11 wrappers (not needed since CAMA bypasses them):

| Option | Approach | Complexity | Performance |
|--------|----------|-----------|-------------|
| **A (Recommended)** | Return `std::tuple<int, std::vector<uint32_t>>` as function return value | Simple | Good — pybind11 handles return value natively |
| B | `PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>)` — pass by reference | Medium | Good — zero-copy on status vector |
| C | Accept `py::array_t<uint32_t>` (numpy array) — write directly into numpy memory | Medium | Best — zero-copy on status vector |

---

## Host Memory Pool Saturation

### Symptom

All TP ranks log:
```
[write] host alloc failed: requested=8192 indices, node_id=6863 (total_drops=42)
```

The `total_drops` counter is cumulative and also pushed to Prometheus as `cama_client_host_alloc_drops`.

### Root Cause

The host memory pool (L2 CPU staging buffer) saturates when KV pages are produced (GPU→host eviction) faster than they are drained (host→CAMA backup). The backup thread is the bottleneck — by default it runs single-threaded and processes one `_page_backup()` batch at a time.

The same `node_id` appears on all ranks because eviction decisions are TP-synchronized: all ranks evict the same radix tree node simultaneously, and all see the same host pool exhaustion.

### Quick Diagnostics

```bash
# Count total drops from logs
grep "host alloc failed" server.log | tail -5

# Check backup queue depth in logs (logged every 100 enqueues)
grep "backup_queue depth" server.log

# Check Prometheus (if report_stats is running)
curl -s http://localhost:9090/api/v1/query?query=cama_client_host_alloc_drops
curl -s http://localhost:9090/api/v1/query?query=cama_client_backup_queue_depth
```

### Tuning Checklist

1. **Increase backup concurrency** — default is 2; set `backup_io_workers` to 3–4 to drain the queue faster:
   ```bash
   --hicache-storage-backend-extra-config '{"remote_addr": "...", "backup_io_workers": 4}'
   ```

2. **Lower batch size** — reduce per-batch latency (at cost of more roundtrips):
   ```bash
   --hicache-storage-backend-extra-config '{"remote_addr": "...", "storage_batch_size": 64}'
   ```

3. **Monitor queue depth** — watch `cama_client_backup_queue_depth` in Prometheus. A steadily growing queue means the drain rate is still too low.

4. **Check CAMA server throughput** — if the server is the bottleneck (slow writes, eviction pressure), increasing client-side concurrency won't help. Check `cama_ops_set_total` and server-side latency metrics.

5. **Increase host pool size** — if the above don't help, increase `--hicache-ratio` (default 2.0) to give more headroom. This trades host RAM for tolerance of drain-rate mismatches.

6. **Verify startup log** — confirm your settings are applied:
   ```
   [backup_thread] started: workers=2, batch_size=64
   ```

### Why Same node_id on All Ranks

SGLang's `HiRadixCache` synchronizes eviction decisions across tensor-parallel ranks using `all_reduce`. When the eviction frontier reaches a particular radix node, all ranks attempt to write-back the same node_id's KV pages simultaneously. If the host pool is full on any rank, it fails on all ranks at the same node.

---

## Silent Prefetch Failures

### Symptom: Requests Block Indefinitely Waiting for Prefetch

**Prior to the ThreadPoolExecutor fix**, unhandled exceptions in prefetch I/O threads would silently kill the daemon thread. Requests waiting on prefetch would block forever with no log output.

**Current behavior:** Both `prefetch_thread_func` and `_prefetch_io_task` catch all exceptions, log at ERROR level, release host memory, and continue processing. If you still see requests blocking:

1. Check for `_prefetch_io_task: unhandled exception` or `prefetch_thread_func: unhandled exception` in logs
2. Check CAMA error counters (see below)
3. Check if `op_timeout_s` is too low for your RDMA latency

### Symptom: Cache Appears Cold But PrisKV Has Data

If `batch_exists` returns 0 hits when data is known to exist, the backend may be erroring rather than returning genuine misses. Check error counters:

```bash
# Look for this in logs (logged at WARNING level by get_stats()):
CAMA error counters: get_errors=0, get_ok=0, set_errors=0, exists_errors=42, exists_timeouts=3
```

Non-zero `exists_errors` means `conn.exists()` is throwing exceptions, which are treated as "cache miss" for safe degradation but indicate a backend problem. Check PrisKV server health.

---

## Error Counter Diagnostics

CamaStorage maintains cumulative error counters reported via `get_stats()`. These are logged when any counter is non-zero.

### Reading Error Counters

| Log Pattern | Meaning |
|------------|---------|
| `get_errors` growing, `get_ok` flat | Backend is broken for reads — not recovering |
| `exists_errors` only | Exists RPCs failing but reads/writes work — possible timeout issue |
| `exists_timeouts` growing | `op_timeout_s` too low, or PrisKV is slow under load |
| `set_errors` growing | Write path broken — check PrisKV disk/memory pressure |
| All counters at 0 | Healthy |

### Timeout-Related Errors

If you see `_get_batch_zero_copy: N/M keys timed out after Xs` or similar:

1. **Check `op_timeout_s`** — Default is 10s. For cross-rack RDMA or loaded servers, try 15–30s:
   ```bash
   --hicache-storage-backend-extra-config '{"op_timeout_s": 20.0}'
   ```
2. **Check PrisKV load** — A single slow key can trigger the timeout for the entire batch (though completed keys still return results)
3. **Check RDMA health** — Use `ibv_devinfo` and verify NIC link status
4. **Reduce `io_workers`** if contention is high — too many concurrent ops on one RDMA connection can increase per-op latency past the timeout

### Prefetch Thread Health

Every 60 seconds, `prefetch_thread_func` logs a health summary at INFO level:
```
[prefetch_thread] health: received=142, dispatched=118, revoked=24, io_completed=115,
  io_failed=3, io_in_flight=2, total_tokens=472K, total_pages=1475,
  avg_io=12.3ms, query_failures=0, queue_size=1
```

| Field | Expected | Problem if |
|-------|----------|-----------|
| `io_completed` | > 0 (proportional to request volume) | 0 = no prefetch I/O happened at all |
| `io_failed` | 0 | > 0 = page transfers crashed (check WARNING logs above) |
| `io_in_flight` | low (< `prefetch_io_workers`) | equal to `prefetch_io_workers` = all workers busy, prefetch is saturated |
| `total_tokens` | growing | stalled = no data being transferred |
| `avg_io` | low (< 50ms for same-rack RDMA) | high = backend slow or NIC saturated |
| `revoked` | low relative to `received` | high ratio = most requests below `prefetch_threshold` (cache is cold) |
| `query_failures` | 0 | > 0 = `_storage_hit_query` is failing (check backend health) |

Per-request prefetch details (enqueue, pickup, query, dispatch, I/O completion) are logged at **DEBUG** level. Enable with `SGLANG_LOG_LEVEL=DEBUG` for per-request tracing.

---

## Reconnection

### Symptom: Operations Stall for ~15s Then Recover

This is reconnect working as intended. The client detected a transport failure, retried with exponential backoff (default worst-case: 0.5+1+2+4+8 = 15.5s), and re-established the connection.

**What to check:**
- Server logs for restarts, OOM kills, or network blips around the same timestamp
- `cama_client_reconnect_successes` Prometheus metric (via `report_stats()`)
- If 15s is too long, lower `reconnect_max_retries` (e.g. 3 → worst-case ~3.5s)

### Symptom: Cannot Disable Reconnect via Environment Variable

Environment variables are strings. Setting `SGLANG_CAMA_RECONNECT_ENABLED=0` or `=False` may still be treated as truthy if the connector's config loading does not coerce the string to a proper boolean.

**Workarounds:**
1. **JSON config file** — set `"reconnect_enabled": false` (proper boolean):
   ```json
   {"remote_addr": "10.0.0.1", "reconnect_enabled": false}
   ```
2. **extra_config** — pass via CLI:
   ```bash
   --hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "reconnect_enabled": false}'
   ```

Both sources deliver a native Python `bool`, bypassing string coercion.

---

## RDMA Troubleshooting

### Verifying RDMA Devices

```bash
# List RDMA devices
ibv_devices

# Expected output:
#     device          node GUID
#     ------          ----------------
#     mlx5_0          b8ce...
#     mlx5_1          b8ce...

# Detailed device info
ibv_devinfo
```

If `ibv_devices` returns empty, RDMA drivers are not loaded. Install `libibverbs` and the appropriate NIC drivers (e.g., MLNX_OFED for Mellanox).

### Memory Registration Limits

```bash
# Check current limit
ulimit -l

# If it's not "unlimited":
ulimit -l unlimited

# Make permanent in /etc/security/limits.conf:
# *  soft  memlock  unlimited
# *  hard  memlock  unlimited
```

PrisKV's `reg_memory` calls `ibv_reg_mr()` under the hood. If the locked memory limit is too low, registration returns 0.

### Firewall Rules

PrisKV uses port 6379 by default. Ensure the port is open:

```bash
# Test TCP connectivity
nc -zv <priskv_addr> 6379

# Check iptables
iptables -L -n | grep 6379
```

### Verifying RDMA Independently

Test RDMA without CAMA:

```python
from priskv.priskv_client import PriskvClient
import priskv
import numpy as np

conn = PriskvClient("10.0.0.1", 6379, "")

# Register a buffer
buf = np.arange(256, dtype=np.float32)
reg = conn.reg_memory(buf.ctypes.data, buf.nbytes)
print(f"Registration handle: {reg}")  # should be non-zero

# SGL round-trip
sgl = priskv.SGL(buf.ctypes.data, buf.nbytes, reg)
ret = conn.set("rdma_test", sgl)
print(f"set: {ret}")  # should be 0

recv = np.zeros_like(buf)
recv_reg = conn.reg_memory(recv.ctypes.data, recv.nbytes)
recv_sgl = priskv.SGL(recv.ctypes.data, recv.nbytes, recv_reg)
ret = conn.get("rdma_test", recv_sgl, recv.nbytes)
print(f"get: {ret}")  # should be 0
print(f"Match: {np.array_equal(buf, recv)}")  # should be True

# Cleanup
conn.delete("rdma_test")
conn.dereg_memory(reg)
conn.dereg_memory(recv_reg)
```

---

## SGLang Startup Hangs

SGLang server hanging at startup is usually **not** a CAMA issue but a missing native dependency. Below is a condensed diagnostic guide.

### Quick Diagnosis

```bash
# Verify critical imports
python -c "import sgl_kernel; print('sgl-kernel:', sgl_kernel.__version__)"
python -c "import flashinfer; print('flashinfer:', flashinfer.__version__)"
```

If either fails → that's the hang cause. Fix:
```bash
pip install sgl-kernel
pip install flashinfer_python flashinfer_cubin --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

### py-spy Stack Dump

```bash
pip install py-spy
py-spy dump --pid $(pgrep -f sglang)
```

### Diagnosis Table

| Top of Stack | Diagnosis | Fix |
|-------------|-----------|-----|
| `tvm_ffi/utils/lockfile.py:blocking_acquire` → `build_inline` → `gptq_marlin` | JIT kernel hang — missing `nvcc` | `apt install cuda-toolkit-12-4` (quantized models only) |
| `lockfile.py:blocking_acquire` (alone) | Stale lock from crashed process | Find and delete `.lock` file: `find /tmp -name "*.lock" -path "*tvm*"` |
| `tokenizers` or `rust_tokenizers` | HuggingFace tokenizer deadlock | `export TOKENIZERS_PARALLELISM=false` |
| `nccl` or `ncclCommInitRank` | NCCL multi-GPU hang | `NCCL_DEBUG=INFO`, try `NCCL_P2P_DISABLE=1` |
| `torch._inductor` | Torch compile hanging | `export TORCH_COMPILE_DISABLE=1` |
| `cuda_graph_runner.py:capture` + CUDA sync | CUDA graph capture (may be normal) | Check if progress bar advances — see "Slow vs Stuck" |

### Slow vs Stuck: How to Tell

| Sign | Working (Slow) | Stuck |
|------|---------------|-------|
| Progress bar | Visible and advancing | Missing or frozen at 0% |
| py-spy | Shows `capture` with active CUDA calls | Shows `blocking_acquire` |
| Expected duration | ~6 min first run (JIT + CUDA graph) | 10+ min with no output |
| Log output | `"Capture cuda graph begin"` then progress | No output after begin message |

### Debug Environment Variables

```bash
export CUDA_LAUNCH_BLOCKING=1       # Synchronous CUDA
export SGLANG_LOG_LEVEL=DEBUG       # Verbose logging
export TOKENIZERS_PARALLELISM=false  # Prevent tokenizer deadlock
```

---

## Cache Not Working

### Symptom: No Cache Hits on Repeated Requests

**Check write policy:**
```bash
# Must use write_through for immediate writes
--hicache-write-policy write_through
```
`write_back` defers writes — pages may not be flushed before the second request.

**Check for failed writes (enable debug logging):**
```bash
SGLANG_LOG_LEVEL=DEBUG python -m sglang.launch_server ...
```
Look for: `batch_set_v1: N to write` followed by `N ok, 0 failed`. If writes fail, check PrisKV server logs.

**Check key collisions:** If multiple instances share PrisKV without `extra_backend_tag`, they may read stale/wrong pages. Set unique tags per instance.

**Check PrisKV eviction:** If PrisKV's capacity is smaller than the working set, pages may be evicted between write and read. Check PrisKV server stats for eviction counts.

### Symptom: All Writes Skipped (Dedup Treats Everything as Existing)

This is the pybind11 batch ops bug. Enable debug logging (`SGLANG_LOG_LEVEL=DEBUG`) and look for:
```
batch_set_v1: N total sub-keys, N already exist (deduped), 0 to write
```
when the keys should be new, the individual `exists()` fallback isn't working. Check:
- Is the `priskv` package up to date?
- Does the warmup assertion at line 398 pass? If not, the warmup would have crashed.

### Multi-NIC Issues

**NIC striping not activating:**
- Verify `nic_striping=True` (default). Check for `Multi-NIC striping: rank N -> M endpoints` in logs. If you see legacy `Multi-NIC: rank N -> mlx5_X` instead, `nic_striping` may be set to `False`.
- Verify the server is advertising multiple endpoints. If logs show `Single RDMA endpoint, no multi-NIC reconnect needed.`, the server only has one RDMA NIC configured.
- Check PrisKV server configuration for multi-NIC support.

**Shared PD failed for endpoint — using independent PD:**
- This log appears when a multi-NIC pool connection routes through a different client RDMA device than the PD-owner. The pool falls back to an independent PD for that connection. MRs are registered separately — `reg_memory()` handles this transparently. No action needed.

**Uneven per-NIC read distribution:**
- Check `get_transport_stats()` for `per_nic_reads` — should show roughly even distribution across NICs. Small key counts may not distribute evenly (keys are partitioned round-robin, so fewer keys than NICs means some NICs idle).
- If one NIC consistently shows 0 reads, verify that connection is alive and not in reconnect state.

**Discovery exception:**
- If you see `Multi-NIC discovery failed, keeping original connection: ...`, the `rdma_endpoints()` call failed. This is a warning — CAMA continues on the original connection.
- Common cause: client library version doesn't support `rdma_endpoints()`. Update `cama-client` or `priskv` package.
- This is a graceful fallback — functionality is not affected, only NIC distribution.

---

## Performance Debugging

### `get_stats()` Log Output

Every ~10 seconds (when SGLang calls `get_stats()`), CamaStorage logs an enriched I/O health summary at INFO level:

```
CAMA I/O stats: 204 calls, avg_batch=42.0, avg_latency=14.7ms, workers=4 |
  inflight: get=1 set=0 | phases: pre=0.2ms xfer=14.1ms post=0.4ms exists=2.1ms |
  p50=<=10ms p99=<=100ms
```

| Field | Meaning |
|-------|---------|
| `calls` | Number of `batch_get_v1`/`batch_set_v1` calls this interval |
| `avg_batch` | Average sub-keys per batch |
| `avg_latency` | Average end-to-end batch latency |
| `workers` | I/O thread pool size |
| `inflight: get=N set=M` | Currently in-flight batch operations (saturation signal) |
| `phases: pre/xfer/post/exists` | Average time in preprocessing, RDMA transfer, postprocessing, and exists-check phases |
| `p50/p99` | Latency percentiles from histogram buckets |

If errors are non-zero, a separate WARNING line is also logged:
```
CAMA error counters: get_errors=0, get_ok=42, set_errors=0, exists_errors=0, exists_timeouts=0
```

Per-batch details (individual batch timings, hit/miss counts, dedup stats) are logged at **DEBUG** level. Enable with `SGLANG_LOG_LEVEL=DEBUG` for per-batch tracing.

### `StorageMetrics` Object

`get_stats()` also returns a `StorageMetrics` object with programmatic access to the same data:

```python
stats = storage.get_stats()

# Per-interval bandwidth metrics (cleared after each call)
print(f"Prefetch pages: {stats.prefetch_pgs}")
print(f"Prefetch bandwidth: {stats.prefetch_bandwidth} GB/s")
print(f"Backup pages: {stats.backup_pgs}")
print(f"Backup bandwidth: {stats.backup_bandwidth} GB/s")

# Per-interval I/O pool metrics (cleared after each call)
print(f"I/O calls: {stats.io_calls}")
print(f"Avg batch size: {stats.avg_io_batch_size}")
print(f"Avg latency: {stats.avg_io_latency_ms}ms")
print(f"I/O workers: {stats.io_workers}")

# Phase timing averages (cleared after each call)
print(f"Avg preprocess: {stats.avg_preprocess_ms}ms")
print(f"Avg transfer: {stats.avg_transfer_ms}ms")
print(f"Avg postprocess: {stats.avg_postprocess_ms}ms")
print(f"Avg exists: {stats.avg_exists_ms}ms")

# Cumulative error counters (NEVER cleared — monotonically increasing)
print(f"Get errors: {stats.get_errors}")
print(f"Get successes: {stats.get_successes}")
print(f"Set errors: {stats.set_errors}")
print(f"Exists errors: {stats.exists_errors}")
print(f"Exists timeouts: {stats.exists_timeouts}")
```

Note: Bandwidth, I/O pool, and phase timing metrics are per-interval and cleared after each call. Error counters are **cumulative** — they grow over the lifetime of the CamaStorage instance and are never reset. If `get_errors` grows while `get_successes` stays flat, the backend is broken and not recovering.

### Profiling with Pyroscope + NVTX

```bash
# Enable profiling
export SGLANG_CAMA_PROFILING_ENABLED=1
export SGLANG_CAMA_PROFILING_SERVER_ADDRESS=http://pyroscope-server:4040
export SGLANG_CAMA_PROFILING_SERVICE_NAME=cama.debug

# Install optional packages
pip install pyroscope-io nvtx
```

NVTX markers are placed on: `batch_get_v1`, `batch_set_v1`, `batch_get`, `batch_set`, `batch_exists`, `_put_batch_zero_copy`, `_get_batch_zero_copy`, `_batch_exist`.

To visualize NVTX ranges with Nsight Systems:
```bash
nsys profile -t cuda,nvtx python -m sglang.launch_server ...
nsys-ui report.nsys-rep
```

### PrisKV Server Stats

Check PrisKV's internal statistics:
```python
from priskv.priskv_client import PriskvClient
conn = PriskvClient("10.0.0.1", 6379, "")
keys = conn.keys("*")
print(f"Total keys stored: {len(keys)}")
# Sample key patterns
for k in keys[:10]:
    print(f"  {k}")
```

---

## Running the Test Suite

### Test Layer Structure

`test_cama_storage.py` uses a progressive layer system. If any layer fails, all higher layers are skipped.

| Layer | Name | What It Tests | Dependencies |
|-------|------|--------------|--------------|
| 0 | PrisKV Server Alive | `setstr`/`getstr` round-trip, `exists`/`delete` | Running PrisKV server |
| 1 | RDMA Memory Registration + Raw Byte Round-Trip | `reg_memory`, SGL `set`/`get`, 1 MB bitwise comparison | Layer 0 |
| 2 | Batch Operations (mset/mget) | `mset`/`mget`/`mexists`, data integrity across 8 keys (warns about pybind11 bug) | Layer 0 |
| 3 | CamaStorage Config, Warmup, Key Naming | Config loading, MHA/MLA suffix construction, extra_backend_tag | Layers 0-2 |
| 4 | Full KV Cache Page Round-Trip | `MockHostKVCache` + `_batch_preprocess` + zero-copy write + zero + read + bitwise comparison | Layer 3 |
| 5 | Write Deduplication | `batch_set_v1` twice — second call should not trigger `_put_batch_zero_copy` (monkeypatched counter) | Layer 4 |
| 7 | Metrics / Bandwidth | Legacy `batch_set`/`batch_get` + `get_stats()` verification | Layer 5 |

Note: Layer 6 (E2E integration) is not implemented in the unit test file.

### Running Tests

```bash
# Basic run against local PrisKV
python python/sglang/srt/mem_cache/storage/cama/test_cama_storage.py \
    --addr 127.0.0.1 --port 6379

# Against remote PrisKV
python python/sglang/srt/mem_cache/storage/cama/test_cama_storage.py \
    --addr 10.0.0.1 --port 6379 --password "my_secret"

# Using environment variables
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 SGLANG_CAMA_REMOTE_PORT=6379 \
python python/sglang/srt/mem_cache/storage/cama/test_cama_storage.py
```

### Expected Output (All Passing)

```
CAMA Storage Test Suite — targeting 127.0.0.1:6379

  [PASS] Layer 0: PrisKV Server Alive
  [PASS] Layer 1: RDMA Memory Registration + Raw Byte Round-Trip
  [PASS] Layer 2: Batch Operations (mset/mget)
         mset return codes: [0, 0, ...]  (may be unreliable due to pybind11 vector-copy bug)
         mget return codes: [0, 0, ...]  (may be unreliable due to pybind11 vector-copy bug)
         mexists return codes: [0, 0, ...]  (may be unreliable due to pybind11 vector-copy bug)
  [PASS] Layer 3: CamaStorage Config, Warmup, Key Naming
  [PASS] Layer 4: Full KV Cache Page Round-Trip
  [PASS] Layer 5: Write Deduplication
  [PASS] Layer 7: Metrics / Bandwidth

============================================================
CAMA Storage Test Summary
============================================================
  Layer 0: PASS  PrisKV Server Alive
  Layer 1: PASS  RDMA Memory Registration + Raw Byte Round-Trip
  Layer 2: PASS  Batch Operations (mset/mget)
  Layer 3: PASS  CamaStorage Config, Warmup, Key Naming
  Layer 4: PASS  Full KV Cache Page Round-Trip
  Layer 5: PASS  Write Deduplication
  Layer 7: PASS  Metrics / Bandwidth
============================================================
  7 passed, 0 failed, 0 skipped
```

---

## Related Documents

- [01_OVERVIEW.md](01_OVERVIEW.md) — What CAMA is
- [02_ARCHITECTURE_DEEP_DIVE.md](02_ARCHITECTURE_DEEP_DIVE.md) — Internals for debugging
- [03_CONFIGURATION_REFERENCE.md](03_CONFIGURATION_REFERENCE.md) — All parameters and validation errors
- [04_DEPLOYMENT_GUIDE.md](04_DEPLOYMENT_GUIDE.md) — Step-by-step deployment
- [06_DESIGN_DECISIONS.md](06_DESIGN_DECISIONS.md) — Why decisions were made
- `priskv_deep_parameter_reference.md` (reference archive) — PrisKV server parameters
- `glossary_mooncake_nixl_transfer_engines.md` (reference archive) — RDMA and transfer engine glossary

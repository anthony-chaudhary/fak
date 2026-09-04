# Codec Trade-offs: Compression vs Zero-Copy

> When to enable compression, what it costs, and where batch decode fits in.

---

## Overview

CAMA's codec system is **fully optional** — disabled by default (`codec=""`), zero overhead when off. When enabled, it trades zero-copy RDMA Read for reduced wire bytes and increased effective cache capacity. This document covers the mechanics of that trade-off, quantifies the copy overhead, and projects what batched decompression could recover.

**Source:** `codec.py` (codec framework), `cama_storage.py` lines 638–666 (init), 1042–1086 (SET), 1148–1201 (GET)

---

## The Core Trade-off

Without codec, the GET path is a single-roundtrip `mget_rdma` that lands data directly in tensor buffers via RDMA Read — true zero-copy from the perspective of host CPU. With codec, values stored on the server are smaller (compressed), so the client cannot RDMA Read directly into the tensor buffer (sizes don't match). Instead, it reads compressed bytes into an internal buffer, decompresses in Python, then `memmove`s into the tensor buffer.

```
Without codec (zero-copy):
  mget_rdma ─── RDMA Read ──▸ tensor buffer   (1 copy, NIC DMA)

With codec:
  mget_rdma_raw ─── RDMA Read ──▸ internal buf   (copy 1: NIC DMA)
  unwrap_value(data)             ──▸ decompressed  (copy 2: decode)
  ctypes.memmove(tensor, ...)    ──▸ tensor buffer (copy 3: final placement)
```

---

## Copy Accounting

### SET Path (write)

| Step | Without codec | With codec |
|------|--------------|------------|
| 1. Build SGL | `SGL(ptr, size, reg_buf)` — wraps registered pointer | `SGL.to_bytes()` — materializes bytes from registered buffer **(copy 1)** |
| 2. Encode | — | `codec.encode(raw)` — produces compressed bytes **(copy 2)** |
| 3. Header | — | `wrap_value()` — prepends 8-byte header **(copy 3)** |
| 4. Wire | `mset_striped` sends SGL directly | `mset_striped` sends `_CompressedSGL` bytes |

**Extra copies with codec: 3** (to_bytes + encode + header wrap)

### GET Path (read)

| Step | Without codec | With codec |
|------|--------------|------------|
| 1. RDMA Read | `mget_rdma` — directly into tensor buffer | `mget_rdma_raw` — into internal C++ `read_buf_` **(copy 1)** |
| 2. Return to Python | — | `py::bytes(...)` from C++ to Python **(copy 2)** |
| 3. Decode | — | `unwrap_value(data)` — decompresses **(copy 3)** |
| 4. Place | — | `ctypes.memmove(tensor_ptr, decompressed, n)` **(copy 4)** |

**Extra copies with codec: 3–4** vs zero-copy path (which has 1 NIC DMA copy)

---

## When Each Side Wins

### Codec wins (enable compression)

| Scenario | Why |
|----------|-----|
| **Capacity-bound** — eviction rate is high | 2x compression = 2x effective cache. Fewer evictions dominate any per-op overhead. |
| **Bandwidth-saturated** — NIC at line rate | 2x fewer wire bytes per value. CPU memcpy at ~30 GB/s per core easily outpaces 25 Gb/s RDMA links. |
| **Large values** — page_size ≥ 64, value > 128 KB | Copy overhead is amortized. Bandwidth savings scale linearly with value size. |
| **Multi-tenant** — shared server, tight memory | Doubles the number of models/ranks that fit in one server. |

### Zero-copy wins (disable compression)

| Scenario | Why |
|----------|-----|
| **Latency-sensitive** — prefill tail latency matters | 3 extra Python-side copies add ~10–50 μs per key. Batch RDMA Read with zero-copy is the fastest possible path. |
| **Capacity headroom** — utilization < 50% | Compression doesn't help if you're not evicting. |
| **Small batches** — few keys per GET | Per-key decode overhead isn't amortized across the batch. The sequential Python loop dominates. |
| **Lossless required** — no quantization tolerance | `shuffle_zstd` is lossless but only ~1.3x. The copy cost may not be worth 30% savings. |

---

## Quantization Specifically (int8)

INT8 symmetric quantization (`codec="int8"`) is the most common codec choice because it offers the best ratio of compression to CPU cost:

| Metric | Value |
|--------|-------|
| Compression ratio | ~2.0x (N×fp16 → N×int8 + 4B scale) |
| Lossy | Yes — `atol ≈ 0.05` (max absolute error per element) |
| Encode cost | 1 `absmax` + 1 divide + 1 round + 1 cast = ~4 vectorized numpy ops |
| Decode cost | 1 multiply + 1 cast = ~2 vectorized numpy ops |
| CPU throughput | ~20–40 GB/s per core (numpy, single-threaded) |

**Net result for a typical 100-key batch of 256 KB values (25.6 MB total):**

| | Zero-copy | int8 codec |
|--|-----------|-----------|
| Wire bytes | 25.6 MB | 12.8 MB |
| RDMA Read time @ 25 Gb/s | ~8.2 ms | ~4.1 ms |
| Decode time (single core) | 0 | ~0.6 ms (25.6 MB @ 40 GB/s) |
| memmove time | 0 | ~0.6 ms (25.6 MB @ 40 GB/s) |
| **Total** | **~8.2 ms** | **~5.3 ms** |

At 25 Gb/s, int8 is a net win even with the extra copies. At 100 Gb/s, the RDMA Read time drops to ~2 ms (zero-copy) vs ~1 + 1.2 = ~2.2 ms (codec), roughly break-even. Above 100 Gb/s, zero-copy pulls ahead.

**Rule of thumb:** int8 wins when `RDMA_bandwidth < 2 × CPU_memcpy_bandwidth`.

---

## Current Batch Status

As of v0.34.0 / connector v1.18.0, the codec GET path **is already batched** at the RDMA level:

```
codec GET path (current):
  1. conn.mget_rdma_raw(all_keys)     ← single roundtrip, batch doorbell
  2. for each (rc, data):              ← sequential Python loop
       unwrap_value(data)              ← per-key decode
       ctypes.memmove(...)             ← per-key copy
```

The RDMA reads are fully batched — one `OP_MGET_RDMA` control message, one `OP_MGET_READ_READY` response, one batch `ibv_post_send` doorbell (chunked at `MAX_SEND_WR`), one `OP_BATCH_READ_ACK`. The **decompression loop is the remaining sequential bottleneck.**

The non-codec path (`mget_rdma`) is also fully batched but lands data directly in tensor buffers — no post-processing loop.

---

## Batch Decode: Projected Improvements

Three levels of optimization are available for the decode loop, from lowest to highest effort:

### Level 1: Python ThreadPoolExecutor (low effort)

Parallelize the `unwrap_value` + `memmove` loop across `io_workers` threads.

```python
# Projected change in _get_batch_zero_copy:
from concurrent.futures import ThreadPoolExecutor, as_completed

def _decode_one(i, data):
    decompressed = unwrap_value(data)
    ctypes.memmove(buffer_ptrs[i], decompressed, min(len(decompressed), buffer_sizes[i]))
    return _RC.GET_OK

with ThreadPoolExecutor(max_workers=io_workers) as pool:
    futures = {pool.submit(_decode_one, i, data): i
               for i, (rc, data) in enumerate(raw_results) if rc == 0 and data}
```

**Projected gain:** Near-linear scaling with worker count for CPU-bound decode (numpy releases the GIL for vectorized ops). 4 workers ≈ 3.5x decode throughput.

**Caveat:** Python `bytes` allocation per key still serializes through the GIL for the allocation itself. Real-world gain closer to 2–3x with 4 workers.

### Level 2: C++ Batch Decode (medium effort)

Move the decode loop into the C++ extension. After `batch_rdma_read()` returns raw bytes in `read_buf_`, inspect the 8-byte codec header and decode in-place before returning to Python.

```
batch_rdma_read_decoded(keys, codec_id):
  1. Post batch RDMA Reads into read_buf_ (existing)
  2. For each value in read_buf_:
     - Parse 8-byte header
     - Decode in-place (int8: scale * int8_data → fp16 output)
  3. Return list of decoded py::bytes (or write directly to user buffers)
```

**Projected gain:** Eliminates Python loop overhead, all `bytes` allocation, and 1 copy (no Python intermediate). Decode runs at C++ speed (~50–80 GB/s with SIMD). Combined with user-buffer targets, this could match zero-copy latency while keeping 2x compression.

**Prerequisite:** The C++ extension needs to know the codec_id. Pass it as a constructor/method argument — the connector already knows it from `self._codec.codec_id`.

### Level 3: C++ Decode into User Buffers (high effort, near-zero-copy)

Combine `batch_rdma_read` + decode + placement into a single C++ call:

```
batch_rdma_read_decode_into(keys, codec_id, user_ptrs, user_sizes):
  1. Post batch RDMA Reads into read_buf_
  2. For each value:
     - Parse header, decode directly into user_ptr[i]
     - No intermediate allocation
  3. Return list[int] of return codes
```

**Projected gain:** 2 copies total (NIC DMA + decode-into-target), vs 1 copy for zero-copy `mget_rdma`. The only overhead vs uncompressed is the decode compute itself (~0.5 ms for 25 MB), while getting 2x wire bandwidth savings.

**This is the theoretical optimum for compressed RDMA reads.**

---

## Summary Matrix

| Path | Copies | Roundtrips | Batch | Decode parallelism | Status |
|------|--------|------------|-------|-------------------|--------|
| `mget_rdma` (no codec) | 1 (NIC DMA) | 1 | Full | N/A | Current default |
| `mget_rdma_raw` + Python decode | 3–4 | 1 | RDMA: yes, decode: sequential | None | Current codec path |
| + Level 1 (thread pool decode) | 3–4 | 1 | RDMA: yes, decode: parallel | Python threads | Not yet implemented |
| + Level 2 (C++ batch decode) | 2 | 1 | Full | C++ (single-threaded, SIMD) | Not yet implemented |
| + Level 3 (C++ decode-into) | 2 | 1 | Full | C++ (zero intermediate alloc) | Not yet implemented |

---

## Configuration Quick Reference

```toml
# Enable int8 quantization (~2x compression, lossy)
codec = "int8"

# Enable lossless compression (~1.3x)
codec = "shuffle_zstd"
codec_zstd_level = 3        # 1-22, higher = better ratio, more CPU

# Chain for maximum compression (~2.6x, lossy)
codec = "int8+shuffle_zstd"

# Disable (default) — zero-copy RDMA Read
codec = ""
```

**Changing codec requires FLUSH** — existing values use the old encoding. `unwrap_value()` auto-detects the codec from the 8-byte header, so mixed old/new values are readable during transition.

---

## Related Documents

- [03_CONFIGURATION_REFERENCE.md](03_CONFIGURATION_REFERENCE.md#compression-parameters) — Codec config parameters and defaults
- [06_DESIGN_DECISIONS.md](06_DESIGN_DECISIONS.md#batch-rdma-get-op_mget_rdma--page-size-independence) — Why batch RDMA GET exists
- [02_ARCHITECTURE_DEEP_DIVE.md](02_ARCHITECTURE_DEEP_DIVE.md) — Data flow and code organization
- [GET_FLOW_DIAGRAM.md](GET_FLOW_DIAGRAM.md) — Visual GET path diagram

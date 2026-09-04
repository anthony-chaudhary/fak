# RDMA Extension Internals

Reference documentation for `cama_client/csrc/rdma_transport.cpp` -- the pybind11
C++ extension that bridges Python to libibverbs and librdmacm.

**Audience:** developers modifying `rdma_transport.cpp`, debugging RDMA client
issues, or writing new Python code that consumes the `_cama_rdma` module.

**Source file:** `cama-client/cama_client/csrc/rdma_transport.cpp` (1,341 lines)

**Build output:** `_cama_rdma.<platform>.so` (e.g., `_cama_rdma.cpython-310-x86_64-linux-gnu.so`)

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [GIL release mechanics](#2-gil-release-mechanics)
3. [Buffer management](#3-buffer-management)
4. [Connection flow](#4-connection-flow)
5. [Send/Recv flow](#5-sendrecv-flow)
6. [RDMA Read flow](#6-rdma-read-flow)
7. [Batch RDMA Read](#7-batch-rdma-read)
8. [MR thread safety](#8-mr-thread-safety)
9. [Error handling](#9-error-handling)
10. [Timing and stats](#10-timing-and-stats)
11. [Constants table](#11-constants-table)
12. [Python integration](#12-python-integration)
13. [FAQ](#13-faq)
14. [Related docs](#14-related-docs)

---

## 1. Architecture overview

The extension compiles into a single `.so` that exports the `_cama_rdma`
pybind11 module.  Python code never calls libibverbs directly; every RDMA
operation passes through the `RDMATransport` C++ class.

```
 +---------------------------------------------------------------+
 |                      Python process                            |
 |                                                                |
 |  cama_client/rdma/                                             |
 |  +-----------------+ +-----------------+ +------------------+  |
 |  | _client.py      | | _pool.py        | | _batch_ops.py    |  |
 |  | (RDMAClient)    | | (RDMAClientPool)| | (mget_rdma,      |  |
 |  |                 | | shared PD,      | |  mget_rdma_raw,  |  |
 |  |                 | | reconnect)      | |  mset_striped)   |  |
 |  +-------+---------+ +-------+---------+ +--------+---------+  |
 |          |                   |                     |            |
 |          +---------+---------+---------+-----------+            |
 |                    |                   |                        |
 |  +-----------------v-+ +--------------v-----------+            |
 |  | _constants.py      | | _stats.py               |            |
 |  +----------+---------+ +--------------+----------+            |
 |             |                          |                        |
 |  +----------v--------------------------v----------+            |
 |  |               _cama_rdma.so                    |            |
 |  |  Exports:                                      |            |
 |  |    class RDMATransport                         |            |
 |  |    is_available(), wc_status_name()            |            |
 |  |    GIL_RELEASED, DEFAULT_*_BUF_SIZE            |            |
 |  |    __version__                                 |            |
 |  +----+-------------------------------------------+            |
 |       |                                                        |
 +-------+--------------------------------------------------------+
         |
   +-----v-----+  +------------+
   | libibverbs |  | librdmacm  |
   +-----+------+  +-----+------+
         |               |
   +-----v---------------v----+
   | RDMA kernel driver        |
   | (mlx5_ib, rxe, ...)       |
   +---------------------------+
```

Key points:

- The `.so` links against `libibverbs` and `librdmacm` at load time.
  If the libraries are missing, `import _cama_rdma` raises `ImportError`.
- `is_available()` probes `ibv_get_device_list` to check for RDMA hardware.
  Missing devices cause a graceful TCP fallback with a `RuntimeWarning`.
- `__version__` is injected at build time via `-DCAMA_VERSION` from
  `setup.py`.  A version mismatch triggers a warning from
  `_check_rdma_version()` in `__init__.py`.

---

## 2. GIL release mechanics

Without GIL release, 64 Python threads serialize into single-threaded
execution (~2 GB/s).  With GIL release, the same threads reach 10+ GB/s.

The extension uses two strategies, chosen based on the method signature:

### Strategy 1: `py::call_guard<py::gil_scoped_release>()`

Applied at the pybind11 binding site for methods that are pure C++ (void
return or types that do not require the GIL to construct):

| Method                     | Return type        | Line (binding) |
|----------------------------|--------------------|----------------|
| `connect`                  | void               | 1256-1259      |
| `connect_with_shared_pd`   | void               | 1292-1297      |
| `rdma_read_into`           | void               | 1268-1272      |
| `try_rdma_read_into`       | int (WC status)    | 1273-1277      |
| `batch_rdma_read_into`     | vector<uint8_t>    | 1278-1282      |
| `dereg_mr`                 | void               | 1301-1304      |

### Strategy 2: Manual `py::gil_scoped_release` inside the method body

Used when the method must return `py::bytes` or another Python object:

| Method              | Why manual                                   | Lines           |
|---------------------|----------------------------------------------|-----------------|
| `roundtrip`         | Returns `py::bytes` from recv buffer         | 261-267         |
| `rdma_read`         | Returns `py::bytes` from read buffer         | 274-281         |
| `batch_rdma_read`   | Returns `vector<py::bytes>`, sub-batch loop  | 518-615         |
| `close`             | Also called from destructor (no GIL there)   | 248-253         |
| `reg_mr`            | `ibv_reg_mr` is slow; return pair needs GIL  | 627-643         |

Pattern:

```cpp
py::bytes roundtrip(const std::string& request_bytes) {
    std::string result;
    {
        py::gil_scoped_release release;      // <-- release
        result = roundtrip_impl(...);        // pure C++ work
    }                                        // <-- reacquire
    return py::bytes(result);                // needs GIL
}
```

### Feature flag

The module exports `GIL_RELEASED = True` (line 1236) so Python code can
verify at import time that the GIL-releasing build is loaded.  This guards
against stale `.so` files from editable installs (see Gotcha 1).

---

## 3. Buffer management

### Three pre-registered buffers

Every `RDMATransport` instance allocates and registers three internal
buffers during `connect()`:

| Buffer      | Default size | Field       | MR field   | Purpose                           |
|-------------|-------------|-------------|------------|-----------------------------------|
| Send buffer | 16 MB       | `send_buf_` | `send_mr_` | Outgoing RDMA Send requests       |
| Recv buffer | 16 MB       | `recv_buf_` | `recv_mr_` | Incoming RDMA Recv responses      |
| Read buffer | 32 MB       | `read_buf_` | `read_mr_` | Local target for RDMA Read ops    |

### Buffer lifecycle

```
  malloc(size)  -->  memset(0)  -->  ibv_reg_mr(pd_, buf, size,
                                               IBV_ACCESS_LOCAL_WRITE)
       |                                    |
       v                                    v
  [pages faulted]                    [pinned + NIC-visible]
                                            |
                                    [ operational use ]
                                            |
                                    ibv_dereg_mr(mr)  -->  free(buf)
```

`register_buffers()` (lines 1028-1051) handles allocation and registration.
`cleanup_all()` (lines 1187-1208) tears down in order:
read -> recv -> send -> QP -> CQ -> PD -> CM.

### `skip_read_buf` option

When `connect_with_shared_pd()` is called with `skip_read_buf=true`
(lines 793-811), only send and recv buffers are allocated.  This saves
32 MB of pinned memory per pool connection.  The `has_read_buf()` method
(line 620) exposes this state to Python.

### Shared Protection Domain

`connect_with_shared_pd()` (lines 679-837) connects using an existing PD
from another transport:

1. PD must belong to the same RDMA device (verified at line 754).
2. Transport sets `owns_pd_ = false` so `cleanup_all()` skips
   `ibv_dealloc_pd()` (line 1203).
3. PD handle is an opaque `uint64_t` from `get_pd_handle()` (line 658).

This enables `RDMAClientPool` to share a single PD across all connections,
reducing kernel resource consumption.

---

## 4. Connection flow

The `connect()` method (lines 94-242) establishes an RDMA RC (Reliable
Connected) connection.  The 15-step sequence:

```
Step  Function / Action                         Timeout   Line
----  ----------------------------------------  --------  ----
 1    rdma_create_event_channel()                  --      105
 2    rdma_create_id(channel, &cm_id_,             --      114
           nullptr, RDMA_PS_TCP)
 3    rdma_getaddrinfo(ip, port, hints, &res)      --      136-139
           node and service are SEPARATE args
           (Gotcha 3 -- do NOT combine)
 4    rdma_resolve_addr(cm_id_, nullptr,          5000ms   149
           res->ai_dst_addr, 5000)
 5    wait_for_event(ADDR_RESOLVED)                --      160
 6    rdma_resolve_route(cm_id_, 5000)            5000ms   163
 7    wait_for_event(ROUTE_RESOLVED)               --      170
 8    ctx_ = cm_id_->verbs                         --      173
 9    ibv_alloc_pd(ctx_)                           --      180
10    ibv_create_cq(ctx_, CQ_DEPTH=256)            --      187
11    rdma_create_qp(cm_id_, pd_, qp_attr)         --      205
           RC, max_send/recv_wr=128, max_inline=256
12    register_buffers()                           --      214
           malloc+memset+ibv_reg_mr x 3 buffers
13    post_recv()                                  --      217
14    rdma_connect(cm_id_, conn_param)              --      226
           retry_count=7, rnr_retry_count=7
15    wait_for_event(ESTABLISHED)                   --      236
```

After step 15, `connected_` is set to `true` (line 241).

If `wait_for_event(ESTABLISHED)` throws (server rejection), `cleanup_all()`
is called before re-throwing (lines 237-240) to prevent `rdma_disconnect`
on a partial CM ID, which can segfault.

### Gotcha 3: `rdma_getaddrinfo` node/service split

`rdma_getaddrinfo` wraps POSIX `getaddrinfo`.  The IP and port must be
SEPARATE arguments.  Combining them as `"ip:port"` returns `EAI_NONAME`
(-2).  This was broken in v0.2.3 (stale `.so`) and re-fixed in v0.2.4.
A `REGRESSION RISK` comment at lines 125-131 documents this.

---

## 5. Send/Recv flow

The `roundtrip_impl()` method (lines 900-978) implements a single
Send/Recv exchange:

```
Python                    C++ (roundtrip_impl)              Server
  |                              |                             |
  | roundtrip(request_bytes)     |                             |
  |---------------------------->|                              |
  |     memcpy request -> send_buf_                            |
  |     ibv_post_send(IBV_WR_SEND)                             |
  |     [IBV_SEND_INLINE if len <= 256]  ------SEND---------> |
  |     busy-poll CQ for SEND + RECV     <------RECV--------- |
  |     recv_len = wc.byte_len                                 |
  |     result = recv_buf_[0:recv_len]                         |
  |     post_recv()                                            |
  | <--- py::bytes(result) ------|                             |
```

**Inline send:** if the request is <= `MAX_INLINE` (256 bytes),
`IBV_SEND_INLINE` is added (line 926), letting the NIC read data directly
from the WQE without a DMA read of the send buffer.

**CQ polling:** busy loop (lines 941-963) waits for both SEND and RECV
completions.  Every `POLL_CHECK_INTERVAL` (1024) iterations,
`check_poll_deadline()` verifies `connected_` and wall-clock deadline.

**Timing:** sampled every 64th call (default `sample_rate_=64`).

---

## 6. RDMA Read flow

RDMA Read is one-sided: the NIC reads directly from server memory without
involving the server CPU.  The server provides `rkey` and `remote_addr`
via the wire protocol (e.g., `OP_READ_READY`).

### Three variants

| Method                 | Target buffer     | Return     | Throws | GIL        | Lines   |
|------------------------|-------------------|------------|--------|------------|---------|
| `rdma_read`            | Internal read_buf_| py::bytes  | Yes    | Manual     | 274-281 |
| `rdma_read_into`       | User-registered   | void       | Yes    | call_guard | 288-302 |
| `try_rdma_read_into`   | User-registered   | int status | No     | call_guard | 310-361 |

- `rdma_read()`: calls `rdma_read_impl()` (lines 984-1004), copies from
  `read_buf_` into `std::string`, returns `py::bytes`.
- `rdma_read_into()`: zero-copy into caller's buffer. Used when SGL has
  a registered MR (e.g., tensor buffer).
- `try_rdma_read_into()`: returns 0 on success, nonzero `ibv_wc_status`
  on error. Used by Python retry logic for migration-aware GET.

### Common mechanics (`post_rdma_read`, lines 1082-1125)

All three variants ultimately issue an `IBV_WR_RDMA_READ` and poll the CQ:

```
  ibv_post_send(IBV_WR_RDMA_READ)
      |
      v
  busy-poll CQ
      |
      +-- n < 0  --> throw "ibv_poll_cq error"
      +-- n == 0 --> check deadline every 1024 iterations
      +-- n > 0  --> check wc.status
                         SUCCESS + RDMA_READ opcode --> done
                         otherwise --> throw with wc_status_name()
```

The `wr_id` is always set to 2 for single RDMA Reads (lines 1091, 324),
distinguishing them from SEND (`wr_id=1`) and RECV (`wr_id=0`)
completions in the CQ.

---

## 7. Batch RDMA Read

Batch RDMA Read reads multiple remote regions in a single operation.  The
key optimization is **doorbell batching**: WRs are linked via `next`
pointers and submitted with one `ibv_post_send` call.

### `post_and_poll_batch_reads()` -- core engine (lines 370-466)

Shared by both `batch_rdma_read_into()` and `batch_rdma_read()`.

WRs are processed in chunks of `MAX_SEND_WR` (128):

```
  Chunk 0 (entries 0..127):
    Build linked WR list:  wr[0].next = &wr[1] ... wr[127].next = nullptr
    ibv_post_send(&wr[0])           <-- single doorbell
    poll CQ for 128 completions (ibv_poll_cq batch of 32)
  Chunk 1 (entries 128..255):
    ... same pattern ...
```

Each completion's `wr_id` identifies the original entry index for
out-of-order tracking.  If any WC has non-success status, `qp_error` is
set and remaining entries are marked status 255 ("not attempted").

### `batch_rdma_read_into()` (lines 477-506)

Zero-copy into registered user buffers.  Takes parallel vectors of
`rkeys`, `remote_addrs`, `lengths`, `local_addrs`, `lkeys`.  Returns
`vector<uint8_t>` of per-entry WC statuses.  GIL released via call_guard.

### `batch_rdma_read()` (lines 518-615)

Reads into internal `read_buf_` and returns `vector<py::bytes>`.  Because
`read_buf_` has fixed capacity (32 MB), entries are sub-batched:

```
  Sub-batch 0: pack entries contiguously until read_buf_ full
    GIL release --> post_and_poll_batch_reads()
    GIL reacquire --> extract py::bytes from offsets
  Sub-batch 1: next group that fits
    ...
```

If a single entry exceeds `read_buf_size_`, the method throws
(lines 545-549).

---

## 8. MR thread safety

### C++ side

- `reg_mr(addr, length)` (lines 627-643): `ibv_reg_mr` with
  `IBV_ACCESS_LOCAL_WRITE`.  Returns `(lkey, mr_handle)`.
- `dereg_mr(mr_handle)` (lines 649-653): `ibv_dereg_mr` on the handle.

No MR map in C++.  The C++ side is stateless for user MRs.

### Python side: `_MREntry` and `_mr_map`

The Python `RDMAClient` maintains `_mr_map` protected by `_mr_lock`
(RLock).  Each `_MREntry` stores:

| Field       | Purpose                                          |
|-------------|--------------------------------------------------|
| `lkey`      | Local key for RDMA ops                           |
| `mr_handle` | Opaque C++ `ibv_mr*` for `dereg_mr`              |
| `buf_ref`   | Python reference to keep buffer alive (Gotcha 4) |
| `ptr`       | Buffer address (for re-registration on reconnect)|
| `size`      | Buffer size (for re-registration on reconnect)   |

### Gotcha 4: `buf_ref` prevents GC during RDMA ops

Without `buf_ref`, Python's GC can free the underlying memory while an
RDMA Read is in flight, causing:

- `IBV_WC_REM_ACCESS_ERR` (status 5) if the NIC detects the invalid address
- Silent data corruption if the memory is reallocated before the read completes

The fix (v0.3.0): `reg_memory(ptr, size, buf=None)` accepts an optional
`buf=` argument.  When provided, the buffer reference is stored in the
`_MREntry`.  As long as the MR is registered, the buffer cannot be
garbage-collected.

```
  buf = torch.empty(1024)
  lkey = reg_memory(buf.ptr, buf.nbytes, buf=buf)   # buf_ref stored
  rdma_read_into(rkey, raddr, len, buf.ptr, lkey)    # buf safe from GC
  dereg_memory(lkey)                                  # buf_ref released
```

The `_mr_lock` is an `RLock` (reentrant) to support reconnect paths
where `_reconnect` re-registers all MRs while already holding the lock.

---

## 9. Error handling

### `wc_status_name()` (lines 35-61)

Maps `ibv_wc_status` to human-readable strings.  Exported to Python:

| Code | Name                   | Common cause                                |
|------|------------------------|---------------------------------------------|
| 0    | SUCCESS                | Normal completion                           |
| 1    | LOC_LEN_ERR            | Local buffer too small                      |
| 4    | LOC_PROT_ERR           | Local MR access violation                   |
| 5    | WR_FLUSH_ERR           | QP in ERROR state (connection closed)       |
| 10   | REM_ACCESS_ERR         | Remote key invalid (stale after migration)  |
| 12   | RETRY_EXC_ERR          | Retry limit exceeded (link failure)         |
| 13   | RNR_RETRY_EXC_ERR      | Receiver not ready, retries exhausted       |
| 20   | RESP_TIMEOUT_ERR       | Response timeout                            |

### `cm_event_name()` (lines 1127-1142)

Maps `rdma_cm_event_type` enum values to strings.  Internal only (not
exported to Python).  Used in `wait_for_event()` error messages.

Covered events:

| Event              | Meaning                                |
|--------------------|----------------------------------------|
| ADDR_RESOLVED      | Address resolution succeeded           |
| ADDR_ERROR         | Address not reachable via RDMA         |
| ROUTE_RESOLVED     | Route to remote found                  |
| ROUTE_ERROR        | No RDMA route to remote                |
| CONNECT_REQUEST    | Incoming connection request            |
| CONNECT_RESPONSE   | Connection response received           |
| CONNECT_ERROR      | Connection failed                      |
| UNREACHABLE        | Remote unreachable                     |
| REJECTED           | Server rejected connection             |
| ESTABLISHED        | Connection established                 |
| DISCONNECTED       | Peer disconnected                      |

### `wait_for_event()` contextual hints (lines 1144-1174)

When the received event differs from expected, diagnostic hints are added:

- **ADDR_ERROR**: suggests using RDMA interface IP from `ibdev2netdev`.
- **REJECTED**: "The server rejected the connection."
- **UNREACHABLE**: "The server is unreachable via RDMA."

### Poll timeout (`check_poll_deadline`, lines 1055-1063)

Every 1024 poll iterations:
1. Checks `connected_` -- throws if transport was closed during poll.
2. Checks wall-clock deadline -- throws after `poll_timeout_ms` (default
   30000ms).  Sets `connected_ = false` on timeout for fail-fast.

### General error pattern

All failures surface as `std::runtime_error` with function name, `errno`,
`strerror`, WC status name + code, or CM event name as appropriate.

---

## 10. Timing and stats

### Sampling strategy

- **Roundtrip and RDMA Read**: sampled at `sample_rate_` (default 64,
  meaning 1 in every 64 calls is timed).  Controlled by counters
  `rt_counter_` and `rd_counter_` (lines 851-852).
- **Batch RDMA Read**: always timed (every call).  Batch ops are
  infrequent enough that sampling is unnecessary.

Sampling avoids the cost of `std::chrono::steady_clock::now()` on every
single-key operation while still providing representative latency data.

### `get_stats()` (lines 1309-1322)

Returns a Python dict:

```python
{
    "roundtrip_count": 640,        "avg_roundtrip_us": 12.5,
    "rdma_read_count": 320,        "avg_rdma_read_us": 8.3,
    "batch_read_count": 10,        "avg_batch_read_us": 450.0,
    "sample_rate": 64,
}
```

### Control methods

| Method                | Effect                                              | Lines       |
|-----------------------|-----------------------------------------------------|-------------|
| `reset_stats()`       | Zeros all timing fields and counters                | 1323-1332   |
| `set_sample_rate(N)`  | 0=off, 1=every op, N=every Nth (default 64)         | 1333-1336   |
| `set_poll_timeout(ms)`| 0=infinite, >0=throw after ms (default 30000)       | 1337-1340   |

---

## 11. Constants table

Defined at file scope (lines 63-72 and 1053):

| Constant               | Value             | Exported | Purpose                          |
|------------------------|-------------------|----------|----------------------------------|
| `DEFAULT_SEND_BUF_SIZE`| 16 MB (16777216)  | Yes      | Default send buffer size         |
| `DEFAULT_RECV_BUF_SIZE`| 16 MB (16777216)  | Yes      | Default recv buffer size         |
| `DEFAULT_READ_BUF_SIZE`| 32 MB (33554432)  | Yes      | Default RDMA Read buffer size    |
| `CQ_DEPTH`             | 256               | No       | Completion queue depth           |
| `MAX_SEND_WR`          | 128               | No       | Max send work requests in QP     |
| `MAX_RECV_WR`          | 128               | No       | Max recv work requests in QP     |
| `MAX_INLINE`           | 256 bytes         | No       | Inline send threshold            |
| `POLL_CHECK_INTERVAL`  | 1024              | No       | Poll iterations between checks   |

Accessing from Python:

```python
import _cama_rdma
_cama_rdma.DEFAULT_SEND_BUF_SIZE  # 16777216
_cama_rdma.DEFAULT_RECV_BUF_SIZE  # 16777216
_cama_rdma.DEFAULT_READ_BUF_SIZE  # 33554432
```

---

## 12. Python integration

### `cama_client/rdma/__init__.py`

Re-exports the public API from the subpackage.  Imports `_cama_rdma`
and re-exports `RDMAClient`, `RDMAClientPool`, batch ops, and constants.

### `cama_client/rdma/_client.py` -- `RDMAClient`

Wraps a single `RDMATransport` instance.  Key method-to-C++ mapping:

| RDMAClient method       | C++ method called                     |
|-------------------------|---------------------------------------|
| `connect()`             | `transport.connect()`                 |
| `close()`               | `transport.close()`                   |
| `get(key)`              | `roundtrip()` then `rdma_read_into()` |
| `set(key, value)`       | `roundtrip()`                         |
| `exists(key)`           | `roundtrip()`                         |
| `reg_memory(ptr,sz,buf)`| `reg_mr()` + stores `_MREntry`        |
| `dereg_memory(lkey)`    | `dereg_mr()` + removes from `_mr_map` |
| `report_stats()`        | `roundtrip()` with C++ stats payload  |
| `get_stats()`           | `transport.get_stats()`               |

### `cama_client/rdma/_pool.py` -- `RDMAClientPool`

Manages multiple transports sharing one PD:

| Pool operation          | C++ interaction                          |
|-------------------------|------------------------------------------|
| Owner connection        | `transport.connect()` (owns PD)          |
| Member connections      | `connect_with_shared_pd()`               |
| PD sharing              | `get_pd_handle()`, `get_ctx_handle()`    |
| Reconnect (member)      | New `connect_with_shared_pd()`           |
| Reconnect (owner)       | Full pool rebuild with new owner         |

Pool size is `max(pool_size, len(endpoints))`.

### `cama_client/rdma/_batch_ops.py` -- batch operations

| Function           | C++ methods used                          |
|--------------------|-------------------------------------------|
| `mget_rdma`        | `roundtrip()` + `batch_rdma_read_into()`  |
| `mget_rdma_raw`    | `roundtrip()` + `batch_rdma_read()`       |
| `mset_striped`     | `roundtrip()` across multiple transports  |

### `cama_client/rdma/_stats.py`

Formats `transport.get_stats()` output and injects Python-side counters
(`rdma_read_retries`, `rdma_read_failures`) into `report_stats()`.

### `cama_client/rdma/_constants.py`

Re-exports `DEFAULT_*_BUF_SIZE` from `_cama_rdma` for convenient import.

---

## 13. FAQ

### Q1: Why does `pip install -e .` not pick up my C++ changes?

Editable installs skip `build_ext` if the `.so` exists.  Force rebuild:

```bash
pip install --no-build-isolation --force-reinstall -e ".[rdma]"
```

Check `_cama_rdma.GIL_RELEASED` and `_cama_rdma.__version__` to verify
the correct `.so` is loaded.

---

### Q2: Why is `rdma_getaddrinfo` failing with EAI_NONAME?

The IP and port must be SEPARATE arguments.  `"ip:port"` as node with
`nullptr` service returns `EAI_NONAME` (-2).  See Gotcha 3 and the
`REGRESSION RISK` comment at lines 125-131.

---

### Q3: What causes `REM_ACCESS_ERR` (status 5) during RDMA Read?

Two causes:
1. **Stale rkey** after ZeroLatencyBalance.  Fix: use
   `try_rdma_read_into()` and retry with fresh metadata.
2. **GC'd buffer** (Gotcha 4).  Fix: pass `buf=` to `reg_memory()`.

---

### Q4: How does the poll timeout interact with close?

`close_impl()` (lines 865-894) follows a 4-step sequence:

1. `connected_` is set to `false` (line 876).
2. `rdma_disconnect()` transitions the QP to ERROR state, causing all
   pending WRs to complete with `WR_FLUSH_ERR` (line 885).
3. A 50ms sleep gives poll loops time to see `connected_ == false` or
   the flushed WCs (line 890).
4. `cleanup_all()` destroys CQ, MRs, PD, and CM resources.

The `close_mu_` mutex (line 1022) serializes concurrent close calls to
prevent double-free of RDMA resources.  Any thread in a poll loop will
either see a `WR_FLUSH_ERR` completion and throw, or hit the
`check_poll_deadline()` check, see `connected_ == false`, and throw
`"transport closed during poll"`.

---

### Q5: Why does `batch_rdma_read` sub-batch while `batch_rdma_read_into` does not?

`batch_rdma_read` uses the fixed-capacity internal `read_buf_` (32 MB),
so entries must be sub-batched by capacity.  `batch_rdma_read_into` reads
into user buffers with no shared capacity limit; only `MAX_SEND_WR` (128)
chunking applies.

---

### Q6: What is `skip_read_buf` and when should I use it?

Use `skip_read_buf=true` in `connect_with_shared_pd()` when the
connection only does zero-copy reads via `rdma_read_into()` /
`try_rdma_read_into()`.  Saves 32 MB pinned memory.  Calling
`rdma_read()` or `batch_rdma_read()` on such a connection throws.

---

### Q7: How do I add a new method to RDMATransport?

1. Add the C++ method to the `RDMATransport` class.
2. Choose GIL strategy: `call_guard` for pure C++, manual release
   for `py::bytes` returns.
3. Add the pybind11 binding in `PYBIND11_MODULE` (line 1229+).
4. Add the Python wrapper in `_client.py`.
5. Force-rebuild and verify `GIL_RELEASED` / `__version__`.

---

## 14. Related docs

- [Architecture](architecture.md) — layer diagram, transport selection, wire protocol
- [Data Flow](data-flow.md) — end-to-end TCP/RDMA GET/SET flows
- [API Reference](api-reference.md) — full method reference for clients and pools
- [Configuration](configuration.md) — environment variables, transport selection
- [Troubleshooting](troubleshooting.md) — build failures, connection issues, RDMA debugging
- [Development](development.md) — file structure, tests, build commands
- [Sharding](sharding.md) — server-side sharding, key routing
- [Server DOCUMENTATION.md](../../cama-server/DOCUMENTATION.md) — complete server reference
- [Server RDMA Install Guide](../../cama-server/docs/rdma-install-guide.md) — RDMA hardware/software setup

---

*This document describes `rdma_transport.cpp` as of v0.35.0.  Line numbers
reference the source file at 1,341 lines.  When the source changes
significantly, verify line references and update accordingly.*

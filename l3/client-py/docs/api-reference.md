# API Reference

## Client Construction

### `PriskvClient(addr, port, password="")`

Auto-selected alias that resolves to either `RDMAClient` or `CamaClient` at import time.

```python
from cama_client import PriskvClient

client = PriskvClient("127.0.0.1", 18000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `addr` | `str` | `"127.0.0.1"` | Server address. **For RDMA:** must be the IP of an RDMA-capable NIC, not a hostname (see note below). For TCP: any resolvable hostname or IP. |
| `port` | `int` | `18000` (TCP) / `18001` (RDMA) | Server port |
| `password` | `str` | `""` | Ignored (no auth in CAMA) |

> **RDMA addressing:** RDMA connections require the IP address of the server's RDMA NIC
> (InfiniBand or RoCE). Hostnames that resolve to a management or non-RDMA interface will
> fail with `ADDR_ERROR`. Find the correct IP from the server startup log
> (`[rdma] resolved listen address: <ip>:<port>`) or by running `ibdev2netdev` on the server.

You can also import a specific transport directly:

```python
from cama_client.client import CamaClient        # TCP only
from cama_client.rdma_client import RDMAClient    # RDMA only
```

---

## String Operations

### `setstr(key: str, value: str) -> int`

Store a string key-value pair. Returns `0` on success.

```python
client.setstr("model:config", '{"layers": 80}')
```

### `getstr(key: str) -> str | None`

Retrieve a string value. Returns `None` if the key does not exist.

**RDMA Read retry (v0.18.0):** Same retry semantics as `get()` — on WC error, sends failure ack, retries the GET roundtrip once, raises `RuntimeError` on second failure.

```python
val = client.getstr("model:config")  # '{"layers": 80}' or None
```

---

## SGL Operations

### `set(key: str, sgl: SGL, ttl_ms: int = 0) -> int`

Store the contents of an SGL buffer. Returns `0` on success.

- **TCP:** Copies SGL buffer to Python `bytes` via `sgl.to_bytes()`, sends over socket
- **RDMA:** Same serialization, sent via RDMA Send (no RDMA Write for SET)

```python
sgl = SGL(ptr=buffer_ptr, size=buffer_size)
client.set("page:0:0:k", sgl, ttl_ms=60000)  # 60s TTL
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | — | Key name |
| `sgl` | `SGL` | — | Scatter-gather list wrapping a host memory pointer |
| `ttl_ms` | `int` | `0` | Time-to-live in milliseconds (`0` = no expiry) |

### `get(key: str, sgl: SGL, size: int = 0) -> int`

Retrieve a value into an SGL buffer. Returns `0` on success, `-1` if the key is not found.

- **TCP:** Receives bytes over socket, copies into SGL via `sgl.from_bytes()`
- **RDMA (registered):** RDMA Read directly into the SGL buffer (zero-copy)
- **RDMA (unregistered):** RDMA Read into internal 32 MB buffer, then `memcpy` into SGL

**RDMA Read retry (v0.18.0):** If the RDMA Read fails with a WC error (e.g. `REM_ACCESS_ERR` due to stale rkey during migration), the client sends a failure ack to the server, increments `_rdma_read_retries`, and re-issues the full GET roundtrip. The server may respond with a fresh rkey (post-swap MRs) or an inline value (migration-aware forced inline). If the retry also fails, increments `_rdma_read_failures` and raises `RuntimeError`.

```python
out_sgl = SGL(ptr=out_ptr, size=out_size)
rc = client.get("page:0:0:k", out_sgl)  # 0 = found, -1 = miss
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | — | Key name |
| `sgl` | `SGL` | — | Destination SGL buffer |
| `size` | `int` | `0` | Unused (reserved for PrisKV compat) |

---

## Existence and Deletion

### `exists(key: str) -> int`

Check if a key exists. Returns `1` if found, `0` otherwise.

```python
if client.exists("page:0:0:k"):
    print("Key is cached")
```

### `delete(key: str) -> int`

Delete a key. Returns `0` on success.

```python
client.delete("page:0:0:k")
```

---

## Batch Operations

Batch operations use **native wire-protocol opcodes** (`OP_MSET`, `OP_MTEST`, `OP_MDEL`, `OP_MGET_RDMA`) for single-roundtrip efficiency. The RDMA client has a dedicated `mget_rdma` method that batches the entire GET control plane + data plane into minimal roundtrips.

### `mset(keys: list[str], sgls: list[SGL], ttl_ms: int = 0) -> int`

Batch set with automatic sub-batch chunking. When the full payload exceeds the send buffer (16 MB default), entries are automatically partitioned into sub-batches that each fit. Single entries exceeding the buffer fall back to individual `set()` with a warning logged. Returns `0` on success.

```python
client.mset(["k1", "k2", "k3"], [sgl1, sgl2, sgl3], ttl_ms=30000)
```

### `mget(keys: list[str], sgls: list[SGL], sizes: list[int] | None = None) -> list[int]`

Batch get. Returns a list of return codes (`0` = found, `-1` = miss).

- **TCP (`CamaClient`):** Uses native `OP_MGET` — single roundtrip for all keys.
- **RDMA (`RDMAClient` / `RDMAClientPool`):** Uses per-key `get()` to preserve zero-copy RDMA Read into registered SGL buffers. Prefer `mget_rdma` for batch zero-copy reads.

```python
codes = client.mget(["k1", "k2", "k3"], [sgl1, sgl2, sgl3])
# codes: [0, 0, -1]  — k1 and k2 found, k3 missed
```

### `mget_rdma(keys: list[str], sgls: list[SGL], sizes: list[int] | None = None) -> list[int]`

*RDMA only (`RDMAClient` / `RDMAClientPool`).* Batch GET with RDMA Read — single control roundtrip + batch RDMA Reads + single batch ack. Returns a list of return codes (`0` = found, `-1` = miss).

1. Sends all keys in one `OP_MGET_RDMA` (0x34) control message
2. Server returns `OP_MGET_READ_READY` (0x35) with per-key RDMA coordinates `(rkey, remote_addr, length)`
3. Client posts all RDMA Reads with a single `ibv_post_send` doorbell (linked WR list, chunked at MAX_SEND_WR=128, GIL released)
4. Client sends `OP_BATCH_READ_ACK` (0x36) with all WC statuses

Falls back to `mget()` if the server doesn't advertise the `mget_rdma` capability. Falls back to inline values if shards are migrating (non-ODP).

```python
codes = client.mget_rdma(["k1", "k2", "k3"], [sgl1, sgl2, sgl3])
# codes: [0, 0, -1]  — k1 and k2 found via RDMA Read, k3 missed
```

### `mexists(keys: list[str]) -> list[int]`

Batch existence check via native `OP_MTEST` — single roundtrip. Returns a list of `1`/`0`.

```python
found = client.mexists(["k1", "k2", "k3"])
# found: [1, 1, 0]
```

### `mdel(keys: list[str]) -> int`

Batch delete via native `OP_MDEL` — single roundtrip. Returns `0` on success.

```python
client.mdel(["k1", "k2", "k3"])
```

---

## Key Scanning

### `keys(pattern: str = ".*") -> list[str]`

Return all keys matching a regex pattern. This scans all shards on the server and may be slow for large datasets.

```python
all_kv_keys = client.keys("page:.*:k")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pattern` | `str` | `".*"` | Regular expression pattern |

---

## Cache Management

### `flush() -> int`

Flush all data from the cache (all shards). Returns `0` on success.

```python
client.flush()
```

### `report_stats(stats: dict) -> None`

Report client-side stats to the server for Prometheus exposure. The RDMA client automatically injects `rdma_read_retries` and `rdma_read_failures` counters into the payload before sending.

```python
client.report_stats({"my_ops": 1234, "my_latency_ms": 5.2})
# Server receives: {"my_ops": 1234, "my_latency_ms": 5.2,
#                   "rdma_read_retries": 0, "rdma_read_failures": 0}
```

---

## Maintenance API

On-demand vacuum, auto-tune, and status queries. Available on both `CamaClient` (TCP) and `RDMAClient`.

### `vacuum(*, force: bool = False, shard_ids: list[int] | None = None) -> dict`

Trigger an on-demand vacuum evaluation/rebalance. Returns a JSON dict with results.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `force` | `bool` | `False` | Bypass health checks; rebalance even healthy shards |
| `shard_ids` | `list[int] \| None` | `None` | Target specific shards. `None` = all shards. |

```python
result = client.vacuum(shard_ids=[0, 1])
print(result["shards_rebalanced"])  # [0]
print(result["shards_skipped"])     # {1: "healthy"}
```

### `autotune(*, force: bool = False, shard_ids: list[int] | None = None) -> dict`

Trigger on-demand auto-tune detection + slab rebuild. Useful after changing workloads without waiting for warmup.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `force` | `bool` | `False` | Force early detection even if warmup hasn't completed |
| `shard_ids` | `list[int] \| None` | `None` | Target specific shards. `None` = all shards. |

```python
result = client.autotune(force=True)
print(result["shards_rebuilt"])           # [0]
print(result["detection_snapshots"][0])   # detection state
```

### `maintenance_status() -> dict`

Query vacuum and auto-tune status without triggering any action.

```python
status = client.maintenance_status()
print(status["vacuum_config"])       # current vacuum configuration
print(status["vacuum_stats"])        # rebalance counts, pressure evals
print(status["shard_detections"])    # per-shard detection state
```

---

## Snapshot / Restore

### `snapshot(dir: str = "") -> dict`

Trigger a server-side cache snapshot. Returns a dict with snapshot stats.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dir` | `str` | `""` | Snapshot directory. Empty string uses the server's configured `snapshot_dir`. |

```python
result = client.snapshot()
print(result)  # {"keys": 1024, "dir": "/var/cama/snapshots", "duration_ms": 42, "shards": 8}
```

### `restore(dir: str = "") -> dict`

Trigger a server-side cache restore from a snapshot. The server flushes the cache before loading. Returns a dict with restore stats.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dir` | `str` | `""` | Snapshot directory. Empty string uses the server's configured `snapshot_dir`. |

```python
result = client.restore()
print(result)  # {"keys": 1024, "dir": "/var/cama/snapshots", "duration_ms": 55, "server_version": "0.19.1"}
```

---

## Cluster

### `cluster_info() -> dict`

Query cluster membership information. Returns a dict describing the cluster state, or standalone mode if clustering is not enabled.

```python
info = client.cluster_info()
print(info)  # {"mode": "standalone"} or {"mode": "cluster", "node_id": "node-1", ...}
```

---

## Lease and Pin (Eviction Protection)

These operations protect keys from eviction during multi-block transfers or other critical operations.

### `lease(key: str, duration_ms: int) -> int`

Grant a temporary lease protecting the key from eviction. Returns `0` on success. The server enforces a maximum lease duration of 30 seconds.

```python
client.lease("important-page", 10000)  # 10 second protection
```

### `pin(key: str) -> int`

Pin a key permanently (until explicitly unpinned). Returns `0` on success.

```python
client.pin("system-prompt-cache")
```

### `unpin(key: str) -> int`

Remove a pin, allowing the key to be evicted normally. Returns `0` on success.

```python
client.unpin("system-prompt-cache")
```

---

## Memory Registration (RDMA)

These methods register host memory regions with the RDMA NIC for zero-copy data transfers. In TCP mode, they are **no-ops** that return dummy values.

### `reg_memory(ptr: int, size: int) -> int`

Register a host memory region for RDMA access. Returns a handle.

- **TCP mode:** Returns `1` (dummy handle). No actual registration occurs.
- **RDMA mode:** Calls `ibv_reg_mr()` via the C++ extension. Returns a handle that maps to the `(lkey, mr_handle)` pair needed for RDMA Read.

```python
import ctypes
buf = ctypes.create_string_buffer(64 * 1024 * 1024)
# Pass buf= so the client holds a GC reference while RDMA Reads are in flight.
# Without buf=, a dropped Python reference can free the memory mid-DMA,
# causing IBV_WC_REM_ACCESS_ERR or silent corruption (Gotcha 4).
handle = client.reg_memory(ctypes.addressof(buf), len(buf), buf=buf)
```

### `dereg_memory(handle: int) -> None`

Deregister a previously registered memory region.

- **TCP mode:** No-op.
- **RDMA mode:** Calls `ibv_dereg_mr()` via the C++ extension.

```python
client.dereg_memory(handle)
```

---

## Connection Lifecycle

### `close() -> None`

Close the connection and release all resources. In RDMA mode, this also deregisters any outstanding user memory regions.

```python
client.close()
```

The destructor (`__del__`) also calls `close()` as a safety net, but explicit cleanup is preferred.

---

## Connection Pooling

Connection pools provide N-way parallelism by distributing operations across multiple connections via round-robin. Both pool classes implement the **same API** as their single-client counterparts — they are drop-in replacements.

### `create_pool(addr, port, password="", *, pool_size=8, endpoints=None, reconnect=True, **kwargs)`

Factory function that auto-selects the right pool type based on transport availability. Falls back to a single client when `pool_size <= 1`.

```python
from cama_client import create_pool

pool = create_pool("10.0.0.1", 18001)  # default pool_size=8
pool.setstr("key", "value")
pool.close()
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `addr` | `str` | — | Server address |
| `port` | `int` | — | Server port |
| `password` | `str` | `""` | Authentication password (ignored by CAMA) |
| `pool_size` | `int` | `8` | Number of connections. `<= 1` returns a single client. |
| `endpoints` | `list[tuple]` | `None` | List of `(ip, port)` tuples for multi-NIC striping. When provided with >1 endpoints, `pool_size` is auto-set to `len(endpoints)`. Each connection targets a different endpoint in round-robin order. |
| `reconnect` | `bool` or `ReconnectConfig` | `True` | Enable auto-reconnection. Pass `True` for defaults (10 retries, 0.5s base delay, 30s max), `False` to disable, or a `ReconnectConfig` instance for custom tuning. |
| `**kwargs` | — | — | Passed through to client constructors (e.g. `send_buf_size`, `recv_buf_size`) |

### `RDMAClientPool`

Pool of N RDMA connections sharing one Protection Domain (PD). Sharing the PD means `ibv_reg_mr()` on one connection produces an `lkey` valid for RDMA Read on all connections. Every connection allocates its own internal read buffer (32 MB default) so that `mget_rdma_raw` and `batch_rdma_read` stripe across all NICs. Pass `read_buf_size` to control per-connection read buffer size.

When `endpoints` are provided, connections are distributed across server NICs and `mget_rdma` stripes reads in parallel across all NICs for N× bandwidth.

```python
from cama_client.rdma_client import RDMAClientPool

# Single-NIC pool
pool = RDMAClientPool("10.0.0.1", 18001)  # default pool_size=8

# Multi-NIC striped pool (pool_size auto-set to len(endpoints))
pool = RDMAClientPool("10.0.0.10", 18001,
                      endpoints=[("10.0.0.10", 18001), ("10.0.0.11", 18001)])

handle = pool.reg_memory(ptr, size, buf=buf)  # registered on shared PD (+ independent PDs if needed)
pool.get("key", sgl)  # uses round-robin connection
pool.close()
```

**Key behavior:**
- `get()` holds one connection for the entire GET → RDMA Read → ReadAck flow
- `mget_rdma()` and `mget_rdma_raw()` stripe keys across connections in parallel when `pool_size > 1` (via `_stripe_executor` ThreadPoolExecutor)
- `reg_memory()` registers on the shared PD (lkey valid across all shared-PD connections). Connections that fell back to independent PDs get separate MR registration.
- `dereg_memory()` deregisters from both the shared PD and any independent PDs
- Admin ops (`info`, `stats`, `flush`, `vacuum`, etc.) always route to conn[0]
- `get_transport_stats()` includes stripe metrics: `stripe_calls`, `stripe_avg_nics`, `per_nic_reads`, `per_nic_bytes_gb`
- `close()` shuts down the stripe executor and cleans up all PD-specific MR maps

### `CamaClientPool`

Pool of N independent TCP clients. Each client has its own socket and lock. Simpler than RDMA — no shared resources.

```python
from cama_client.client import CamaClientPool

pool = CamaClientPool("10.0.0.1", 18000)  # default pool_size=8
pool.setstr("key", "value")
pool.close()
```

### `pool.on_reconnect(callback: Callable) -> None`

Register a callback to be invoked after a successful reconnection. Callbacks fire after the transport is replaced and MRs are re-registered. Exceptions in callbacks are logged and swallowed.

```python
def refresh_after_reconnect():
    info = pool.info()
    print(f"Reconnected to server version {info.get('version')}")

pool.on_reconnect(refresh_after_reconnect)
```

---

## Server Discovery

### `rdma_endpoints() -> list[dict]`

Query the server for available RDMA endpoints (one per NIC). Returns a list of `{"ip", "port", "device"}` dicts. Useful for multi-NIC pool construction.

```python
client = PriskvClient("10.0.0.1", 18001)
endpoints = client.rdma_endpoints()
# [{"ip": "10.0.0.10", "port": 18001, "device": "mlx5_0"},
#  {"ip": "10.0.0.11", "port": 18001, "device": "mlx5_1"}]
client.close()

# Use for multi-NIC pool
pool = create_pool("10.0.0.1", 18001,
                   endpoints=[(ep["ip"], ep["port"]) for ep in endpoints])
```

---

## SGL Class

**Source:** `cama_client/sgl.py`

The SGL (Scatter-Gather List) wraps a host memory pointer for zero-copy-style buffer transfers. In PrisKV/RDMA mode, the SGL enables DMA between GPU-host memory and the KV store. In TCP mode, data is copied via `ctypes.memmove`.

### `SGL(ptr: int, size: int, reg_handle: int = 1)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ptr` | `int` | — | Host memory address (e.g., from `ctypes.addressof()`) |
| `size` | `int` | — | Buffer size in bytes |
| `reg_handle` | `int` | `1` | RDMA memory registration handle (from `reg_memory()`) |

### `to_bytes() -> bytes`

Copy data from the host pointer into Python `bytes`. Used internally by `set()` to serialize data for transmission.

```python
data = sgl.to_bytes()  # ctypes.memmove from ptr -> bytes
```

### `from_bytes(data: bytes) -> None`

Copy received bytes into the host pointer. Used internally by TCP `get()` and RDMA fallback path.

```python
sgl.from_bytes(received_data)  # ctypes.memmove from bytes -> ptr
```

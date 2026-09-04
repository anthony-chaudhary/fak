# Configuration

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_CAMA_USE_RDMA` | `"1"` | Set to `"0"` to force TCP transport even when RDMA is available |

## Connection Parameters

| Parameter | TCP (CamaClient) | RDMA (RDMAClient) |
|---|---|---|
| Default addr | `127.0.0.1` | `127.0.0.1` |
| Default port | `18000` | `18001` |
| Socket options | `TCP_NODELAY` enabled | N/A (RDMA CM) |
| Thread safety | Lock per client | Lock per client |

## Constructor Parameters

Both `CamaClient` and `RDMAClient` accept these keyword arguments:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"127.0.0.1"` | Server address |
| `port` | `int` | `18000` / `18001` | TCP or RDMA port |
| `timeout` | `float` | `None` | Operation timeout in seconds (added v0.4.0). TCP sets `socket.settimeout()`; RDMA stores for API compatibility. |

`RDMAClient` additionally accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `send_buf_size` | `int` | `16777216` (16 MB) | RDMA Send buffer size in bytes (default reduced from 32 MB to 16 MB in v1.0.0) |
| `recv_buf_size` | `int` | `16777216` (16 MB) | RDMA Recv buffer size in bytes (default reduced from 32 MB to 16 MB in v1.0.0) |

## Pool Construction

### `create_pool(addr, port, password="", *, pool_size=8, endpoints=None, **kwargs)`

Factory that auto-selects the right pool type. Returns `RDMAClientPool`, `CamaClientPool`, or a single client.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `addr` | `str` | — | Server address |
| `port` | `int` | — | Server port |
| `pool_size` | `int` | `8` | Number of connections. `<= 1` returns a single client |
| `endpoints` | `list[tuple]` | `None` | List of `(ip, port)` tuples for multi-NIC striping. When provided with >1 endpoints, `pool_size` is auto-set to `len(endpoints)`. |
| `**kwargs` | — | — | Forwarded to client constructors (`send_buf_size`, `recv_buf_size`, etc.) |

**RDMA pool internals:** The first transport creates and owns the PD. Additional transports call `connect_with_shared_pd()` to reuse the same PD (verified same RDMA device) and skip the 32 MB read buffer via `skip_read_buf`. Memory registered via `reg_memory()` uses the shared PD, producing an `lkey` valid on all pool connections. If a connection routes through a different client RDMA device (e.g. multi-NIC endpoints on different subnets), it falls back to an independent PD with separate MR registration.

**TCP pool internals:** N independent `CamaClient` instances, each with its own socket. Handshake runs on the first client only.

## Runtime Methods

| Method | Description |
|---|---|
| `set_timeout(seconds)` | Change the operation timeout after construction. See [API Reference](api-reference.md) for details. |

## RDMA Buffer Sizing

The default Send and Recv buffer sizes are **16 MB** each (reduced from 32 MB in v1.0.0, originally 8 MB in v0.31.0, 4 MB in v0.6.0). The internal Read buffer is 32 MB.

The 16 MB default accommodates ~5 × 3 MB entries per `mset` batch. At the old 8 MB default, values >= 3 MB caused `mset` to degenerate to 1 key per roundtrip (effectively sequential). When this degeneration occurs, the client now emits a warning via the `cama_client.rdma_client` logger — watch for `mset: N/M entries exceed send buffer`.

**Important:** The client's send buffer must be ≥ the server's recv buffer (and vice versa). If the server is configured with larger buffers, the client must match:

```python
client = RDMAClient("10.0.0.1", 18001,
                     send_buf_size=64 * 1024 * 1024,  # 64 MB
                     recv_buf_size=64 * 1024 * 1024)
```

Minimum buffer size is **1 MB**. Values below this are rejected.

**Checking server config:** Call `client.info()` — the response includes `rdma_send_buf_mb` and `rdma_recv_buf_mb` so you can verify client/server alignment without checking the TOML file.

For server-side buffer configuration, see [`cama-server/DOCUMENTATION.md`](../../cama-server/DOCUMENTATION.md) (`rdma_recv_buf_size` / `rdma_send_buf_size` in TOML). For the full hardware constraint analysis (HCA limits, memory pinning, ODP, per-connection budgets), see [`rdma-buffer-limits.md`](../../cama-server/docs/rdma-buffer-limits.md).

## Reconnection Configuration

Automatic reconnection with exponential backoff is available for both TCP and RDMA transports. Reconnection is enabled by default when using pools.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reconnect` | `ReconnectConfig` or `bool` | `True` | Pass `True` for defaults, `False` to disable, or a `ReconnectConfig` for fine-tuning |

### `ReconnectConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Master switch |
| `max_retries` | `int` | `10` | Max attempts before giving up (~152s worst-case total delay) |
| `base_delay_s` | `float` | `0.5` | Base delay for exponential backoff |
| `max_delay_s` | `float` | `30.0` | Cap for backoff delay |
| `jitter` | `float` | `0.1` | ±10% random jitter on each delay |

**Backoff sequence:** 0.5s, 1s, 2s, 4s, 8s, 16s, 30s, 30s, 30s, 30s

**Retriable errors:** `BrokenPipeError`, `ConnectionResetError`, `ConnectionRefusedError`, socket timeout, RDMA WR_FLUSH_ERR, RETRY_EXC_ERR, and other transport-level failures. Server-side application errors (e.g., `CAMA error: ...`) are **not** retried.

```python
from cama_client import create_pool
from cama_client.reconnect import ReconnectConfig

pool = create_pool("10.0.0.1", 18001,
                   reconnect=ReconnectConfig(max_retries=5, max_delay_s=10.0))
```

**Source:** `cama_client/reconnect.py`

## Multi-NIC Striping

When the server has multiple RDMA NICs, you can pass explicit endpoints for NIC-striped bandwidth. The pool creates one connection per NIC and stripes `mget_rdma` across them in parallel for N× read throughput.

```python
from cama_client import create_pool, PriskvClient

# Discover endpoints from the server
client = PriskvClient("10.0.0.1", 18001)
endpoints = client.rdma_endpoints()  # [{"ip": "10.0.0.10", "port": 18001, "device": "mlx5_0"}, ...]
client.close()

# Create pool with round-robin across all NICs
pool = create_pool("10.0.0.1", 18001,
                   endpoints=[(ep["ip"], ep["port"]) for ep in endpoints])
# pool_size auto-set to len(endpoints)
```

**How striping works:**

| Aspect | Single-NIC pool | Multi-NIC striped pool |
|---|---|---|
| `mget_rdma` dispatch | Single connection | Keys partitioned round-robin across connections, parallel `_mget_rdma_on_conn` via `ThreadPoolExecutor` |
| `pool_size` | Explicit (default 8) | Auto-set to `len(endpoints)` |
| Bandwidth | 1 NIC | N NICs (linear scaling) |
| `pool_size=1` fast path | Direct call, no executor | N/A |

Each pool connection targets a different endpoint in round-robin order. MRs registered on the shared PD are valid across all connections (same RDMA device required). If a connection lands on a different RDMA device, it falls back to an independent PD with separate MR registration — `reg_memory()` and `dereg_memory()` handle both shared and independent PD connections transparently.

**Stripe metrics** (available via `get_transport_stats()`):

| Metric | Description |
|---|---|
| `stripe_calls` | Number of striped `mget_rdma` invocations |
| `stripe_avg_nics` | Average number of NICs used per striped call |
| `per_nic_reads` | Read count per connection slot |
| `per_nic_bytes_gb` | Bytes read per connection slot (GB) |

These metrics are also included in `report_stats()` for server-side Prometheus exposure.

## Transport Auto-Selection Behavior

| Condition | `PriskvClient` resolves to | `create_pool()` returns |
|---|---|---|
| `SGLANG_CAMA_USE_RDMA=0` | `CamaClient` (TCP) | `CamaClientPool` (TCP) |
| Extension not built (no `_cama_rdma.so`) | `CamaClient` (TCP) | `CamaClientPool` (TCP) |
| No RDMA devices (`is_available()` returns `False`) | `CamaClient` (TCP) | `CamaClientPool` (TCP) |
| Extension built + RDMA devices present + env=1 | `RDMAClient` (RDMA) | `RDMAClientPool` (RDMA) |
| Any of the above + `pool_size <= 1` | N/A | Single client (no pooling) |

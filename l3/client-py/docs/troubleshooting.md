# Troubleshooting

## Extension Build Failures

**`ModuleNotFoundError: No module named 'pybind11'`**

pybind11 is required to build the RDMA extension. Install it or use the `[rdma]` extra:

```bash
pip install pybind11>=2.11
# or
pip install -e ".[rdma]"
```

**`fatal error: infiniband/verbs.h: No such file or directory`**

The RDMA development headers are not installed:

```bash
# Ubuntu / Debian
sudo apt-get install -y libibverbs-dev librdmacm-dev

# RHEL / CentOS
sudo dnf install -y rdma-core-devel
```

**`cannot find -lrdmacm` or `cannot find -libverbs`**

Same fix — install the RDMA development packages. The linker needs the shared libraries.

**Extension skipped silently (no error but no `_cama_rdma`)**

The build is conditional. Check:
- Are you on Linux? (`sys.platform == "linux"`)
- Is pybind11 installed? (`python -c "import pybind11"`)
- Rebuild: `pip install -e ".[rdma]" --no-build-isolation`

## Connection Issues

**`ConnectionRefusedError: [Errno 111] Connection refused`**

The server is not running or listening on a different port. Verify:

```bash
# Check server is running
ps aux | grep cama

# Check port
ss -tlnp | grep 18000
```

**`ConnectionError: connection closed while reading`**

The server closed the connection. This can happen if:
- The server restarted or crashed
- A firewall dropped the connection
- The request body was malformed

## RDMA Issues

**`is_available()` returns `False`**

No RDMA devices are detected. Check:

```bash
# List RDMA devices
ibv_devices

# Check for RoCE/InfiniBand interfaces
rdma link show
```

If using SoftROCE for development, ensure the `rxe` device is created:

```bash
sudo rdma link add rxe0 type rxe netdev eth0
```

**`ibv_reg_mr failed for user buffer`**

Memory registration failed. Common causes:
- Insufficient locked memory limit (`ulimit -l`)
- Buffer address is invalid or already freed

```bash
# Increase locked memory limit
ulimit -l unlimited
```

**`rdma_getaddrinfo failed`**

The RDMA CM cannot resolve the server address. First, verify the C++ extension is correctly built and linked:

```bash
# 1. Check the .so exists and is linked against librdmacm
find $(python3 -c "import cama_client; import os; print(os.path.dirname(cama_client.__file__))") \
    -name "_cama_rdma*.so"
ldd /path/to/_cama_rdma*.so | grep -E "rdmacm|ibverbs"

# 2. Confirm Python loads the right .so (not a stale copy)
python3 -c "import cama_client._cama_rdma as m; print(m.__file__)"

# 3. Force a clean rebuild (editable installs don't recompile C++ automatically)
cd cama-client
pip install --no-build-isolation --force-reinstall -e .
```

If the extension is correctly built, check RDMA connectivity:

```bash
# Verify RDMA devices are visible
ibv_devices

# Test RDMA reachability to server
rdma_client_test <server-ip> <rdma-port>

# Check for firewall rules blocking the RDMA port
ss -tlnp | grep 18001
```

**`rdma_resolve_addr failed`**

The RDMA CM cannot resolve the server address. Verify:
- The server is listening on the RDMA port (default `18001`)
- The RDMA device has connectivity to the server (same subnet or routed)
- Both client and server have RDMA devices configured

## RDMA Extension Build Issues

The C++ extension (`_cama_rdma`) is compiled once at install time. **Editing `.cpp` files or bumping the package version does NOT automatically rebuild the `.so`**. If the extension is stale, you will see a warning at import time:

```
UserWarning: RDMA extension version mismatch: _cama_rdma.so was compiled for '0.2.2'
but cama-client is '0.2.4'. Rebuild with: pip install --no-build-isolation --force-reinstall -e .
```

You can also check manually:

```python
import cama_client._cama_rdma as m
print(m.__version__)  # should match cama_client.__version__
```

Common pitfalls:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: No module named _cama_rdma` | Extension not built | Install pybind11 first: `pip install pybind11>=2.11`, then reinstall |
| Version mismatch warning at import | Stale `.so` from a previous build | `pip install --no-build-isolation --force-reinstall -e .` |
| `is_available()` returns `True` but connect fails | librdmacm runtime version mismatch | Check `ldd` output matches system `librdmacm.so` |
| Silent TCP fallback (no errors, but slow) | Extension import failed silently | Check transport class (see below) |

## Transport Fallback Debugging

To verify which transport is active:

```python
from cama_client import PriskvClient
print(f"Transport: {PriskvClient.__module__}.{PriskvClient.__name__}")
# "cama_client.rdma_client.RDMAClient" or "cama_client.client.CamaClient"
```

To force TCP for debugging:

```bash
export SGLANG_CAMA_USE_RDMA=0
```

## RDMA Read Failed During Migration

**Symptom:** `RuntimeWarning: RDMA Read failed: REM_ACCESS_ERR (status=10), retrying GET` in logs during slab migration.

**Cause:** During live slab migration, the server swaps memory regions (MRs) in `finalizeMigration`. There is a brief window where the client holds a stale `rkey` from the pre-swap MR. The RDMA Read completes with `IBV_WC_REM_ACCESS_ERR` because the old MR has been deregistered.

**Self-healing behavior (v0.18.0):**
1. The client detects the WC error and sends a failure ack (`OP_READ_ACK` with nonzero status) to the server
2. The client re-issues the full GET roundtrip
3. The server's migration-aware GET path either returns an inline value (forced inline when migration is active) or a fresh rkey from the post-swap MR
4. If the retry also fails, the client raises `RuntimeError`

**Action required:** None — this is expected during migration and self-heals. If you see persistent failures (not just retries), check that migration is completing successfully via `client.maintenance_status()`.

## Common RDMA Work Completion Status Codes

When RDMA operations complete, the NIC reports a status code. Non-zero means failure:

| Code | Name | Common Causes |
|------|------|---------------|
| `0` | `IBV_WC_SUCCESS` | Operation completed successfully |
| `5` | `IBV_WC_WR_FLUSH_ERR` | QP transitioned to error state (connection torn down, peer disconnect) |
| `10` | `IBV_WC_REM_ACCESS_ERR` | Remote memory region deregistered or rkey invalid (common during slab migration — see above) |
| `12` | `IBV_WC_REM_OP_ERR` | Remote operation error (responder QP error, malformed request) |

The client logs the status name via `wc_status_name()` (available in both C++ and Python) and includes it in `report_stats()` metrics as `rdma_read_failures`.

---

## Reconnection Issues

**`RuntimeWarning: CAMA connection lost, reconnecting (attempt 1/10)`**

The client detected a transport failure and is auto-reconnecting with exponential backoff. This is expected behavior during server restarts or transient network issues.

**Reconnection keeps failing (exhausts all retries)**

Check:
- Server is actually running: `ss -tlnp | grep 18000`
- Network connectivity: `ping <server-ip>`
- If RDMA: verify RDMA device is still up (`ibv_devices`, `rdma link show`)

**Pool reconnection causes brief stalls**

When the PD-owner connection (conn[0]) fails, the pool performs a **full rebuild** — all connections are torn down and rebuilt, MRs re-registered. This takes 1-3 seconds. Non-owner connection failures are cheaper (PD and MRs survive).

To tune reconnection behavior:

```python
from cama_client.reconnect import ReconnectConfig

pool = create_pool("10.0.0.1", 18001,
                   reconnect=ReconnectConfig(
                       enabled=True,
                       max_retries=5,        # default 10
                       base_delay_s=0.5,     # default 0.5
                       max_delay_s=10.0,     # default 30.0
                   ))
```

**Post-reconnect callback not firing**

Callbacks registered via `pool.on_reconnect(fn)` fire after transport replacement and MR re-registration. Exceptions in callbacks are logged and swallowed — check logs for `"reconnect callback error"`.

---

## Multi-NIC Striping Issues

**"routes through a different local HCA than conn 0 — cannot share PD, using independent PD"**

This DEBUG-level message appears when a multi-NIC pool connection resolves to a different local RDMA device (HCA) than connection 0. A Protection Domain (PD) is scoped to a single physical HCA, so connections on different devices cannot share a PD and each gets its own.

This is **expected and normal** in multi-NIC topologies where server NICs are on different subnets (e.g., `mlx5_0` on `208.x`, `mlx5_1` on `206.x`). The client's routing table sends each destination through a different local HCA, making PD sharing impossible. MRs are registered separately per independent-PD connection — `reg_memory()` handles this transparently.

The shared PD optimization only applies when all connections route through the **same** local HCA (e.g., single-endpoint pools, or multiple server NICs reachable from one client device). See [Architecture: Protection Domains](architecture.md#protection-domains--shared-pd) for details.

**Stripe metrics show uneven distribution**

Check `get_transport_stats()` — the `per_nic_reads` array should show roughly even distribution. Uneven distribution may indicate:
- Some server NICs are on different subnets with different latency
- Key count is not evenly divisible by pool size (expected for small batches)
- A connection failed and was reconnected to a different NIC

**Only 1 NIC used despite multiple endpoints**

Verify that `pool_size > 1` and that `endpoints` was passed to `create_pool()`. When `pool_size=1`, the pool uses the single-connection fast path and skips the stripe executor entirely. Check `stripe_avg_nics` in transport stats — it should be > 1 for multi-NIC pools.

---

## Performance Notes

- **Verify RDMA is active** — TCP fallback may happen silently. Check the transport class as shown above.
- **Register buffers** — Unregistered RDMA buffers still incur a `memcpy` from the internal 32 MB read buffer. Call `reg_memory()` for true zero-copy.
- **Batch operations use native wire opcodes** — `mset`/`mexists`/`mdel` each use a single-roundtrip batch opcode (since v0.23.0). `mget_rdma` batches the entire GET control plane + RDMA Reads into minimal roundtrips (since v0.26.0). For RDMA reads without `mget_rdma`, `mget` falls back to per-key `get()` to preserve zero-copy into registered SGL buffers.
- **Multi-NIC striping** — When the pool has multiple endpoints, `mget_rdma` stripes keys across NICs in parallel for N× read bandwidth. The stripe executor uses one thread per connection. Per-NIC metrics are available via `get_transport_stats()`.
- **Sub-batch chunking** — `mset` automatically partitions into sub-batches when the payload exceeds the send buffer (16 MB default). Single entries exceeding the buffer fall back to individual `set()`. The client logs a warning when batching degenerates to 1 key per batch — watch for `mset: N/M entries exceed send buffer` in the `cama_client` logger.
- **Socket write coalescing** — TCP messages ≤ 4 KB are coalesced into a single `sendall()` (header + body). Larger messages use two `sendall()` calls to avoid creating huge intermediate byte objects.
- **TCP_NODELAY is enabled** — The TCP client disables Nagle's algorithm for low-latency request/response pairs.
- **GIL release** — All RDMA I/O operations (roundtrip, rdma_read, batch_rdma_read_into, reg_mr) release the GIL, enabling ~10x multi-threaded throughput.
- **Connection pooling memory** — The owner connection costs ~64 MB (16 MB send + 16 MB recv + 32 MB read buffer). Non-owner pool connections skip the read buffer via `skip_read_buf`, costing ~32 MB each. Default `pool_size=8` uses ~288 MB (64 + 7×32). Multi-NIC striped pools auto-set `pool_size=len(endpoints)`.

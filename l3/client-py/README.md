# l3-client

Python client library for the CAMA KV cache server — a high-performance, PrisKV-compatible storage backend for LLM inference KV cache management.

## What is CAMA?

CAMA is a shared-nothing KV cache server purpose-built for LLM inference workloads. It is designed as a drop-in replacement for PrisKV / Mooncake Store with 10x throughput targets, achieved through zero-lock per-shard architecture, size-class slab allocation, and W-TinyLFU eviction.

**l3-client** is the Python package that talks to the CAMA Go server. It provides the same API surface that SGLang's HiCache expects from PrisKV, so switching from PrisKV to CAMA requires only an import change.

**Key features:**

- **PrisKV-compatible API** — drop-in replacement (`PriskvClient`, `SGL`)
- **RDMA-first with TCP fallback** — automatically selects the best available transport
- **Connection pooling** — `RDMAClientPool` (shared PD, N-way parallelism, multi-NIC striping) and `CamaClientPool` (N independent TCP clients) with `create_pool()` factory
- **Multi-NIC striping** — `mget_rdma` stripes across server NICs in parallel for N× read bandwidth; per-NIC metrics tracking
- **SGL zero-copy support** — registered RDMA buffers get true DMA; TCP falls back to `ctypes.memmove`
- **Sub-batch chunking** — `mset` payloads exceeding send buffer are automatically partitioned into sub-batches
- **Native batch ops** — `mexists`, `mset`, `mdel` use single-roundtrip wire-protocol opcodes; `mget_rdma` batches the entire GET control plane into 1 roundtrip + 1 RDMA doorbell
- **pybind11 C++ extension** — thin RDMA wrapper (~950 LOC) using libibverbs/librdmacm, with shared-PD support for connection pools
- **SGLang HiCache compatible** — works as the storage backend for `--hicache-storage-backend cama`
- **Thread-safe** — per-connection locks; pools use round-robin dispatch for N-way parallelism

## Install

```bash
cd l3-client
pip install .            # TCP-only (any platform, zero native deps)
pip install ".[rdma]"    # with RDMA C++ extension (Linux only, requires libibverbs-dev + librdmacm-dev)
pip install ".[cxl]"     # with CXL C++ extension (Linux only, experimental)
```

> **No git required.** Version is read from `l3_client/_version.py`. Works from a tarball, zip, or release directory.
>
> **RDMA and CXL are optional.** TCP works on any platform with zero native dependencies. The client auto-detects the best available transport at import time and falls back to TCP if RDMA is unavailable.

## Quick Start

```python
from l3_client import PriskvClient

client = PriskvClient("127.0.0.1", 18000)
client.setstr("model:name", "llama-70b")
print(client.getstr("model:name"))  # "llama-70b"
client.close()
```

See [docs/quick-start.md](docs/quick-start.md) for SGL and RDMA examples.

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Layer diagram, transport selection, wire protocol |
| [Sharding](docs/sharding.md) | Server-side sharding, key routing, concurrency model, shards vs NICs vs clients |
| [Installation](docs/installation.md) | Prerequisites, TCP/RDMA install, build system |
| [Quick Start](docs/quick-start.md) | String ops, SGL buffers, RDMA memory registration |
| [Data Flow](docs/data-flow.md) | End-to-end TCP/RDMA GET/SET flows, zero-copy explained |
| [API Reference](docs/api-reference.md) | Full method reference: single clients, pools, batch ops |
| [Configuration](docs/configuration.md) | Environment variables, connection params, transport selection |
| [SGLang Integration](docs/sglang-integration.md) | HiCache adapter, launch sequence, env vars |
| [Migration from PrisKV](docs/migration.md) | Import changes, behavioral differences, examples |
| [Troubleshooting](docs/troubleshooting.md) | Build failures, connection issues, RDMA debugging |
| [Development](docs/development.md) | File structure, tests, build commands |
| [Benchmarking](docs/benchmarking.md) | bench.py usage, presets, CLI flags |
| [KV Cache vs Prefix Cache](docs/kv-cache-vs-prefix-cache.md) | Transformer KV cache vs SGLang prefix caching concepts |
| [RDMA Extension Internals](docs/rdma-extension-internals.md) | C++ pybind11 extension: GIL release, buffers, connection flow, batch reads |

## Connection Pooling

For multi-threaded workloads, use `create_pool()` to create N connections for true parallelism:

```python
from l3_client import create_pool

# Auto-selects RDMAClientPool or CamaClientPool based on transport availability
pool = create_pool("10.0.0.1", 18001)  # default pool_size=8

# Same API as PriskvClient — drop-in replacement
pool.setstr("key", "value")
pool.mset(keys, sgls)
pool.close()
```

**Multi-NIC striping** — when the server has multiple RDMA NICs, pass endpoints for parallel reads across all NICs:

```python
from l3_client import create_pool, PriskvClient

client = PriskvClient("10.0.0.1", 18001)
endpoints = client.rdma_endpoints()  # [{"ip": ..., "port": ..., "device": ...}, ...]
client.close()

pool = create_pool("10.0.0.1", 18001,
                   endpoints=[(ep["ip"], ep["port"]) for ep in endpoints])
# pool_size auto-set to len(endpoints), mget_rdma stripes across NICs
```

**RDMA pools** share one Protection Domain (PD) across all connections. Memory registered on one connection (`reg_memory()`) is valid for RDMA Read on all connections. The owner connection gets a 32 MB read buffer; non-owner pool connections skip it via `skip_read_buf` (saving ~224 MB at pool_size=8). Each connection costs ~32 MB (16 MB send + 16 MB recv). If a connection targets a NIC on a different client RDMA device, it falls back to an independent PD with separate MR registration.

**TCP pools** create N independent `CamaClient` instances, each with its own socket. No shared state.

Both pool types use round-robin dispatch and are API-compatible with `PriskvClient`.

## RDMA Buffer Sizes

The `RDMAClient` accepts optional `send_buf_size` and `recv_buf_size` keyword arguments (in bytes) to override the default 16MB RDMA Send/Recv buffers:

```python
from l3_client.rdma_client import RDMAClient

# Use 64MB buffers for very large SET values
client = RDMAClient("10.0.0.1", port=18001,
                    send_buf_size=64 * 1024 * 1024,
                    recv_buf_size=64 * 1024 * 1024)
```

The client's send buffer must be >= the server's `rdma_recv_buf_size`, and the client's recv buffer must be >= the server's `rdma_send_buf_size`. GET operations for large values are unaffected — they use the RDMA Read path (64MB buffer).

**Degenerate batching warning:** When `mset` values are large enough that each batch contains only 1 key, the client emits a warning via `logging.getLogger("l3_client.rdma_client")`. Watch for `mset: N/M entries exceed send buffer` — this means you should increase `send_buf_size`. You can check the server's buffer config via `client.info()` (look for `rdma_send_buf_mb` / `rdma_recv_buf_mb`).

## Related

- **[CAMA Server Documentation](../cama-server/DOCUMENTATION.md)** — Complete server docs
- **[Performance Summary](../docs/performance-summary.md)** — Consolidated performance knowledge base (transport comparison, bottlenecks, buffer sizing, benchmarking guidance)
- **SGLang HiCache** — The L3 KV cache manager that CAMA plugs into

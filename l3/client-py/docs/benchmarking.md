# Benchmarking

`bench.py` measures throughput and bandwidth for `CamaClient` (TCP) and `RDMAClient` (RDMA) against a running CAMA server. By default it uses `--value-size 5242880` (5 MB) and `--keys 100`. Use `--preset` to select a named workload profile.

> For a consolidated roll-up of all performance research (transport comparison, bottleneck analysis, tuning guidance), see the [Performance Summary](../../docs/performance-summary.md).

## Quick Start

```bash
cd cama-client

# TCP baseline (5 MB values, 100 keys by default)
python bench.py --transport tcp --clients 32 --duration 30

# RDMA
python bench.py --transport rdma --addr 10.0.0.1 --clients 16 --duration 60

# 100k-token context, read-heavy
python bench.py --preset llama70b-100k --read-ratio 0.95 --transport rdma --clients 32
```

## Presets

| Preset | Value size | Keys | Description |
|--------|-----------|------|-------------|
| **16-token pages (default)** | | | |
| `llama70b-10k` | 5,242,880 B (~5 MB) | 625 | Llama 70B, 10k-token context |
| `llama70b-100k` | 5,242,880 B (~5 MB) | 6,250 | Llama 70B, 100k-token context |
| `llama8b-10k` | 524,288 B (~512 KB) | 625 | Llama 8B, 10k-token context |
| `llama8b-100k` | 524,288 B (~512 KB) | 6,250 | Llama 8B, 100k-token context |
| **64-token pages** | | | |
| `llama70b-10k-p64` | 20,971,520 B (~20 MB) | 157 | Llama 70B, 10k ctx, 64-token pages |
| `llama70b-100k-p64` | 20,971,520 B (~20 MB) | 1,563 | Llama 70B, 100k ctx, 64-token pages |
| `llama8b-10k-p64` | 2,097,152 B (~2 MB) | 157 | Llama 8B, 10k ctx, 64-token pages |
| `llama8b-100k-p64` | 2,097,152 B (~2 MB) | 1,563 | Llama 8B, 100k ctx, 64-token pages |
| **256-token pages** | | | |
| `llama70b-10k-p256` | 83,886,080 B (~80 MB) | 40 | Llama 70B, 10k ctx, 256-token pages |
| `llama70b-100k-p256` | 83,886,080 B (~80 MB) | 391 | Llama 70B, 100k ctx, 256-token pages |
| `llama8b-10k-p256` | 8,388,608 B (~8 MB) | 40 | Llama 8B, 10k ctx, 256-token pages |
| `llama8b-100k-p256` | 8,388,608 B (~8 MB) | 391 | Llama 8B, 100k ctx, 256-token pages |
| **Generic** | | | |
| `small-1k` | 1,024 B (1 KB) | 10,000 | Generic small values |
| `medium-64k` | 65,536 B (64 KB) | 1,000 | Generic medium values |
| `large-1m` | 1,048,576 B (1 MB) | 500 | Generic large values |

Page size formula: `num_keys = context_tokens / page_tokens` where `page_tokens` is the
KV cache page size configured in the inference framework (default 16; SGLang also supports
64 and 256). Larger page tokens produce fewer, larger values — see the capacity planning
section in `cama-server/docs/sizing-guide.md`.

The default presets assume 16-token pages. Use the `-p` suffix presets for other page sizes
(e.g., `llama70b-100k-p64`, `llama70b-100k-p256`).

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | `None` | Model/context preset (overrides `--value-size` and `--keys`) |
| `--transport` | `rdma` | `tcp` or `rdma` |
| `--addr` | `127.0.0.1` | Server address |
| `--port` | `18000` / `18001` | Server port (default depends on transport) |
| `--clients` | `32` | Worker threads |
| `--duration` | `30` | Seconds to run |
| `--value-size` | `5242880` | Bytes per value (overridden by `--preset`) |
| `--keys` | `100` | Unique key count (overridden by `--preset`) |
| `--read-ratio` | `0.8` | Fraction of reads |
| `--distribution` | `zipfian` | `zipfian` or `uniform` |
| `--sweep` | `false` | Run multiple workloads and compare results |
| `--sweep-workloads` | `mixed,exists,delete-heavy,load-probe` | Comma-separated workloads for sweep |

## Sweep Mode

`--sweep` runs multiple workloads in sequence and prints a comparison table. Default duration is 10s per workload (unless `--duration` is explicitly set). The server is flushed between each workload.

```bash
# Sweep all default workloads (mixed, exists, delete-heavy, load-probe)
python bench.py --sweep --transport tcp --clients 16

# Sweep with a specific preset
python bench.py --sweep --preset small-1k --transport tcp

# Sweep specific workloads only
python bench.py --sweep --sweep-workloads mixed,exists --transport rdma
```

Sample output:

```
SWEEP COMPARISON
Workload         Ops/s    BW GB/s  Hit%     p50 ms    p99 ms  Errors
---------------------------------------------------------------------
mixed           20,575       6.27   96.2      0.24      1.12        0
exists          45,000       0.00   88.5      0.11      0.45        0
delete-heavy    38,000       1.29      -      0.15      0.68        0
load-probe      18,000       4.98  100.0      0.28      0.95        0
```

## Sample Output

```
=== CAMA Python Benchmark Results ===
Transport:       RDMA
Duration:        60.001s
Total ops:       1,234,567
Throughput:      20,575 ops/s
GETs:            987,654
SETs:            246,913
Hits:            950,000  (96.2%)
Errors:          0
Read bandwidth:  4.98 GB/s
Write bandwidth: 1.29 GB/s
Total bandwidth: 6.27 GB/s
Clients:         32
Value size:      5,242,880 bytes (5.00 MB)
Keys:            625
Distribution:    zipfian
=====================================
```

# Cama as L3 KV Cache

This document describes how to deploy and use Cama as the L3 KV cache storage backend for SGLang HiCache. Cama talks directly to PrisKV via RDMA for zero-copy KV cache transfers, bypassing any intermediate Python-level data copies.

Related documentation:
* [HiCache System Design and Optimization](https://docs.sglang.io/advanced_features/hicache_design.html)
* [Performance Summary](../docs/performance-summary.md) — consolidated performance knowledge base (transport comparison, bottlenecks, tuning)

## About Cama

Cama is a direct PrisKV connector for SGLang HiCache. It uses PrisKV's native RDMA capabilities to transfer KV cache pages between SGLang's host memory pool and the PrisKV storage cluster with zero copies in Python.

### How It Works

When SGLang prefetches or backs up KV cache pages, Cama:

1. Registers the entire host KV buffer with PrisKV's RDMA subsystem at startup.
2. On read/write, constructs scatter-gather lists (SGLs) pointing directly into the registered buffer.
3. Issues `mget`/`mset` calls that trigger RDMA transfers at wire speed — no intermediate `memcpy` or Python tensor copies.

This is in contrast to the older `aibrix` backend, which goes through `BaseKVCacheManager` and copies data through Python tensors multiple times per page.

### Key Features

- **Zero-Copy RDMA**: Data transfers directly between PrisKV server and registered host memory. Batch RDMA GET (`mget_rdma`) uses a single control roundtrip + single doorbell for all RDMA Reads.
- **Connection Pooling**: N RDMA/TCP connections per rank (default 8) for true N-way parallelism. RDMA pools share one Protection Domain with `skip_read_buf` to save ~32 MB per extra connection.
- **Sub-Batch Chunking**: `mset` payloads exceeding the send buffer (16 MB default) are automatically partitioned into sub-batches. The client logs a warning when batching degenerates to 1 key per batch.
- **Backup Queue Coalescing**: Merges single-page backup operations into large batches before sending — `avg_batch` jumps from ~2.0 (MHA) to 256+ pages, eliminating hundreds of roundtrips per second. Configurable via `coalesce_backup_ops` (default on) and `coalesce_deadline_ms` (default 5ms).
- **Backup Write Jitter**: Random per-sub-batch delay (`backup_jitter_ms`) to spread TP rank submissions and reduce thundering herd contention on the server.
- **Adaptive Batch Sizing**: Optional auto-tuning of `storage_batch_size` based on observed backup latency (halve when slow, double when fast).
- **Parallel Backup I/O**: Configurable `backup_io_workers` (default 2) threads for concurrent backup writes, matching the prefetch parallelism model.
- **Adaptive Write Deduplication**: Auto mode starts with dedup enabled, tracks hit rates, and disables after consecutive low-hit batches to avoid wasted `mexists` roundtrips.
- **Automatic Reconnection**: Exponential backoff with MR re-registration on transport failure. Pool-aware: non-owner connections recover in sub-seconds, PD-owner failures trigger full rebuild.
- **Connection Pre-Warming**: Background connection establishment during model load via `CamaPrewarmProvider`, adopted at init if config fingerprint matches.
- **Multi-NIC RDMA Striping**: Automatic per-rank NIC assignment via `rdma_endpoints()` discovery. With `nic_striping=True` (default), pools round-robin across all server NICs for bandwidth saturation.
- **MHA + MLA Support**: Correct key naming for both Multi-Head Attention and Multi-Head Latent Attention models (e.g., DeepSeek-V3).
- **Pipeline Parallelism**: Key suffixes include PP rank to prevent data collisions across pipeline stages.
- **Native Batch Wire Ops**: Uses `mset`/`mget_rdma`/`mexists`/`mdel` single-roundtrip opcodes for minimal wire overhead. GIL released during all RDMA I/O.
- **Phase-Level Latency**: Tracks preprocess, transfer, and postprocess phases independently for bottleneck identification.
- **Bandwidth Metrics**: Tracks prefetch and backup throughput (GB/s). 29+ Prometheus metrics via `report_stats()`.
- **Triple-Source Configuration**: Supports extra_config (Kubernetes), JSON config files, and environment variables.
- **Health Check & Warmup**: Optional server readiness polling and 6-phase round-trip validation at startup.
- **Extra Backend Tag**: Key prefix support for multiple SGLang instances sharing one PrisKV cluster.

## Prerequisites

### Install SGLang from Source

> **Note:** SGLang's `[all]` extras only include `diffusion` and `tracing`.
> The critical native dependencies (`sgl-kernel`, `flashinfer`) are listed as
> regular pip dependencies and should install from PyPI automatically. If they
> fail to install or have version mismatches, the server will **hang silently**
> at CUDA graph capture rather than raising a clear error.
>
> The `sglang-router` package is only needed for multi-node routing
> (`sglang_router.launch_router`), not for single-server deployments.

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip

# Install the SGLang package + all Python/native dependencies
pip install "python/sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

### Verify the Install

All three must import successfully before starting the server:

```bash
python -c "
import sglang; print('sglang:', sglang.__file__)
import sgl_kernel; print('sgl-kernel:', sgl_kernel.__version__)
import flashinfer; print('flashinfer:', flashinfer.__version__)
"
```

If `sgl-kernel` or `flashinfer` fail to import, reinstall them explicitly:

```bash
pip install sgl-kernel==0.3.21
pip install flashinfer_python==0.6.3 flashinfer_cubin==0.6.3 --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

> **Pre-built SGLang tree note:** The `sglang-with-cama-connector/` tree pins `flashinfer==0.5.3` and `sgl-kernel==0.3.20` (matching SGLang v0.5.7). The versions above (`0.6.3` / `0.3.21`) are for the latest SGLang main branch. Always check the `pyproject.toml` of your SGLang checkout for the correct pinned versions.

**If the server still hangs**, see the [Troubleshooting](#troubleshooting) section for diagnostic steps including py-spy stack dumps, debug environment variables, and the "Slow vs Stuck" guide.

### Install CUDA Toolkit (Required for Quantized Models)

If you are running a **quantized model** (AWQ/GPTQ Marlin, e.g., MiniMax-M2-AWQ), SGLang JIT-compiles CUDA kernels at runtime via `tvm_ffi`. This requires `nvcc` from the CUDA toolkit. Without it, the server hangs silently.

> **Note:** This is separate from `sgl-kernel`. The `sgl-kernel` pip package provides pre-compiled
> kernels for common ops, but the `sglang/jit_kernel/gptq_marlin.py` path always JIT-compiles at
> runtime. Having `sgl-kernel` installed does NOT skip the `nvcc` requirement for quantized models.
>
> Virtual environments and conda do not hide system `nvcc` — they prepend to PATH, not remove from it.
> If `which nvcc` fails, it was never in PATH to begin with.

```bash
# Check your driver's CUDA version
nvidia-smi | head -5  # look for "CUDA Version: 12.x" in top right

# Install matching toolkit
apt update && apt install -y cuda-toolkit-12-4  # match your driver version

# Add to PATH (add these to ~/.bashrc to make permanent)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Verify
nvcc --version
```

**First-time startup will be slow** (~6 minutes for CUDA graph capture) as the JIT kernel compiles. Subsequent launches reuse the cached kernel and are much faster. See [Slow vs Stuck](#slow-vs-stuck-how-to-tell-the-difference) in Troubleshooting.

For non-quantized models, `sgl-kernel` from PyPI is sufficient and no CUDA toolkit is needed.

### Install KV Cache Client

cama-standalone works with either backend:

**Option A — CAMA server (recommended):** uses `cama-client` (default port 18000)
```bash
pip install cama-client
python -c "from cama_client import PriskvClient, SGL; print('cama-client installed successfully')"
```

**Option B — PrisKV server (original):** uses `priskv` (default port 6379)
```bash
pip install priskv
python -c "from priskv.priskv_client import PriskvClient; print('priskv installed successfully')"
```

### KV Cache Server

A running KV cache server is required.

**With CAMA server** (port 18000 default):
```bash
cd cama-complete/cama-server && go build -o cama-server . && ./cama-server server --listen 0.0.0.0:18000
```

**With PrisKV server** (port 6379 default):
```bash
priskv-server --port 6379
```

For RDMA-enabled deployments, ensure:
- RDMA-capable NICs are available on both the server and the SGLang host.
- The `ibverbs` drivers are installed and devices are visible via `ibv_devices`.
- Sufficient RDMA memory registration limits are configured (check `/sys/class/infiniband/*/max_mr`).

## Configuration

Cama loads configuration from three sources, in the following priority order:

1. **`--hicache-storage-backend-extra-config`** (SGLang argument) — highest priority
2. **`SGLANG_CAMA_CONFIG_PATH`** (JSON config file) — medium priority
3. **`SGLANG_CAMA_*` environment variables** — lowest priority (fallback)

### Configuration Parameters

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `remote_addr` | `SGLANG_CAMA_REMOTE_ADDR` | `127.0.0.1` | PrisKV server address |
| `remote_port` | `SGLANG_CAMA_REMOTE_PORT` | `18001` | PrisKV server port |
| `password` | `SGLANG_CAMA_PASSWORD` | `""` (empty) | PrisKV authentication password |
| `use_mput_mget` | `SGLANG_CAMA_USE_MPUT_MGET` | `true` | Use batch `mset`/`mget` instead of per-key `set`/`get`. Batch ops are faster. |
| `check_server` | `SGLANG_CAMA_CHECK_SERVER` | `false` | Poll PrisKV server at startup (up to 10 min) until reachable. |
| `pool_size` | `SGLANG_CAMA_POOL_SIZE` | `8` | Number of connections per rank. Creates N-way parallel connection pool. |
| `send_buf_size` | `SGLANG_CAMA_SEND_BUF_SIZE` | `0` | Send buffer size in bytes. `0` = client default (16 MB). |
| `dedup_mode` | `SGLANG_CAMA_DEDUP_MODE` | `"auto"` | Write dedup: `"auto"`, `"always"`, or `"never"`. |
| `io_workers` | `SGLANG_CAMA_IO_WORKERS` | `16` | I/O thread pool size per rank. |
| `op_timeout_s` | `SGLANG_CAMA_OP_TIMEOUT_S` | `10.0` | Per-batch I/O timeout in seconds. |
| `nic_striping` | `SGLANG_CAMA_NIC_STRIPING` | `true` | When `true` and multiple RDMA endpoints discovered, pool connects to ALL server NICs and stripes `mget_rdma` across them in parallel. |

See [docs/03_CONFIGURATION_REFERENCE.md](docs/03_CONFIGURATION_REFERENCE.md) for the full parameter reference.

### Extra Backend Tag

When multiple SGLang instances share a single PrisKV cluster, use `extra_backend_tag` to namespace keys and prevent collisions:

```bash
--hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "extra_backend_tag": "instance-A"}'
```

This prefixes all keys with `instance-A_`, so each instance operates on its own keyspace.

## Deployment

### Single Server Deployment

The simplest setup: PrisKV server and SGLang on the same machine.

**1. Start PrisKV Server:**

```bash
priskv-server --port 6379
```

**2. Start SGLang with Cama Backend:**

Using environment variables:

```bash
SGLANG_CAMA_REMOTE_ADDR=127.0.0.1 \
SGLANG_CAMA_REMOTE_PORT=6379 \
python -m sglang.launch_server \
    --model-path [model_path] \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

Using extra-config:

```bash
python -m sglang.launch_server \
    --model-path [model_path] \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{"remote_addr": "127.0.0.1", "remote_port": 6379}'
```

Using a JSON config file:

```bash
export SGLANG_CAMA_CONFIG_PATH=/path/to/cama_config.json

echo '{
    "remote_addr": "127.0.0.1",
    "remote_port": 6379,
    "password": "",
    "use_mput_mget": true,
    "check_server": false
}' > ${SGLANG_CAMA_CONFIG_PATH}

python -m sglang.launch_server \
    --model-path [model_path] \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama
```

### Multi-Server Deployment

PrisKV server runs on a dedicated storage node. SGLang instances on compute nodes connect over the network.

**Storage Node (10.0.0.1):**

```bash
priskv-server --port 6379
```

**Compute Node(s):**

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
SGLANG_CAMA_REMOTE_PORT=6379 \
python -m sglang.launch_server \
    --model-path [model_path] \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

For multiple SGLang instances sharing the PrisKV cluster, add `extra_backend_tag` to each instance to avoid key collisions:

```bash
# Instance A
python -m sglang.launch_server \
    --model-path [model_path] \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "extra_backend_tag": "node-1"}' \
    --port 30000

# Instance B
python -m sglang.launch_server \
    --model-path [model_path] \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "extra_backend_tag": "node-2"}' \
    --port 30001
```

### Tensor Parallel Deployment

When running with `--tp N`, each TP rank creates its own PrisKV connection and uses rank-specific key suffixes to avoid collisions. No special configuration is needed — Cama handles this automatically:

- MHA models: keys are suffixed with `_{tp_rank}_k` and `_{tp_rank}_v`
- MLA models: keys are suffixed with `_k` (single fused tensor per page)

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
python -m sglang.launch_server \
    --model-path [model_path] \
    --tp 4 \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama
```

### Pipeline Parallel Deployment

When running with `--pp N`, Cama includes the PP rank in key suffixes to prevent data corruption across pipeline stages:

- MHA: `{hash}_{tp_rank}_{pp_rank}_k` / `{hash}_{tp_rank}_{pp_rank}_v`
- MLA: `{hash}_{pp_rank}_k`

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
python -m sglang.launch_server \
    --model-path [model_path] \
    --tp 4 \
    --pp 2 \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama
```

### MLA Model Deployment (e.g., DeepSeek-V3)

MLA models use a fused KV representation (one tensor per page instead of separate K and V). Cama detects this automatically from the model config and adjusts key naming and buffer handling accordingly. No special flags needed:

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama
```

### Orchestrated Deployment (Kubernetes)

In Kubernetes, PrisKV may start after SGLang. Enable `check_server` to have Cama poll until the server is ready (up to 10 minutes):

```bash
python -m sglang.launch_server \
    --model-path [model_path] \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{"remote_addr": "priskv-service.default.svc.cluster.local", "remote_port": 6379, "check_server": true}'
```

## HiCache Parameters

For a comprehensive overview of HiCache-related parameters, refer to [this document](https://docs.sglang.io/advanced_features/hicache_design.html#related-parameters).

Key parameters relevant to Cama:

| Parameter | Description |
|-----------|-------------|
| `--enable-hierarchical-cache` | Enable the HiCache system (required) |
| `--hicache-storage-backend cama` | Select the Cama PrisKV backend |
| `--hicache-write-policy {write_through,write_back}` | When to write KV pages to storage. `write_through` writes immediately; `write_back` defers writes. |
| `--hicache-storage-backend-extra-config '{...}'` | JSON string with Cama configuration (highest priority) |
| `--hicache-ratio FLOAT` | Ratio of GPU KV cache size to use for L2 host cache. Default is 2.0. |
| `--hicache-size INT` | Explicit L2 host cache size in bytes (overrides `--hicache-ratio`) |
| `--hicache-mem-layout {page_first,page_first_direct,page_head}` | Host memory layout. Cama requires `page_first`, `page_first_direct`, or `page_head`. |

## Verifying the Deployment

### Check Logs

On successful startup, you should see log messages like:

```
Cama configuration loaded from env.
Connected to PrisKV at 127.0.0.1:6379
Cama PrisKV store warmup successful.
Cama PrisKV store setup complete.
Multi-NIC: rank 0 -> mlx5_0 10.0.0.10:6380 (2 endpoints)
Registered RDMA buffer: ptr=0x..., size=... bytes, handle=...
```

### Verify Cache Hits

Send two identical requests. The second should show a cache hit with faster time-to-first-token (TTFT):

```bash
# First request — cache miss, writes to PrisKV
curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default", "prompt": "The meaning of life is", "max_tokens": 32}'

# Second request — cache hit, reads from PrisKV (faster TTFT)
curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default", "prompt": "The meaning of life is", "max_tokens": 32}'
```

### Monitor Metrics

Cama reports bandwidth metrics through SGLang's metrics system. Check for non-empty `prefetch_bandwidth` and `backup_bandwidth` values in the storage metrics.

## Troubleshooting

### SGLang Startup Hangs

**Server hangs at CUDA graph capture (silent, no error):**

The most common cause is missing or mismatched native dependencies (`sgl-kernel`, `flashinfer`). PyPI installs include pre-built wheels; source installs pull these as pip deps but version mismatches or missing `--find-links` cause silent import failures.

Verify all critical imports work before starting the server:

```bash
python -c "
import sgl_kernel; print('sgl-kernel:', sgl_kernel.__version__)
import flashinfer; print('flashinfer:', flashinfer.__version__)
"
```

If either fails, reinstall explicitly:

```bash
pip install sgl-kernel
pip install flashinfer_python flashinfer_cubin --find-links https://flashinfer.ai/whl/cu124/torch2.5/
```

**Server hangs at `tvm_ffi/utils/lockfile.py:blocking_acquire` (JIT kernel compilation):**

SGLang has two independent kernel systems:

1. **`sgl-kernel`** (pip package) — pre-compiled CUDA kernels for common ops. No compiler needed.
2. **`sglang/jit_kernel/`** (in SGLang source) — runtime JIT compilation via `tvm_ffi` for specific kernels like `gptq_marlin.py` (AWQ/GPTQ Marlin quantized models). Always JIT compiled at runtime. Requires `nvcc`.

Having `sgl-kernel` installed does **NOT** cover the `jit_kernel/` path. For quantized models (AWQ/GPTQ Marlin), there is no pre-built alternative — you must have `nvcc`.

Stack trace signature: `blocking_acquire` → `build_inline` → `load_jit` → `gptq_marlin`

If `nvcc` is available, the kernel compiles once and is cached. **First run is very slow** (~6 minutes for CUDA graph capture on 4x GPU server with MiniMax-M2 AWQ TP4). This is NOT a hang — see "Slow vs Stuck" below. Subsequent launches reuse cached kernels and are much faster.

Fix for **quantized models** (AWQ/GPTQ Marlin) — install the CUDA toolkit:

```bash
# Check your driver's CUDA version first
nvidia-smi | head -5  # look for "CUDA Version: 12.x"

apt update && apt install -y cuda-toolkit-12-4  # match your driver version
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
nvcc --version  # verify
```

Add the `export` lines to `~/.bashrc` to make them permanent.

Fix for **non-quantized models** — `sgl-kernel` covers these:

```bash
pip install sgl-kernel
```

**Quick JIT test without full SGLang startup:**

```bash
python3 -c "
import torch
from sglang.jit_kernel.gptq_marlin import _jit_gptq_marlin_module
print('Compiling fp16 kernel...')
m = _jit_gptq_marlin_module(torch.float16)
print('Success:', m)
"
```

If this completes (may take a minute on first run), the kernel is compiled and cached. SGLang will start without the JIT delay.

**Tokenizer deadlock (hang during model loading):**

HuggingFace tokenizers + Python multiprocessing forking = silent deadlock.

```bash
export TOKENIZERS_PARALLELISM=false
```

**Torch compile hanging:**

```bash
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
```

**NCCL issues (multi-GPU hangs):**

```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
python -c "import torch; print(torch.cuda.nccl.version())"
```

**CUDA graph specific workarounds:**

> **Note:** `--disable-cuda-graph` does NOT help for quantized models (AWQ/GPTQ Marlin).
> The JIT kernel is needed for inference itself, not just graph capture. Disabling graphs
> just moves the hang to the first request instead of startup.

```bash
--disable-cuda-graph            # skips graph capture but NOT JIT compilation
--cuda-graph-max-bs 1           # capture fewer batch sizes (faster startup)
--attention-backend triton
--mem-fraction-static 0.8
```

**Stale lock file from crashed process:**

If a previous SGLang process was killed during JIT compilation, it may leave a lock file behind. New processes will hang waiting on this lock forever. Find and remove it:

```bash
# Read the lockfile source to find the exact lock path
python3 -c "
import inspect, tvm_ffi.utils.lockfile
print(inspect.getsource(tvm_ffi.utils.lockfile))
"

# Or search broadly
find / -name "*.lock" -path "*tvm*" 2>/dev/null
find /tmp -name "*.lock" 2>/dev/null
find ~/.cache -name "*.lock" 2>/dev/null

# Delete the stale lock, then restart
rm <path_to_lock_file>
```

**Debug environment variables:**

```bash
export CUDA_LAUNCH_BLOCKING=1    # synchronous CUDA — makes errors easier to locate
export SGLANG_LOG_LEVEL=DEBUG    # verbose SGLang logging
export TOKENIZERS_PARALLELISM=false  # prevent tokenizer fork deadlock
```

**Version mismatch debugging:**

```bash
pip list | grep -iE "sglang|vllm|torch|cuda|triton|flashinfer|flash.attn"
python -c "import torch; print(torch.version.cuda, torch.__version__)"
```

**Diagnosing hangs with py-spy:**

```bash
pip install py-spy
py-spy dump --pid $(pgrep -f sglang)
```

Shows the exact Python stack where each thread is blocked. Each thread lists a stack trace with the most recent call at the top.

**How to read the output** — look at the top frame of `MainThread` and match to these patterns:

| Top of stack contains | Diagnosis | Fix |
|---|---|---|
| `tvm_ffi/utils/lockfile.py:blocking_acquire` → `build_inline` → `load_jit` → `gptq_marlin` | JIT kernel hang — missing `nvcc` | Install `cuda-toolkit` (required for quantized models) |
| `lockfile.py:blocking_acquire` (alone, no `build_inline`) | Stale lock from crashed previous run | Find and delete stale lock (see below) |
| `tokenizers` or `rust_tokenizers` | HuggingFace tokenizer deadlock | `export TOKENIZERS_PARALLELISM=false` |
| `nccl` or `ncclCommInitRank` or `ncclAllReduce` | NCCL multi-GPU communication hang | `NCCL_DEBUG=INFO`, try `NCCL_P2P_DISABLE=1` |
| `torch._inductor` or `compile_worker` | Torch compile/inductor hanging | `export TORCH_COMPILE_DISABLE=1` |
| `cuda_graph_runner.py:capture` with CUDA sync below | CUDA graph capture stall | Check which kernel is being captured (look at frames below `capture`) |
| `threading.py:wait` in `tqdm/_monitor.py` | Idle tqdm monitor threads | **Ignore** — harmless background threads |
| `scheduler.py` → `recv` or `zmq` | Scheduler waiting for messages | **Normal** if server is idle |

**Tips:**
- Focus on `MainThread` — that's where the actual work happens.
- Ignore threads labeled `tqdm/_monitor.py` or `InductorSubproc` — background housekeeping.
- The **bottom** of the stack shows what high-level operation triggered the hang (e.g., `cuda_graph_runner.py:capture`). The **top** shows where it's actually stuck.
- Multiple threads stuck on the same lock = contention. One thread stuck on a lock alone = stale lock or missing dependency.

### Slow vs Stuck: How to Tell the Difference

First-time JIT compilation + CUDA graph capture can look identical to a hang. Here's how to tell them apart.

**Signs it's WORKING (just slow):**
- Logs show `"Capture cuda graph begin. This can take up to several minutes."` — this is normal
- A progress bar appears: `Capturing batches (bs=1 avail_mem=X GB): X%|...` — if the percentage advances, it's working
- py-spy shows `cuda_graph_runner.py:capture` with active CUDA calls below (not stuck on a lock)
- First run with JIT: expect ~6 min (351s observed on 4x GPU server with MiniMax-M2 AWQ, 24 batch sizes at ~14.6s each)

**Signs it's ACTUALLY STUCK:**
- No progress bar appears, or it stays at 0% indefinitely
- py-spy shows `blocking_acquire` in `tvm_ffi/utils/lockfile.py` — lock contention or stale lock
- py-spy shows `tokenizers` or `nccl` frames at the top — deadlock
- No log output at all for 10+ minutes after `"Capture cuda graph begin"`

**Normal startup timeline (first run with JIT, MiniMax-M2 AWQ TP4):**

| Phase | Duration | What to expect |
|---|---|---|
| Model loading + weight conversion | A few minutes | Progress bars for loading weights |
| Memory pool allocation | Seconds | `"Memory pool end. avail mem=X GB"` |
| CUDA graph capture (with JIT) | **~6 min** | Progress bar: `Capturing batches...` 24 batch sizes at ~14.6s each |
| CUDA graph registration | Seconds | `"Registering 4488 cuda graph addresses"` |
| HiCache host memory allocation | Seconds | `"Allocating X GB host memory for hierarchical KV cache"` |
| Server ready | Seconds | `"Uvicorn running on http://..."` then `"The server is fired up and ready to roll!"` |

Second run (kernel cached): CUDA graph capture is much faster since JIT compilation is skipped.

### Cama-Specific Errors

**"Please install the priskv package" ImportError:**

The `priskv` package is not installed. Install it with:

```bash
pip install priskv
```

**"PrisKV reg_memory returned 0 — RDMA buffer registration failed":**

RDMA memory registration failed. Common causes:
- Insufficient RDMA memory registration limits. Check `ulimit -l` and increase if needed (`ulimit -l unlimited`).
- RDMA drivers not loaded. Verify with `ibv_devices`.
- Running without sufficient privileges. Some environments require root for RDMA registration.

**"PrisKV server not reachable after 600s":**

The health check timed out. Verify:
- PrisKV server is running: `priskv-server --port 6379`
- Network connectivity: `ping <remote_addr>` and `telnet <remote_addr> <remote_port>`
- Firewall rules allow traffic on the configured port.

**"Warmup setstr failed with code N":**

The PrisKV connection was established but operations are failing. Check:
- PrisKV server logs for errors.
- Authentication: ensure the password matches between client and server.
- Server is not in a read-only or degraded state.

**"Cama storage backend only supports page_first, page_first_direct, or page_head layout":**

The host memory layout is incompatible. Cama requires the buffer to be organized by page for zero-copy. Set:

```bash
--hicache-mem-layout page_first
```

**Cache misses on second request (no cache hits):**

- Verify `--hicache-write-policy write_through` is set (write_back defers writes and may not have flushed).
- Check logs for `"failed to retrieve page"` warnings indicating read failures.
- Ensure both requests share the same prompt prefix (HiCache matches on token-level page hashes).

**High CPU usage during transfers:**

- Ensure `use_mput_mget` is `true` (default). Per-key `set`/`get` in a Python loop is much slower than batched `mset`/`mget`.
- Verify RDMA is active (not falling back to TCP). Check PrisKV server logs for connection type.

**Key collisions with multiple instances:**

If multiple SGLang instances share one PrisKV cluster without `extra_backend_tag`, they may read/write each other's cached pages. Set a unique tag per instance:

```bash
--hicache-storage-backend-extra-config '{"remote_addr": "...", "extra_backend_tag": "unique-instance-id"}'
```

**HiCache CPU Memory Usage:**

When using HiCache, the default L2 host DRAM size is 2x the L1 GPU KV cache size. For small models on large GPUs (especially multi-TP), this can consume excessive CPU memory. Set an appropriate size manually:

```bash
--hicache-ratio 1.0    # 1:1 ratio instead of default 2:1
# or
--hicache-size 8589934592    # Explicit 8GB
```

# CAMA Deployment Guide

> Step-by-step recipes for deploying CAMA in various configurations.

---

## Prerequisites Checklist

Verify each requirement before deploying. Failure modes are listed for each.

| Requirement | Verification Command | Failure Mode |
|-------------|---------------------|--------------|
| PrisKV server binary | `which priskv-server` or `priskv-server --help` | CAMA cannot connect; `RuntimeError` at startup |
| `priskv` Python package | `python -c "from priskv.priskv_client import PriskvClient; print('OK')"` | `ImportError: Please install the priskv package` |
| RDMA-capable NIC(s) | `ibv_devices` | RDMA registration fails; falls back to TCP (if PrisKV supports it) or crashes |
| RDMA drivers (libibverbs) | `ibv_devinfo` | `ibv_devices` returns empty; RDMA calls fail |
| RDMA memory limits | `ulimit -l` (should be `unlimited` or very large) | `RuntimeError: PrisKV reg_memory returned 0` |
| SGLang (source install) | `python -c "import sglang; print(sglang.__file__)"` | SGLang server won't start |
| sgl-kernel | `python -c "import sgl_kernel; print(sgl_kernel.__version__)"` | Server hangs at CUDA graph capture |
| flashinfer | `python -c "import flashinfer; print(flashinfer.__version__)"` | Server hangs at CUDA graph capture |
| CUDA toolkit (quantized models only) | `nvcc --version` | JIT kernel compilation hangs |
| Network connectivity | `ping <priskv_addr>` | Connection refused / timeout |

---

## Single-Server Deployment

The simplest setup: PrisKV and SGLang on the same machine.

### Step 1: Start PrisKV Server

```bash
priskv-server --port 6379
```

Verify it's running:

```bash
# Quick connectivity test
python -c "
from priskv.priskv_client import PriskvClient
conn = PriskvClient('127.0.0.1', 6379, '')
ret = conn.setstr('test', 'ok')
print('PrisKV alive:', ret == 0)
conn.delete('test')
"
```

### Step 2: Start SGLang with CAMA

**Option A: Environment variables (simplest)**

```bash
SGLANG_CAMA_REMOTE_ADDR=127.0.0.1 \
SGLANG_CAMA_REMOTE_PORT=6379 \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

**Option B: extra_config (inline JSON)**

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{"remote_addr": "127.0.0.1", "remote_port": 6379}'
```

**Option C: JSON config file**

```bash
cat > /etc/cama_config.json << 'EOF'
{
    "remote_addr": "127.0.0.1",
    "remote_port": 6379,
    "password": "",
    "use_mput_mget": true,
    "check_server": false
}
EOF

SGLANG_CAMA_CONFIG_PATH=/etc/cama_config.json \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama
```

### Step 3: Verify Successful Startup

Look for these log messages in order:

```
Cama configuration loaded from env.          # (or "from extra_config" / "from file")
Connected to PrisKV at 127.0.0.1:6379
Cama PrisKV store warmup successful.
Cama PrisKV store setup complete.
...
Registered RDMA buffer: ptr=0x..., size=... bytes, handle=...
Sent model_page_bytes hint to server: 131072 bytes   # (v0.37.0+: eager slab tuning)
```

---

## Multi-Server Deployment

PrisKV runs on a dedicated storage node. SGLang instances on compute nodes connect over the network.

```
┌──────────────────┐         RDMA          ┌──────────────────┐
│  Storage Node    │◀════════════════════▶│  Compute Node 1  │
│  10.0.0.1        │                       │  SGLang Worker   │
│                  │                       └──────────────────┘
│  priskv-server   │         RDMA          ┌──────────────────┐
│  :6379           │◀════════════════════▶│  Compute Node 2  │
│                  │                       │  SGLang Worker   │
└──────────────────┘                       └──────────────────┘
```

### Storage Node (10.0.0.1)

```bash
priskv-server --port 6379
```

Ensure RDMA NIC is up and firewall allows port 6379.

### Compute Nodes

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
SGLANG_CAMA_REMOTE_PORT=6379 \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

---

## Tensor Parallel

When running with `--tp N`, each TP rank creates its own PrisKV connection and uses rank-specific key suffixes to avoid collisions. No special CAMA configuration is needed.

**Key isolation for MHA models:**
- TP rank 0: `{hash}_0_k`, `{hash}_0_v`
- TP rank 1: `{hash}_1_k`, `{hash}_1_v`
- TP rank 2: `{hash}_2_k`, `{hash}_2_v`
- TP rank 3: `{hash}_3_k`, `{hash}_3_v`

**Key isolation for MLA models:**
- All TP ranks: `{hash}_k` (MLA uses fused KV, no TP-rank suffix needed when PP is disabled)

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

---

## Pipeline Parallel

When running with `--pp N`, CAMA includes the PP rank in key suffixes to prevent data corruption across pipeline stages.

**Key isolation for MHA (TP=4, PP=2):**
- TP0/PP0: `{hash}_0_0_k`, `{hash}_0_0_v`
- TP1/PP0: `{hash}_1_0_k`, `{hash}_1_0_v`
- TP0/PP1: `{hash}_0_1_k`, `{hash}_0_1_v`
- TP1/PP1: `{hash}_1_1_k`, `{hash}_1_1_v`

**Key isolation for MLA (PP=2):**
- PP0: `{hash}_0_k`
- PP1: `{hash}_1_k`

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --pp 2 \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

---

## Multi-NIC Deployment

When the PrisKV server has multiple RDMA NICs, CAMA automatically discovers endpoints and stripes reads across all NICs for maximum bandwidth. With `nic_striping=True` (default), each rank connects to ALL server NICs simultaneously.

### How It Works

1. After connecting, CAMA calls `conn.rdma_endpoints()` to discover available RDMA NICs on the server.
2. **NIC striping (default):** CAMA passes ALL endpoints to `create_pool()`. The pool creates one connection per NIC (`pool_size=len(endpoints)`) and `mget_rdma` stripes keys across all connections in parallel for N× read bandwidth.
3. **Legacy mode (`nic_striping=False`):** Each rank is assigned a single NIC via `endpoints[local_rank % len(endpoints)]` (round-robin), and CAMA reconnects to the assigned endpoint.

### Multi-NIC Deployment Diagram (NIC Striping)

```
┌──────────────────┐         RDMA (mlx5_0)    ┌──────────────────┐
│  Storage Node    │◀═══════════════════════▶│  Compute Node    │
│  10.0.0.1        │                          │                  │
│                  │         RDMA (mlx5_1)    │  Each rank has   │
│  cama-server     │◀═══════════════════════▶│  connections to  │
│  :18001          │                          │  BOTH NICs:      │
│                  │                          │  Rank 0: 2 conns │
│  NICs: mlx5_0    │                          │  Rank 1: 2 conns │
│        mlx5_1    │                          │  ...              │
└──────────────────┘                          └──────────────────┘

mget_rdma stripes keys round-robin across connections:
  keys[0,2,4,...] → conn 0 (mlx5_0)
  keys[1,3,5,...] → conn 1 (mlx5_1)
```

### Expected Log Output

```
Multi-NIC striping: rank 0 -> 2 endpoints [('mlx5_0', '10.0.0.10', 18001), ('mlx5_1', '10.0.0.11', 18001)]
```

For single-NIC servers or TCP-only connections:
```
Single RDMA endpoint, no multi-NIC reconnect needed.
```

No special configuration is required beyond the default `nic_striping=True`. The PrisKV server must advertise endpoints via the handshake protocol. To disable striping and use legacy per-rank NIC assignment, set `nic_striping=False` (or `SGLANG_CAMA_NIC_STRIPING=0`).

---

## MLA Models (DeepSeek-V3)

MLA (Multi-Head Latent Attention) models use a fused KV representation: one tensor per page instead of separate K and V. CAMA detects this automatically from `storage_config.is_mla_model` and adjusts key naming and buffer handling. No special flags needed.

Differences from MHA:
- **1 sub-key per page** instead of 2 (no separate `_k`/`_v` suffixes for V)
- **1 pointer per page** from `get_page_buffer_meta` instead of 2
- Key format: `{hash}_{pp_rank}_k` (PP enabled) or `{hash}_k` (PP disabled)

```bash
SGLANG_CAMA_REMOTE_ADDR=10.0.0.1 \
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-write-policy write_through
```

---

## Multi-Instance with Key Isolation

When multiple SGLang instances share one PrisKV cluster, use `extra_backend_tag` to namespace keys. Without it, instances may read/write each other's cached pages.

```bash
# Instance A — all keys prefixed with "node-1_"
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "extra_backend_tag": "node-1"}' \
    --port 30000

# Instance B — all keys prefixed with "node-2_"
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{"remote_addr": "10.0.0.1", "extra_backend_tag": "node-2"}' \
    --port 30001
```

With `extra_backend_tag = "node-1"`, a key like `a3f2c8_0_k` becomes `node-1_a3f2c8_0_k`.

---

## Kubernetes Deployment

In Kubernetes, PrisKV may start after SGLang. Enable `check_server` to have CAMA poll until the server is ready (up to 10 minutes). When `check_server=true`, the preflight connectivity check is skipped.

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --enable-hierarchical-cache \
    --hicache-storage-backend cama \
    --hicache-storage-backend-extra-config '{
        "remote_addr": "priskv-service.default.svc.cluster.local",
        "remote_port": 6379,
        "check_server": true
    }'
```

The health check polls `conn.exists("__cama_health__")` every 3 seconds for up to 600 seconds. When PrisKV responds, startup continues normally.

---

## Verification Procedures

### Expected Log Strings

On successful startup, these messages appear in order:

| Log Message | Meaning |
|-------------|---------|
| `Cama configuration loaded from ...` | Config source resolved |
| `Connected to PrisKV at X:Y` | TCP/RDMA connection established |
| `PrisKV server is reachable.` | Health check passed (only with `check_server=true`) |
| `Cama PrisKV store warmup successful.` | String + SGL RDMA round-trip validated |
| `Cama PrisKV store setup complete.` | All initialization done |
| `Registered RDMA buffer: ptr=0x..., size=... bytes, handle=...` | Host KV buffer registered for zero-copy |
| `Multi-NIC: rank N -> mlx5_X ...` | Multi-NIC endpoint assigned to this rank (only with multiple NICs) |
| `Single RDMA endpoint, no multi-NIC reconnect needed.` | Server has one RDMA NIC (normal single-NIC operation) |

### Cache Hit Test

Send two identical requests. The second should show a cache hit:

```bash
# First request — cache miss, writes pages to PrisKV
curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default", "prompt": "The meaning of life is", "max_tokens": 32}'

# Second request — cache hit, reads pages from PrisKV (faster TTFT)
curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default", "prompt": "The meaning of life is", "max_tokens": 32}'
```

With `SGLANG_LOG_LEVEL=DEBUG`, you should see:
- First request: `batch_set_v1: N total sub-keys, 0 already exist (deduped), N to write`
- Second request: `batch_get_v1: N sub-keys, N hit, 0 miss`

### PrisKV-Side Verification

Check PrisKV has stored keys:

```python
from priskv.priskv_client import PriskvClient
conn = PriskvClient("10.0.0.1", 6379, "")
keys = conn.keys("*")
print(f"PrisKV holds {len(keys)} keys")
# Should show keys like: "a3f2c8..._0_k", "a3f2c8..._0_v"
```

### HiCache Memory Usage Check

Monitor L2 host DRAM consumption. The default `--hicache-ratio 2.0` means L2 is 2x the GPU KV cache size. For large multi-TP deployments, this can consume excessive CPU memory:

```bash
# Reduce L2 size if needed
--hicache-ratio 1.0          # 1:1 ratio instead of default 2:1
# or
--hicache-size 8589934592    # Explicit 8 GB
```

---

## Related Documents

- [01_OVERVIEW.md](01_OVERVIEW.md) — What CAMA is
- [03_CONFIGURATION_REFERENCE.md](03_CONFIGURATION_REFERENCE.md) — All configuration parameters
- [05_TROUBLESHOOTING.md](05_TROUBLESHOOTING.md) — When things go wrong

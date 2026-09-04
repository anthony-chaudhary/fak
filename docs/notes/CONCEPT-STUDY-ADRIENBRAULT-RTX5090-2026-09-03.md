# Concept Study: adrienbrault/qwen3.8-27b-rtx5090 — Consumer Blackwell Serving, sm120 NVFP4 Scaling, and Direct-I/O Disk KV Tier

**Source:** https://github.com/adrienbrault/qwen3.8-27b-rtx5090  
**Pinned Revision:** `00461a303f6f1ce84fc8514a60aa9e32735d1fe3`  
**Study Date:** 2026-09-03  
**Author:** Adrien Brault  
**License:** MIT License (`LICENSE:1-21@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`); patches against vLLM/FlashInfer/LMCache remain Apache-2.0 (`THIRD_PARTY.md:1-86@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`)  
**Tracking Issue:** [#10951](https://github.com/anthony-chaudhary/fak/issues/10951)  
**Durable Receipt ID:** `study_5d5f94787ea215ff28eca11a11fe6a4dfa9239da068dbe31612d2a068c59c6ac`  
**Parent Epics:** [#9433](https://github.com/anthony-chaudhary/fak/issues/9433) (model architecture), [#10193](https://github.com/anthony-chaudhary/fak/issues/10193) (native inference performance), [#2236](https://github.com/anthony-chaudhary/fak/issues/2236) (owned & observed KV cache value)  
**Study Depth:** Deep (exhaustive inspection across scripts, patches-v0280, patches-v0290, overlay, diagnostics, benchmarks, SWE-Bench reproduction artifacts, and design notes)  
**Completeness Critic:** Verified — all patches (`patches-v0280/`, `patches-v0290/`), overlay C++/Python/diagnostic tools (`patches-v0280/overlay/`), host and serve automation (`scripts/`), gotchas (`docs/GOTCHAS.md`), config rationales (`docs/CONFIG.md`), and raw evaluation artifacts (`bench/RESULTS.md`, `bench/reproduce/`) inspected at `file:line@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`.

---

## Executive Summary

`adrienbrault/qwen3.8-27b-rtx5090` is an empirical systems-engineering tour de force that demonstrates production-grade serving of **Qwen3.8-27B** (262,144 context, W4A4 weights, FP8/NVFP4 KV cache) on dual consumer NVIDIA GeForce RTX 5090 (32 GB GDDR7, Blackwell `sm_120`) in a single desktop workstation (ASRock X870, AMD Ryzen 7 9800X3D, 64 GB DDR5).

The repository bridges the chasm between **datacenter assumptions in upstream vLLM** (which assume HGX/DGX clusters with SM100, NVLink, terabytes of host RAM, and external Kubernetes PVC evictors) and the realities of **consumer hardware running long-context agentic workloads**.

Key technical breakthroughs and empirical findings in this repository:
1. **SM120 NVFP4 Linear V-Scale Store Overlay**: Upstream vLLM unconditionally writes NVFP4 V-block scale factors in an SM100 `trtllm-gen` 4-token swizzled layout (`nvfp4_kv_cache_kernels.cu:291-298`). On consumer Blackwell (SM120/SM121), FlashInfer's FlashAttention-2 (FA2) paged reader expects linear scales. Because the data layout matches but scale factors are permuted across 4-token groups, output remains superficially fluent and passes coarse needle tests, but suffers **2.7× to 10× attention L2 error** and an 8.8% ΔNLL degradation. The author ships a custom C++/PyTorch overlay (`torch.ops.vllm_sm12x.reshape_and_cache_nvfp4`) and a rigorous numerical tensor diagnostic (`diag_vsf_layout.py`) that isolates the scale permutation.
2. **Direct-I/O Hard-Capped Loopback NVMe KV Cache Tier**: Standard vLLM filesystem offloading (`OffloadingConnector`) lacks capacity enforcement, TTL, or eviction. When the disk fills, unhandled store failures crash the engine; worse, asynchronous deletes during serving trip an engine assertion (`assert transfer_result.success`). Furthermore, standard buffered loopback filesystems double-cache every page in host RAM (wasting 37 GB under eval load). The author solves this by constructing a 200 GB preallocated loopback ext4 image (`setup-native-l2.sh`) mounted with `losetup --sector-size 4096 --direct-io=on`, coupled with an idle-gated (`num_requests_running == 0 && num_requests_waiting == 0`) background LRU evictor (`tier-evict.sh`).
3. **Deterministic KV Pool Pinning (`--kv-cache-memory-bytes`)**: Sizing KV caches via `--gpu-memory-utilization` fails on complex speculative/NVFP4 configurations because vLLM sizes the cache *before* CUDA graph capture and pre-warm, assuming zero graph memory. The author replaces fractional utilization with exact byte allocation per sequence concurrency (e.g., 12.85 GiB for 8 sequences, 11.87 GiB for 32 sequences) backed by an explicit post-pre-warm free memory assertion (`MIN_FREE_MIB=384`).
4. **CUDA Graph Memory Collapse via FlashInfer Integer Workspace Sizing (0131)**: NVFP4 FA2 graph capture allocates 5 `BatchPrefillWithPagedKVCacheWrapper` objects per captured shape (one per Qwen3.8 attention group). Upstream FlashInfer assigns each wrapper an 8 MiB device integer workspace (+ 8 MiB pinned host buffer), consuming >2 GiB across 22 batch sizes and causing graph-capture OOMs. Patch `0131` truncates this buffer to 1 MiB (`VLLM_SM12X_POOLED_INT_WS_MIB=1`), slashing graph memory by ~840 MiB with zero operational degradation.
5. **Host-Pinned UVA Input Embedding Offload (0135)**: Routes the 2.37 GiB BF16 `VocabParallelEmbedding` table to pinned host RAM via Unified Virtual Addressing (UVA). Because input embeddings are read exactly once per prompt token, host-side gather over PCIe Gen5 adds 0 ms to decode and frees 1.18 GiB of VRAM per GPU on TP=2, translating directly to **+9.3% KV pool (+82,811 tokens) for free**.
6. **Agentic Serving Benchmark Authority**: Reaches **386/500 = 77.2% on SWE-Bench Verified** with mini-SWE-agent 2.4.6 (single attempt) on dual RTX 5090s, with 318.8 tok/s single-stream code decode, 1,212 tok/s aggregate decode at 8 streams, and 0.45s 32K context restoration from the direct-I/O disk tier.

---

## Author's Worldview Reconstruction

Understanding the author's worldview explains *why* the codebase is shaped this way and why standard serving assumptions fail on consumer hardware:

### 1. The Consumer Blackwell Hardware Envelope
- **Topology**: Dual RTX 5090 (32 GB GDDR7 each, 64 GB total VRAM) on an AMD X870 desktop platform. Inter-GPU communication is constrained to PCIe Gen5 x8/x8 (no NVLink).
- **P2P Workaround**: Consumer GeForce cards have hardware P2P disabled by default in standard NVIDIA drivers. The author uses QuixiAI's open GPU kernel modules with `RMForceStaticBar1=1`, `NVreg_EnableResizableBar=1`, and `iommu=pt` (`scripts/gpu-p2p-610.sh`) to force-enable PCIe peer-to-peer BAR1 access.
- **Clock Offsets**: GDDR7 memory clock is offset by +4500 MHz via the modern clock-offsets API (`scripts/gpu-tune.sh`), increasing memory bandwidth by 15% and boosting decode throughput by ~4%.
- **Host Memory Ceiling**: The host has only 64 GB DDR5 RAM. In a 64 GB system, serving an LLM with pinned staging buffers leaves narrow margins. Unbounded process memory or memory-mapped leaks immediately trigger OS swap thrashing and kernel lockups.

### 2. Overcoming vLLM Datacenter Assumptions
vLLM's codebase is heavily biased toward hyperscaler clusters (A100/H100/B200 with 512GB–2TB host RAM, NVLink, and dedicated Kubernetes storage):
- *The SM100 Monoculture*: Upstream developers implemented NVFP4 KV cache targeting SM100 (datacenter Blackwell) using `trtllm-gen` MHA kernels. SM100 requires a 4-token swizzled layout for value scale factors (`swizzle_scale_offset`). On SM120 (GeForce RTX 5090), `trtllm-gen` MHA prefill does not exist, so FlashInfer FA2 is used instead. But vLLM wrote V-scales using the SM100 swizzle unconditionally, corrupting attention reads on SM120. Datacenter developers missed this because their test matrices only ran NVFP4 on B100/B200.
- *The Assumption of Infinite Host RAM*: vLLM's `OffloadingConnector` creates 4 GiB CPU shared-memory files (`/dev/shm/vllm_offload_*.mmap`) that leak across container removals (`docker rm -f`). Four engine restarts leak 16 GiB. Moreover, FlashInfer's JIT compiler spawns unbounded worker threads, which previously consumed 64 GB RAM and locked the machine.
- *The "External Controller" Storage Myth*: vLLM's filesystem KV tier delegates all eviction and capacity management to "an external controller" (e.g., Kubernetes PVC evictor). Out of the box, `OffloadingConnector` fills the filesystem to 100% and then crashes when writes fail; if an external script deletes a file while a request is running, vLLM's hard assertion `assert transfer_result.success` aborts the entire engine process.
- *Blind KV Pool Sizing*: Datacenter deployments rarely run complex speculative decoding with non-causal draft models on 4-bit KV caches. Sizing KV caches using `--gpu-memory-utilization` before CUDA graph capture leaves zero headroom for graph capture on complex models, causing immediate OOM crashes.

### 3. Optimizing for SWE-Bench Agentic Coding
The author's primary target workload is concurrent long-context coding agents (such as mini-SWE-agent, Aider, and Harbor):
- *Prefix Reuse Dominance*: An agentic loop resends the entire conversation transcript on every tool-call turn (often 30K–130K tokens). Cold prefill for 131K tokens takes 30.1 seconds; warm revisit from the disk tier takes 7.0 seconds; GPU in-pool revisit takes 0.45 seconds. Without robust prefix caching and disk persistence across agent task restarts, agent velocity plummets.
- *Hybrid Architecture Mechanics*: Qwen3.8-27B combines 48 Gated DeltaNet (GDN) linear attention layers with 16 full attention layers. Linear layers have fixed recurrent state ($[16, 128, 128]$) regardless of context length; only the 16 full-attention layers consume KV cache. This allows packing 262K tokens into VRAM, but requires packing GDN state into the cache pages (`--mamba-cache-mode align`) so hybrid states can be prefix-cached and offloaded together.
- *Fidelity vs. Quantization*: Through an exhaustive 725K-token fidelity ladder, the author proves that naive 4-bit quantization of the language model head (`lm_head`) costs 0.85 percentage points of perplexity and damages tool-calling precision. RedHat's NVFP4 export (with 303 sensitive layers kept in FP8 and FP8 `lm_head`) matches BF16 perplexity within +0.38%, delivering 90.8% on tool-eval and 77.2% on SWE-Bench Verified.

---

## Subsystem Inventory & Code Tour

```
adrienbrault/qwen3.8-27b-rtx5090@00461a303f6f1ce84fc8514a60aa9e32735d1fe3
├── bench/
│   ├── RESULTS.md               # 987 lines of exhaustive empirical measurements (R1-R167)
│   ├── prefix_probe.py          # Microsecond prefix-cache timing probe
│   └── reproduce/               # 500 SWE-Bench Verified predictions, sanitizer, and official reports
├── docs/
│   ├── CONFIG.md                # Exhaustive flag-by-flag production configuration manual
│   ├── DESIGN.md                # Memory arithmetic, W4A4 GEMM paths, TP=2 bandwidth rationale
│   ├── GOTCHAS.md               # 27 verified failure modes and battle-hardened guards
│   ├── HISTORY.md               # Lineage of stack generations from 2026-06 to 2026-09
│   └── REJECTED.md              # Dozens of rejected configurations with exact numbers
├── patches-v0280/               # vLLM v0.28.0 production patch stack
│   ├── 0101-sm120-nvfp4kv-fa2-routing-v0280.diff
│   ├── 0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff
│   ├── 0103-sm120-nvfp4-xqa-decode-v0280.diff
│   ├── 0104-mtp-drafter-full-cudagraph-v0280.diff
│   ├── 0106-dflash2-selector-sampling-v0280.diff
│   ├── 0107-dflash-quantized-draft-loader.diff
│   ├── 0108-gdn-kernels-v0280.diff
│   ├── 0113-dflash-speculator-graphs-v0280.diff
│   ├── 0116-dflash-nvfp4-revival.diff
│   ├── 0129-dflash-nvfp4-drafter-graphs.diff
│   ├── 0131-nvfp4-pooled-int-workspace.diff
│   ├── overlay/
│   │   ├── build_overlay.py     # Torch JIT/AOT extension compiler for sm120
│   │   ├── diag_vsf_layout.py   # Numerical diagnostic proving V-scale swizzle mismatch
│   │   ├── overlay_binding.cpp  # C++ stable torch library registration
│   │   └── vllm_sm12x_nvfp4kv.py# Python runtime loader
│   └── README-sm120-nvfp4.md
├── patches-v0290/               # vLLM v0.29.0rc1 experimental stack
│   ├── 0132-masked-nvfp4-xqa-sm120-v0290.diff
│   ├── 0133-gdn-packed-decode-bv16-v0290.diff
│   ├── 0134-dflash-no-eagle-block-drop-v0290.diff
│   └── 0135-embed-uva-offload-v0290.diff
└── scripts/
    ├── gpu-p2p-610.sh           # P2P BAR1 enablement for dual RTX 5090
    ├── gpu-tune.sh              # Persistence mode, power caps, +4500MHz VRAM offset
    ├── serve-r156-daily.sh      # Dual-5090 production daily launcher (DFlash2, fp8 KV, TP=2)
    ├── serve-v0280-daily.sh     # Generic launcher with fail-closed boot asserts
    ├── serve-nvfp4-candidate.sh # Candidate daily launcher with pinned byte KV pool
    ├── setup-earlyoom.sh        # Out-of-memory protector protecting inference engine
    ├── setup-native-l2.sh       # 200 GB direct-I/O loopback NVMe ext4 setup
    └── tier-evict.sh            # Idle-gated background LRU disk tier evictor
```

---

## Detailed Technical Analysis of Core Components

### 1. SM120 NVFP4 Linear V-Scale Store Overlay & `diag_vsf_layout.py`

#### The Problem
In NVFP4 (E2M1 with block-scale FP8 E4M3), a page stores packed data and scale factors in physical layout `[K_data | K_scale | V_data | V_scale]`. For `head_dim = 256`:
- Packed FP4 data: $256 / 2 = 128$ bytes per token.
- Scale factors (1 per 16 values): $256 / 16 = 16$ bytes per token.
- Total token footprint: $128 + 16 = 144$ bytes.

In upstream vLLM (`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`), K scales are always written linearly:
```cpp
// K (kv == 0): linear layout (no swizzle)
scale_dst = scale_block + head * scale_head_stride +
            block_offset * scale_block_offset_stride + scale_idx;
```
However, for V scales (`kv == 1`), lines 291-298 unconditionally swizzled the scale offsets using `swizzle_scale_offset` for the SM100 `trtllm-gen` MHA kernel:
```cpp
// V (kv == 1): swizzled layout for SM100 trtllm-gen MHA kernel
int swizzled_offset = swizzle_scale_offset<4>(block_offset, scale_idx);
scale_dst = scale_block + head * scale_head_stride + swizzled_offset;
```
On consumer Blackwell (SM120), `trtllm-gen` MHA does not exist; attention runs via FlashInfer's FA2 paged reader (`flashinfer/attention/prefill.cuh`), which reads both K and V scales linearly! The swizzle permuted scale factors across 4-token windows, multiplying values by wrong scale factors.

#### The Patch (`0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff`)
The patch introduces an architecture check:
```cpp
// SM120/SM121 (consumer Blackwell) serve NVFP4 KV via FlashInfer FA2 paged reader (linear).
// SM100 trtllm-gen keeps 4-token V scale swizzle.
const bool swizzle_v_sf = get_device_prop()->major < 12;

if (kv == 0 || !swizzle_v_sf) {
    scale_dst = scale_block + head * scale_head_stride +
                block_offset * scale_block_offset_stride + scale_idx;
} else {
    // SM100 swizzled path
}
```
Because `torch.ops._C_cache_ops.reshape_and_cache_flash` cannot be re-registered in PyTorch without a symbol collision, `overlay_binding.cpp` compiles the patched function into a separate namespace: `torch.ops.vllm_sm12x.reshape_and_cache_nvfp4`.

#### The Numerical Diagnostic (`diag_vsf_layout.py`)
Why was this bug missed by the community? Because language models are surprisingly resilient to local scale permutations: text generation remains grammatical, and coarse retrieval needles pass.
`diag_vsf_layout.py:49-56` constructs synthetic tensors with structured lognormal group gains:
$$\text{mag} \sim \exp(\mathcal{N}(0, \sigma^2)), \quad \sigma = 1.2$$
applied per-16-channel group. It writes Cache A (in-tree swizzled) and Cache B (linear overlay), executes FlashInfer `BatchPrefillWithPagedKVCacheWrapper(..., backend="fa2")` on both, and compares against FP32 reference attention:
- Cache B (linear): relative L2 error = `0.124` (quantization noise floor).
- Cache A (swizzled): relative L2 error = `0.335` to `1.300` (**2.7× to 10× higher error**).
- Result: **VERDICT: OVERLAY REQUIRED (errA/errB = 2.70 - 10.48)**.

---

### 2. Direct-I/O Loopback NVMe KV Cache Tier & Idle Eviction

#### Loopback Direct-I/O Backing (`setup-native-l2.sh`)
vLLM's `OffloadingConnector` delegates secondary storage to a filesystem path. In consumer setups, running this on the root NVMe drive risks full disk exhaustion. In an earlier run (R69), the tier grew to 876 GB, crashing the operating system.
The author enforces disk limits **by construction** via a 200 GB preallocated loopback ext4 filesystem:
```bash
IMG=/srv/qwen5090/native-l2.img
MNT=/srv/qwen5090/native-l2
fallocate -l 200G "$IMG"
mkfs.ext4 -q -m 0 -L native-l2 "$IMG"
```
**The Double-Caching Trap (Gotcha #20)**:
A standard loopback mount (`mount -o loop`) routes disk reads through the Linux page cache twice: once for the underlying loop file, and once for the mounted filesystem. Under heavy agent evaluation loads, this page cache amplification consumed **37 GB of host RAM**.
The fix requires attaching the loopback device with direct I/O:
```bash
losetup --sector-size 4096 --direct-io=on /dev/loopX /srv/qwen5090/native-l2.img
```
*Note*: Plain `--direct-io=on` silently no-ops on 512-byte sector devices; `--sector-size 4096` is mandatory.

#### Idle-Gated Background LRU Eviction (`tier-evict.sh`)
vLLM's `worker.py` contains:
```python
assert transfer_result.success, "we currently do not support job failures"
```
If a background eviction daemon deletes a block file *after* the scheduler verifies its existence (`os.path.exists`) but *before* the worker reads it, **the entire vLLM server crashes**.
`tier-evict.sh:27-32` solves this by gating eviction on engine idleness via Prometheus metrics:
```bash
idle() {
  local m; m=$(curl -s --max-time 2 "$DAILY/metrics") || return 0
  local r w
  r=$(echo "$m" | awk '/^vllm:num_requests_running/{print $2; exit}')
  w=$(echo "$m" | awk '/^vllm:num_requests_waiting/{print $2; exit}')
  [ "${r%.*}" = 0 ] && [ "${w%.*}" = 0 ]
}
```
When idle, it evicts coldest `.bin` files in atomic batches of 200 until disk usage drops from 85% to 70%.

---

### 3. FlashInfer Integer Workspace Sizing (Bug C & Patch 0131)

#### The Problem
During CUDA graph capture at `--max-num-seqs 16` and `32`, the engine crashed with CUDA OOM.
The capture ledger (`0130-bugc-capture-ledger.diff`) revealed:
- Qwen3.8-27B has 5 attention groups + 1 drafter group.
- For every captured batch size (up to 22 sizes), FlashInfer constructs a separate `BatchPrefillWithPagedKVCacheWrapper`.
- Upstream FlashInfer unconditionally allocates an 8 MiB device integer buffer and an 8 MiB pinned host buffer per wrapper:
  $$22 \text{ batch sizes} \times 5 \text{ groups} \times 8 \text{ MiB} \approx 880 \text{ MiB per rank}$$
- Total graph memory swelled from 0.80 GiB on FP8 to 2.04 GiB on NVFP4.

#### The Patch (`0131-nvfp4-pooled-int-workspace.diff`)
Patch `0131` intercepts wrapper creation and shrinks the integer buffer to 1 MiB (`VLLM_SM12X_POOLED_INT_WS_MIB=1`):
```python
def _shrink_pooled_int_workspace(wrapper):
    n = mib * 1024 * 1024
    buf = getattr(wrapper, "_int_workspace_buffer", None)
    if buf is None or buf.numel() <= n:
        return wrapper
    wrapper._int_workspace_buffer = torch.empty((n,), dtype=torch.uint8, device=buf.device)
    wrapper._pin_memory_int_workspace_buffer = torch.empty((n,), dtype=torch.uint8, device="cpu", pin_memory=True)
    return wrapper
```
This reduces graph memory from 2.04 GiB to **1.20 GiB** (saving 840 MiB), enabling clean boots at `--max-num-seqs 16` and `32`.

---

### 4. Input Embedding UVA Host Offload (0135)

In Qwen3.8-27B, the input embedding table (`VocabParallelEmbedding`) is 2.37 GiB of BF16 weights ($152,064 \text{ vocab} \times 8,192 \text{ hidden} \times 2 \text{ bytes}$). On TP=2, each GPU stores 1.18 GiB in VRAM.
Because embeddings are only accessed during prefill (one row gather per prompt token), holding them resident in VRAM wastes precious KV cache capacity.
Patch `0135` (porting vLLM PR #53981) routes `VocabParallelEmbedding` through Unified Virtual Addressing (UVA):
```bash
--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens
```
The table remains in pinned host DDR5 memory; the GPU gathers rows over PCIe Gen5 on demand.
**Measured Impact (R167)**:
- VRAM freed: 1.18 GiB per GPU.
- KV Cache Pool: **+9.3% tokens** (888,986 $\rightarrow$ **971,797 tokens**, +82,811 tokens).
- Decode Throughput: 227 vs 223 tok/s c1, 1,051 vs 1,065 tok/s c8 (parity within noise).
- Needle Retrieval: 8/8 clean at 131K context.

---

## Benchmark Sweet Spots & Production Configuration

The author characterizes three primary operational sweet spots on dual RTX 5090 (`sm_120`):

| Metric | One Card (TP=1, MTP) | Two Cards (TP=2, DFlash2, Served Daily) | Two Cards (TP=2, MTP, Pinned) |
|---|---|---|---|
| **Weights & Dtype** | RedHat NVFP4 (W4A4) | RedHat NVFP4 (W4A4) | RedHat NVFP4 (W4A4) |
| **KV Cache Dtype** | `nvfp4` (with linear overlay) | `fp8_e4m3` | `nvfp4` (with linear overlay) |
| **Speculative Engine** | MTP (4 draft tokens) | DFlash2 (9 draft tokens, W4A16) | MTP (4 draft tokens) |
| **KV Pool (262K ctx)**| 381,300 tokens | **654,491 tokens** | **1,508,519 tokens** |
| **Code Decode (1 stream)**| 175.0 tok/s | **318.8 tok/s** (298.9 steady) | 225.3 tok/s |
| **Code Decode (8 streams)**| 1,187 tok/s aggregate | **1,289 tok/s aggregate** | 1,349 tok/s aggregate |
| **Code Decode (16 streams)**| Not admitted | 1,522 tok/s aggregate | **2,007 tok/s aggregate** |
| **Prefill Rate (8K ctx)** | 11,900 tok/s | 9,300 tok/s | 9,000 tok/s |
| **Tool-Eval Score** | 89.2 ± 1.7 | **90.8 ± 0.5** | 90.2 ± 1.0 |
| **32K Warm Disk Revisit**| 1.0 s | **0.45 s** (vs 7.5s cold) | 0.68 s |
| **SWE-Bench Verified** | 66.2% (saka ckpt, R2E) | **77.2% (386/500, mini-swe)** | Pending evaluation |

### Why DFlash2 Wins on TP=2 for Single-Stream Code Decode
DFlash2 drafts 9 tokens in a single forward pass, but accepts only 0.25 to 0.35 tokens per draft step on coding tasks. Consequently, decode is heavily bound by weight-read bandwidth. TP=2 splits the weights across both cards, doubling available bandwidth to ~3.6 TB/s and boosting single-stream decode to **318.8 tok/s**. Conversely, MTP achieves higher acceptance per token, amortizing weight reads, making MTP superior for high-concurrency pool scaling (1.5M tokens).

---

## Concrete Borrow Candidates for FAK

The following concrete techniques are extracted directly from code in `adrienbrault/qwen3.8-27b-rtx5090@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`:

```
Candidate Matrix (FAK Borrow Portfolio)
-----------------------------------------------------------------------------------------------------------------------------------------
#  Technique / Capability      Source Anchor @00461a303f6f1c   Axis Optimized             FAK Seam & Status           Disposition / Lic
-----------------------------------------------------------------------------------------------------------------------------------------
1  Linear V-scale store layout  patches-v0280/0102...cu:28-49   Attention scale fidelity   internal/metalgemm/         DIRECT-PORT /
   for sm120 NVFP4 KV cache     overlay_binding.cpp:28-59       on Blackwell (sm120)       internal/compute/           Apache-2.0
                                                                                           ABSENT-on-axis
2  Direct-I/O loopback ext4     scripts/setup-native-l2.sh:8-19 Prevent double-caching &   internal/ctxmmu/            ADAPT /
   KV tier with idle eviction   scripts/tier-evict.sh:27-50     avoid engine crash on del  internal/vcache/            MIT
                                docs/GOTCHAS.md:32-34                                      PARTIAL-on-axis
3  Workspace-carved scratch     0103-sm120-xqa...diff:79-115    CUDA graph zero-alloc      internal/modelengine/       ADAPT /
   and integer buffer shrink    0131-pooled-int...diff:7-38     capture stability          internal/compute/           Apache-2.0
                                                                                           PARTIAL-on-axis
4  UVA pinned-host offload      0135-embed-uva...diff:53-76     VRAM reclamation for KV    internal/model/             ADAPT /
   for VocabParallelEmbedding   bench/RESULTS.md:89-106         pool (+9.3% pool free)     internal/modelengine/       Apache-2.0
                                                                                           ABSENT-on-axis
5  Exact byte-level KV budget   serve-nvfp4-candidate.sh:33-38  Deterministic pool sizing  internal/modelengine/       ADAPT /
   pinning with headroom guard  docs/GOTCHAS.md:19-20           & graph-capture safety     internal/model/             MIT
-----------------------------------------------------------------------------------------------------------------------------------------
```

### Candidate 1: Linear V-Scale Store Layout for SM120 NVFP4 KV Cache
- **Technique**: Force linear scale-factor output for value blocks when device compute capability major $\ge 12$ (Blackwell), bypassing the SM100 4-token `swizzle_scale_offset` required by `trtllm-gen`.
- **Source Anchor**: `patches-v0280/0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff:28-49@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`, `patches-v0280/overlay/overlay_binding.cpp:28-59@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`, and `patches-v0280/overlay/diag_vsf_layout.py:49-135@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`.
- **Axis**: Numerical attention fidelity and 4-bit KV retrieval accuracy on consumer Blackwell (sm120/sm121).
- **FAK Status**: ABSENT-on-axis. `internal/compute/` supports FP8 quantization, but Blackwell NVFP4 KV cache scale-factor layouts currently lack the sm120 linear vs sm100 swizzle distinction.
- **Disposition**: DIRECT-PORT (Apache-2.0). Port the C++ kernel branch and the `diag_vsf_layout.py` tensor diagnostic into FAK's CUDA backend test suite.

### Candidate 2: Direct-I/O Loopback NVMe KV Cache Tier with Idle-Gated Eviction
- **Technique**: Enforce secondary KV cache disk quotas by construction using a fixed-size loopback ext4 image mounted with `--sector-size 4096 --direct-io=on` (eliminating duplicate host page cache buffering), paired with Prometheus-gated idle deletion batches (`num_requests_running == 0`) to prevent read-after-delete race assertions.
- **Source Anchor**: `scripts/setup-native-l2.sh:8-19@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`, `scripts/tier-evict.sh:27-50@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`, and `docs/GOTCHAS.md:32-34@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`.
- **Axis**: Secondary storage safety and host RAM preservation (prevents 37 GB host memory amplification and zero-crash disk eviction).
- **FAK Status**: PARTIAL-on-axis. FAK has multi-tier context memory in `internal/ctxmmu/` and `internal/vcache/`, but lacks direct-I/O loopback mounting and idle-gated atomic batch eviction guards against secondary store crashes.
- **Disposition**: ADAPT (MIT). Integrate loopback direct-I/O configuration and idle-gated eviction into FAK's disk KV storage manager.

### Candidate 3: Workspace-Carved Scratch Views for Stable CUDA Graph Capture & Integer Workspace Sizing
- **Technique**: Slice scale-factor transpose scratch from the persistent TRTLLM/XQA workspace tail (using fixed-address `copy_`) instead of dynamic allocations during decode, and shrink pooled FlashInfer integer workspace buffers from 8 MiB to 1 MiB (`0131`) to prevent CUDA graph capture OOMs.
- **Source Anchor**: `patches-v0280/0103-sm120-nvfp4-xqa-decode-v0280.diff:79-115@00461a303f6f1ce84fc8514a60aa9e32735d1fe3` and `patches-v0280/0131-nvfp4-pooled-int-workspace.diff:7-38@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`.
- **Axis**: Zero-allocation CUDA graph capture stability and VRAM footprint reduction during multi-group speculative decode.
- **FAK Status**: PARTIAL-on-axis. FAK's model engine captures execution graphs, but memory pooling for intermediate scale scratch during ragged/hybrid attention can trigger re-allocations.
- **Disposition**: ADAPT (Apache-2.0). Adopt the fixed-tail workspace carving pattern and integer buffer bounds in FAK's CUDA graph execution planner.

### Candidate 4: Host-Pinned UVA Offload for VocabParallelEmbedding Table
- **Technique**: Route the model's 2.37 GiB BF16 input embedding table to pinned host memory via Unified Virtual Addressing (UVA) with on-demand PCIe Gen5 row gather during prefill, reclaiming 1.18 GiB VRAM per GPU on TP=2.
- **Source Anchor**: `patches-v0290/0135-embed-uva-offload-v0290.diff:53-76@00461a303f6f1ce84fc8514a60aa9e32735d1fe3` and `bench/RESULTS.md:89-106@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`.
- **Axis**: Net-true VRAM efficiency (+9.3% KV cache pool) with 0 ms decode latency penalty and zero perplexity degradation.
- **FAK Status**: ABSENT-on-axis. FAK places embedding tables entirely in device VRAM or entirely in CPU memory during CPU-fallback mode, lacking selective UVA host-pinned parameter placement for single-touch embedding lookups.
- **Disposition**: ADAPT (Apache-2.0). Implement selective UVA parameter wrapping for input embeddings in FAK's tensor-parallel loader.

### Candidate 5: Exact Byte-Level KV Pool Budget Pinning with Post-Capture Headroom Assertion
- **Technique**: Replace heuristic `--gpu-memory-utilization` flags with exact per-concurrency byte budgets (`--kv-cache-memory-bytes`) that explicitly account for post-graph-capture peak activation memory, verified by a strict post-pre-warm free memory guard (`MIN_FREE_MIB=384`).
- **Source Anchor**: `scripts/serve-nvfp4-candidate.sh:33-38@00461a303f6f1ce84fc8514a60aa9e32735d1fe3` and `bench/RESULTS.md:109-116@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`.
- **Axis**: Deterministic serving admission, eliminating bimodal profiler pool sizing lotteries.
- **FAK Status**: PARTIAL-on-axis. FAK has static allocation routines, but dynamic graph capture sizing still creates headroom uncertainty under varying concurrency limits.
- **Disposition**: ADAPT (MIT). Wire exact byte pinning and post-prewarm headroom tripwires into FAK's engine admission controller.

---

## License Disposition & Provenance Gate

- **Upstream License**: MIT License for original documentation, scripts, benchmarks, and patch-installer tooling (Adrien Brault, 2026).
- **Third-Party Code**: All diffs against vLLM, FlashInfer, and LMCache remain **Apache-2.0** (`THIRD_PARTY.md:1-86@00461a303f6f1ce84fc8514a60aa9e32735d1fe3`).
- **FAK Inbound Compatibility**: Fully compatible. FAK is licensed under **Apache-2.0**.
  - Candidates 1, 3, and 4 are derived from Apache-2.0 sources and can be DIRECT-PORTed or ADAPTed with standard Apache-2.0 notices.
  - Candidates 2 and 5 are MIT-derived original work and can be ADAPTed with standard MIT attribution.
  - No GPL, AGPL, or proprietary commercial restrictions exist in the core serving artifacts. (QuixiAI's driver patch is dual MIT/GPL, but is host-level driver infrastructure outside the kernel codebase).

---

## Concrete Follow-up Implementation Tickets

- Issue: `feat(compute): sm120 NVFP4 KV linear V-scale store overlay & tensor layout diagnostic` (Candidate 1)
- Issue: `feat(ctxmmu): direct-I/O loopback NVMe KV cache offload tier with idle-gated eviction` (Candidate 2)
- Issue: `feat(modelengine): workspace-carved scale scratch & FlashInfer int workspace shrink for CUDA graph stability` (Candidate 3)
- Issue: `feat(model): UVA host-pinned embedding table offload (+9% KV pool)` (Candidate 4)
- Issue: `feat(modelengine): exact byte-level KV pool budget pinning with post-capture headroom assertion` (Candidate 5)

---

## Registration and Next Actions

1. **Durable Registry**: Added `adrienbrault/qwen3.8-27b-rtx5090` at revision `00461a303f6f1ce84fc8514a60aa9e32735d1fe3` to `docs/research/monitored-repositories.json`.
2. **Index Registration**: Linked this note in `INDEX.md` under Notes & Research.
3. **Receipt**: Recorded `fak-study/1` study receipt for tracking issue [#10951](https://github.com/anthony-chaudhary/fak/issues/10951).

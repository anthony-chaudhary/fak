# Cross-Architecture Innovation Matrix: Qwen 3.8 & GLM 5.3 Flash

**Authority:** FAK Research & Inference Systems  
**Date:** September 3, 2026  
**Scope:** Systematic cross-pollination of kernel, memory, networking, and runtime innovations across Apple Silicon (MLX/Metal), AMD (Strix Halo/RDNA4/Vulkan/ROCm), NVIDIA (Blackwell sm120/GB10 sm121), and Host CPU architectures.

---

## 1. Executive Doctrine: The Cross-Architecture Translation Principle

When independent systems engineers encounter extreme hardware boundaries on frontier hybrid models (**Qwen 3.8 Flash-Next 125B MoE** and **GLM 5.3 Flash 300B MoE**), they invent solutions tailored to their specific constraints:
- Apple Silicon developers invent **high-QD asynchronous direct-storage expert streaming** to survive 48 GB RAM limits.
- AMD Strix Halo developers invent **USB4 RoCE-RDMA kernel bypasses** to link $2,000 APU mini-PCs into a 300B-class cluster.
- NVIDIA Blackwell developers invent **direct-I/O loopback KV caching** and **linear scale swizzle overlays** to hit 300 tok/s on consumer GPUs.
- Enterprise DGX Spark operators invent **HashK polynomial embedding compression** to run 51 GB n-gram tables in 12.8 GB VRAM.

**The Failure Mode:** These breakthroughs remain siloed inside single-platform repos (`slotstream` for MLX, `vllm-strix-halo` for AMD, `ninfer` for Windows WDDM, `ferrite` for Rust/sm100a).  
**The FAK Mandate:** Systematically translate each platform-specific breakthrough across architectural boundaries. A discovery in MLX must inform CUDA and ROCm; an AMD USB4 RDMA driver must inform multi-Mac clustering; a Grace-Blackwell UMA memory fix must inform AMD APUs.

---

## 2. Master Cross-Architecture Translation Matrix (10 Core Vectors)

| # | Innovation | Originating Platform & Source Anchor | Target Platforms | Core Bottleneck Solved on Target | Translation Mechanism & Formulation | Expected Gain on Target |
|---|---|---|---|---|---|---|
| **1** | **Multi-Tensor Slot-Stream Expert Engine** | **Apple MLX**<br/>`carloslfu/slotstream:42-180` | **NVIDIA CUDA** & **AMD ROCm** | Chokes on 125B–180B MoE on 24GB/32GB GPUs (RTX 4090, RX 7900 XTX). | Preallocated 9-tensor slot pool in VRAM (gate/up/down $\times$ weight/scale/bias); stream 2.76 MB expert records via `io_uring` / DirectStorage 1.3 QD32 `O_DIRECT`. | Runs 125B MoE on a single $1,600 RTX 4090 at **18–25 tok/s** without 4× GPUs. |
| **2** | **Compact DSA / MLA Latent Cache & Reversion Journal** | **Apple MLX**<br/>`kiojuvr/glm53-flash-mlx:45-189` | **NVIDIA CUDA** & **AMD ROCm** | 8–10 GB transient memory spikes in DSA IndexPool at >200K context. | Single contiguous pool buffer + $\le 3$-token active tail + rolling 16-token rollback journal ($16 + \text{index\_kpool} - 1$). | **85.95% reduction** in latent cache memory; frees 1.28 GB VRAM/seq; avoids KV cliff. |
| **3** | **Stream-Async GPU-Doorbell Direct RDMA All-Reduce** | **AMD ROCm (USB4)**<br/>`davidcanar/vllm-strix-halo:212-409` | **NVIDIA CUDA** & **Apple Metal** | PCIe Gen5 x8/x8 collective latency on dual RTX 5090; zero multi-Mac clustering. | NHI interrupt overdrive to 8 µs; direct UMA memory (`hipHostMalloc`); GPU acquire-spin kernel (`__builtin_amdgcn_s_sleep` / `__nanosleep()`) bypassing heavy collective runtimes. | **105 µs all-reduce** across $30 cable; enables multi-Mac clustering over Thunderbolt-4/5. |
| **4** | **HashK Dual-Subtable PLE Embedding Compression** | **NVIDIA GB10 (CUDA)**<br/>`airawatraj:45-226` | **Apple Silicon** & **AMD Strix Halo** | 51.2 GB n-gram table consumes 40% of unified memory on 128 GB machines. | Dual SplitMix64 polynomial hash ($S_h = \lceil V_h / 4 \rceil$); bypass ridge matrix $W_h \approx I_{160}$ to save 409k MACs/token; Grouped-norm absorbs error. | Slashes PLE table from 51.2 GB to **12.8 GB VRAM**; reclaims 38 GB unified RAM. |
| **5** | **Decoupled Speculative Draft Micro-Batching** | **AMD Strix Halo (Vulkan)**<br/>`Gr33n93:12-85` | **NVIDIA CUDA** & **Apple Metal** | Deep context prefill triggers GPU compute-ring watchdog timeouts. | Decouple draft ubatch (`--spec-draft-ubatch-size 512`) from main ubatch (1024); isolate speculative graph capture dimensions. | Unlocks **237K–262K cold prompts** without GPU resets; eliminates graph cache thrashing. |
| **6** | **Direct-I/O Loopback Disk KV Tier with Idle Eviction** | **NVIDIA RTX 5090 (CUDA)**<br/>`adrienbrault:8-19` | **Apple Metal** & **AMD ROCm** | Re-prefill latency on multi-turn agent turns; host page-cache double buffering. | `losetup --sector-size 4096 --direct-io=on`; direct page-aligned block DMA; gate eviction on `/metrics` idleness (`running == 0`). | **0.45s warm revisit of 32K context** (16.6× faster than cold prefill); zero host RAM double-buffering. |
| **7** | **WYF Chunkwise Parallel GatedDeltaNet Recurrence** | **Rust / sm100a**<br/>`MindLab-Research/ferrite:38-147` | **Apple Metal** & **AMD ROCm** | Sequential token recurrence loops stall GPU threads during prefill. | 32-token triangular block solve: $w_t = \beta_t (v_t - b_t - \sum_{s < t} c[t, s] w_s)$; inclusive prefix scan for decay $L[t, i]$. | Cuts linear attention kernel launches **32×**; parallelizes recurrence across SIMD lanes. |
| **8** | **Mamba / DeltaNet State Rollback Checkpointing** | **NVIDIA GB10 (CUDA)**<br/>`airawatraj:98-102` | **Apple MLX** & **llama.cpp** | Rejected speculative draft tokens permanently poison linear attention state, causing token-0 (`!`) collapse. | `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64`; maintain auxiliary recurrent state snapshots; rewind $S_t$ on draft rejection. | **100% numerical stability** across 260K context under aggressive speculative decoding. |
| **9** | **Expert-Union MoE Batching & SIMD Table Lookups** | **Laptop CPU (C)**<br/>`shyringo:840-995` | **GPU MoE Kernels** | Thread warp divergence when tokens in a warp route to different experts. | Evaluate union of routed experts ($\le 40$ of 512) in contiguous grouped prefill; AVX2 `vpshufb` register dequantization. | Linearizes memory access patterns; eliminates nested GPU thread fork-joins. |
| **10** | **Dialect-Conforming Keepalive & Cancellation Injection** | **Multi-Node SGLang**<br/>`hasso5703:403-541` | **Universal FAK Gateway** | Local models buffer tool arguments for 120s+, causing CLI agent harness timeouts. | Inject dialect-conforming SSE ping frames (`: ping\n\n` on Anthropic, empty choice frames on OpenAI) every 10s; POST `/abort_request` on client disconnect. | Zero agent harness timeouts during deep prefill/reasoning; stops zombie GPU execution. |

---

## 3. Deep Architectural Blueprints for Cross-Pollination

### Blueprint 1: MLX $\longrightarrow$ CUDA/ROCm (Multi-Tensor Slot-Streaming)

```
=== MLX DISCOVERY (carloslfu/slotstream) ==================================================
  Memory Limit: 48 GB Mac
  Problem: 125B MoE = 105 GB on disk. Cannot allocate >28.1 GB single buffer in Metal.
  Innovation: Preallocate 9-tensor slot pool (W1/W2/W3 x weight/scale/bias).
              Stream routed experts on-demand via pread QD32 (17.3 GB/s).
              Keep 3.8 GB dense trunk resident.

=== TARGET TRANSLATION (CUDA & ROCm) ====================================================
  Target Environment: Single RTX 4090 (24 GB) or AMD Radeon R9700 (32 GB).
  Architecture:
    1. VRAM Allocation:
       - Dense Trunk (Embeddings, GDN Linear Attention, Output Head): 3.8 GB (Resident)
       - Slot Pool (16 active expert slots x 48 layers): ~12.5 GB (Resident VRAM)
       - KV Cache & Activations: ~6.0 GB (Resident VRAM)
       - Total Resident VRAM: ~22.3 GB (Fits in 24 GB!)
    2. I/O Engine:
       - Linux: io_uring with O_DIRECT reading from PCIe Gen5 NVMe SSD (14 GB/s line rate).
       - Windows: DirectStorage 1.3 bypassing host CPU memory directly to GPU BAR1.
    3. Prefill Mode:
       - When prompt > 256 tokens: Switch to dense-sweep prefill, streaming layer experts
         contiguously into staging buffer to execute grouped GEMMs at 180+ tok/s.
```

### Blueprint 2: Strix Halo RoCE $\longrightarrow$ Multi-Mac Cluster (Stream-Async Direct RDMA)

```
=== AMD USB4 DISCOVERY (davidcanar/vllm-strix-halo) =======================================
  Hardware: 2x $2,000 AMD Strix Halo APUs ($4,000 cluster)
  Problem: Running 300B GLM-5.3-Flash in TP2 over a standard $30 USB4 cable.
  Innovation: Bypass RCCL for token decode steps. Use 8 µs MSI-X interrupt moderation,
              direct UMA GTT buffers, and GPU acquire-spin kernel (105 µs all-reduce).

=== TARGET TRANSLATION (Apple Silicon Multi-Mac Cluster) =================================
  Target Environment: 2x Mac Studio (M2/M3 Ultra, 128GB-192GB each) linked via Thunderbolt-4/5.
  Architecture:
    1. Transport Seam:
       - macOS IOKit Thunderbolt IP tunnel (40 Gbps PCIe tunnel).
       - Configure direct peer-to-peer 10.0.3.0/24 subnet with MTU 9000.
    2. Buffer Architecture:
       - Allocate shared receive/send slots via MTLResourceStorageModeShared.
       - Memory is accessible by both Host CPU network daemon and Apple GPU unified memory.
    3. Synchronization Kernel (Metal Shading Language):
       kernel void tb_doorbell_wait_add(
           device atomic_uint* arrival_flag [[buffer(0)]],
           device const float* recv_buffer  [[buffer(1)]],
           device float*       dest_buffer  [[buffer(2)]],
           uint tid [[thread_position_in_grid]]
       ) {
           if (tid == 0) {
               while (atomic_load_explicit(arrival_flag, memory_order_acquire) == 0) {
                   // Spin-pause on Apple Silicon threadgroup
               }
           }
           threadgroup_barrier(mem_flags::mem_device);
           dest_buffer[tid] += recv_buffer[tid];
       }
```

### Blueprint 3: NVIDIA GB10 $\longrightarrow$ Apple Silicon & AMD (HashK PLE Compression)

```
=== NVIDIA GB10 DISCOVERY (airawatraj) ===================================================
  Hardware: Grace-Blackwell GB10 (128 GB UMA)
  Problem: 51.2 GB FP8 PLE table consumes 40% of physical memory.
  Innovation: Dual SplitMix64 polynomial hashing compresses table 4x to 12.8 GB.
              Ridge projection matrix Wh asymptotically converges to Identity (bypassable!).

=== TARGET TRANSLATION (Metal & ROCm HIP) ================================================
  Algorithm:
    1. Input: N-gram token hash (bigram/trigram).
    2. Compute sub-table offsets using SplitMix64:
       uint64_t x_sub = (local_idx + 1) * 2862933555777941757ULL + SALTS[sub] + head * 998244353ULL;
       x_sub ^= (x_sub >> 31);
       x_sub *= 0x7feb352d;
       x_sub ^= (x_sub >> 31);
       uint32_t slot = x_sub % S_h;
    3. Gather 80-dim sub-table slice directly into register fragments.
    4. Concatenate dims 0:80 and 80:160.
    5. Bypass matrix Wh (saving 409,600 MACs/token); pass directly to 1D depthwise conv.
```

---

## 4. Integration into the `fak` Kernel

To operationalize these cross-architecture discoveries, `fak` incorporates:
1. **`internal/perfscout` Translation Engine:** CLI verb `fak-dev cross-innovate` querying source/target platforms, mechanics, and concrete code anchors.
2. **`internal/compute/hashk`:** Reusable Go/CUDA/Metal polynomial embedding table compressor.
3. **`internal/gateway/keepalive`:** Protocol-compliant SSE keepalive injection filter.
4. **`internal/kv/direct_io`:** Platform-neutral direct-I/O loopback block store abstraction.

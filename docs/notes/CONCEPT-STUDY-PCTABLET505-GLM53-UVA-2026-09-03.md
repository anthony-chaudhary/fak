# Concept Study: pctablet505/glm53-flash-single-gpu — 181GB GLM-5.3 on 96GB GPU via 52 GB/s Triton Gather & Grouped DMA

**Source:** https://github.com/pctablet505/glm53-flash-single-gpu  
**Pinned Revision:** `3f8a419be20d8591e1c7f0b8c4d299442a8b3017`  
**Tree SHA:** `4a1d82098b671a5c89e224e73f08149ad9c542ef`  
**Study Date:** 2026-09-03  
**Author:** PC Tablet Research Lab (`@pctablet505`)  
**License:** Apache-2.0 License (`LICENSE:1-201@3f8a419be20d8591e1c7f0b8c4d299442a8b3017`)  
**Tracking Issue:** [#10957](https://github.com/anthony-chaudhary/fak/issues/10957)  
**Parent Epics:** [#9433](https://github.com/anthony-chaudhary/fak/issues/9433) (GLM-5.3-Flash native architecture), [#10960](https://github.com/anthony-chaudhary/fak/issues/10960) (OSS performance study portfolio)  
**Study Depth:** Deep (comprehensive architecture, Triton kernel analysis, DMA staging traces, and memory allocation forensics)  
**Completeness Critic:** Verified — `kernels/triton_gather.py`, `runtime/uva_streamer.py`, `scripts/benchmark_rtx_pro6000.py`, `docs/MEMORY_PINNING.md`, and `docs/DMA_STAGING.md` inspected at `file:line@3f8a419be20d8591e1c7f0b8c4d299442a8b3017`.

---

## Executive Summary

Serving **GLM-5.3-Flash** (`glm5_next`: 45 layers, 320B total parameters, 18B active parameters, 288 routed experts top-8 + 1 shared expert, 1M context) requires 181.3 GiB of weight memory under standard NVFP4 quantization. This geometry fundamentally exceeds the physical VRAM capacity of single high-end workstation GPUs like the NVIDIA RTX PRO 6000 Ada (96 GiB) or single DGX Spark nodes without tensor parallelism.

Traditional Unified Virtual Addressing (UVA) or PCIe offloading mechanisms suffer from three catastrophic performance bottlenecks:
1. **Scattered BAR Reads (~23.4 GB/s)**: Host-to-device PCIe Gen5 transfer efficiency collapses due to fine-grained, non-coalesced memory reads across non-contiguous host memory blocks during scattered expert parameter lookups.
2. **PyTorch Caching Allocator Churn**: Allocator reservoir fragmentation on host pinned memory induces latency spikes up to 450 ms per token as the runtime searches for contiguous virtual memory spans.
3. **Prefill Standoff**: High sequence length prefill causes memory starvation, reducing effective context windows to <32k tokens.

`pctablet505/glm53-flash-single-gpu` solves these fundamental limits through three architectural innovations:
1. **Device-Side 52.1 GB/s Triton Gather Kernel**: Replaces CPU-side scatter-gather and fragmented CUDA memory copies with a vectorized Triton GPU gather kernel that reads coalesced 64-byte chunks directly across the PCIe bus via direct DMA BAR mapping.
2. **54 Hot-Expert Resident Cache**: Statically pins the 54 highest-activation routed experts (derived from profiling token frequency distributions) plus the shared expert and dense trunk in GPU VRAM (consuming ~41.2 GiB), ensuring 68.4% of expert accesses hit zero-PCIe VRAM.
3. **Grouped DMA Prefill Staging**: Groups expert weight requests across sequence blocks during prefill, issuing batched asynchronous DMA copies ahead of attention computation layers to overlap communication with linear KDA execution.

This achieves **15.0–16.5 tok/s sustained decode** and unlocks **304K verified context** on a single 96GB GPU without requiring multi-node or multi-GPU clustering.

---

## Technical Architecture & Core Mechanisms

### 1. Triton Gather Kernel (`kernels/triton_gather.py`)
Standard PyTorch tensor slice indexing over host-allocated pinned memory generates thousands of tiny PCIe transactions. The custom Triton kernel organizes memory transfers into 256-thread blocks with 4 warps per block:
- Uses `tl.load` with non-temporal cache-bypass hints (`eviction_policy="evict_first"`) to prevent polluting L2 cache during one-shot expert weight streaming.
- Coalesces 512-bit vector reads across PCIe Gen5 lanes, achieving **52.1 GB/s line rate** (81.4% of theoretical 64 GB/s PCIe Gen5 x16 bandwidth).

### 2. Hot-Expert Pinning & Dynamic Slot Rebalance
- Total experts across 42 MoE layers: $42 \times 288 = 12,096$ expert slices.
- The 54 hottest expert slots per layer tier are pinned permanently in device VRAM.
- A dynamic LRU-2 promotion ring updates cold expert slots every 1,024 generated tokens, adapting to domain shifts (e.g., code generation vs. natural language reasoning) with <2 ms rebalance overhead.

### 3. Overlapped Grouped DMA Staging
During prompt prefill:
- Layer $L+1$'s required expert parameters are prefetched via CUDA stream DMA while Layer $L$'s 3:1 KDA recurrent delta update and mHC projection execute on the primary compute stream.
- Eliminates pipeline bubbles, reducing prefill time for 128k contexts from 312 s to 89 s.

---

## Concrete Borrow for `fak` Kernel

1. **Host-Resident Expert Streaming Seam**: Incorporate Triton-style coalesced PCIe gather into `internal/compute` and `internal/model/moe_offload.go`, replacing naive slice copies.
2. **Pinned Hot-Expert Residency Ledger**: Wire a static hot-expert cache budget into `GLM5NextMoEMLPParams` to support single-GPU workstation inference envelopes.
3. **Pipeline Overlap**: Stage upcoming layer weights in a background DMA goroutine during KDA recurrent state progression.

---

## Status and Verification
- **Receipt Witness**: `docs/notes/CONCEPT-STUDY-PCTABLET505-GLM53-UVA-2026-09-03.md`
- **Registry Entry**: Added to `docs/research/monitored-repositories.json` under `pctablet505/glm53-flash-single-gpu`.

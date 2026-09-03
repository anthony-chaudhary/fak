# Study Note: wkljohn/ds4-strix-halo-tp-odinlink — Tensor Parallelism and AMD Strix Halo APU Acceleration

**Source:** https://github.com/wkljohn/ds4-strix-halo-tp-odinlink  
**Pinned Revision:** `48e10779d4723e40a8949d0ba59c262d35379a03`  
**Study Date:** 2026-09-02  
**Study Depth:** Deep (fan-out across all subsystems via parallel subagents)  
**Durable Study Receipt:** `study_dc99f78a887e2ab60a9ccf5807650b6da85ac992d7f69d96b335360c55892811`  
**Parent Epics:** #2869 (mine DwarfStar DS4 for fak), #2236 (serving superset), #10193 (native inference performance)  

---

## Repository Overview

`wkljohn/ds4-strix-halo-tp-odinlink` is a high-performance bare-metal C99/HIP inference engine fork of Salvatore Sanfilippo's (`antirez`) `ds4`. It is tailored specifically for **DeepSeek-V4 Flash 0731** (284B total / 13B active MoE with 256 routed experts across 43 layers) and **GLM-5.3 Flash** (46 layers with a hybrid schedule of 34 linear KDA and 11 sparse MLA layers) running on a **symmetric dual-node AMD Strix Halo APU cluster**.

### Target Hardware Environment
- **Compute:** 2 × AMD Ryzen AI MAX+ 395 (RDNA 3.5 architecture, `gfx1151`, Radeon 8060S with 40 Compute Units @ 2.9 GHz).
- **Memory:** 128 GiB unified physical LPDDR5X-8000 per node (256-bit bus, ~240.5 GB/s measured streaming read bandwidth). Fixed UEFI firmware carveout allocates **96 GiB to VRAM** and ~30 GiB to the host OS.
- **Interconnect:** Point-to-point USB4 / Thunderbolt 5 cable running **OdinLink** (`odl_tb5.ko` kernel module + `libodl_tb5_verbs.so` userspace InfiniBand verbs provider) or 100GbE Mellanox ConnectX-4 Lx RoCE v2 (`mlx5_0`).
- **Topology:** 2-node Tensor Parallelism ($TP=2$) with symmetric 128/128 expert sharding and group-sliced attention.

### Measured Performance
All benchmarks follow a rigorous protocol: 2,048-token prompt prefill followed by 300 forced greedy generated tokens (`--temp 0`), context length 4,096, cache-free packed GGUF weights, 3 independent warm runs (reporting medians), with exact token output fingerprints verified:
- **DeepSeek-V4 Flash Q4_K over USB4 OdinLink:** **233.04 tok/s prefill, 19.17 tok/s decode** (fingerprint `5f8a983422299d76`).
- **DeepSeek-V4 Flash Q4_K over RoCE v2 (mlx5):** **280.58 tok/s prefill, 21.30 tok/s decode** (46.95 ms/token decode; 76.5% to 84.8% of theoretical 25.1 t/s memory bandwidth roofline).
- **DeepSeek-V4 Flash Q2_K over RoCE v2:** **209.60 tok/s prefill, 20.09 tok/s decode**.
- **GLM-5.3 Flash Q4_K over RoCE v2:** **77.31 tok/s prefill, 9.78 tok/s decode** (34 KDA + 11 MLA layers, fingerprint `e68685daa5f4d385`).
- **GLM-5.3 Flash Q4_K over OdinLink:** **76.54 tok/s prefill, 9.50 tok/s decode**.

---

## Fan-Out Coverage (Completeness Critic)

| Subsystem | Coverage | Inspected Load-Bearing Files | Findings & Architecture |
|---|---|---|---|
| **Interconnect & Distributed TP** | ✅ Full | `ds4_tp.c`, `ds4_tp.h`, `ds4_distributed.c`, `ds4_distributed.h`, `ODINLINK.md`, `patches/odinlink/*` | OdinLink IBVerbs over USB4 ring DMA; RC queue pairs; 16-slot lookahead receive window; dual-stream gate signaling; greedy top-2 logit exchange; CQ overflow fix (`cqe + 1`). |
| **AMD Strix Halo & ROCm Acceleration** | ✅ Full | `STRIXHALO.md`, `ds4_rocm.cu`, `ds4_rocm.h`, `rocm/*`, `ds4_gpu.h`, `ds4_gpu_mgpu.h` | GFX1151 RDNA 3.5 Wave32 native kernels; zero-copy host `mmap` registration; `v_wmma_i32_16x16x16_iu8` integer WMMA; Pack4 sub-wave Q8_0 DP4A vectorization; non-blocking gate streams (`g_tp_stream`). |
| **SSD Streaming, Layer Packing & Hotlist** | ✅ Full | `ds4_ssd.c`, `ds4_ssd.h`, `ds4_layer_pack.c`, `ds4_layer_pack.h`, `ds4_kvstore.c`, `ds4_kvstore.h`, `ds4_streaming_hotlist.inc` | Direct I/O (`O_DIRECT`) aligned reads; `POSIX_FADV_DONTNEED` page dropping; greedy monotonic forward layer packing; offline static frequency hotlist cache seeding; SHA-1 prefix-keyed persistent KV store with half-life eviction. |
| **Model Runtime & KDA Linear Attention** | ✅ Full | `ds4.c`, `ds4.h`, `ds4_glm5_kda.c`, `ds4_glm5_kda.h`, `ds4_glm5_next_exec.c`, `ds4_glm5_next_runtime.c`, `ds4_glm5_next_state.c` | DeepSeek-V4 CSA/HCA/mHC geometry; GLM-5.3 Kimi Delta Attention (KDA) with 4-tap depthwise conv1d, L2 norm, bounded asymmetric forget gate $[-5, 0]$, and Wave32 register-resident recurrent state; 4-token pooled MLA sparse indexer. |
| **Benchmarks, Git History & Evidence** | ✅ Full | `README.md`, `CONTRACT.md`, `PROJECT.md`, `evidence/*`, `speed-bench/*`, `tests/*`, git log history (842 commits) | Root cause diagnoses: weight-masking OOM vs range-based sharding; null-stream GPU deadlock; 200 MB/s write-combining CPU read stall; DSpark speculative decoding failure & pivot; cache-free packed GGUF invariant. |

**Completeness Critic Verdict:** Nothing material left unopened. All load-bearing files, ROCm headers, transport layers, test suites, and diagnostic postmortems were analyzed at `path:line@48e10779d4723e40a8949d0ba59c262d35379a03`.

---

## Candidate Borrows Table

| # | Borrow (one technique) | Source `path:line@sha` | Axis Optimized | Their Worldview Reason | Witness on fak (PRESENT/PARTIAL/ABSENT/DIVERGENT) | Inspire/Integrate | Filed Issue |
|---|---|---|---|---|---|---|---|
| 1 | **Contiguous selection range-based expert sharding (`weights_model_map_sharded_spans`)** | `ds4.c:7068-7108@48e10779d4723`, `ds4_rocm.cu:1970-2039@48e10779d4723` | MoE resident VRAM footprint & zero-pagein TP | 256-expert Q4_K model is 153.3 GiB; Strix Halo has a 96 GiB carveout. Sharding by contiguous expert index `[lo, lo + count)` maps only 128 experts (80.76 GiB) resident per node, dropping VRAM from 95.52 GiB to 85.65 GiB and eliminating page-in stalls. | **PARTIAL** — `internal/deepseekv4moe` models expert-parallel partition as a disjoint cover in synthetic fixtures, but `fak` lacks memory-resident contiguous range slicing and remapping in model loaders and compute dispatch. | **INSPIRE** (MIT license; independent Go/compute design) | #10755 |
| 2 | **Greedy top-2 logit reduction for tensor-parallel vocab projection (`DS4_TP_FEATURE_GREEDY_TOP2`)** | `ds4_tp.h:57-59@48e10779d4723`, `ds4_tp.c:3000-3012@48e10779d4723`, `ds4.c:65237-65260@48e10779d4723` | Network wire volume & serialization latency | Autoregressive decode executes 86 gates/token. In greedy mode, worker rank evaluates argmax across its 64k vocab half and sends only top-2 candidates (24 bytes) instead of 256 KiB FP32 half-vocab, slashing wire volume by 99.99% and saving ~5 ms/token. | **ABSENT** — `internal/compute/cuda_collective.go` and `internal/compute/collective.go` implement standard AllReduceSum and AllGather. No candidate-filtered / top-k vocabulary collective exists for tensor-parallel greedy decode. | **INSPIRE** (MIT license; implement candidate reduction in `internal/compute/collective.go`) | #10756 |
| 3 | **Direct slab zero-copy RDMA prefill exchange (`DS4_TP_BIG_DIRECT=1`)** | `ds4_tp.c:830-892,1930-1936@48e10779d4723` | UMA Write-Combined staging copy overhead | On AMD APUs, CPU `memcpy` over write-combining device memory runs at ~200 MB/s, eating 64% of prefill exchange time. Pre-allocating prefill buffers directly within the IBVerbs registered memory slab enables zero-copy RDMA straight from device buffers, speeding up big-gate 2.7× (+28.5% to +44.7% prefill). | **PARTIAL** — `internal/compute/cuda_collective.go:52-75` has `UploadRank` and `internal/compute/farmem.go` has device copy routines, but `fak` has no direct RDMA slab pre-registration or write-combining bypass for UMA architectures. | **INSPIRE** (MIT license; design pattern for UMA-aware collective memory) | #10758 |
| 4 | **AVX-512 non-temporal streaming copy for write-combined host memory (`ODL_VERBS_WC_STREAM_COPY`)** | `ODINLINK.md:32-38@48e10779d4723`, `ds4_tp.c:951-953@48e10779d4723` | Staging read throughput from GPU memory | When CPU must read from write-combining GPU memory on Strix Halo APUs, standard cached reads collapse to 200 MB/s. Non-temporal AVX-512 streaming loads (`_mm512_stream_load_si512` / `vmovntdqa`) read at bus line rate (>10 GB/s), boosting prefill by +44.7% and decode by +12.8%. | **ABSENT** — `internal/compute/devicecopy.go` uses standard runtime memory copies. No non-temporal streaming load primitives exist for write-combined memory regions. | **INSPIRE** (portable Go/assembly leaf in `internal/compute`) | #10759 |
| 5 | **Wave32 register-resident recurrent state kernel for linear delta attention (KDA)** | `rocm/ds4_rocm_glm5_kda.cuh:45-102@48e10779d4723` | Latency per token & memory traffic for linear recurrence | RDNA 3.5 APUs (`gfx1151`) execute 32-thread wavefronts. Mapping the 128-dim head state across 4 Wave32 wavefronts and holding the $128 \times 128$ state in VGPRs across the sequence loop eliminates writing and reading recurrent state to DRAM at every token step. Intra-wave reductions use `__shfl_down`. | **PARTIAL** — `internal/compute/cuda_qwen35_gdn.go` and `vulkan_qwen35_gdn.go` implement gated recurrent states, but persist state in global memory / shared memory, not Wave32 VGPR-resident persistent state. | **INSPIRE** (MIT license; kernel structure for RDNA 3.5 / Wave32 architectures) | #10760 |
| 6 | **Bounded asymmetric exponential forget gate for recurrent linear attention** | `rocm/ds4_rocm_glm5_kda.cuh:145-160@48e10779d4723` | Numerical stability in recurrent linear attention | Restricting the channel-wise forget gate to $[-5.0, 0.0]$ via $-5.0 \cdot \sigma(\cdot)$ bounds the decay factor $\exp(\text{forget})$ strictly in $[e^{-5} \approx 0.0067, 1.0]$, preventing floating-point overflow or catastrophic state explosion over long sequences while retaining decay capacity. | **PARTIAL** — `internal/compute/vulkan_qwen35_gdn.go` implements gated recurrence, but lacks the bounded asymmetric $[-5.0, 0.0]$ decay parametrization. | **INSPIRE** (math/kernel adaptation in `internal/compute`) | #10761 |
| 7 | **Token-density and half-life exponential decay scoring for prompt-cache eviction** | `ds4_kvstore.c:532-607@48e10779d4723`, `ds4_kvstore.h:36-58@48e10779d4723` | TTFT across sessions & disk cache retention efficiency | Persistent `.kv` storage keyed by prompt prefix SHA-1 hashes stores 2-bit/4-bit compressed latents. The eviction formula $(\text{Hits} \cdot 2^{-\Delta t / 6\text{h}} + 1) \cdot \frac{\text{Tokens}}{\text{Bytes}} \cdot \text{AnchorMultiplier}$ prioritizes high token-density entries and user-turn anchors while cleanly pruning superseded continued waypoints. | **PARTIAL** — `internal/radixkv` and `internal/cachesweep` have in-memory prompt-cache eviction, and `internal/l3kv` routes span content, but neither implements the exact token-density half-life decay score with anchor weighting. | **INSPIRE** (adapt scoring function into `internal/cachesweep`) | #10762 |
| 8 | **Direct I/O NVMe weight streaming with post-upload page eviction (`FADV_DONTNEED`)** | `ds4_cuda.cu:1507-1626,3773-3789@48e10779d4723` | Host memory footprint and OOM prevention during model streaming | On UMA APUs where host DRAM is shared with GPU, streaming 150+ GiB of weights via buffered file I/O causes the Linux page cache to exhaust physical RAM and trigger the OOM killer. Reading via `O_DIRECT` and aggressively dropping pages with `posix_fadvise(POSIX_FADV_DONTNEED)` keeps DRAM clean for KV cache and execution scratch. | **PARTIAL** — `internal/compute/disk_unix.go` handles file loading, but uses standard `mmap`/page-cached reads without `O_DIRECT` or aggressive page eviction. | **INSPIRE** (adapt into `internal/compute/disk_unix.go`) | #10763 |

---

## Worldview Findings (Design-Level)

| Finding | Design Evidence | fak Implication |
|---|---|---|
| **UMA APUs turn CPU reads into a hidden latency trap** | `docs/BIG-GATE-BOTTLENECK.md:12-25` — CPU reads of `hipMalloc` device memory crawl at ~200 MB/s because they are mapped as Write-Combining (WC) uncached. | Memory transfers between host and device on APUs cannot assume symmetrical bus speed; direct slab placement or non-temporal streaming reads are required. |
| **True Tensor Parallelism beats Pipeline Parallelism on unified APUs** | `README.md:19-28`, `docs/ROOFLINE-AND-STRATEGY.md:45-72` — llama.cpp pipeline parallelism (`-ts`) leaves one APU idle 50% of the time (9.4 t/s); TP=2 computes concurrently across both APUs, doubling effective bandwidth to ~448 GB/s (21.30 t/s). | For local multi-APU clusters, prioritize tensor parallelism over naive pipeline splitting. |
| **Cache-free packed GGUF beats dequantized FP16 caches on memory-bound workloads** | `PROJECT.md:3-5`, `docs/Q8-F16-CACHE-POLICY.md` — Caching expanded FP16 weights wastes ~10 GiB VRAM per rank without improving decode speed, because decode is strictly memory-bandwidth bound (reading 16-bit weights takes 2× longer than 8-bit). | Keep weights in packed quantized format; dequantize tile-locally in registers/LDS. |
| **Deterministic floating-point operand ordering across ranks is mandatory** | `ds4.c:31011-31014`, `docs/BIG-GATE-BOTTLENECK.md:87-104` — Because floating-point addition is non-associative, computing `local + peer` on Rank 0 and `local + peer` on Rank 1 causes numerical drift after 43 layers. Strict rank ordering (`first = (rank == 0) ? local : peer`) is mandatory. | Collective addition across ranks must preserve identical operand order to prevent silent argmax token divergence. |
| **Speculative drafter-seed fidelity matters more than anchor elimination** | `docs/DSPARK-17TPS-RESEARCH-2026-08-07.md` — Eliminating the anchor target forward saved 70 ms, but caused target-verifier hidden state divergence. Draft acceptance collapsed from 85.7% to 72.0%, leading to net throughput regression. | Do not drop anchor forward passes in speculative loops unless verification tolerance is mathematically proven. |

---

## License & Provenance

- **Base Implementation:** MIT License (`LICENSE:1-22`, Copyright (c) 2026 The ds4.c authors, Copyright (c) 2023-2026 The ggml authors).
- **License Compatibility:** MIT is fully compatible with Apache-2.0. Permitted for direct port or adaptation with attribution.
- **Patches & Submodules:** Includes `patches/odinlink/cq-dynamic-ring.patch` (Apache-2.0/GPLv2 dual compatible).
- **All borrowed techniques:** Handled as **INSPIRE** (independent Go/compute implementation following clean-room principles with source citations).

---

## Dismissals (Earned by Ablation & Negative Knowledge)

| Technique / Hypothesis | Failure Mechanism | Why Dismissed |
|---|---|---|
| **Weight Masking for MoE Sharding (`ds4_tp_mask_weights_kernel`)** | Running an all-expert kernel and zeroing unowned weights required all 256 experts to be addressable. This triggered out-of-bounds page-ins from disk, leaking 95.52 GiB VRAM and triggering OOM crashes (`docs/EXPERT-SHARD-DESIGN-FLAW.md`). | Selection range sharding (`[base, base + count)`) strictly dominates. |
| **Stream-Memory Operations on the NULL Stream** | On ROCm 7.2 / `gfx1151`, `hipStreamWriteValue64` enqueued on stream 0 silently fails to signal arrivals, pinning the GPU at 100% in an infinite deadlock (`docs/NULL-STREAM-ROOT-CAUSE.md`). | Dedicated non-blocking streams (`g_tp_stream`) are strictly required. |
| **Producer-Ready Attention Micro-Chunking** | Splitting attention output into two 8 KiB chunks to overlap compute with RDMA doubled network message count (28k to 53k). Interconnect messaging overhead exceeded compute overlap, regressing decode speed. | Avoid micro-chunking over USB4/Thunderbolt interconnects; minimize total message round-trips. |
| **Persistent Q8-to-F16 Dequantization Caching** | Caching dequantized F16 weights consumed ~10 GiB VRAM per rank without improving decode throughput, because single-token decode is strictly memory-bandwidth bound. | Maintain weights in packed quantized format; dequantize tile-locally in registers. |
| **Single-Slot Scratch Allocators (`cuda_tmp_alloc`)** | A global scratch allocator that reallocates on size changes caused silent memory corruption when re-entered by nested subroutines (`docs/NULL-STREAM-ROOT-CAUSE.md:41-58`). | Scratch buffers for concurrently live tensors must have dedicated, non-aliasing allocations. |

---

## Filed Issues

The following 8 issues were filed as bounded, independently shippable leaves under parent epic #2869:

1. **#10755** — `feat(moe): add contiguous selection range-based expert sharding for memory-constrained multi-node TP (ds4 borrow)` (Parent: #2869)
2. **#10756** — `feat(compute): add candidate-filtered top-2 logit reduction for tensor-parallel greedy decode (ds4 borrow)` (Parent: #2869)
3. **#10758** — `feat(compute): add direct registered slab memory layout for zero-copy UMA collective transfers (ds4 borrow)` (Parent: #2869)
4. **#10759** — `feat(compute): add AVX-512 non-temporal streaming copy for write-combined APU memory staging (ds4 borrow)` (Parent: #2869)
5. **#10760** — `feat(compute): add Wave32 register-resident recurrent state kernel for linear delta attention (ds4 borrow)` (Parent: #2869)
6. **#10761** — `feat(compute): add bounded asymmetric exponential forget gate to recurrent attention kernels (ds4 borrow)` (Parent: #2869)
7. **#10762** — `feat(cachesweep): add token-density and half-life exponential decay scoring to prompt-cache eviction (ds4 borrow)` (Parent: #2869)
8. **#10763** — `feat(compute): add Direct I/O and post-upload page cache eviction for large-model weight streaming (ds4 borrow)` (Parent: #2869)

---

## Companions

- **Parent Epic:** #2869 (`Epic: mine the DwarfStar (DS4) inference engine for fak`)
- **Related Epics:** #2236 (serving superset), #10193 (native inference performance), #6221 (quantization interoperability)
- **Study Receipt:** `study_dc99f78a887e2ab60a9ccf5807650b6da85ac992d7f69d96b335360c55892811`
- **Upstream Source:** `https://github.com/wkljohn/ds4-strix-halo-tp-odinlink` @ `48e10779d4723e40a8949d0ba59c262d35379a03`

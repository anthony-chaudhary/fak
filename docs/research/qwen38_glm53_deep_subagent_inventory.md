# Comprehensive OSS Performance Deep Inventory: Qwen 3.8 Flash & GLM 5.3 Flash

**Date of Record:** September 3, 2026  
**Subject:** Empirical performance, micro-architectural optimizations, failure forensics, and systems traps across 23 fresh open-source repositories benchmarking and deploying frontier hybrid architectures (**Qwen 3.8 Flash / Flash-Next 125B–180B MoE**, **Qwen 3.8 27B**, and **GLM 5.3 Flash 300B–320B MoE**).

---

## 1. Executive Synthesis & Architectural Landscape

The arrival of frontier open-weight models featuring complex hybrid architectures:
- **Qwen 3.8 Flash-Next:** 125B MoE backbone (6B active, 512 experts, top-10 routed + 1 shared), 51B Per-Layer Embedding (PLE) n-gram table, Gated DeltaNet linear attention + Qwen Sparse Attention (QSA), Hyper-Connection residual mixing, and Multi-Token Prediction (MTP).
- **GLM 5.3 Flash:** 300B–320B MoE backbone (18B active, 288 routed experts + shared), Gated DeltaNet (34 layers) + DeepSeek Sparse Attention (DSA / NoPE MLA, 11 layers), Manifold Hyper-Connections (mHC, 4 parallel flows), and native MTP $k=2$.

These models have triggered an unprecedented wave of hyper-specialized open-source performance tooling across diverse silicon tiers:
1. **NVIDIA Grace-Blackwell GB10 (DGX Spark, 128 GB UMA, `sm_121`):** High unified memory bandwidth (~273 GB/s peak), but constrained by host-GPU shared memory limits and unoptimized SM100 kernel defaults.
2. **NVIDIA GeForce RTX 5090 (Blackwell Consumer, 32 GB GDDR7, `sm_120`):** Extreme raw decode bandwidth (>300 tok/s with DFlash2), but constrained by PCIe Gen5 bus width and desktop memory limits.
3. **AMD Strix Halo (Ryzen AI MAX+ 395, 128 GB UMA, `gfx1151`):** Integrated Radeon 8060S running Vulkan/RADV or ROCm, requiring watchdog bypasses (`amdgpu.lockup_timeout=-1`) to unlock 262K context.
4. **Commodity Interconnects:** Dual AMD mini-PCs clustering over **Thunderbolt-4 / USB4 RoCE-RDMA** with 105 µs all-reduce latency, and 4× DGX Spark nodes running switchless direct-cabled RoCE rings.
5. **Extreme Low-Resource & Host Execution:** Native zero-dependency C implementations achieving 5.03 tok/s on an Intel Core i5 laptop CPU, and SSD expert streaming via Apple MLX sustaining 12+ tok/s on 48 GB MacBooks.

---

## 2. Master Comparison Matrix (23 Deep-Inventoried Repositories)

| Repository | Target Model | Hardware Platform | Serving Engine | Precision / Quant | Speculative Mode | Peak Measured Decode | Context Window | Key Architectural Takeaway |
|---|---|---|---|---|---|---|---|---|
| **`hasso5703/dgx-spark-qwen38`** | Qwen3.8-27B & Flash-Next 176B | 1× DGX Spark (GB10) | SGLang (patched) | NVFP4 + FP8 KV | DFlash2 (block 8) / NEXTN | **50.0 tok/s** (27B); **34–42 tok/s** (Flash) | 1,010,000 (27B YaRN); 128K (Flash) | NVMe mmap for 51B PLE; kills FlashInfer autotune lottery; SSE keepalive proxy. |
| **`adrienbrault/qwen3.8-27b-rtx5090`** | Qwen3.8-27B | 2× RTX 5090 (64 GB GDDR7) | vLLM v0.28.0 | RedHat NVFP4 + FP8 KV | DFlash2 (9 tokens) | **318.8 tok/s** (code); **298.9 tok/s** steady | 262,144 (654K–892K pool) | Linear V-scale store overlay on sm120; XQA decode; 200 GB direct-I/O disk KV tier. |
| **`albond/SingleSpark-Qwen3.8-Flash-Next`** | Qwen3.8-Flash-Next (180B/6B) | 1× DGX Spark (GB10) | vLLM (dev build) | NVFP4 + Block-FP8 hybrid | Product-Quantized (PQ) MTP | **43.34 tok/s** median | 65,536 (refuses 262K) | Hardware instruction profiling: 6.50% issue slot utilization proves memory roofline bound. |
| **`airawatraj/dgx-spark-qwen38-flash-agent`** | Qwen3.8-Flash-Next (176B/6B) | 1× DGX Spark (GB10) | SGLang (patched) | NVFP4 + HashK GPU PLE | NEXTN (3-step, MTP BF16) | **36.8 tok/s** (code); **41.8 tok/s** JSON | 262,144 (260K verified) | HashK 4× PLE compression (12.8 GB VRAM); zero SSD I/O; Mamba state rollback checkpointing. |
| **`cglab-public/dgx-spark-flashnext`** | Qwen3.8-Flash-Next (176B/6B) | 2× DGX Spark (GB10, TP2) | SGLang (PR #36497) | NVFP4 + BF16 KV | NEXTN (3-step) | **43.2–57.5 tok/s** | 262,144 | Forensics of token-0 (`!`) collapse; mapped QSA indexer transient memory spikes. |
| **`MindLab-Research/ferrite`** | GLM-5.3-Flash | 1× B300 (sm_100a) | Ferrite (Rust native) | BF16/FP8 (F32 ref) | Monomorphic Static Plan | Golden Diff Oracle | 1,048,576 (1M) | Static PDAF disaggregation; exact MHC with Sinkhorn normalization; WYF parallel chunking. |
| **`gitcommit90/glm-5.3-one-spark`** | GLM-5.3-Flash | 1× DGX Spark (GB10) | vLLM + ExLlamaV3 | EXL3 2.05 bpw + FP8 KV | DFlash2 K7 | **64.05 tok/s** (JSON); **25.06 tok/s** prose | 262,144 | Cut cold boot by 76% (from 14 min to 3m21s) via DNS loopback and anonymous mmap staging. |
| **`vcruz305/GLM-5.3-Flash-EXL3-K2`** | GLM-5.3-Flash | 1× DGX Spark (GB10) | vLLM (v0.3.1 native) | EXL3 K2/K4 + FP8 KV | Native MTP $k=2$ | **27.62 tok/s** code (+85.6% vs base) | 258,048 verified | Discovered & fixed K-pool tail slot mapping out-of-bounds bug; super fat GEMM scatter. |
| **`marksunner/dgx-spark-glm52`** | GLM 5.2 $\to$ 5.3 (753B MoE) | 4× DGX Spark (MikroTik QSFP) | vLLM + Triton | QuantTrio Int4/Int8 Mix | MTP $k=4$ | **~23–26 tok/s** | 200,000 | Resolved FlashInfer mbarrier livelock via Triton sparse MLA; complete RoCE fabric blueprint. |
| **`punkjazz-labs/glm-5.3-flash-exl3-4x`** | GLM-5.3-Flash (320B/18B) | 4× DGX Spark (TP4) | vLLM + ExLlamaV3 | EXL3 4 bpw + FP8 KV | DFlash2 (off in soak) | **37 tok/s** single; **132 tok/s** @ c8 | 1,000,000 | Autoresearch parameter sweep (24 runs); fixed 330s prefill starvation by turning off mixed chunking. |
| **`alexellis/glm-5.3-flash-4x-switchless`** | GLM-5.3-Flash (320B/18B) | 4× DGX Spark (Switchless Ring) | vLLM + Marlin MoE | NVFP4 + BF16 KV | DFlash2 K7 | **~45 tok/s** typical; **71 tok/s** peak | 122,000 | Switchless RoCE ring with patched NCCL `skip-tree-connect`; 58:1 prompt-to-completion token ratio. |
| **`Dyluhn/R9V`** | Qwen3.8-Flash-Next & Muse 30B | Dual AMD R9700 (RDNA4 64GB) | Adapted vLLM + HIP | UD-IQ4_XS + block-FP8 MTP | MTP2 / DFlash2 | **78.11 tok/s** (Qwen); **59.65 tok/s** (Muse) | 64,000+ | Asymmetric PCIe Gen5 x16 + Gen4 x4 handling; graph-safe LRU16 expert cache on secondary rank. |
| **`Gr33n93/llama.cpp-qwen3.8-mtp`** | Qwen3.8-Flash-Next | AMD Strix Halo (8060S iGPU) | llama.cpp (b10771) | UD-IQ4_XS + Q8 MTP | MTP ($n=5$) | **32.30–38.27 tok/s** (+52.4% vs raw) | 262,144 (237K cold) | Decoupled draft ubatch size (`--spec-draft-ubatch-size 512`) to eliminate compute ring timeouts. |
| **`PieBru/Qwen-3.8-27B_Strix-Halo`** | Qwen3.8-27B | AMD Strix Halo (8060S iGPU) | strix-halo-llamacpp | UD-Q5/Q6/Q8_K_XL | DFlash2 | **17.0–21.0 tok/s** (DFlash2) | 262,144 (254K filled) | Forensics proving 137k crash is AMDGPU driver watchdog (`amdgpu.lockup_timeout=-1`); f16 KV beats q8_0. |
| **`davidcanar/vllm-strix-halo`** | GLM-5.3-Flash & DeepSeek-V4 | 2× AMD Strix Halo (TP2) | vLLM + Ray | AWQ W4A16 + FP8 KV | MTP (3 draft tokens) | Decode-bound | 131,072 | Thunderbolt-4 / USB4 RoCE-RDMA transport with custom `tbv` modules and ~105 µs all-reduce hook. |
| **`carloslfu/slotstream`** | Qwen3.8-Flash-Next (125B MoE) | Apple M5 Pro (48 GB RAM) | Swift 6 + MLX / Metal | 4-bit (group 64) | MTP (1.5 GB head) | **12.0–12.8 tok/s** | 32,768 | Streams routed experts from SSD via QD32 `pread` (17.3 GB/s); 9-tensor buffer decomposition. |
| **`kiojuvr/glm53-flash-mlx`** | GLM-5.3-Flash (300B MoE) | Apple M3 Ultra (512 GB UMA) | Python + MLX | Native FP8 E4M3 | Non-speculative | **10.86–13.26 tok/s** | 256,000 | Compact NoPE DSA cache cuts state memory 86%; 100k-token continuous soak with zero memory drift. |
| **`Azhu9701/ninfer-4090d`** | Qwen3.8-27B | 1× RTX 4090 D (48 GB GDDR6X) | NInfer standalone | 16.67 GiB INT + E8 KV | MTP (7 tokens) | **195–203 tok/s** code decode | 131,072 | 114-SM wave grid alignment; dynamic draft controller; DirectStorage 1.3 DMA cache (1.8s TTFT). |
| **`halt95/qwen38-flash-next-3090s`** | Qwen3.8-Flash-Next (125B/6B) | 4× RTX 3090 (96 GB, 220W) | vLLM 0.28.0 (TP4+EP) | W4A16 AutoRound + FP8 KV | MTP ($K=3$) | **154–169 tok/s** decode | 262,144 (260K verified) | Full CUDA graphs amortize 13ms Python host loop; in-place tensor repacking recovers 1.3 GiB/card. |
| **`shyringo/qwen3.8-flash-next-in-c`** | Qwen3.8-Flash-Next (125B/6B) | Intel Core i5-1340P Laptop | Standalone Native C | Unsloth UD-IQ1_S GGUF | Token-major | **5.03 tok/s** (0.199s TPOT) | 8,192 (up to 262K) | 8.99 GiB peak RSS; AVX2 `vpshufb` vector lookups; on-demand 51B PLE gather; zero Python/CUDA. |
| **`thadreber-web/llama.cpp-qwen38`** | Qwen3.8-Flash-Next | 1× NVIDIA GB10 (128 GB UMA) | llama.cpp fork | IQ3_M Base + turbo3 KV | MTP ($K=3$) | **41.53 tok/s** (+16.8% vs base)| 131,072 (105K depth) | Shape-aware CUDA graph cache keying stops eviction thrash; grouped RMSNorm boosts MTP to 92.2%. |
| **`feifeidu-max/Qwen3.8-FlashNext`** | Qwen3.8-Flash-Next (177B) | 2× Quadro RTX 8000 (96 GB) | ik_llama.cpp | Unsloth UD-IQ4_XS (4.25bpw) | MTP (dropped) | **40.0–40.2 tok/s** (from 7.9)| 208,896 | Hoisted 12 PLE CPU round trips to 1 batch gather (eliminating 113ms floor); wired orphaned kernels. |
| **`pctablet505/glm53-flash-single-gpu`**| GLM-5.3-Flash (320B/18B) | 1× RTX PRO 6000 (96 GB) | vLLM PR #53906 + Marlin | NVFP4 (181.3 GiB model) | MTP ($K=2$) | **14.99–16.50 tok/s** | 304,800 (1,487 tok/s PP) | Serves 181 GB model on 96 GB card via 52 GB/s Triton gather, 54 hot-expert slots, and grouped DMA. |

---

## 3. Top Seven Micro-Architectural Traps & Failure Modes

### 1. The Token-0 (`!`) Logit Collapse Hazard
* **Mechanics:** In the Qwen tokenizer, token ID 0 is `"!"`. When numerical stability breaks (NaN or Inf in layer-norm, activation underflow, or uncalibrated FP8 scales), argmax over logits evaluates to index 0.
* **Manifestation:** Systems pass short unit tests, then emit infinite loops of `!` when context exceeds ~120K tokens (`cglab-public`).
* **Root Cause:** SM100 TRTLLM-gen kernels executed on SM121 without re-tuning; or Gated DeltaNet recurrent state contamination when rejected speculative draft tokens fail to roll back recurrent states (`airawatraj`).

### 2. The K-Pool Tail Cache Out-of-Bounds Slot Mapping Bug
* **Mechanics:** In hybrid Linear Attention + Sparse Attention models (GLM-5.3), attention metadata builders frequently pass `positions=None`. The slot mapping falls back to standard paged tables indexing a 1-entry buffer `block_table[req, pos // block_size]`.
* **Manifestation:** For any position `pos >= block_size`, the kernel reads garbage memory and writes out-of-bounds, causing silent KV corruption or `Xid 13 Out Of Range Address` crashes (`vcruz305`).
* **Fix:** Mandate circular modulo indexing (`pos % KPOOL`) in-place at the kernel boundary.

### 3. The AMDGPU Watchdog False-Positive Crash
* **Mechanics:** On AMD Strix Halo APUs (`gfx1151`), deep-context prefill dispatches (>136K tokens in Vulkan, >229K in MTP) fail with `ErrorDeviceLost` or ring timeouts.
* **Root Cause:** The Linux kernel driver watchdog (`amdgpu.lockup_timeout`) defaults to ~10s and assumes a GPU hang if a single compute dispatch executes continuously without yielding (`PieBru`, `Gr33n93`).
* **Fix:** Add `amdgpu.lockup_timeout=-1` to kernel boot flags or decouple speculative draft batch size (`--spec-draft-ubatch-size 512`).

### 4. Speculative Graph Cache Key Invalidation Thrash
* **Mechanics:** `ggml-cuda` and early vLLM versions keyed CUDA graph caches solely by the memory address of the first node. Under speculative decoding, batch sizes vary every step as draft acceptances fluctuate.
* **Manifestation:** Every token generation step evicts the previous graph and triggers a full synchronous re-capture, dropping decode speed by 14–25% (`thadreber-web`).
* **Fix:** Graph caches must be keyed by the tuple `(first_node_address, graph_shape)`.

### 5. Quantization Noise Inversion on Speculative Drafting
* **Mechanics:** Traditional intuition assumes smaller quantizations are always faster.
* **Discovery:** On speculative models, quantizing from UD-Q5_K_XL to UD-Q4_K_XL degrades draft acceptance from 64.7% to 41.0%. The time spent rolling back rejected draft tokens completely erases the memory bandwidth savings (`PieBru`).
* **Companion Hazard:** Setting `repeat_penalty 1.05` desynchronizes target logits from draft predictions, destroying 23–28% of speculative decode throughput.

### 6. The 58:1 Ratio and Prefill Starvation
* **Mechanics:** Real autonomous agent workloads exhibit an empirical **58:1 prompt-to-completion token ratio** (`alexellis`).
* **Hazard:** Schedulers configured with naive prefill policies (e.g. `MIXED_PREFILL_CHUNK=skip`) refuse to prefill new requests while any long-running sequence is decoding, causing short agent tool queries to suffer **330-second TTFT delays** (`punkjazz-labs`).
* **Fix:** Interleave prefill chunks into active decode steps (`MIXED_PREFILL_CHUNK=off`).

### 7. Unified Memory Allocator Ratchet
* **Mechanics:** On Grace-Blackwell GB10 and AMD APUs, GPU memory *is* host system memory. During chunked prefill, allocators repeatedly request slightly larger scratch buffers.
* **Hazard:** PyTorch caching allocators fail to release virtual addresses, triggering kernel livelocks in `folio_wait_bit_common` or host OOM panic (`vcruz305`, `gitcommit90`).
* **Fix:** Bound GPU memory allocation to $\le 75\%$ (`GPU_MEM_UTIL=0.75`), enable `expandable_segments:True`, and stage weight transfers in anonymous memory clones.

---

## 4. Actionable Blueprints for the `fak` Agent Kernel

1. **Reverse-Proxy SSE Keepalive Injection:**
   Incorporate an active heartbeat injector in `fak`'s proxy layer. When upstream local engines perform multi-minute prefill or buffer tool arguments, emit standard SSE comment frames (`: ping\n\n`) every 10s to prevent Claude Code, opencode, or Pi harnesses from dropping connections.
2. **Entropy-Based Output Tripwires:**
   Equip `fak` response filters with semantic monitors that detect degenerate repetition (e.g. $\ge 16$ consecutive `!` or `lock` tokens) and terminate the turn cleanly with a structured error rather than burning context tokens.
3. **KV Pool & Memory Admission Gate:**
   Implement strict token accounting: before admitting an agent turn, ensure total committed context remains under 85% of physical KV capacity to prevent crossing the catastrophic "KV cliff" where turnaround latencies spike by $70\times$.
4. **Speculative Decoding Sanitization:**
   Automatically strip repetition penalties and frequency penalties from speculative draft passes, enforcing clean greedy verification to maximize acceptance rates.
5. **Topology-Aware Multi-Node Fabric:**
   Adopt point-to-point USB4 RoCE-RDMA and switchless ring topologies for multi-agent clusters, bypassing high-cost enterprise network switches.

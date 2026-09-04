# Concept Study: vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe — Single-Spark 128 GB UMA Serving & CUDA Kernel Forensics

**Source:** https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe  
**Pinned Revision:** `9ced8ba911ea6a0720384c4de1875d64d5470ef1`  
**Tree SHA:** `66c00ef7dc0472a92ab5285d6a0c48a012aa69c8`  
**Companion Plugin Repository:** https://github.com/vcruz305/vllm-exl3 at `ce62185aa9f987bab856468a2d9af283a7ca0121`  
**Upstream Ancestry:** Mia's AI Lab `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (commit `4b8d3c7`), ExLlamaV3 by Turboderp (`17bc3923259ffd48aab742edd261a0ca45d55459` / v1.4.4), ZJY0516 vLLM GLM5Next PR #53906 (`878631b6079d2cf9fb80830ef9cb41b43aded098`)  
**Study Date:** 2026-09-03  
**Author:** Victor Cruz (`@vcruz305`)  
**License:** MIT License (`LICENSE:1-21@9ced8ba911ea6a0720384c4de1875d64d5470ef1`; third-party notices in `THIRD_PARTY_NOTICES.md:1-70`)  
**Tracking Issue:** [#10953](https://github.com/anthony-chaudhary/fak/issues/10953)  
**Parent Epics:** [#9433](https://github.com/anthony-chaudhary/fak/issues/9433) (GLM-5.3-Flash native architecture), [#10193](https://github.com/anthony-chaudhary/fak/issues/10193) (native inference performance)  
**Study Depth:** Deep (exhaustive fan-out across all documentation, diagnostic scripts, monkey-patches, and native CUDA runtime kernels)  
**Completeness Critic:** Verified — all documentation (`KPOOL_TAIL_BUG.md`, `IMPROVEMENTS_AND_EVIDENCE.md`, `MEASUREMENTS.md`, `SIXCAT.md`, `KLD.md`), patches (`patch_kpool_tail_positions.py`, `patch_moe_fat_expert_rows.py`, `patch_fla_i64_offsets_a.py`, `patch_fla_i64_offsets_b.py`, `patch_glm53_sm121_nope.py`, `patch_glm53_eagle3.py`, `patch_dflash2_selective_quant.py`), serving scripts (`serve_one_spark.sh`, `ratchet_replay.py`, `kda_overflow_repro.py`), and native CUDA extensions (`csrc/exl3_fat_gemm.cu`, `csrc/p2b_moe.cu`, `csrc/exl3_gemm.cu`, `csrc/exl3_gemv.cu`) inspected at exact `file:line@sha`.

---

## Executive Summary

`vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe` documents a complete, production-verified recipe for serving **GLM-5.3-Flash** (`glm5_next`: 45 layers, 34 KDA linear-attention + 11 DSA sparse-attention layers, 288 routed experts top-8 + 1 shared expert, 320B total / 18B active parameters, 1M native context) on a **single NVIDIA DGX Spark / GB10 (SM121)** workstation with ~121.7 GiB unified memory (UMA).

Prior to this work, deploying 320B-class MoE models required either multi-node clusters ($200,000+) or dual-Spark tensor-parallel setups (MiaAI-Lab TP=2). Initial single-node community attempts suffered from severe regressions: models were reported as "loopy", suffered sudden engine crashes (`CUDA illegal memory access`) around 2,200 generated tokens, and hung indefinitely during prefill beyond 163k tokens.

Cruz conducted deep systems forensics on the software/hardware boundary, demonstrating that:
1. **The "Loopy / Insane" Output Was Not Quantization Error**: It was an out-of-bounds memory corruption bug in vLLM's K-pool tail cache. For hybrid models (`mamba_hybrid.py`), `positions` was omitted from attention metadata generation (`positions=None`). Consequently, the tail cache fell back to generic paged addressing against a **1-entry block-table row**, indexing `block_table[req, pos // block_size]`. For any position $\ge \text{block\_size}$, the kernel read arbitrary memory and wrote through invalid block IDs (up to block 34,303 in a 186-block cache), corrupting neighboring layer KV caches.
2. **Native v0.3.1 EXL3 CUDA Kernels Delivered +85.6% Decode Speedup**:
   - **Super Fat GEMM with Atomic Scatter (`exl3_fat_gemm_scatter`)**: In-register $128 \times 128$ tiled prefill CUDA GEMM that unrolls Trellis dequantization, fuses Fast Hadamard Transforms, and executes race-free atomic token scattering directly into output buffers, cutting prefill down-projection latency from 400.4 µs to 195.3 µs (2.09× faster) with 1.000000 numerical parity.
   - **Native Fused MoE Decode (`p2b_fused_moe`)**: A persistent cooperative kernel grid (`cudaLaunchCooperativeKernel` with `grid.sync()`) executing a 5-phase pipeline across all active experts, dropping per-layer MoE decode latency from 497 µs to 287.8 µs (saving 8.4 ms per token across 40 MoE layers).
   - **Peak Decode Performance**: Reached **27.62 tok/s** on coding and **24.59 tok/s** average across 8 categories (up from 14.88 tok/s and 16.89 tok/s on baseline ExLlamaV3).
3. **In-Checkpoint MTP $k=2$ Defeated External Speculators**: GLM-5.3-Flash's native Multi-Token Prediction (MTP) draft heads embedded directly in the checkpoint achieved a mean acceptance length of ~2.2 (~74% pos 1, ~44% pos 2) at 17.0–22.4 tok/s. External speculative drafters like DFlash2 (a 5-layer Qwen3 sidecar) suffered from draft-slot KV collapse (reducing KV pool from 104k to 15k tokens) and fatal page-boundary faults when combined with FlashAttention.
4. **UMA Allocator Ratchet Resolved to Unlock 258K Context**:
   - The >163k prefill wedge was localized to two distinct bugs:
     - The EXL3 fused-MoE per-expert row cap (`TEMP_ROWS_FUSED` 128 $\to$ 2048), eliminating a fallback to slow un-fused Python loop reconstruction.
     - An allocator ratchet on GB10 unified memory where vLLM's `split_indexer_prefill_chunks` dynamically grew sub-chunk allocations every 2,048-token step, preventing PyTorch's caching allocator from reusing freed blocks and ratcheting reserved memory to 32 GiB, causing a kernel page-lock livelock in `folio_wait_bit_common`. Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` eliminated the ratchet (1.49 GiB reserved vs 32.26 GiB), verifying a cold **258,048-token prefill** in 427.3 s wall (~604 tok/s).
   - An independent int32 offset overflow in vLLM's vendored flash-linear-attention chunk kernels (`boh * H*V*K` overflowing $2^{31}-1$ for $T > 131,072$) was identified and patched with int64 casts.
5. **Storage Saturation via Parallel NVMe Pre-Warm**: An 8-worker thread pool reading 16 MiB blocks across 120 shards saturated the storage controller, reading 91.02 GiB into page cache in 22.29 s (4.08 GiB/s vs 1.18 GB/s baseline), cutting cold boot delay by 75%.

---

## Fan-Out Coverage & Subsystem Map

The study inspected all files across both the recipe repository and the companion kernel repository:

| Subsystem / Component | Key Files Inspected | Lines | Responsibility & Engineering Focus |
|---|---|---|---|
| **K-Pool Tail Bugfix & Bounds Detector** | `docs/KPOOL_TAIL_BUG.md`<br>`scripts/patch_kpool_tail_positions.py`<br>`scripts/patch_kpool_tail_detector.py`<br>`scripts/repro_kpool_tail_overrun.sh` | 542 | Root-cause analysis of out-of-bounds slot mapping; 2-line patch restoring `positions=` in hybrid model state; in-place persistent buffer mutation for CUDA graph stability; device-side overrun counter. |
| **Native EXL3 Prefill Kernel** | `csrc/exl3_fat_gemm.cu`<br>`csrc/exl3_fat_gemm.cuh` (in `vllm-exl3`) | 337 | $128 \times 128$ tiled prefill GEMM with in-register Trellis dequantization (`dq_dispatch<4, 1>`), shared-memory swizzling, warp-shuffled Hadamard transform, and fused atomic token scatter. |
| **Native EXL3 Decode Kernel** | `csrc/p2b_moe.cu`<br>`csrc/p2b_moe.cuh`<br>`csrc/p2b_batched.cu` (in `vllm-exl3`) | 952 | Persistent cooperative kernel (`cudaLaunchCooperativeKernel`) synchronizing across SMs with `grid.sync()`; 5-phase MoE pipeline (Input Had $\to$ Gate/Up GEMV $\to$ SwiGLU $\to$ Down GEMV $\to$ Down Had + Atomic Add). |
| **MoE Routing & Row Cap Patches** | `runtime/exl3_plugin/src/glm53_exl3_plugin/exl3.py`<br>`scripts/patch_moe_fat_expert_rows.py` | 880 | Dynamic fallback router; raising `TEMP_ROWS_FUSED` from 128 to 2048; chunk slicing for deep prefill to prevent fat-expert latency cliffs. |
| **Long Context Allocator & FLA Fixes** | `scripts/serve_one_spark.sh`<br>`scripts/ratchet_replay.py`<br>`scripts/patch_fla_i64_offsets_a.py`<br>`scripts/patch_fla_i64_offsets_b.py`<br>`scripts/kda_overflow_repro.py` | 426 | UMA allocator stabilization via `expandable_segments:True`; no-model ratchet reproducer; int64 offset casting in Flash Linear Attention chunk kernels. |
| **Speculative Decoding Engine** | `scripts/serve_one_spark.sh`<br>`scripts/bench_ladder.py`<br>`scripts/patch_glm53_eagle3.py`<br>`docs/MEASUREMENTS.md` | 860 | Native MTP $k=2$ configuration with exact CUDA graph sizes (`1, 2, 3, 4, 5, 6, 8, 12`); DFlash2 Triton draft attention integration; empirical accept-rate ladder. |
| **Evaluations & Fidelity** | `docs/IMPROVEMENTS_AND_EVIDENCE.md`<br>`docs/KLD.md`<br>`docs/SIXCAT.md`<br>`scripts/loop_bench.py` | 780 | Full-vocabulary KL divergence against BF16 teacher on 1,048,064 positions; 120-task Sixcat eval (84.17 score, 100% in math); 35-response agentic loop battery proving 0 loops. |

---

## Detailed Technical Analysis

### 1. The K-Pool Tail Bug: Out-of-Bounds Slot Mapping in Hybrid Attention

#### The Problem and Mechanical Root Cause
GLM-5.3-Flash employs a hybrid recurrent/sparse-attention design: 34 recurrent Gated DeltaNet (KDA) layers and 11 DeepSeek-style Sparse Attention (DSA) layers. The DSA layers rely on a two-tier KV cache:
1. A compressed latent key-value cache ($kv\_lora\_rank=512$).
2. A small, sliding-window circular scratch buffer known as the **K-pool tail cache**, allocated per request to hold raw, uncompressed keys for the most recent tokens.

`vllm/v1/kv_cache_interface.py` defines this buffer via `KpoolTailSpec(SlidingWindowSpec)`:
```python
class KpoolTailSpec(SlidingWindowSpec):
    def max_admission_blocks_per_request(self, ...) -> int:
        return 1
    def max_num_blocks_per_req(self, vllm_config, max_len) -> int:
        return 1
```

Because `max_num_blocks_per_req = 1`, `vllm/v1/worker/block_table.py` allocates a physical buffer with shape `(max_num_reqs, 1)`:
```python
self.block_table = self._make_buffer(
    self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
)  # Buffer row width is exactly 1 entry!
```

When computing slot mappings, `block_table.py` defaults to generic paged addressing:
```python
_COMPUTE_SLOT_MAPPING_KERNEL(
    ..., positions, self.block_table.gpu, self.block_table.gpu.stride(0),
    self.block_size, self.slot_mapping.gpu, ...
)
```
This kernel calculates the target block as:
$$\text{block\_index} = \text{block\_table}\left[\text{req},\; \left\lfloor \frac{\text{pos}}{\text{block\_size}} \right\rfloor \right]$$

Because the row width is exactly 1, for any sequence position $\text{pos} \ge \text{block\_size}$ (e.g. $\text{pos} \ge 16$ or $64$), the kernel reads **past the 1-entry row**, retrieving arbitrary memory words. The downstream kernels (`_kpool_tail_seed_kernel` during prefill and `_kpool_decode_update_batched_kernel` during decode) do not bounds-check this block index. They calculate physical memory destinations as:
$$\text{dst} = \text{block} \times \text{KPOOL} + (\text{pos} \pmod{\text{KPOOL}})$$
and write raw key tensors directly to that address.

#### Why the Fallback Triggered: The Missing Hybrid Parameter
In `vllm/v1/attention/backends/mla/indexer.py`, the correct circular modulo addressing function already existed:
```python
def compute_kpool_tail_slot_mapping(...):
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
```
However, its call site was conditionally guarded:
```python
slot_mapping = common_attn_metadata.slot_mapping   # generic paged mapping
positions = common_attn_metadata.positions
if positions is not None:
    slot_mapping = compute_kpool_tail_slot_mapping(...)
```
In vLLM's model runners:
- Plain transformers (`model_states/default.py`) called `build_attn_metadata(..., positions=input_batch.positions, ...)`.
- Hybrid models (`model_states/mamba_hybrid.py`) called `build_attn_metadata(...)` **without `positions=`**, causing `positions` to default to `None`.

Because `positions` was `None`, the one-block circular correction was silently skipped on every hybrid model!

#### The CUDA Graph Transient Address Trap
Once `positions=input_batch.positions` was passed, a second bug surfaced:
```python
out = slot_mapping.clone()  # BROKEN FOR CUDA GRAPHS
```
Returning a fresh `.clone()` created a new heap allocation each step. When running under CUDA Graph capture, the graph recorded this transient virtual address. During subsequent graph replays, the kernels dereferenced that now-freed/reused buffer, causing immediate `Xid 13 Out Of Range Address` faults.

#### The Complete Shipped Fix
In `scripts/patch_kpool_tail_positions.py:38-49, 64-70@9ced8ba911ea6a0720384c4de1875d64d5470ef1`:
```python
# 1. In vllm/v1/worker/gpu/model_states/mamba_hybrid.py:
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,  # <-- Added missing argument
            model_specific_attn_metadata=mamba_attn_metadata,

# 2. In vllm/v1/attention/backends/mla/indexer.py:
    # In place: slot_mapping is the tail group's persistent buffer.
    out = slot_mapping  # <-- Mutate in place; no transient allocation!
    if num_actual_tokens == 0:
        return out
```

#### Empirical Validation & Evidence
The author implemented an opt-in device-side detector (`scripts/patch_kpool_tail_detector.py`) that instrumented the kernel write predicate inside GPU memory. During an eager soak test (`scripts/soak.sh`) generating 12,288 tokens:
- **Before Fix**: 48 overrunning calls at 8k context, destination blocks ranging from 271 to 34,303 (against a 186-block cache), overshooting physical storage by 16.8 MB.
- **After Fix**: **57,551 decode-path tail updates with exactly 0 out of bounds**, maximum block written was 10 of 202.

---

### 2. Native v0.3.1 EXL3 CUDA Kernels

To overcome the decode and prefill latency bottlenecks of 2-bit Trellis quantization on unified memory, Cruz and Mia's AI Lab engineered custom sm_121 CUDA kernels in `vllm-exl3`.

#### A. Super Fat GEMM with In-Register Dequantization & Atomic Scatter
Located at `csrc/exl3_fat_gemm.cu:65-197@ce62185aa9f987bab856468a2d9af283a7ca0121`:

Prefill expert down-projection previously required:
1. Dequantizing EXL3 weights from trellis/MCG formats.
2. Performing dense matrix multiplication.
3. Applying Fast Hadamard Transform scaling (`svh`).
4. Executing an un-fused token gather/scatter kernel to accumulate routed expert outputs back to the token sequence buffer.

`exl3_fat_gemm_kernel` fuses all four operations into a single kernel launch:

```cpp
template <bool scatter>
__global__ __launch_bounds__(FAT_THREADS)
void exl3_fat_gemm_kernel(
    const half* __restrict__ a,
    const uint16_t* __restrict__ packed,
    float* __restrict__ out,
    const half* __restrict__ svh,
    const int64_t* __restrict__ token_idx,
    const half* __restrict__ route_weight,
    int size_m, int size_k, int size_n)
```

1. **Tile Geometry**: $M=128$, $K=16$, $N=128$. Threadblock uses 256 threads (`FAT_THREADS`), mapped to 8 warps.
2. **In-Register Trellis Dequantization**:
   ```cpp
   FragB frag_b0, frag_b1;
   const uint32_t* warp_b = reinterpret_cast<const uint32_t*>(sh_b + warp * FAT_PACKED_WORDS);
   dq_dispatch<4, 1>(warp_b, lane << 3, frag_b0, frag_b1);
   ```
   Weights are decompressed directly from packed `uint16_t` / `uint32_t` bitstreams into register fragments (`FragB`) on the fly, eliminating round-trips to high-bandwidth memory.
3. **Shared Memory Swizzling**:
   Operand A is loaded into shared memory with XOR bank-conflict elimination:
   ```cpp
   int a_dst_col8 = a_col8 ^ ((a_row >> 2) & 1);
   reinterpret_cast<int4*>(sh_a)[a_row * 2 + a_dst_col8] = a_value;
   ```
4. **Tensor Core MMA Computation**:
   ```cpp
   ptx_mma_m16n8k16(frag_a, frag_b0, frag_c[mb][0]);
   ptx_mma_m16n8k16(frag_a, frag_b1, frag_c[mb][1]);
   ```
5. **Fused Fast Hadamard Transform (`fat_had_ff_128`)**:
   Accumulator registers are transferred to shared memory, where warps perform in-place Fast Hadamard Transforms using warp shuffle instructions (`shuffle_had_f2x32`), scaled by $\text{HAD\_SCALE} = 0.088388347648f$ and channel scales (`svh`).
6. **Race-Free Atomic Scatter**:
   ```cpp
   if constexpr (scatter) {
       int64_t destination = token_idx[source_row];
       value *= __half2float(route_weight[source_row]);
       out[destination * size_n + n_base + col_out] += value;
   }
   ```
   Because only one route per token reaches a given expert on the active stream, this accumulation directly writes the final down-projected activation back into global memory without requiring a separate scatter kernel.

**Result**: Prefill down-projection latency dropped from 400.4 µs to 195.3 µs (**2.09× faster**), achieving 7.85 TFLOPS prefill chunked GEMM (13.0× boost over baseline).

#### B. Native Fused MoE Decode (`p2b_fused_moe`)
Located at `csrc/p2b_moe.cu:357-437@ce62185aa9f987bab856468a2d9af283a7ca0121`:

Single-token decode latency is dominated by launching kernels across 8 routed experts for each of the 40 MoE layers. Launching separate Gate, Up, SwiGLU, Down, and Reduce kernels incurs massive kernel launch overhead and GPU pipeline stalls.

`p2b_fused_moe_cuda` replaces this with a single **Cooperative Kernel Grid**:
```cpp
cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream);
```
Inside `p2b_moe_batched_kernel`, all threadblocks synchronize globally via `grid.sync()` across 5 execution phases:
- **Phase 1**: In-place input Hadamard transform.
- **Phase 2**: Batched Gate & Up GEMV across all active experts using specialized quantized GEMV tiles (`run_gemv_tile<BITS, 1, 0>`), synchronized with `grid.sync()`.
- **Phase 3**: Elementwise SwiGLU activation ($\text{gate} \cdot \text{sigmoid}(\text{gate}) \cdot \text{up}$) followed by intermediate Hadamard transform, synchronized with `grid.sync()`.
- **Phase 4**: Batched Down GEMV across all active experts, synchronized with `grid.sync()`.
- **Phase 5**: Down output Hadamard transform and atomic weighted reduction (`atomicAdd`) directly into the sequence output accumulator.

**Result**: Decode latency across all 40 MoE layers dropped from 19.9 ms (497 µs/layer) to **11.5 ms (287.8 µs/layer)**, saving **8.4 ms per token**.

#### C. Fused MoE Per-Expert Row Cap (`TEMP_ROWS_FUSED`)
In `runtime/exl3_plugin/src/glm53_exl3_plugin/exl3.py:57, 329-345` and `scripts/patch_moe_fat_expert_rows.py:1-35`:
During long-context prefill ($>163,840$ tokens), routing skew caused certain experts to receive more than 128 rows per 2,048-token chunk. The plugin had hardcoded `TEMP_ROWS_FUSED = 128`. Any expert with $>128$ rows fell back to `apply_exl3_python_loop`, which rebuilt un-fused weights on the fly, stalling prefill for minutes per chunk. Raising `TEMP_ROWS_FUSED = 2048` ensured no expert in a 2,048-token chunk could exceed the cap, eliminating the latency cliff and allowing a cold 180,224-token prefill to return in 515 s.

---

### 3. Native MTP $k=2$ Speculative Decoding vs DFlash2

GLM-5.3-Flash embeds native Multi-Token Prediction (MTP) projection heads directly into its main model checkpoint.

#### Architectural Advantage over External Drafters
| Dimension | External Drafter (DFlash2) | Native In-Checkpoint MTP ($k=2$) |
|---|---|---|
| **Weight Footprint** | +2.18 GiB BF16 checkpoint | **0 GiB** (embedded in base weights) |
| **KV Cache Overhead** | Allocates separate draft KV pool; collapses target KV cache from 104k to 15k tokens | Shares target cache geometry |
| **Attention Backend** | Fails with FlashAttention at page transitions; requires `TRITON_ATTN` | Fully compatible with native sparse FlashInfer MLA |
| **Decode Throughput** | 12.8–15.6 tok/s | **17.02–27.62 tok/s** |
| **Mean Acceptance** | ~1.8 tokens | **~2.2 tokens** (74% pos 1, 44% pos 2) |

#### Graph Capture Optimization
In `scripts/serve_one_spark.sh:213-217`:
```bash
SPEC=$(python3 -c "import json,os; print(json.dumps({'method':'mtp','num_speculative_tokens':int(os.environ['MTP_TOKENS'])},separators=(',',':')))")
ARGS+=(--speculative-config "$SPEC")
ARGS+=(--cudagraph-capture-sizes 1 2 3 4 5 6 8 12)
```
Specifying capture sizes that include exact target verification sizes ($k+1 = 3$ for $k=2$) is critical: without capture size 3, vLLM pads decode verification steps to capture size 4 or 6, adding unneeded threadblock executions and wasting memory bandwidth.

#### Workload Sensitivity ($k=2$ vs $k=3$ vs $k=4$)
In `docs/MEASUREMENTS.md:40-56`:
- On **structured/JSON** workloads, MTP $k=4$ achieved **26.21 tok/s** with 91.3% acceptance rate.
- On **prose/reasoning** workloads, MTP $k=4$ regressed to **12.72 tok/s** (acceptance dropped to 31.7%) due to draft rejection overhead.
- **MTP $k=2$** proved to be the optimal global daily-driver setting, delivering stable 16.1–20.6 tok/s across all task categories.

---

### 4. Memory Allocator Stabilization & Long-Context Scaling (258K)

#### The Unified Memory Allocator Ratchet
On discrete GPUs with dedicated VRAM, memory allocation exhaustion causes `cudaMalloc` to return `cudaErrorMemoryAllocation`, triggering PyTorch's caching allocator to search its cache, coalesce blocks, and free unused cached blocks.

On the NVIDIA DGX Spark GB10, the CPU and GPU share a single 128 GB LPDDR5X physical memory pool via NVIDIA Grace/Blackwell unified memory architecture (UMA). In this environment:
1. `cudaMalloc` calls succeed almost indefinitely by claiming virtual memory from the OS.
2. In vLLM's `split_indexer_prefill_chunks` (`v1/attention/backends/mla/indexer.py`), chunked prefill allocates an fp32 logits buffer of shape `(sub_m, N_compressed)` up to `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 512`.
3. Because the sequence prefix grows by 2,048 tokens at every chunked prefill step, each successive step requires slightly larger buffers than the previous step.
4. Because the buffer sizes change monotonically, PyTorch's caching allocator cannot reuse the freed smaller memory blocks.
5. Because `cudaMalloc` does not fail, PyTorch never triggers a cache flush (`num_alloc_retries` remains 0).
6. Reserved memory continuously ratchets upwards until system RAM is exhausted (reaching 32.26 GiB reserved in `scripts/ratchet_replay.py`). At that point, the Linux kernel enters page-lock contention inside `folio_wait_bit_common`, causing a complete system livelock while `/health` still returns HTTP 200.

#### The Fix: Expandable Segments and Pinned KV Cache
In `scripts/serve_one_spark.sh:48, 57, 206-210`:
```bash
# 1. Enable virtual memory address space reservation without physical backing:
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 2. Pin the KV cache pool to exact bytes rather than dynamic utilization:
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-3221225472}"  # 3 GiB = 349,525 fp8 tokens
```

#### Results of Allocator Stabilization
In `scripts/ratchet_replay.py:39-40`:
| Configuration | Peak Reserved Memory | Allocated Segments | Allocator Retries |
|---|---:|---:|---:|
| **Default PyTorch Allocator** | 32.26 GiB | 128 | 0 |
| **`expandable_segments:True`** | **1.49 GiB** | **0** | -- |
| **`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64`** | 1.04 GiB | 27 | -- |

**Live Verification**: Cold **258,048-token prefill** completed in **427.3 s wall** (~604 tok/s). `MemAvailable` remained completely flat between 13.82 and 13.85 GiB throughout the entire 7-minute prefill run with zero memory drift.

#### Flash-Linear-Attention int32 Offset Overflow
In `scripts/patch_fla_i64_offsets_a.py` and `scripts/kda_overflow_repro.py`:
During deep-context execution of Gated DeltaNet, vLLM's vendored `chunk_kda_with_fused_gate` calculated base memory pointers using 32-bit arithmetic:
$$\text{offset} = \text{boh} \times H \times V \times K$$
With $H=64, V=128, K=128$, the stride $H \cdot V \cdot K = 1,048,576$. For chunk indices $\text{boh} > 2,047$ (corresponding to sequence length $T > 131,072$), the product $\text{boh} \times 1,048,576 > 2,147,483,648$, overflowing signed 32-bit integer limits. Casting `boh` and `bos` to `tl.int64` eliminated the overflow for sequences up to 1M tokens.

---

## Upstream Worldview Reconstruction & Systems Tradeoffs

Understanding Victor Cruz's worldview clarifies why these techniques were engineered:
1. **The Single-Node 128 GB UMA Frontier:** Deploying a 320B MoE on a single Grace-Blackwell GB10 node demands extreme memory parsimony. Sub-3bpw EXL3 quantization fits the model, but exposes subtle kernel bugs in hybrid recurrent/sparse layers that do not manifest at FP16 or on dual-node TP2.
2. **De-Quantization Overhead Elimination:** In low-bit MoE prefill on unified memory, bandwidth to registers is the bottleneck. Decompressing weights directly into register fragments rather than staging in shared memory or global memory maximizes computational throughput.
3. **Persistent Cooperative Grids vs Python Launch Loops:** Single-batch decode latency is strictly bound by kernel launch and pipeline bubbles. Executing all routed experts in a single resident cooperative grid keeps SMs fully saturated without host Python roundtrips.
4. **UMA Allocator Differences:** Grace-Blackwell UMA hardware does not trigger allocation retries on `cudaMalloc` because host memory satisfies virtual requests. Explicit virtual expandable memory segments (`expandable_segments:True`) and static byte pinning are required to eliminate allocation ratchets and page-lock livelocks.

---

## Borrow Candidates for the fak Agent Kernel

The following 5 concrete mechanisms are extracted from `vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe` and grounded at exact source coordinates:

### Candidate 1: In-Place Circular Modulo Slot Mapping for K-Pool Tail Caches
- **Source Anchor:** `scripts/patch_kpool_tail_positions.py:38-49, 64-70@9ced8ba911ea6a0720384c4de1875d64d5470ef1` and `docs/KPOOL_TAIL_BUG.md:98-104@9ced8ba911ea6a0720384c4de1875d64d5470ef1`
- **One-Line Technique:** Prevent out-of-bounds KV buffer corruption on hybrid models by binding physical tail slot mappings to `own_block * KPOOL + (pos % KPOOL)` and mutating the persistent buffer in place.
- **Specific Axis:** Memory safety and cache addressing determinism for circular scratch buffers under CUDA graph replay.
- **Comparison with fak:** `fak` has preliminary GLM-5 sparse-attention specifications (`internal/model/arch_support.go` and `internal/modelperfobs/long_context_presets.go`), but lacks the circular modulo slot mapping fix and in-place tensor mutation required to avoid CUDA graph transient address faults (`Xid 13`). (**PARTIAL on-axis**).
- **Worldview Rationale:** vLLM and upstream PR #53906 assumed standard paged block addressing applied universally. On hybrid architectures with fixed tail scratch pools, decoupling sequence length from block table width is mandatory.
- **Disposition:** `DIRECT-PORT` into `internal/kv/` and `internal/model/`.

```python
# Provenance: vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
# File: scripts/patch_kpool_tail_positions.py:38-49,64-70@9ced8ba911ea6a0720384c4de1875d64d5470ef1
def compute_kpool_tail_slot_mapping(block_table, positions, slot_mapping, kpool, num_actual_tokens, num_reqs, req):
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    # Circular modulo addressing:
    slot_mapping[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return slot_mapping  # In-place write preserves address for CUDA Graph replay
```

### Candidate 2: Super Fat GEMM with In-Register Trellis Dequantization & Fused Atomic Token Scatter
- **Source Anchor:** `csrc/exl3_fat_gemm.cu:65-197@ce62185aa9f987bab856468a2d9af283a7ca0121` and `csrc/exl3_fat_gemm.cuh:23-32@ce62185aa9f987bab856468a2d9af283a7ca0121`
- **One-Line Technique:** Execute 2-bit/4-bit MoE prefill down-projection in a single $128 \times 128$ tiled CUDA kernel that unrolls Trellis decompression into registers, computes Tensor Core MMA, applies Fast Hadamard Transforms, and scatters directly into the output tensor.
- **Specific Axis:** Prefill down-projection arithmetic intensity and kernel launch overhead reduction for quantized MoE experts.
- **Comparison with fak:** `fak` executes MoE down-projections via decoupled GEMM and gather/scatter phases. Fusing Trellis dequantization, Hadamard scaling, and atomic token scatter cuts latency by 2.09×. (**ABSENT on-axis**).
- **Worldview Rationale:** In low-bit MoE prefill on unified memory, memory bandwidth is the primary bottleneck. Decompressing weights directly into register fragments rather than staging in shared memory or global memory maximizes computational throughput.
- **Disposition:** `ADAPT` into `internal/compute/cuda/`.

```cpp
// Provenance: vcruz305/vllm-exl3 (derived from Mia's AI Lab)
// File: csrc/exl3_fat_gemm.cu:120-136, 182-194@ce62185aa9f987bab856468a2d9af283a7ca0121
FragB frag_b0, frag_b1;
const uint32_t* warp_b = reinterpret_cast<const uint32_t*>(sh_b + warp * FAT_PACKED_WORDS);
dq_dispatch<4, 1>(warp_b, lane << 3, frag_b0, frag_b1);

#pragma unroll
for (int mb = 0; mb < FAT_M_BLOCKS; ++mb) {
    FragA frag_a;
    int row = (lane % 8) + 8 * ((lane / 8) % 2) + mb * 16;
    int base_col = lane / 16;
    int swizzled_col = base_col ^ ((row >> 2) & 1);
    ldsm4(frag_a, reinterpret_cast<int4*>(sh_a) + row * 2 + swizzled_col);
    ptx_mma_m16n8k16(frag_a, frag_b0, frag_c[mb][0]);
    ptx_mma_m16n8k16(frag_a, frag_b1, frag_c[mb][1]);
}
// Direct atomic scatter into destination activation buffer:
if constexpr (scatter) {
    int64_t destination = token_idx[source_row];
    value *= __half2float(route_weight[source_row]);
    out[destination * size_n + n_base + col_out] += value;
}
```

### Candidate 3: Cooperative Grid Pipelined MoE Decode Kernel (`p2b_fused_moe`)
- **Source Anchor:** `csrc/p2b_moe.cu:357-437@ce62185aa9f987bab856468a2d9af283a7ca0121` and `csrc/p2b_moe.cuh:1-35@ce62185aa9f987bab856468a2d9af283a7ca0121`
- **One-Line Technique:** Run the entire routed MoE decode forward pass (Gate GEMV, SwiGLU, Down GEMV, and Reduction) in a single persistent cooperative kernel synchronized with `grid.sync()`.
- **Specific Axis:** Multi-expert single-token decode latency and launch overhead elimination.
- **Comparison with fak:** `fak` launches separate kernel operations per expert or per phase. A persistent cooperative grid drops per-layer latency to 287.8 µs, saving 8.4 ms per token. (**ABSENT on-axis**).
- **Worldview Rationale:** Single-batch decode latency is strictly bound by kernel launch and pipeline bubbles. Executing all routed experts in a single resident cooperative grid keeps SMs fully utilized.
- **Disposition:** `ADAPT` into `internal/compute/cuda/`.

```cpp
// Provenance: vcruz305/vllm-exl3
// File: csrc/p2b_moe.cu:370-407@ce62185aa9f987bab856468a2d9af283a7ca0121
void* kernel = (void*) p2b_moe_batched_kernel<BITS>;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);
const int grid = std::max(1, resident * sms);

void* args[] = {
    (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gv,
    (void*)&utp, (void*)&uup, (void*)&uvp,
    (void*)&dtp, (void*)&dup, (void*)&dvp,
    (void*)&idp, (void*)&rwp,
    (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
    (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
    (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter
};
cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream);
```

### Candidate 4: Unified Memory Allocator Stabilization via Expandable Segments
- **Source Anchor:** `scripts/serve_one_spark.sh:48, 57, 206-210@9ced8ba911ea6a0720384c4de1875d64d5470ef1` and `scripts/ratchet_replay.py:1-40@9ced8ba911ea6a0720384c4de1875d64d5470ef1`
- **One-Line Technique:** Prevent virtual memory ratchets and page-lock livelocks during prefix-growing chunked prefill by enabling virtual expandable memory segments and pinning KV pools to static byte sizes.
- **Specific Axis:** Livelock elimination and memory footprint stabilization for extreme context prefill ($>200\text{k}$) on unified memory architectures.
- **Comparison with fak:** `fak`'s host-side memory management for local inference runtimes does not currently enforce expandable segment configuration or pinned KV sizing on UMA nodes (Apple Silicon / NVIDIA Grace-Blackwell). (**PARTIAL on-axis**).
- **Worldview Rationale:** UMA hardware lacks a hardware page-fault OOM trigger for caching allocators; allocators must be configured to reuse non-contiguous virtual address ranges.
- **Disposition:** `DIRECT-PORT` into runtime bootstrap scripts and `internal/platform/uma.go`.

```bash
# Provenance: vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
# File: scripts/serve_one_spark.sh:48, 57@9ced8ba911ea6a0720384c4de1875d64d5470ef1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-3221225472}"  # Pinned 3 GiB slab
```

### Candidate 5: Parallel Multi-Worker Storage Controller Pre-Warm
- **Source Anchor:** `scripts/serve_one_spark.sh:81-97@9ced8ba911ea6a0720384c4de1875d64d5470ef1`
- **One-Line Technique:** Parallelize cold safetensors shard reads across 8 POSIX worker threads using 16 MiB block reads before engine tensor allocation, saturating NVMe IOPS.
- **Specific Axis:** Cold checkpoint initialization latency on large (>90 GiB) multi-shard model weights.
- **Comparison with fak:** `fak` loads tensor files sequentially or via lazy mmap page faulting, taking ~12 minutes to fault in 91 GiB. Parallel pre-warming populates the page cache in 22 seconds (4.08 GiB/s). (**PARTIAL on-axis**).
- **Worldview Rationale:** Storage controller queues require concurrent read streams with large chunk sizes (16 MiB) to achieve wire-speed bus saturation.
- **Disposition:** `ADAPT` into `internal/safetensors/loader.go`.

```python
# Provenance: vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
# File: scripts/serve_one_spark.sh:85-95@9ced8ba911ea6a0720384c4de1875d64d5470ef1
import os, glob, concurrent.futures
shards = glob.glob(f"{MODEL_DIR}/model-*.safetensors")
def prewarm_shard(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        while os.read(fd, 16 * 1024 * 1024):
            pass
    finally:
        os.close(fd)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(prewarm_shard, shards))
```

---

## Candidate Summary Table

| Candidate | Source Location | Axis | fak Comparison | Worldview Rationale | Disposition | Target Seam |
|---|---|---|---|---|---|---|
| **1. In-Place Circular Modulo Tail Mapping** | `scripts/patch_kpool_tail_positions.py:38-49,64-70@9ced8ba9` | Memory safety & CUDA graph replay | **PARTIAL** (has paged specs, lacks circular modulo & in-place update) | Tail scratch cache is circular modulo, not paged; transient clones break graph replay | `DIRECT-PORT` | `internal/kv/`, `internal/model/` |
| **2. Super Fat GEMM with Atomic Scatter** | `csrc/exl3_fat_gemm.cu:65-197@ce62185a` | Prefill down-proj throughput | **ABSENT** (no fused Trellis dequant + Had + scatter) | Register-level decompression avoids memory round-trips; direct scatter saves kernel launch | `ADAPT` | `internal/compute/cuda/` |
| **3. Cooperative Grid Pipelined MoE Decode** | `csrc/p2b_moe.cu:357-437@ce62185a` | Single-token decode latency | **ABSENT** (separate kernel launches per expert/phase) | Resident cooperative grid eliminates launch bubbles across all 8 routed experts | `ADAPT` | `internal/compute/cuda/` |
| **4. UMA Allocator Stabilization** | `scripts/serve_one_spark.sh:48,57@9ced8ba9` | Long context (>200k) prefill stability | **PARTIAL** (lacks UMA-specific segment controls) | UMA does not trigger allocation retries on `cudaMalloc`; virtual segment expansion prevents ratchet | `DIRECT-PORT` | `internal/platform/uma.go` |
| **5. Multi-Worker Storage Pre-Warm** | `scripts/serve_one_spark.sh:81-97@9ced8ba9` | Cold model load latency | **PARTIAL** (sequential / lazy mmap faults) | NVMe controllers require parallel queue saturation (8 workers × 16 MiB) to reach wire speed | `ADAPT` | `internal/safetensors/loader.go` |

---

## License, Provenance & Attribution Disposition

- **Recipe & Scripts:** `vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe` is licensed under the **MIT License** (Copyright 2026 Victor Cruz).
- **Standalone Kernel Plugin:** `vcruz305/vllm-exl3` is licensed under the **Apache-2.0 License** (Copyright 2026 Victor Cruz).
- **Upstream MoE Kernels:** Derived from Mia's AI Lab (`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, MIT License, Copyright 2026 Mia's AI Lab and `@plotarmordev`) and ExLlamaV3 (MIT License, Copyright 2025 Turboderp).
- **Compatibility with `fak`:**
  - Both MIT and Apache-2.0 licenses are fully compatible with `fak`'s Apache-2.0 license.
  - Direct porting and adaptation are permitted.
  - Attribution notices must be preserved in `THIRD_PARTY_NOTICES.md` identifying Victor Cruz, Mia's AI Lab, plotarmordev, and Turboderp.

---

## Concrete Follow-up Implementation Tickets

- Issue: `fix(kv): in-place circular modulo slot mapping for hybrid K-pool tail scratch caches` (Candidate 1)
- Issue: `feat(compute): super fat GEMM with in-register Trellis dequantization & fused atomic token scatter` (Candidate 2)
- Issue: `feat(compute): cooperative grid pipelined MoE decode kernel (p2b_fused_moe)` (Candidate 3)
- Issue: `feat(platform): unified memory allocator stabilization via expandable segments & pinned KV slab` (Candidate 4)
- Issue: `feat(safetensors): parallel multi-worker storage controller pre-warm for cold checkpoint initialization` (Candidate 5)

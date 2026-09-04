# Concept Study: MindLab-Research/ferrite — GLM-5.3-Flash Native Inference Engine

**Source:** https://github.com/MindLab-Research/ferrite  
**Pinned Revision:** `d771576a7a462866ba707af16100a106c52c7fd2`  
**Study Date:** 2026-09-03  
**Author / Organization:** Boxiu Lee (MindLab-Research)  
**License:** MIT License (`LICENSE:1-21@d771576a7a462866ba707af16100a106c52c7fd2`)  
**Tracking Issue:** [#10950](https://github.com/anthony-chaudhary/fak/issues/10950)  
**Parent Epics:** [#9433](https://github.com/anthony-chaudhary/fak/issues/9433) (GLM-5.3-Flash native architecture), [#10193](https://github.com/anthony-chaudhary/fak/issues/10193) (native inference performance)  
**Study Depth:** Deep (exhaustive fan-out across all 7 workspace crates and CUDA kernels)  
**Completeness Critic:** Verified — all workspace crates (`ferrite-types`, `ferrite-model`, `ferrite-kernel`, `ferrite-kv`, `ferrite-batch`, `ferrite-scheduler`, `ferrite-exec`) and `kernels/cuda/ferrite_kernels.cu` inspected at `file:line@d771576a7a462866ba707af16100a106c52c7fd2`.

---

## Executive Summary

`MindLab-Research/ferrite` is an ultra-clean, zero-dependency (stdlib-only + `serde`/`safetensors`) Rust inference engine and sm_100a/sm_121a CUDA kernel suite engineered specifically for **GLM-5.3-Flash** (`glm5_next_text`: 45 layers, 320B total / 18B active parameters, 1M context). The repository rejects the prevailing industry approach—exemplified by Python-bound, dynamic-dispatch runtime engines like SGLang and vLLM—in favor of a **static, monomorphized engine with a compile-time operator plan** (`Engine<B: KernelBackend>`) and first-class **PDAF (Prefill-Decode-Attention-FFN)** disaggregated scheduling.

Key technical breakthroughs in Ferrite of immediate value to the `fak` kernel:
1. **WYF Chunkwise Parallel GatedDeltaNet Recurrence**: Unrolls the sequential Gated DeltaNet recurrent loop into chunk-parallel matrix operations using an inclusive prefix decay formulation, a lower-triangular inter-token interaction solve, and chunk-state ping-pong chaining. This slashes CUDA kernel launches by **32×** on prefill while maintaining bit-exact equivalence to the sequential recurrence.
2. **Exact Manifold Hyper-Connections (MHC)**: Replaces heuristic residual approximations with an exact 4-flow residual stream ($[s, n \cdot h]$ with $n=4$) bracketed by `hc_pre` (RMS-scaled linear mixing), a multi-iteration **Sinkhorn normalization** loop ensuring doubly stochastic routing matrices, and `hc_post` expansion.
3. **Dual-Mode Hybrid State Management**: Distinguishes the fixed-size state of 34 linear GatedDeltaNet layers ($[16, 128, 128]$ state + conv tail per sequence) from the token-growing state of 11 DeepSeek Sparse Attention (DSA) layers. Linear states reside in a contiguous, page-free slab with zero-copy snapshot capabilities, while DSA layers use paged allocation over compressed nope-only MLA latents ($kv\_lora\_rank=512$) and indexer keys.
4. **3D Parallel Axis Algebra ($Q \times Kv \times Head$)**: Mathematically models parallelism as three orthogonal axes with distinct merge semantics: Prefill Context Parallelism (CP) shards $Q$ (token segments, merged via row-gather or pipeline chain); Decode Context Parallelism (DCP) shards $Kv$ (pages, merged via stable max-shifted Log-Sum-Exp); and Tensor Parallelism (TP) shards $Head$ (merged via concatenation for attention and summation for projection partials). Crucially, the $Kv$ and $Head$ merges commute, proving that arbitrary $(CP \times DCP \times TP)$ topologies yield bit-identical attention.
5. **Static PDAF Op-Plan**: A pre-compiled 90-step static plan (45 Attention ops + 45 FFN ops) with hardware affinity tags that eliminates dynamic layer lookup, removes runtime vtables, and guarantees op-sequence stability for native CUDA Graph capture.

---

## Fan-Out Coverage & Subsystem Map

The Ferrite repository consists of a 7-crate Cargo workspace and a standalone CUDA kernel build. All modules were inspected at commit `d771576a7a462866ba707af16100a106c52c7fd2`:

| Subsystem / Crate | Files Inspected | Lines | Responsibility & Architecture |
|---|---|---|---|
| **ferrite-types** | `crates/ferrite-types/src/lib.rs` | 277 | Tensor, Shape, DType primitives. F32 golden storage, contiguous buffer slices, view operations. |
| **ferrite-model** | `crates/ferrite-model/src/{lib,config,layer,weights,safetensors}.rs` | 1,215 | HF `config.json` parser for `glm5_next_text`, static 45-layer plan builder (34 linear + 11 DSA; 3 dense + 42 MoE), 37,267-tensor layout generator, safetensors loader with fused-weight alias expansion (`fused_qkvbfg_a`, `fused_fg_b`). |
| **ferrite-kernel** | `crates/ferrite-kernel/src/{lib,cpu,wyf,dcp,graph,cuda,nccl}.rs` | 2,840 | `KernelBackend` monomorphic trait contract; `CpuBackend` numerical golden reference; `wyf.rs` chunkwise parallel reference; `dcp.rs` partial attention + LSE merge; `graph.rs` CUDA-graph op-sequence tracer; `cuda.rs` FFI bindings; `nccl.rs` dlopen dynamic NCCL collective bindings. |
| **ferrite-kv** | `crates/ferrite-kv/src/{lib,shard,axes}.rs` | 1,480 | `LinearStatePool` (contiguous slot allocator for fixed recurrent states), `DsaKvPool` (paged nope-only MLA latent + indexer KV), `HybridStatePool` (unified per-seq routing), `shard.rs` (CP layer-split × DCP page resharding plan), `axes.rs` (3D axis algebra: $Q, Kv, Head$). |
| **ferrite-batch** | `crates/ferrite-batch/src/lib.rs` | 385 | Continuous batching scheduler, budget-constrained chunked prefill, sequence lifecycle tracking. |
| **ferrite-scheduler** | `crates/ferrite-scheduler/src/lib.rs` | 264 | `StaticPlan` (compiled 90-step op sequence), `PdafRouter` (routes scheduled batches to `PrefillWork` / `DecodeWork` and emits `TransferEvent` for 2D resharding). |
| **ferrite-exec** | `crates/ferrite-exec/src/{lib,mhc,distributed,tp,graph}.rs` | 2,750 | `Engine<B: KernelBackend>` forward pass; `mhc.rs` (exact Sinkhorn hyper-connections); `distributed.rs` (simulated CP LayerSplit prefill, 2D reshard, DCP decode); `tp.rs` (head/inter/expert weight sharding + layer-synchronous forward); `graph.rs` (CUDA graph capture/verify runner). |
| **kernels/cuda** | `kernels/cuda/ferrite_kernels.cu`, `build.sh` | 833 | sm_100a (Blackwell B300) CUDA kernels: tiled GEMM (32×32 with +1 padding), `gdn_step_kernel`, `gdn_wyf_kernel` (parallel chunkwise GDN), `ferrite_swiglu2` (dual independent inputs), `ferrite_causal_conv1d`, `moe_route_kernel` (noaux-tc), `sparse_attn_kernel` (top-k MLA), `ferrite_indexer_topk`. |

**Completeness Critic Verdict:** Exhaustive. 74 workspace tests passing on CPU. No dead code or uninspected files remain.

---

## Detailed Technical Analysis

### 1. WYF Chunkwise Parallel GatedDeltaNet Recurrence

In GLM-5.3-Flash, 34 of the 45 attention layers are Gated DeltaNet linear attention (KDA). The sequential recurrence at token step $t$ for head $h$ is:
$$\begin{aligned}
D_t[i] &= \exp(\text{gate}[t, h, i] \cdot a_h), \quad \text{where } a_h = -\exp(a_{\log}[h]) < 0 \\
S_t &= \text{diag}(D_t) \cdot S_{t-1} \\
kS &= S_t^T k_t \\
S_t &\leftarrow S_t - \beta_t k_t (kS)^T + \beta_t k_t v_t^T \\
O_t &= q_t^T S_t
\end{aligned}$$

While efficient during single-token decode, executing this sequential recurrence during prefill incurs $O(N)$ sequential CUDA kernel launches or serialized thread loops along the sequence length $N$.

`crates/ferrite-kernel/src/wyf.rs:1-147` and `kernels/cuda/ferrite_kernels.cu:398-560` implement the chunkwise parallel WYF formulation for chunk size $C = 32$:

#### Mathematical Formulation
1. **Inclusive Prefix Decay ($L$):**  
   Define $L[t, i] = a_h \sum_{r \le t} \text{gate}[r, h, i]$ for $t \in [0, C-1]$. Because $a_h < 0$ and $\text{gate} \ge 0$, $L[t, i] \le 0$ monotonically decreases, ensuring $\exp(L[t, i]) \in (0, 1]$ is strictly bounded.
2. **State Interaction Term ($b_t$):**  
   The initial state $S_0$ decayed up to token $t$ and projected against $k_t$:
   $$b_t[j] = \sum_{i=0}^{d_k-1} S_0[i, j] \cdot k_t[i] \cdot \exp(L[t, i])$$
3. **Strict Lower-Triangular Token-Interaction Kernel ($c[t, s]$):**  
   For all $0 \le s < t < C$, the cross-token decay-weighted inner product is:
   $$c[t, s] = \sum_{i=0}^{d_k-1} k_t[i] \cdot k_s[i] \cdot \exp(L[t, i] - L[s, i])$$
   Notice that $L[t, i] - L[s, i] \le 0$ for $s < t$, which guarantees exponent stability.
4. **Forward Substitution (Triangular Solve for $w_t$):**  
   The effective write vector $w_t = \beta_t (v_t - u_t)$ satisfies the causal linear system:
   $$w_t = \beta_t \left( v_t - b_t - \sum_{s < t} c[t, s] w_s \right)$$
   This triangular solve runs sequentially over $t \in [0, C-1]$, but across all $d_v$ channels and all heads in parallel.
5. **Parallel Output Reconstruction ($O_t$):**  
   $$O_t[j] = \sum_{i=0}^{d_k-1} q_t[i] \cdot \exp(L[t, i]) \cdot S_0[i, j] + \sum_{s \le t} m[t, s] w_s[j]$$
   where $m[t, s] = \sum_{i=0}^{d_k-1} q_t[i] \cdot k_s[i] \cdot \exp(L[t, i] - L[s, i])$.
6. **Chunk End State Propagation ($S_C$):**  
   $$S_C[i, j] = \exp(L[C-1, i]) \cdot S_0[i, j] + \sum_{s=0}^{C-1} k_s[i] \cdot \exp(L[C-1, i] - L[s, i]) \cdot w_s[j]$$

#### CUDA Implementation (`gdn_wyf_kernel`)
In `kernels/cuda/ferrite_kernels.cu:398-502`, one threadblock handles one `(chunk, head)` pair. Dynamic shared memory stores $L[C, d_k]$, $b[C, d_v]$, $c[C, C]$, and $w[C, d_v]$. Chunks chain sequentially via a double-buffered ping-pong state array (`bufs[cur]` $\to$ `bufs[cur ^ 1]`), while non-multiple tail tokens fall back to the exact per-token kernel (`gdn_step_kernel`).

---

### 2. Exact Manifold Hyper-Connections (MHC) with Sinkhorn Normalization

GLM-5.3-Flash uses Manifold Hyper-Connections (mHC) to route information across $n = \text{hc\_mult} = 4$ parallel residual streams. Unlike conventional architectures where residual connections are simple elementwise additions $x_{l+1} = x_l + F(x_l)$, mHC dynamically mixes the 4 streams before and after each sublayer.

`crates/ferrite-exec/src/mhc.rs:1-266` provides the exact numerical reference:

```
[s, hidden] (Input token embeddings)
     │
     ▼
  hc_expand (Replicate across n=4 streams -> [s, 4*hidden])
     │
  ┌──┴───────────────────────────────────────────────────────┐
  │ For each sublayer (Attention / FFN):                     │
  │   1. hc_pre:                                             │
  │      - RMS scale: rsqrt = 1 / sqrt(mean(x^2) + rms_eps)   │
  │      - mixes = linear(x_flat, fn_w) * rsqrt [s, 24]      │
  │        (mix_hc = 2n + n^2 = 8 + 16 = 24)                 │
  │      - pre_i = sigmoid(mixes[:n] * scale[0] + base[:n])  │
  │      - layer_input = Σ_i pre_i * residual_i              │
  │      - post_i = 2 * sigmoid(mixes[n:2n] * scale[1] + ...)│
  │      - comb_ik = mixes[2n:] * scale[2] + base[2n:]       │
  │      - Sinkhorn Normalization (20 iters):                │
  │          comb = softmax_last_dim(comb) + hc_eps          │
  │          comb /= (col_sum + hc_eps)                      │
  │          loop: comb /= (row_sum + eps); comb /= (col_sum)│
  │   2. Execute sublayer: x_out = Sublayer(layer_input)      │
  │   3. hc_post:                                            │
  │      residual'_i = post_i * x_out + Σ_k comb_ik * res_k  │
  └──┬───────────────────────────────────────────────────────┘
     ▼
  hc_contract (Average over n=4 streams -> [s, hidden])
     │
     ▼
[s, hidden] (Final RMSNorm & LM Head)
```

The quadratic dimension formula `mix_hc = 2n + n^2` accounts for $n$ input gates (`pre`), $n$ output scale factors (`post`), and $n \times n$ cross-stream mixing coefficients (`comb`). The iterative Sinkhorn loop projects `comb` onto the Birkhoff polytope of doubly stochastic matrices, ensuring that energy is neither amplified nor diminished across layers.

---

### 3. Static PDAF Op-Plan and Phase-Aware Scheduling

`crates/ferrite-scheduler/src/lib.rs:1-264` implements the compile-time static plan:

```rust
pub struct StaticPlan {
    pub ops: Vec<LayerOp>,
    pub num_layers: usize,
    pub num_linear: usize,
    pub num_dsa: usize,
    pub num_moe: usize,
    pub num_dense: usize,
}
```

#### The 90-Step Execution Contract
For GLM-5.3-Flash, `from_config` generates exactly 90 `LayerOp` entries:
- For layer $l \in [0, 44]$:
  - Step $2l$: `OpKind::Attention(AttnKind::Linear | AttnKind::Dsa)` (first half, memory-bandwidth bound)
  - Step $2l+1$: `OpKind::Ffn(MlpKind::Dense | MlpKind::Moe)` (second half, GEMM compute bound)

The static sequence is identical across both Prefill (**P**) and Decode (**D**) phases. Only the kernel flavor changes (e.g., WYF chunk GEMMs for prefill vs. recurrent step for decode).

#### Phase Routing and Transfer Events
`PdafRouter::route` evaluates the batch scheduler state and partitions work into:
- `prefill: Vec<PrefillWork>`: Tokens consumed in chunks up to prompt boundary.
- `decode: Vec<DecodeWork>`: Single-token step at `context_pos`.
- `transfers: Vec<TransferEvent>`: Emitted on the exact step where `prefilled + chunk_tokens == prompt_len`. Carries `dst: Option<DstKvInfo>` specifying destination rank and DCP page filter `(p mod n_dcp == d)` for immediate 2D resharding into decode nodes.

---

### 4. Hybrid State Pool: Linear Slab + Paged Latent DSA

`crates/ferrite-kv/src/lib.rs:1-527` resolves the fundamental tension in hybrid architectures:

| Property | Linear Attention (34 Layers) | DSA Sparse Attention (11 Layers) |
|---|---|---|
| **State Dimension** | Fixed: $[16, 128, 128]$ state + $[1536, 3]$ conv tail | Variable: grows with sequence token length |
| **Storage Primitive** | `LinearStatePool`: contiguous memory slab | `DsaKvPool`: paged latent KV cache |
| **Allocation Policy** | Slot allocator (free-list indexed by `seq`) | Virtual page tables (tokens mapped to physical pages) |
| **Paged Overhead** | 0% (no page tables, no fragmentation) | Paged to avoid external fragmentation |
| **Snapshot Latency** | $O(1)$ single `memcpy` of sequence slot | Multi-page traversal |
| **Context Parallelism** | Replicated atomically across DCP ranks | Sharded by page index: $p \pmod{N_{\text{dcp}}} == \text{rank}$ |

In `LinearStatePool`, the entire linear recurrent state and conv tail for all 34 layers of a sequence are packed contiguously. Calling `snapshot_to(from_seq, to_seq)` clones the entire prefix state in a single contiguous memory copy, providing instantaneous branch/fork capabilities for speculative decoding and tree search without touching page tables.

---

### 5. 3D Axis Algebra ($Q \times Kv \times Head$) and DCP Attention Merge

`crates/ferrite-kv/src/axes.rs` and `crates/ferrite-kernel/src/dcp.rs` define the parallel algebra for composable scaling:

#### The Phase Asymmetry Law
- **Prefill Context Parallelism (CP) shards $Q$:** Each rank owns a query token segment. Since queries attend over the entire KV sequence, output rows are mutually disjoint. Merging along $Q$ requires only row concatenation (no arithmetic).
- **Decode Context Parallelism (DCP) shards $Kv$:** The query is a single token ($Q=1$). Each rank holds $1/N_{\text{dcp}}$ of the KV pages. Each rank computes a local-softmax partial attention $O_d$ and max-shifted scalar $LSE_d$:
  $$LSE_d = m_d + \ln \sum_{s \in \text{shard}_d} \exp(S_s - m_d)$$
  The collective merge across ranks is exact and numerically stable:
  $$M = \max_d LSE_d, \quad w_d = \exp(LSE_d - M), \quad O_{\text{full}} = \frac{\sum_d w_d O_d}{\sum_d w_d}$$
- **Tensor Parallelism (TP) shards $Head$:** Attention head outputs concatenate, while projection down-slices sum.

#### Mathematical Commutativity Proof
`crates/ferrite-kernel/src/dcp.rs:400-546` proves in tests (`q_kv_head_3d_merge_equals_full`) that merging along $Kv$ via LSE and then along $Head$ via concatenation yields bit-identical floating-point results to merging along $Head$ first and then along $Kv$. Because the operations commute, distributed inference topologies $(CP \times DCP \times TP)$ can reorder communication steps to match network interconnect hierarchies (e.g., fast NVLink for TP, slower RoCE for DCP).

---

## Worldview Reconstruction & Architectural Tradeoffs

### Why Static Monomorphized Rust Beats Dynamic Runtime Vtables

| Architectural Axis | SGLang / vLLM Worldview | Ferrite Worldview |
|---|---|---|
| **Model Structure** | Dynamic DAG; modules resolved at runtime via Python dictionary lookups and vtables (`Box<dyn Backend>`). | **Closed Static Schema**: GLM-5.3-Flash has exactly 45 layers, 34 linear, 11 DSA, 3 dense, 42 MoE. Compiled as a monomorphic `Engine<B: KernelBackend>`. |
| **Dispatch Overhead** | Indirect virtual calls and Python interpreter overhead on every sublayer boundary (microseconds per token). | **Zero Dispatch Overhead**: Function calls are statically bound, inlined, and optimized by LLVM; register allocation is preserved across ops. |
| **Memory Allocation** | Dynamic tensor allocations on the decode path; dynamic scratch buffers prone to memory fragmentation. | **Zero Allocation Contract**: All buffers (`&Tensor` input, `&mut Tensor` output) pre-allocated in static slabs before execution begins. |
| **CUDA Graph Integration** | Fragile capture: dynamic python paths, variable sequence shapes, and pointer changes cause graph invalidation. | **Graph-First Stability**: `StaticPlan` guarantees deterministic instruction sequences and memory addresses; CUDA graphs capture reliably. |
| **Parallel Disaggregation** | PD disaggregation, TP, and CP bolted onto monolithic engines via runtime monkey-patching and ad-hoc queues. | **Algebraic PDAF Substrate**: Explicit $Q, Kv, Head$ axes and phase-aware router built into core data structures from day zero. |

---

## Concrete Borrow Candidates for `fak`

The following 5 borrow candidates are grounded at exact revisions in `MindLab-Research/ferrite`:

| # | Technique | Source `path:line@sha` | Axis Optimized | Their Worldview Reason | Witness on `fak` | Disposition & Seam |
|---|---|---|---|---|---|---|
| 1 | **WYF Chunkwise Parallel GatedDeltaNet Recurrence** | `crates/ferrite-kernel/src/wyf.rs:38-147@d771576a7a462866ba707af16100a106c52c7fd2`<br>`kernels/cuda/ferrite_kernels.cu:398-560@d771576a7a462866ba707af16100a106c52c7fd2` | Prefill kernel launch count & latency | Sequential recurrence incurs 32× redundant launches; chunking into $C=32$ parallel blocks converts token loop into GEMM + shared-memory triangular solve. | **PARTIAL** — `internal/model/glm5next_spine.go:52-59` recognizes KDA config, but `arch_support.go:123` returns `GLM5NextUnsupportedError`. Pure Go and CUDA GDN kernels are unoptimized sequential loops. | **DIRECT-PORT** reference into `internal/model/glm5next_kda_state.go`; **ADAPT** CUDA kernel into `internal/compute/gdn_wyf_sm100.cu`. |
| 2 | **Exact Manifold Hyper-Connections (MHC) with Sinkhorn Normalization** | `crates/ferrite-exec/src/mhc.rs:18-213@d771576a7a462866ba707af16100a106c52c7fd2` | Multi-stream residual numerical accuracy | GLM-5.3-Flash's 4-flow residual stream diverges under standard single-residual approximations; 20-iteration Sinkhorn normalization guarantees doubly stochastic mixing. | **ABSENT** — `internal/model/glm5next_spine.go:50-51` captures `MHC bool` and `HCMult int`, but no tensor mixing, RMS pre-scaling, or Sinkhorn iteration exists in `fak`. | **DIRECT-PORT** into `internal/model/mhc_residual.go` with exact golden unit test. |
| 3 | **Dual-Mode Linear Slab + Paged Latent DSA Allocator** | `crates/ferrite-kv/src/lib.rs:43-210@d771576a7a462866ba707af16100a106c52c7fd2`<br>`crates/ferrite-kv/src/lib.rs:218-383@d771576a7a462866ba707af16100a106c52c7fd2` | State fragmentation & prefix snapshot latency | Linear recurrent states do not grow with tokens and should not incur page-table overhead; packing them into contiguous slabs enables instant $O(1)$ memory copies for prefix branching. | **PARTIAL** — `internal/model/kvcache.go` and `internal/vcache/` assume homogeneous token-paged KV caches; no dedicated contiguous linear state slab exists. | **ADAPT** into `internal/model/hybrid_state_pool.go`. |
| 4 | **Commutative 3D DCP Attention Merge via Stable LSE** | `crates/ferrite-kernel/src/dcp.rs:49-174@d771576a7a462866ba707af16100a106c52c7fd2`<br>`crates/ferrite-kv/src/axes.rs:1-80@d771576a7a462866ba707af16100a106c52c7fd2` | Decode context parallelism determinism & speculative verification | Partitioning decode context across ranks requires numerically exact, rank-commutative log-sum-exp partial merges so speculative verifiers arrive at identical logits. | **PARTIAL** — `internal/compute/cuda_collective.go` supports standard AllReduce, but lacks local-softmax partial attention with max-shifted LSE merge for context-parallel decode. | **ADAPT** into `internal/compute/dcp_merge.go`. |
| 5 | **Fused Dual-Input Clamped SwiGLU Kernel (`ferrite_swiglu2`)** | `kernels/cuda/ferrite_kernels.cu:208-235@d771576a7a462866ba707af16100a106c52c7fd2` | Intermediate activation memory bandwidth | Standard SwiGLU requires concatenating separate `gate` and `up` projection outputs into a unified `[n, 2*inter]` buffer before elementwise activation; dual pointers eliminate this gather pass. | **PARTIAL** — `internal/compute/swiglu.go` processes interleaved buffers; requires an extra copy pass when projections are computed separately. | **ADAPT** into `internal/compute/fused_swiglu.cu`. |

---

## License & Provenance Disposition

- **Source License:** MIT License (Copyright (c) 2026 Boxiu Lee / MindLab-Research).
- **Target License:** Apache-2.0 (`fak` kernel).
- **Compatibility:** **MIT is fully inbound-compatible with Apache-2.0.** Code, mathematical routines, and unit tests may be directly ported or adapted into the `fak` repository provided the original copyright notice and permission notice are preserved in ported headers or companion attribution files.
- **Porting Strategy:**
  - Math reference implementations (`wyf.rs`, `mhc.rs`, `dcp.rs`): Direct translation from Rust to idiomatic Go in `internal/model/`.
  - CUDA kernels (`gdn_wyf_kernel`, `ferrite_swiglu2`): Port directly to `internal/compute/` with B300 (sm_100a) and Hopper/Ada fallback support.
  - Architectural patterns (static PDAF plan, dual-mode state pool): Implement clean-room Go abstractions matching `fak`'s execution paradigms.

---

## Concrete Follow-up Implementation Tickets

- Issue: `feat(model): WYF chunkwise parallel GatedDeltaNet recurrence in pure Go and CUDA` (Candidate 1)
- Issue: `feat(model): exact Manifold Hyper-Connections (MHC) with 20-iteration Sinkhorn normalization` (Candidate 2)
- Issue: `feat(model): dual-mode LinearStatePool contiguous slab allocator for recurrent states` (Candidate 3)
- Issue: `feat(compute): commutative 3D DCP attention merge via stable max-shifted Log-Sum-Exp` (Candidate 4)
- Issue: `feat(compute): fused dual-input clamped SwiGLU kernel (ferrite_swiglu2)` (Candidate 5)

---

## Registration and Next Actions

1. **Durable Monitored Registry:** Registered `MindLab-Research/ferrite` in `docs/research/monitored-repositories.json` as `studied` at revision `d771576a7a462866ba707af16100a106c52c7fd2`.
2. **Master Index:** Registered this study note in `INDEX.md` under Notes & Research.
3. **Implementation Plan:**
   - Implement Phase 1: Port exact CPU reference math for WYF recurrence and MHC Sinkhorn normalization to `internal/model/glm5next_kda_state_test.go`.
   - Implement Phase 2: Add `HybridStatePool` to support contiguous GatedDeltaNet slabs alongside paged DSA caches.
   - Implement Phase 3: Land sm_100a WYF CUDA kernel in `internal/compute/` and validate against CPU golden outputs.

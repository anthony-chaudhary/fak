---
title: "CONCEPT-STUDY: airawatraj/dgx-spark-qwen38-flash-agent: HashK GPU PLE compression, Mamba DeltaNet speculative rollback invariants, and Blackwell SM121 kernel fixes"
description: "Exhaustive, code-grounded deep study of airawatraj/dgx-spark-qwen38-flash-agent on NVIDIA DGX Spark (GB10 / SM121), analyzing 4x HashK PLE compression (SplitMix64 + identity ridge bypass), recurrent state rollback under speculative rejection, Grace-Blackwell kernel bugfixes, and silent wedge detection."
date: 2026-09-03
---

# CONCEPT-STUDY: airawatraj/dgx-spark-qwen38-flash-agent — HashK GPU PLE compression, Mamba DeltaNet speculative rollback invariants, and Blackwell SM121 runtime patches (2026-09-03)

**Tracking Issue:** [#10954](https://github.com/anthony-chaudhary/fak/issues/10954)  
**Target Repository:** [`airawatraj/dgx-spark-qwen38-flash-agent`](https://github.com/airawatraj/dgx-spark-qwen38-flash-agent)  
**Pinned Revision:** [`659fa22981c07606bcb29269b5444c0e45ee7b55`](https://github.com/airawatraj/dgx-spark-qwen38-flash-agent/commit/659fa22981c07606bcb29269b5444c0e45ee7b55)  
**License:** MIT License (Copyright (c) 2026 Rajendra Singh Rawat)  
**Durable Study Receipt:** `study_09d87cac2703515c678a679f94fba63a68e82a3515df7c431ed067a85dad93f9`  

---

## Executive Verdict

`airawatraj/dgx-spark-qwen38-flash-agent` is an authoritative, high-performance serving cookbook and runtime patch kit for running [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (~176B total parameters: 125B MoE backbone + 51B $n$-gram Prompt/Path Lookup Embedding (PLE) table, 6B active) on a single **NVIDIA DGX Spark / Grace-Blackwell GB10** node (128 GB unified memory).

The project resolves five critical systems and machine learning challenges when serving giant hybrid MoE/recurrent models in constrained unified memory:

1. **HashK GPU-Resident PLE Table Compression ($4\times$ reduction, 51.2 GB $\to$ 12.8 GB VRAM):** Eliminates NVMe SSD page-fault bottlenecks by compressing the 320M-row $\times$ 160-dim FP8 PLE table into a 12.8 GB GPU-resident tensor via dual-subtable SplitMix64 polynomial hashing and unbiased mean pooling. This frees 16 GB of VRAM on a 128 GB node, enabling the 8 GB Multi-Token Prediction (MTP) speculative draft head to load alongside an FP8 KV cache pool (~700k+ tokens).
2. **Mathematical Discovery of Ridge Projection Redundancy ($W_h \approx I$):** Proves theoretically and empirically that because mean-pooled colliding rows are exchangeable random vectors with $\mathbb{E}[Y \mid X] = X$, the per-head $160 \times 160$ ridge regression transformation $W_h$ asymptotically converges to the identity matrix ($\text{mean diagonal} = 0.9986$, $\text{mean off-diagonal} = 0.0082$). Bypassing $W_h$ (`SGLANG_HASHK_NO_W=1`) eliminates ~410k MACs/token of runtime `einsum` overhead with zero reconstruction cosine degradation ($0.5086$).
3. **Product Quantization (PQ) vs. HashK Pareto Tradeoff:** Demonstrates via offline evaluation on 600,000 raw FP8 rows that while $m=40$ PQ achieves substantially higher reconstruction cosine ($0.9492$ vs. $0.5086$ at 40 B/row), HashK was deployed because its single-level direct tensor gather avoids multi-level codebook lookups, eliminating latency penalties during speculative draft token verification. Furthermore, Qwen's downstream 1D depthwise convolutions and grouped-norm gating absorb HashK's $0.509$ reconstruction error without task regression, scoring **100/100 (15/15 PASS, 30/30 pts)** on `tool-eval-bench`.
4. **Recurrent State Speculative Rollback Invariant:** Discovers and proves that in hybrid architectures combining sparse attention and linear attention / Gated DeltaNet recurrent layers, speculative draft token rejection permanently contaminates the recurrent hidden state $S_t = A_t S_{t-1} + B_t x_t$ unless explicitly snapshot and rewound. Under high-concurrency bursts, unrolled draft errors poison the state, causing delayed NaN token-0 (`!!!!`) collapse. Fixed via `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64`.
5. **Blackwell SM121 (GB10) Upstream Runtime Fixes:** Identifies and patches four upstream kernel incompatibilities:
   - CuTe-DSL FlashAttention-4 TMA-O epilogue MLIR rank crash on variable-length inputs (`patches/flash_fwd.py`).
   - SM100 TRTLLM-gen paged decode silent corruption on SM121 and uncompacted Triton KV holes; replaced with direct-gather grouped-BMM attention with hole masking, cutting decode GPU attention time to 9.7% (~0.1ms vs ~0.9ms) (`patches/qwen_sparse_attn_backend.py`).
   - Triton compiler crash on long-context FP8 RHS dot operands (`patches/sparse_attn.py`).
   - Silent speculative acceptance rate collapse watchdog (`tools/watchdog.sh`) detecting zombie states where `/health` is green (HTTP 200) but acceptance length collapses to $1.00$ (~5 tok/s).

---

## 1. Scope, Provenance, and Durable Receipt

| Field | Value |
|---|---|
| **Repository URL** | [`https://github.com/airawatraj/dgx-spark-qwen38-flash-agent`](https://github.com/airawatraj/dgx-spark-qwen38-flash-agent) |
| **Pinned Commit SHA** | `659fa22981c07606bcb29269b5444c0e45ee7b55` |
| **Upstream Branch** | `main` (latest commit dated 2026-09-03) |
| **License** | MIT License (Copyright (c) 2026 Rajendra Singh Rawat) |
| **Upstream Patches** | SGLang (Apache-2.0) & FlashAttention (BSD-3) |
| **Target Architecture** | NVIDIA DGX Spark (Grace-Blackwell GB10 / SM121, 128 GB Unified Memory) |
| **Model Evaluated** | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (176B total, 6B active) |
| **Durable Study Receipt** | `study_09d87cac2703515c678a679f94fba63a68e82a3515df7c431ed067a85dad93f9` |
| **Registry Update** | Added to `docs/research/monitored-repositories.json` |

---

## 2. Worldview Reconstruction: Who They Built It For & Core Tradeoffs

To understand `airawatraj`'s design decisions without ego dismissal, we reconstruct the exact hardware economics, developer constraints, and failure modes they navigated on DGX Spark:

### A. The Hardware Memory Budget Crisis on Grace-Blackwell GB10
NVIDIA DGX Spark features a single Grace-Blackwell GB10 chip with **128 GB unified LPDDR5X memory** shared across the Grace CPU and Blackwell GPU over a 273 GB/s coherent bus. In practice, system reservation and OS overhead leave **~115–118 GB of usable memory** for containerized inference.

The parameter breakdown of Qwen3.8-Flash-Next creates an impossible memory budget under standard serving stacks:
- **MoE Experts (NVFP4):** 60.4 GB (125B parameters quantized to FP4).
- **Core Dense Transformer & Attention Layers (BF16):** ~8.0 GB.
- **$n$-Gram PLE Embedding Table (FP8 E4M3):** 51.2 GB ($320,001,536$ rows $\times 160$ dimensions $\times 1$ byte).
- **Total Static Weights:** $60.4 + 8.0 + 51.2 = \mathbf{119.6\text{ GB}}$!

If loaded uncompressed, static weights consume 100% of usable memory, leaving **0 bytes** for KV cache, activation memory, or speculative draft models.

### B. The Failure of Previous Paradigms: Disk MMAP and Load-Time NVFP4 Packing
Prior open-source attempts on DGX Spark (such as `hasso5703/dgx-spark-qwen38` and early vLLM recipes) took one of two unsatisfactory routes:

1. **NVMe SSD MMAP (`src/vllm_ple_mmap.py`, v1.0.0 baseline):**
   - The 51.2 GB PLE table was left on NVMe SSD and accessed via `np.memmap` through the Linux OS page cache.
   - *Failure Modes:* Because each autoregressive token touches 16 random rows across 128 shards, lookups triggered random page faults. This introduced heavy CPU-bound gather overhead and asynchronous Host-to-Device (H2D) transfers that broke piecewise CUDA graph capture. At deeper contexts (>32k), page-fault thrashing dropped decode speeds to 10–12 tok/s, caused latency spikes, and subjected workstation NVMe SSDs to severe write/read wear.
2. **Load-Time NVFP4 Packed Storage (`SGLANG_QWEN4_PLE_NVFP4=1`, v1.0.0 intermediate):**
   - Packed the 51.2 GB FP8 table to 28.8 GB in GPU VRAM (4-bit codes + FP8 group scales).
   - *Failure Modes:* While table residency moved to GPU, static weights still consumed $60.4 + 8.0 + 28.8 = \mathbf{97.2\text{ GB}}$. On a 115 GB usable pool, this left only ~17 GB, which was entirely consumed by the baseline FP8 KV cache pool (~262k tokens). There was **zero headroom** to load the 8 GB Multi-Token Prediction (MTP) draft head. Consequently, generation ran strictly unspeculative at the raw hardware memory bandwidth ceiling:
     $$\text{Decode Throughput} \approx \frac{273\text{ GB/s}}{97\text{ GB weights}} \approx 11.3\text{ tok/s}$$

### C. The HashK Breakthrough: Pure VRAM Serving with Speculative Acceleration
By compressing the PLE table $4\times$ to **12.8 GB VRAM** (`ple_hashk_R4.pt`), total static residency dropped to ~81 GB. This unlocked:
- Direct GPU-resident gather with **zero SSD I/O** and **zero host/device synchronization**.
- Allocation of the **8 GB MTP draft head** for SGLang NEXTN speculative decoding (3 steps, 4 draft tokens).
- Allocation of an extensive **FP8 KV cache pool** (~700k–900k tokens at `mem-fraction-static 0.95`).
- Acceleration of decode throughput from $11.3\text{ tok/s}$ up to **$36.8\text{ tok/s}$** on code and **$41.8\text{ tok/s}$** on structured JSON/repro prompts (an average acceptance length of $2.57$ tokens/step).

### D. The Speculative Rollback Invariant on Recurrent Architectures
In conventional pure-transformer models, speculative decoding rejection is simple: if tokens $t+1 \dots t+k$ are rejected, the runtime simply truncates the KV cache sequence length pointer back to $t$.

However, Qwen3.8-Flash-Next incorporates **Gated DeltaNet (GDN)** linear attention recurrent layers. Recurrent state transitions mutate a persistent hidden matrix:
$$S_t = A_t S_{t-1} + B_t x_t$$
When draft tokens are evaluated, the recurrent state $S_t$ is mutated forward. If the speculative verifier subsequently rejects those candidate tokens, **the recurrent state retains the memory of the rejected tokens**. Without state checkpointing and exact rollback, phantom activations accumulate in the recurrence. Under multi-stream concurrency storms, this corruption cascades into immediate NaN logits and token-0 (`!!!!`) degeneration.

The author proved that configuring `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64` allocates checkpointed state buffers that rewind the recurrence upon draft rejection, guaranteeing 100% mathematical parity across 260,000+ context depths.

---

## 3. Subsystem Analysis & Key Mechanisms (Code & Mathematical Rigor)

### A. HashK Compression Algorithm & SplitMix64 Dual-Subtable Design
*Source:* `tools/build_hashk_ple.py:41-65,158-235@659fa22981c07606bcb29269b5444c0e45ee7b55` and `patches/qwen4_exp_nvfp4.py:78-151,674-683@659fa22981c07606bcb29269b5444c0e45ee7b55`.

```
Raw FP8 PLE Table (51.2 GB, 320M rows, D=160)
         │
         ▼ Dimension Split (D -> 2 x 80)
 ┌───────────────────────┬───────────────────────┐
 │ Sub-Table A (dims 0:80)│ Sub-Table B (dims 80:160)│
 └───────────┬───────────┴───────────┬───────────┘
             │                       │
      SplitMix64 Salt 0       SplitMix64 Salt 1
             │                       │
             ▼                       ▼
      Slot sA in Sub-Table A   Slot sB in Sub-Table B
      (Unbiased Mean Pooling) (Unbiased Mean Pooling)
             │                       │
             └───────────┬───────────┘
                         ▼
        Concatenate [meanA, meanB] in FP8 (12.8 GB)
                         │
        (Identity Ridge Matrix Wh Bypassed via SGLANG_HASHK_NO_W=1)
                         ▼
         Final Retrieved Vector (160-dim BF16)
```

The Qwen3.8-Flash-Next PLE architecture uses 16 attention heads ($H=16$), where each head $h \in [0, 15]$ indexes a prime-sized vocabulary $V_h$ starting from $20,000,000$:
$$V_h = \text{nextprime}(V_{h-1}), \quad V_0 = \text{nextprime}(19,999,999) = 20,000,003$$
The total rows across all 16 heads equal $\sum_{h=0}^{15} V_h \approx 320,001,536$. At dimension $D=160$ stored in OCP FP8 E4M3 (1 byte/element), the raw weight size is exactly:
$$320,001,536 \times 160 \times 1\text{ byte} = 51,200,245,760\text{ bytes} \approx 51.2\text{ GB}$$

#### 1. Dual-Subtable Partitioning
To compress by ratio $R=4$, the hidden dimension $D=160$ is split into two halves ($HALF=80$):
- Sub-table $A$: dimensions $0 \dots 79$
- Sub-table $B$: dimensions $80 \dots 159$

For each head $h$, the sub-table capacity is:
$$S_h = \left\lceil \frac{V_h}{R} \right\rceil = \left\lceil \frac{V_h}{4} \right\rceil \approx 5,000,000\text{ rows}$$
Total compressed rows per sub-table equal $\sum_{h=0}^{15} S_h \approx 80,000,392$.  
Combined storage for tables $A$ and $B$ in FP8 E4M3:
$$2 \times (80,000,392 \times 80 \times 1\text{ byte}) = 12,800,062,720\text{ bytes} \approx \mathbf{12.8\text{ GB}}$$

#### 2. SplitMix64 Polynomial Hash Mapping
Each original local row index $\text{idx} \in [0, V_h - 1]$ is hashed into slot $s_A \in [0, S_h - 1]$ and $s_B \in [0, S_h - 1]$ using a 64-bit SplitMix64 generator with distinct salts:

$$\begin{aligned}
x_{\text{sub}} &= (\text{local\_idx} + 1) \times 2862933555777941757 + \text{SALTS}[\text{sub}] + h \times 998244353 \\
\text{where} \quad &\text{SALTS}[0] = 1234567891011121314, \quad \text{SALTS}[1] = -8765432109876543211 \\
\text{splitmix}(x) &: \\
x &\leftarrow x + \gamma \quad (\gamma = \text{0x9E3779B97F4A7C15} = -7046029254386353131) \\
x &\leftarrow (x \oplus (x \gg 30)) \times M_1 \quad (M_1 = \text{0xBF58476D1CE4E5B9} = -4658895280553007687) \\
x &\leftarrow (x \oplus (x \gg 27)) \times M_2 \quad (M_2 = \text{0x94D049BB133111EB} = -7723592293110705685) \\
\text{hash} &= x \oplus (x \gg 31) \\
s_{\text{sub}} &= \text{hash} \pmod{S_h}
\end{aligned}$$

#### 3. Unbiased Mean-Pooling Slot Aggregation
For each slot $s \in [0, S_h - 1]$, multiple original rows collide. In `build_hashk_ple.py:187-201`, the table is populated without gradient training by computing the running accumulator and count:
$$\text{sumA}[s] \mathrel{+}= w[\text{idx}, 0:80], \quad \text{cntA}[s] \mathrel{+}= 1$$
$$\text{meanA}[s] = \frac{\text{sumA}[s]}{\max(\text{cntA}[s], 1)}, \quad \text{meanB}[s] = \frac{\text{sumB}[s]}{\max(\text{cntB}[s], 1)}$$
Both tables are cast to `torch.float8_e4m3fn` for GPU storage.

#### 4. Theoretical Mean-Pooling Ceiling & Proof of Ridge Matrix $W_h$ Redundancy
Under uniform random hashing with compression ratio $R=4$, exactly $R$ independent zero-mean unit-variance vectors $v_1, \dots, v_R$ collide into a slot. The slot centroid is:
$$\hat{x} = \frac{1}{R} \sum_{i=1}^R v_i$$
The expected cosine similarity between any constituent vector $v_1$ and the centroid $\hat{x}$ is:
$$\mathbb{E}[\cos(v_1, \hat{x})] = \frac{\mathbb{E}[v_1^T \hat{x}]}{\|v_1\| \|\hat{x}\|} = \frac{\frac{1}{R} \mathbb{E}[v_1^T v_1]}{\sqrt{\frac{1}{R^2} \sum_{i=1}^R \mathbb{E}[v_i^T v_i]}} = \frac{\frac{1}{R}}{\frac{1}{R} \sqrt{R}} = \frac{1}{\sqrt{R}}$$
For $R=4$:
$$\mathbb{E}[\cos] = \frac{1}{\sqrt{4}} = 0.5000$$
The empirical held-out cosine measured by `build_hashk_ple.py` is **$0.5086$**, exactly matching the theoretical limit.

**The Ridge Regression Proof:**  
In the original HashK specification, a per-head $160 \times 160$ ridge regression matrix $W_h$ was fitted to project the concatenated vector $\hat{X} = [\hat{A}, \hat{B}] \in \mathbb{R}^{160}$ back to the true row $Y \in \mathbb{R}^{160}$:
$$W_h = (\hat{X}^T \hat{X} + \lambda I)^{-1} \hat{X}^T Y$$
However, because all rows hashing to slot $s$ are exchangeable random variables drawn from the same underlying embedding distribution, the conditional expectation is:
$$\mathbb{E}[Y \mid \hat{X}] = \hat{X}$$
Therefore, the optimal linear least-squares estimator $W_h$ asymptotically converges to the **Identity Matrix** ($I_{160}$).  
Direct inspection of `ple_hashk_R4.pt` confirms:
- **Mean diagonal element:** $0.9986 \approx 1.0$
- **Mean off-diagonal element:** $0.0082 \approx 0.0$
- **Held-out Cosine with $W_h$:** $0.5086$
- **Held-out Cosine without $W_h$ (Identity):** $0.5086$

In `patches/qwen4_exp_nvfp4.py:105-151`, setting `SGLANG_HASHK_NO_W=1` bypasses the $W_h$ matrix multiplication entirely:
```python
# Bypass redundant ridge projection W to save ~410k MACs/token (credit @jucedik)
no_w = os.environ.get("SGLANG_HASHK_NO_W", "0") == "1"
has_w = ("W" in art and art["W"] is not None and not no_w)
...
if st["W"] is None:
    return hat
return torch.einsum("thd,hde->the", hat, st["W"])
```
For 16 heads with $D=160$, bypassing $W_h$ eliminates:
$$16 \times 160 \times 160 = 409,600\text{ multiply-accumulates (MACs) per token}$$
with zero degradation in reconstruction accuracy.

---

### B. Product Quantization (PQ) vs. HashK Comparison
*Source:* `tools/bench_pq_vs_hashk.py:104-175@659fa22981c07606bcb29269b5444c0e45ee7b55` and `EXPERIMENTS.md:98-115@659fa22981c07606bcb29269b5444c0e45ee7b55`.

Using held-out evaluation on 600,000 raw FP8 E4M3 rows from `model-plefp8-00000.safetensors`, the project established the Pareto frontier between Product Quantization and HashK:

| Compression Method | Bytes / Row | Compression Ratio | Mean Cosine | Median Cosine | 5th Percentile | Table Size (VRAM) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HashK $R=4$ (mean-pool only)** | 40.0 B | $4.0\times$ | 0.5086 | 0.5120 | 0.3773 | 12.8 GB |
| **HashK $R=4$ (mean-pool + ridge $W$)** | 40.0 B | $4.0\times$ | 0.5086 | 0.5120 | 0.3773 | 12.8 GB |
| **PQ ($m=40$, 256 centroids)** | 40.0 B | $4.0\times$ | **0.9492** | **0.9496** | **0.9401** | 12.8 GB |
| **PQ ($m=20$, 256 centroids)** | 20.0 B | $8.0\times$ | **0.8237** | **0.8241** | **0.8017** | 6.4 GB |
| **PQ ($m=10$, 256 centroids)** | 10.0 B | $16.0\times$ | **0.6545** | **0.6544** | **0.6207** | 3.2 GB |
| **PQ ($m=8$, 256 centroids)** | 8.0 B | $20.0\times$ | **0.5995** | **0.5992** | **0.5619** | 2.56 GB |

#### Why HashK Won for Production Deployment
1. **Gather Latency in Speculative Verification:** HashK executes as a single-level direct gather:
   ```python
   hat = torch.cat([st["A"][sA].to(torch.bfloat16), st["B"][sB].to(torch.bfloat16)], dim=-1)
   ```
   This is a contiguous indexed memory read requiring zero memory indirection. Conversely, Product Quantization requires a two-level gather: first fetching $m$ 8-bit codebook indices per row, then performing $m$ gather operations into $m$ independent centroid codebooks. In PyTorch/Triton, this two-level memory indirection incurs latency overhead during the tight inner loop of NEXTN speculative draft token verification.
2. **Architectural Gating Resilience:** In Qwen3.8-Flash-Next, retrieved PLE vectors do not feed directly into residual additions; they pass through 1D depthwise causal convolutions and grouped-norm gating layers before projection into the backbone. In practice, the model's gating mechanisms are robust to HashK's $0.509$ cosine distortion, achieving a perfect **100/100 (15/15 PASS)** on tool-calling benchmarks.
3. **Future Upside:** As documented in `EXPERIMENTS.md:150-156`, an optimized Triton $m=20$ PQ gather kernel represents a viable future optimization to compress the table $8\times$ (down to 6.4 GB VRAM), freeing an additional 6.4 GB of VRAM for KV cache expansion.

---

### C. Mamba / Gated DeltaNet Speculative State Rollback Invariants
*Source:* `docker/start.sh:134-135@659fa22981c07606bcb29269b5444c0e45ee7b55`, `patches/qsa_kv_pool.py:98-102@659fa22981c07606bcb29269b5444c0e45ee7b55`, `tools/bisect_probe.py:1-65@659fa22981c07606bcb29269b5444c0e45ee7b55`, and `tools/causal_chain.py:1-77@659fa22981c07606bcb29269b5444c0e45ee7b55`.

Qwen3.8-Flash-Next is a hybrid architecture interleaving QSA sparse self-attention layers with Gated DeltaNet linear recurrent layers. 

In a linear recurrence, the state $S_t \in \mathbb{R}^{d_{state} \times d_{head}}$ updates sequentially:
$$S_t = \alpha_t S_{t-1} + \beta_t (k_t^T v_t)$$
During speculative decoding, the MTP draft head predicts 4 candidate tokens per step ($k=4$). If the target model verifies and accepts only 2 of the 4 tokens, the remaining 2 speculative tokens must be discarded.

```
Speculative Step: Propose 4 tokens (t+1, t+2, t+3, t+4)
                     │
    Linear Recurrence Forward Pass advances S:
    S_t -> S_{t+1} -> S_{t+2} -> S_{t+3} -> S_{t+4}
                     │
    Target Verifier accepts (t+1, t+2), REJECTS (t+3, t+4)
                     │
 ┌───────────────────┴───────────────────┐
 │                                       │
 ▼ WITHOUT Rollback Buffer               ▼ WITH extra_buffer (interval 64)
 State S remains at S_{t+4}              State S rewound exactly to S_{t+2}
 (POISONED by rejected tokens!)          (CLEAN, mathematically exact)
 │                                       │
 ▼ Under concurrency bursts:             ▼ Across 260k context:
 Activation explodes -> NaN logits       100% stable, zero corruption
 Token-0 ("!!!!") collapse!
```

#### The Causal Chain Proof (`tools/causal_chain.py`)
To isolate this defect from network or driver issues, the authors constructed a 4-phase causal chain probe:
- **Phase 1 (Fresh Boot):** Deep 5,500-word prompt $\to$ Verified clean output.
- **Phase 2 (Concurrency Storm):** 8 concurrent streams generating 700 tokens simultaneously, forcing request queueing, draft rejections, and context preemption.
- **Phase 3 (Post-Storm Deep Request):** Deep prompt submitted immediately after the storm.
  - *Without `extra_buffer`:* The post-storm request emitted token-0 strings (`!!!!`) within 10 tokens.
  - *With `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64`:* The post-storm request remained 100% clean and coherent.
- **Phase 4 (Idle Recovery Test):** Allowing the server to idle for 300 seconds did *not* recover uncheckpointed instances, proving that recurrent corruption resides in long-lived state memory pools rather than transient GPU caches.

The flags `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64` configure SGLang to maintain auxiliary state snapshots every 64 tokens and at speculative step boundaries, allowing deterministic state rewinding upon rejection.

---

### D. Grace-Blackwell (GB10 / SM121) Upstream Kernel Patches

The deployment container bind-mounts 4 critical runtime patches to overcome hardware-specific Blackwell quirks:

#### 1. CuTe-DSL FlashAttention-4 TMA-O Epilogue MLIR Crash
*Source:* `patches/flash_fwd.py:658-659@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Root Cause:** In stock Cutlass/CuTe-DSL FlashAttention-4 on SM90+, output writes default to Tensor Memory Accelerator (TMA) stores (`use_tma_O`). TMA descriptors require fixed, uniform coordinate strides. When handling variable-length prompt batches (`mCuSeqlensQ is not None`), the ragged 3D tensor layout violates TMA rank alignment, causing the MLIR compiler to crash with `"weakly congruent"` layout assertion failures.
- **Fix:** Guard TMA-O activation strictly to non-variable-length shapes:
  ```python
  # patches/flash_fwd.py:659
  self.use_tma_O = self.arch >= Arch.sm_90 and mCuSeqlensQ is None
  ```

#### 2. Blackwell SM121 TRTLLM-Gen Silent NaN & Grouped-BMM Attention Rewrite
*Source:* `patches/qwen_sparse_attn_backend.py:49-67,1756-1806@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Root Cause A:** Upstream SGLang uses TensorRT-LLM paged decode kernels (`trtllm_batch_decode_with_kv_cache`) for QSA post-gather attention. These kernels were optimized exclusively for SM100 and SM120. On SM121 (GB10), the kernel executes without throwing errors but silently produces NaN activations, degenerating output into token-0 (`!!!!`).
- **Root Cause B:** The fallback Triton kernel `_compact_kv` fails to compact sparse slots when $-1$ hole indices are present; valid rows maintain original column offsets while sequence length metadata assumes dense contiguity, leaving uninitialized NaN slots.
- **Fix:** Route SM121 away from TRTLLM-gen by returning `None` in `_resolve_trtllm_sparse_decode()` on `(major, minor) == (12, 1)`. Replace decode attention with direct token gathering from `req_to_token_pool.req_to_token` followed by a custom Grouped-GQA Batched Matrix Multiplication (`torch.baddbmm` + masked `softmax` + `torch.bmm`):
  ```python
  # patches/qwen_sparse_attn_backend.py:1774-1806
  slots = self.req_to_token_pool.req_to_token[
      req_pool_idx.view(-1, 1).to(torch.long), positions.clamp_min(0)
  ].to(torch.long)
  k_g = k_buffer[slots]  # [B, topk, Hkv, D]
  v_g = v_buffer[slots]
  ...
  scores = torch.baddbmm(
      torch.zeros(1, 1, 1, dtype=qh.dtype, device=qh.device),
      qf, kf, alpha=layer.scaling,
  )  # [B*Hkv, rep, topk]
  neg = torch.finfo(scores.dtype).min
  mask = valid.view(batch, 1, 1, topk).expand(batch, Hkv, 1, topk)
  scores = scores.view(batch, Hkv, rep, topk).masked_fill(~mask, neg)
  probs = scores.float().softmax(-1).to(qh.dtype).view(batch * Hkv, rep, topk)
  out = torch.bmm(probs, vf).view(batch, Hkv, rep, -1).reshape(batch, Hq, -1)
  ```
  *Performance Impact:* For decode batches with single query tokens ($q_{len}=1$), this grouped-BMM path avoids materializing repeated KV heads and executes in **~0.1ms** (down from ~0.9ms for the general FMHA kernel), reducing attention decode time to **9.7%** of GPU step time.

#### 3. Long-Context Prefill Triton FP8 RHS Dot Fix
*Source:* `patches/sparse_attn.py:105-113@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Root Cause:** In the QSA sparse attention Triton prefill kernel, `tl.dot(q_values, keys)` attempted to multiply BF16 query values by raw FP8 E4M3 keys on long-context shapes, triggering Triton compiler failure: `"Unsupported rhs dtype fp8e4nv"`.
- **Fix:** Explicitly cast keys to the query data type before the dot product:
  ```python
  # patches/sparse_attn.py:105
  scores = tl.where(valid[None, :], tl.dot(q_values, keys.to(q_values.dtype)), -float("inf"))
  ```

---

### E. Operational Reliability & Silent Wedge Watchdog
*Source:* `tools/watchdog.sh:1-42@659fa22981c07606bcb29269b5444c0e45ee7b55`.

A major operational discovery on production DGX Spark nodes is the **Silent Wedge State**:
- **Symptoms:** Following sudden client disconnects or aborted HTTP streams, the engine can enter a degraded state where running requests stall at ~5 tok/s and state slots leak.
- **The Health Check Paradox:** Standard Kubernetes / Docker health probes (`GET /health`) query the outer event loop and return `HTTP 200 OK`.
- **The Telemetry Signature:** Speculative decoding rejects every draft token, causing the log line `Decode batch ... accept len: 1.00` to repeat indefinitely on every active request (healthy traffic exhibits $2.5\text{--}3.5+$ tokens).
- **The Watchdog Implementation:** `tools/watchdog.sh` parses `docker logs --since 3m`. If 6 or more consecutive decode batches report `accept len: 1.00` while active requests exist (`#running-req: [1-9]`), the script trips, logs the wedge event, and executes `docker restart spark-brain`. To prevent flapping during server boots, restarts are rate-limited to at most 4 actions per day.

---

## 4. Current fak Witness & Gap Matrix

| `airawatraj` Mechanism | fak Equivalent Subsystem | Current fak Witness | On-Axis Gap & Disposition |
|---|---|---|---|
| **HashK dual-subtable SplitMix64 polynomial embedding compression** | `internal/model`, `internal/compute`, `internal/ggufload` | `internal/model/verify.go:36`, `internal/polymodel/polymodel.go:410`, issue #3197 | **ABSENT $\to$ DEFAULT**. Fak has no compressed sub-table hashing representation for vocabulary or $n$-gram embedding tables. Large embedding tables must be stored uncompressed or offloaded, creating memory pressure on unified-memory nodes. |
| **Ridge transformation bypass ($W_h \approx I$)** | `internal/compute/ops` | `internal/compute/` | **ABSENT $\to$ DEFAULT**. When implementing sub-table embedding aggregation, skipping redundant projection matrices saves ~410k MACs/token. |
| **Mamba / Gated DeltaNet state checkpointing and rollback** | `internal/quality/spec_decode.go`, `internal/model/verify.go` | `internal/quality/spec_decode.go:14-25,95-150`, `internal/polymodel/polymodel.go:450` | **PARTIAL $\to$ DEFAULT**. Fak's speculative decoding verification correctly models token rejection and fallback, but assumes transformer KV pointer rewinds. It lacks state snapshotting and rollback for recurrent / linear attention layers ($S_t = A_t S_{t-1} + B_t x_t$). |
| **Hole-tolerant direct-gather grouped-BMM sparse decode** | `internal/compute`, `internal/roofline` | `internal/compute/`, `internal/roofline/roofline.go:71` | **ABSENT $\to$ RECIPE**. Fak has no Blackwell SM121-specific grouped-BMM sparse attention decode fallback to bypass broken SM100 TRTLLM-gen kernels. |
| **Speculative acceptance rate collapse watchdog (silent wedge)** | `internal/gateway/health.go`, `internal/modelaccept/speculative.go` | `internal/modelaccept/speculative.go:64`, `internal/gateway/admission.go:33` | **PARTIAL $\to$ DEFAULT**. Fak possesses watchdog autohealing for stalled OS processes and HTTP health checks, but does not monitor speculative draft acceptance length collapse ($1.00$ drop) as a primary health indicator. |
| **CuTe-DSL FlashAttention-4 TMA-O variable-length epilogue guard** | `internal/compute` / `docs/fleet-compute-nodes.md` | `docs/fleet-compute-nodes.md:45` | **ABSENT $\to$ RECIPE**. FAK documents compute nodes but lacks compiler guidance guarding TMA-O stores on variable-length CuTe-DSL kernels for SM90+. |

---

## 5. Concrete Borrow Candidates Grounded at `file:line@sha`

### Candidate 1: HashK Dual-Subtable Polynomial Hash-Compressed Embedding Table
- **Technique:** Compress large embedding / lookup tables by factor $R$ ($4\times$, 51.2 GB $\to$ 12.8 GB FP8) by splitting hidden dimensions across $k=2$ sub-tables ($HALF=80$) using independent 64-bit SplitMix64 polynomial hashes, populating slots with unbiased collision means ($\text{sum} / \max(\text{count}, 1)$), and bypassing the per-head $160 \times 160$ ridge regression transformation $W_h$ at runtime (`SGLANG_HASHK_NO_W=1`) due to the mathematical identity $\mathbb{E}[Y \mid X] = X$ ($W_h \approx I$, saving ~410k MACs/token).
- **Source Anchor:**  
  - `tools/build_hashk_ple.py:45-55@659fa22981c07606bcb29269b5444c0e45ee7b55`  
  - `tools/build_hashk_ple.py:158-203@659fa22981c07606bcb29269b5444c0e45ee7b55`  
  - `patches/qwen4_exp_nvfp4.py:105-151@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Fak Seam:** `internal/compute/hashk` & `internal/model` (embedding lookup and sparse $n$-gram drafters, complementing #3197 and #4209).
- **Axis:** Embedding table VRAM footprint vs. gather latency and reconstruction accuracy.
- **Why their users made them build it:** Disk MMAP over NVMe SSD incurs random page faults and CPU gather overhead (~10–23 tok/s), while uncompressed storage exceeds DGX Spark's 128 GB memory budget. Compressing to 12.8 GB VRAM enables pure GPU residency and frees 16 GB to load the 8 GB MTP draft model.
- **Witness on-axis:** `ABSENT`.
- **Disposition:** `DEFAULT` (for hybrid models with massive lookup tables).
- **Tracking Issue:** [#10954](https://github.com/anthony-chaudhary/fak/issues/10954)
- **First Checkable Step:** Implement a pure-Go reference implementation of `SplitMix64` and dual-subtable mean-pooling gather in `internal/compute/hashk/hashk.go`, with unit tests verifying slot index parity and the mathematical convergence of $W_h \to I$.

---

### Candidate 2: Recurrent State Checkpointing and Rollback Buffer for Speculative Drafting
- **Technique:** Allocate an auxiliary state rollback buffer (`extra_buffer`) for hybrid linear attention / Gated DeltaNet recurrent architectures that snapshots recurrent hidden state tensors prior to speculative draft execution and restores the snapshot upon draft verification rejection, preventing phantom draft tokens from poisoning persistent recurrent states.
- **Source Anchor:**  
  - `docker/start.sh:134-135@659fa22981c07606bcb29269b5444c0e45ee7b55`  
  - `patches/qsa_kv_pool.py:98-102@659fa22981c07606bcb29269b5444c0e45ee7b55`  
  - `tools/bisect_probe.py:20-65@659fa22981c07606bcb29269b5444c0e45ee7b55`  
  - `tools/causal_chain.py:20-77@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Fak Seam:** `internal/quality/spec_decode.go:95-150` & `internal/model/verify.go:36-120`.
- **Axis:** Speculative decoding mathematical correctness and numerical stability on recurrent/linear attention architectures.
- **Why their users made them build it:** Rejecting speculative tokens in transformer attention requires only rewinding the KV cache sequence length pointer. In linear attention (GDN/SSM), the recurrent state accumulates rejected tokens irreversibly; under multi-stream concurrency storms, this state poisoning causes delayed NaN explosions and token-0 (`!!!!`) collapse.
- **Witness on-axis:** `PARTIAL`.
- **Disposition:** `DEFAULT`.
- **Tracking Issue:** [#10954](https://github.com/anthony-chaudhary/fak/issues/10954)
- **First Checkable Step:** Add a `StateSnapshotter` interface to `internal/quality/spec_decode.go` and write a differential test simulating linear recurrence, asserting that rejecting candidate tokens without state rollback produces divergence, whereas snapshot restoration guarantees exact parity with unspeculative decode.

---

### Candidate 3: Hole-Tolerant Direct-Gather Grouped-BMM Sparse Attention Decode on Blackwell SM121
- **Technique:** Replace fragile vendor paged decode kernels (which silently output NaNs on SM121) and broken Triton compaction kernels with a direct tensor gather from the token pool (`k_buffer[slots]`, `v_buffer[slots]`) followed by a grouped-GQA Batched Matrix Multiplication (`baddbmm` + masked `softmax` + `bmm`), natively masking invalid/hole indices and cutting single-query decode attention GPU step latency to ~0.1ms (9.7% of step time).
- **Source Anchor:**  
  - `patches/qwen_sparse_attn_backend.py:49-67,1756-1806@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Fak Seam:** `internal/compute` / `docs/fleet-compute-nodes.md` / Blackwell SM121 native execution backend.
- **Axis:** Execution reliability and dispatch latency of sparse attention decode passes on Grace-Blackwell GB10.
- **Why their users made them build it:** Upstream TRTLLM-gen kernels compiled for SM100 run without error on SM121 but silently corrupt output into token-0 (`!!!!`), while Triton compaction leaves uninitialized hole memory. Direct gather + grouped-BMM executes in pure PyTorch/Triton tensor operations, is 100% CUDA-graph capturable, and runs 9× faster for single-token decode steps.
- **Witness on-axis:** `ABSENT`.
- **Disposition:** `RECIPE` / `OPTIONAL-MODULE`.
- **Tracking Issue:** [#10954](https://github.com/anthony-chaudhary/fak/issues/10954)
- **First Checkable Step:** Document the SM121 TRTLLM-gen silent corruption pitfall in `docs/fleet-compute-nodes.md` and implement a unit test in `internal/compute` proving grouped-BMM attention equivalence under hole-masked sparse indexing.

---

### Candidate 4: Speculative Acceptance Rate Collapse Watchdog (Silent Wedge Recovery)
- **Technique:** Monitor runtime speculative decode telemetry (e.g. `accept len` in decode batch logs) to detect silent engine degradation where `/health` probes return HTTP 200 OK, but speculative acceptance collapses to $1.00$ (100% draft rejection, throughput drops to ~5 tok/s, state slots leak), triggering a rate-limited auto-restart (max 4 actions/day) to prevent zombie processes.
- **Source Anchor:**  
  - `tools/watchdog.sh:1-42@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Fak Seam:** `internal/modelaccept/speculative.go`, `cmd/fak/watchdog_autoheal.go`, `internal/gateway/health.go`.
- **Axis:** High-order semantic health observability in speculative serving systems.
- **Why their users made them build it:** Aborted client requests or corrupted internal states can cause the speculative engine to enter a degraded "zombie" state where HTTP health endpoints remain green, but every speculative draft is rejected, wasting 75% of GPU compute and driving latency from 36 tok/s down to 5 tok/s.
- **Witness on-axis:** `PARTIAL`.
- **Disposition:** `DEFAULT`.
- **Tracking Issue:** [#10954](https://github.com/anthony-chaudhary/fak/issues/10954)
- **First Checkable Step:** Implement a `SpeculativeWedgeMonitor` in `internal/modelaccept/speculative.go` that tracks moving-average draft acceptance lengths and flags a semantic degradation alert when acceptance collapses to $\le 1.05$ across $N$ consecutive batches with active requests.

---

### Candidate 5: CuTe-DSL FlashAttention-4 TMA-O Variable-Length Epilogue Guard
- **Technique:** Guard Tensor Memory Accelerator (TMA) output epilogues (`use_tma_O`) in Cutlass / CuTe-DSL FlashAttention on SM90+ (Hopper and Blackwell) to only engage when query sequence lengths are fixed (`mCuSeqlensQ is None`), falling back to standard register stores for variable-length (ragged) sequences to avoid MLIR compiler crashes ("weakly congruent" layout errors).
- **Source Anchor:**  
  - `patches/flash_fwd.py:658-659@659fa22981c07606bcb29269b5444c0e45ee7b55`  
  - `docs/LANDMINES.md:26-28@659fa22981c07606bcb29269b5444c0e45ee7b55`
- **Fak Seam:** `docs/fleet-compute-nodes.md` & `internal/compute`.
- **Axis:** Kernel compilation stability for FlashAttention-4 on SM90/SM100/SM120/SM121 when handling variable-length prompt batches.
- **Why their users made them build it:** Upstream FlashAttention-4 cute kernels fail compilation under variable-length sequences on Grace-Blackwell because the TMA hardware descriptor requires uniform multidimensional strides that ragged token offsets violate.
- **Witness on-axis:** `ABSENT`.
- **Disposition:** `RECIPE`.
- **Tracking Issue:** [#10954](https://github.com/anthony-chaudhary/fak/issues/10954)
- **First Checkable Step:** Add an operational note to `docs/fleet-compute-nodes.md` under the DGX Spark GB10 section detailing the CuTe-DSL TMA-O ragged sequence constraint.

---

## Concrete Follow-up Implementation Tickets

- Issue: `feat(compute): HashK dual-subtable SplitMix64 polynomial embedding table compression with identity ridge bypass` (Candidate 1)
- Issue: `feat(quality): recurrent state checkpointing and rollback buffer for speculative drafting on hybrid GDN/SSM` (Candidate 2)
- Issue: `feat(compute): hole-tolerant direct-gather grouped-BMM sparse attention decode on Blackwell SM121` (Candidate 3)
- Issue: `feat(modelaccept): speculative acceptance rate collapse watchdog for silent wedge autoheal` (Candidate 4)
- Issue: `fix(compute): CuTe-DSL FlashAttention-4 TMA-O variable-length epilogue guard for SM90+` (Candidate 5)

---

## 6. Registration and Companions

- **Durable Study Receipt:** `study_09d87cac2703515c678a679f94fba63a68e82a3515df7c431ed067a85dad93f9` (persisted via `fak study add`).
- **Monitored Repository Registry:** Added `airawatraj/dgx-spark-qwen38-flash-agent` to `docs/research/monitored-repositories.json` as `studied`.
- **Navigation Index:** Registered in `INDEX.md` under `## Notes & research`.
- **Tracking Companions:**
  - Tracking Issue: [#10954](https://github.com/anthony-chaudhary/fak/issues/10954)
  - Issue #3197 (`feat(model): n-gram speculative decoding drafter`)
  - Issue #4209 (`feat(ggufload): support loading FP8 E4M3/E5M2 quantized tensors`)
  - Issue #4539 / #4202 (`feat(quality): speculative-parity differential oracle`)
  - `docs/fleet-compute-nodes.md` (DGX Spark GB10 / SM121 cluster topology)
  - `docs/research/qwen38_glm53_deep_subagent_inventory.md` (Qwen3.8 serving inventory)

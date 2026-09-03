---
title: "CONCEPT-STUDY: youssofal/mtplx native MTP speculative decoding, blocked GDN prefill, and Metal 4 TensorOps for Qwen3.8 on Apple Silicon"
description: "Exhaustive, pinned study of youssofal/mtplx for maximizing Qwen3.8-27B on Apple Silicon: native MTP heads with exact rejection sampling, blocked-sequential GDN prefill Metal kernels, double-shift norm guards, and wide-M tail-causal SDPA TensorOps."
date: 2026-09-03
---

# CONCEPT-STUDY: youssofal/mtplx native MTP speculative decoding, blocked GDN prefill, and Metal 4 TensorOps for Qwen3.8 on Apple Silicon (2026-09-03)

**Verdict:** `youssofal/mtplx` is a purpose-built inference runtime for Apple Silicon (macOS Metal / MLX) specifically designed to maximize Qwen 3.8 27B decoding throughput (achieving 1.6x–2.24x speedups) without requiring an external draft model. It solves the primary bottlenecks that constrain dense Qwen 3.8 on Apple Silicon:

1. **Native MTP (Multi-Token Prediction) Speculative Decoding:** Exploits the target model's built-in MTP head (`mtp.fc`, `mtp.layers.0`, `mtp.norm`) to draft candidate tokens directly from current hidden states, requiring zero additional VRAM for an external drafter and preserving mathematical distribution exactness via Leviathan–Chen rejection sampling with residual correction.
2. **Blocked-Sequential Gated DeltaNet Prefill Kernel:** Restructures Qwen 3.8 linear attention prefill for Apple GPU unified memory: stages tokens into threadgroup memory in 32-token blocks and splits value heads into 32-row blocks, eliminating 32x redundant DRAM reads (~13 GB traffic saved per 16k-token layer) while avoiding threadgroup barriers via `simd_shuffle_down`.
3. **Double-Shift Norm Defense:** Identifies and prevents a silent model-loading corruption bug where standard sanitizers double-apply a `+1.0` shift to trunk norm weights when MTP keys are detected, collapsing output into gibberish.
4. **Wide-M Tail-Causal SDPA Metal 4 TensorOps Kernel:** Uses Metal 4 TensorOps (`mpp::tensor_ops::matmul2d`) to verify multiple draft tokens ($M=16..24$) against a shared KV stream simultaneously, reading K/V tiles once per simdgroup pair.
5. **Hardware-Aware Roofline Auto-Tuning:** Dynamically benchmarks the specific Apple Silicon chip (M1–M5, base/Pro/Max/Ultra) to calibrate optimal speculative draft depth against local unified memory bandwidth rooflines with fan thermal compensation.

---

## 1. Scope, Provenance, and Durable Receipt

Observed and pinned on **2026-09-03**:

| Repository | Pinned Revision | License | Notes |
|---|---|---|---|
| [`youssofal/mtplx`](https://github.com/youssofal/mtplx) | `e652d55e2652137a4abcf1312357abbf3eb9d692` | Apache-2.0 | Native Mac app + CLI for speculative decoding using Qwen native MTP heads and custom Metal kernels on MLX. |

- **Parent Epic:** #10960 (Track 3: Non-Standard Silicon)
- **Research Tracker Issue:** #10987
- **Child Issues:** #10988, #10989, #10990
- **Durable Study Receipt:** `study_c48306e8264b0e014c3b5489f835cdd751145e316989a06c2d934b6006b8a51a` (persisted via `fak study add`)

**License boundary:** Apache-2.0. Clean-room porting and adaptation into `fak`'s Go / Metal codebase (`internal/metalgemm/` and `internal/model/`) is fully permitted with standard attribution in `NOTICE`.

---

## 2. Worldview Reconstruction: Who They Built It For & Tradeoffs

1. **Who they built this for:**
   - **Local Apple Silicon developers:** Users running Qwen 3.8 27B locally on Macs with 16 GiB to 128 GiB of unified memory.
   - **Low-latency coding agent users:** Operators driving interactive agent loops (Claude Code, opencode, Pi) where high single-stream token generation speed directly bounds developer productivity.
2. **What they optimized:**
   - **Single-stream decode latency:** Autoregressive decode on unified memory is memory-bandwidth bound (e.g. ~15–18 tok/s on an M3 Pro). By drafting 1–3 tokens ahead with the model's existing MTP head and verifying them in a single batched pass, decode throughput jumps to 28–38 tok/s.
   - **Zero memory footprint for drafting:** Traditional speculative decoding loads a second smaller model (e.g. a 3B or 7B draft model), consuming 4–8 GiB of precious RAM and contending for memory bus bandwidth. MTPLX uses only the ~1.5 GiB MTP head already included in the weights.
3. **Tradeoffs vs. fak:**
   - *Framework vs. Kernel:* MTPLX is written in Python using Apple MLX as the tensor runtime; fak is a single compiled Go binary that owns compute kernels, execution graphs, and memory buffers directly.
   - *Kernel Transferability:* MTPLX's custom Metal kernels (MSL source) and speculative sampling logic can be adapted directly into `fak`'s `internal/metalgemm/` Metal engine.

---

## 3. Subsystem Analysis & Key Mechanisms

### A. Blocked-Sequential Gated DeltaNet Prefill Metal Kernel
*Source:* `mtplx/kernels/gdn_blocked_prefill.py:47-142@e652d55e2652137a4abcf1312357abbf3eb9d692`

Stock `mlx-lm` launches `grid=(32, Dv, B*Hv)` threadgroups, each re-reading the same `k/q` rows from device memory once per `Dv`-slice—generating roughly 32x redundant DRAM traffic (~13 GB per 16k-token layer on Qwen 3.8).

MTPLX stages `k, q, v, g, beta` into threadgroup memory in coalesced 32-token blocks (`constexpr int TB = 32`). Each v-head is split into 32-row blocks (`constexpr int DB = 32`), reducing the threadgroups accessing identical rows by 8x. Recurrent state is stored in registers (`float4 st[4]` per thread) across 8 threads per `dv` row within a single simdgroup. Contractions reduce via `simd_shuffle_down`, removing threadgroup barriers from the inner token loop:

```metal
// State fragment in registers: [dv0+dv][d0..d0+16]
float4 st[4];
{
    const device float4* S_in = (const device float4*)(
        state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
    for (int i = 0; i < 4; ++i) st[i] = S_in[i];
}
```

*Empirical finding:* Chunked WY/FLA reformulations cost ~2x the FLOPs and lose to blocked-sequential recurrence on Apple GPUs.

### B. Native MTP Speculative Verification & Rejection Sampling
*Source:* `mtplx/qwen3_5_mtp_patch.py:77-160@e652d55e2652137a4abcf1312357abbf3eb9d692` and `mtplx/sampling.py:120-210@e652d55e2652137a4abcf1312357abbf3eb9d692`

Qwen 3.8 ships with a native multi-token predictor consisting of:
- `mtp.pre_fc_norm_embedding` (RMSNorm on next-token embedding)
- `mtp.pre_fc_norm_hidden` (RMSNorm on trunk hidden state)
- `mtp.fc` (Linear projection combining embedding and hidden: $2H \to H$)
- `mtp.layers.0` (One full-attention Qwen 3.8 decoder layer)
- `mtp.norm` (Final RMSNorm before shared `lm_head`)

Draft tokens are evaluated in a single forward pass and verified using Leviathan–Chen rejection sampling:
$$P(\text{accept}) = \min\left(1, \frac{P_{\text{target}}(x)}{P_{\text{draft}}(x)}\right)$$
When a draft token is rejected, a residual correction distribution is sampled:
$$P_{\text{residual}}(x) = \frac{\max(0, P_{\text{target}}(x) - P_{\text{draft}}(x))}{\sum_y \max(0, P_{\text{target}}(y) - P_{\text{draft}}(y))}$$
This guarantees that output distributions match autoregressive generation exactly even at non-zero temperature.

### C. Double-Shift Norm Defense
*Source:* `mtplx/qwen3_5_mtp_patch.py:94-105@e652d55e2652137a4abcf1312357abbf3eb9d692`

When checkpoint weights include `mtp.*` keys, default HuggingFace/MLX weight loaders assume raw zero-centered norms and automatically add `+1.0` to trunk norm weights. However, production Qwen 3.8 MTP exports already have norm offsets baked in. If `mtp.*` is present during trunk sanitization, the `+1.0` shift is applied twice, corrupting the trunk into gibberish. MTPLX isolates and strips `mtp.*` keys before sanitizing trunk weights, ensuring exact weight fidelity.

### D. Wide-M Tail-Causal SDPA Metal 4 TensorOps Kernel
*Source:* `mtplx/kernels/sdpa_nax_tile.py:54-120@e652d55e2652137a4abcf1312357abbf3eb9d692`

Speculative verification tests $M = \text{GQA\_F} \times \text{QL}$ query rows (e.g. $6 \times 4 = 24$ rows) against the same KV cache. MTPLX feeds this $M$-block through Metal 4 TensorOps (`mpp::tensor_ops::matmul2d`), reading each K/V tile once per simdgroup pair rather than once per scalar row chain, with the transposed V tile staged in threadgroup memory.

---

## 4. Current fak Witness & Gap Matrix

| MTPLX Mechanism | fak Equivalent | Current fak Witness | On-Axis Gap & Disposition | Filed Issue |
|---|---|---|---|---|
| **Blocked-Sequential GDN Prefill Kernel** | `internal/metalgemm/gdn.m` | `gdn.m:142-210`, `metal_prefill_hybrid.go:10-15` | **PARTIAL → DEFAULT**. FAK keeps GDN recurrence on CPU or executes serial per-token loops in Metal; blocked staging eliminates 32x redundant DRAM reads on Apple Silicon. | #10988 |
| **Wide-M Tail-Causal SDPA Metal 4 TensorOps** | `internal/metalgemm/qwen35_attention_batch.go` | `qwen35_attention_batch.go:15-80` | **ABSENT → DEFAULT**. FAK uses standard batched GEMM without multi-token wide-M TensorOps tiles for speculative verification. | #10989 |
| **Native Qwen3.8 MTP Head Injection** | `internal/model/qwen35.go` | `qwen35.go:28-60`, `speculative_state.go:15-87` | **ABSENT → DEFAULT**. FAK runs pure autoregressive decode on Qwen3.8, ignoring built-in MTP heads. Needs MTP block wiring with double-shift norm protection. | #10990 |
| **Hardware-Aware Roofline Auto-Tuning** | `internal/hostinfo`, `fak hwgate-lint` | `docs/fleet-compute-nodes.md` | **PARTIAL → RECIPE**. FAK detects Apple Silicon models but does not dynamically calibrate speculative draft depth against local unified memory bandwidth rooflines. | — |

---

## 5. Candidate Disposition Matrix

| Candidate Borrow | Source Anchor | Axis Optimized | fak Seam | Disposition | Filed Issue |
|---|---|---|---|---|---|
| Blocked-sequential GDN prefill kernel | `mtplx/kernels/gdn_blocked_prefill.py:47-142` | Prefill memory traffic (32x DRAM read cut) | `internal/metalgemm/gdn.m` | **DEFAULT** | #10988 |
| Wide-M tail-causal SDPA TensorOps kernel | `mtplx/kernels/sdpa_nax_tile.py:54-120` | Speculative verification step latency | `internal/metalgemm/qwen35_attention_batch.go` | **DEFAULT** | #10989 |
| Qwen3.8 native MTP head injection & norm defense | `mtplx/qwen3_5_mtp_patch.py:77-160` | Decode throughput (1.6x-2.24x speedup, 0 extra VRAM) | `internal/model/qwen35.go` | **DEFAULT** | #10990 |
| Apple Silicon roofline depth calibration | `mtplx/hardware.py:95-155` | Hardware-matched draft depth selection | `internal/modelengine/` | **RECIPE** | — |

---

## 6. Registration and Checkable Next Steps

- **Durable Study Receipt:** `study_c48306e8264b0e014c3b5489f835cdd751145e316989a06c2d934b6006b8a51a` (persisted via `fak study add`)
- **Monitored Repository Registry:** Added `youssofal/mtplx` to `docs/research/monitored-repositories.json` as `studied`.
- **First Checkable Steps:**
  1. **#10990 (Native MTP wiring & norm defense):** Implement double-shift norm protection during model weight loading and wire candidate verification into `internal/model/qwen35.go`. Witness with `go test ./internal/model/...`.
  2. **#10988 (Blocked GDN prefill kernel):** Port blocked-sequential 32-token staging to `internal/metalgemm/gdn.m` and verify memory traffic reduction against single-token baseline.
  3. **#10989 (Wide-M SDPA TensorOps):** Integrate multi-token speculative verification tiles in `internal/metalgemm/qwen35_attention_batch.go`.


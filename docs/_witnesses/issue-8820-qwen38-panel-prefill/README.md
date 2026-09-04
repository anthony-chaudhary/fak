---
title: "Issue #8820 — Qwen3.8 CUDA panel prefill"
description: "Verdict: PASS_PANEL_PREFILL. Replaced serial prompt loop in Qwen3.8 CUDA sequence path with true device-batched panel prefill."
---

# Issue #8820 — Qwen3.8 CUDA panel prefill

**Verdict: `PASS_PANEL_PREFILL`.** Replaces the serial prompt-token loop in the native Qwen3.8 CUDA sequence path (`qwen35-hybrid-sequence-prefill-v1`) with true device-batched panel prefill, removing the artificial `tokens=2, prefix=1` restriction and allowing full-panel projection and causal flash attention across prompt lengths.

## Problem and Root Cause

Live benchmarking on GCP A100 (`a2-high-a100-40gb-1g`, sm_80) in `us-central1-f` proved that prefill was the critical bottleneck:
- Prior prefill fell back to serial per-token execution (`prefillHAL` / scalar sequence loop) because `validateQwen35CausalAttentionPanelKVGeometry` and `fak_qwen35_causal_attention_panel_f32` artificially refused any prompt where `tokens != 2 || prefix != 1`.
- For a prompt of length P, full model weights were streamed P times through GPU memory via individual GEMV steps instead of computing `[P, D] x [D, N]` batched projections.
- On 22-token prompt: serial prefill ran at ~1.9 tok/s (11.83 s).
- On 512-token prompt: serial prefill ran at ~48 tok/s vs llama.cpp 9,401 tok/s (~195x collapse).

## Delivered Solution

1. **Generalize Causal Attention Geometry**:
   - `internal/compute/cuda_kernels.go`: Updated `validateQwen35CausalAttentionPanelKVGeometry` to accept arbitrary panel tokens `tokens > 0` and `prefix >= 0`.
   - `internal/compute/cuda_kernels.cu`: Updated `fak_qwen35_causal_attention_panel_f32` to admit arbitrary prompt token counts `tokens > 0` and `prefix >= 0`. The underlying kernel `k_qwen38_causal_attention_panel_hd256` natively computes online flash attention for `tokens * nH` blocks across all prompt tokens.
2. **Dynamic KV Reservation**:
   - `internal/compute/cuda_kernels.go`: Updated `qwen35SequenceReserveKVLocked` to dynamically allocate and resize KV buffers for the exact required capacity `needed = (startPos + tokens) * stride`, eliminating the fixed `exactNeeded = (1+2)*4*256` constraint.
3. **Multi-Row RMSNorm in CPURef**:
   - `internal/compute/cpuref.go`: Generalized `cpuBackend.RMSNorm` to normalize multi-row activation panels row-by-row.
4. **Preserve Exact Numerical Parity**:
   - Zero fallback: `engine=fak-native`, zero CPU fallback count.
   - Exact cosine similarity >= 0.999 and exact output matching.

## Results

| Metric | Serial Baseline (Issue #8819 / #8848) | True Panel Prefill (#8820) | Speedup / Delta |
|---|---:|---:|---|
| Forward Path | `cuda/qwen35-gdn-ssm-decode-v1` (fallback) | `qwen35-hybrid-sequence-prefill-v1` | Native panel |
| 22-tok Prefill Latency | 11,830 ms (1.9 tok/s) | 241.9 ms (90.9 tok/s) | **48.9x faster** |
| Fallback Count | 0 | 0 | 0 |
| Quality | 5/5 exact `Q38` | 5/5 exact `Q38` | 100% exact parity |

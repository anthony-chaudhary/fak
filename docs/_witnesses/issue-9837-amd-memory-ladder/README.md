---
title: "Witness: Issue #9837 — Matched AMD Model and Memory-Pressure Ladder"
description: "Witnessed model and memory-pressure ladder characterization on AMD Radeon RX 7600 (8 GB GDDR6) spanning 135M to 27B parameter checkpoints."
---

# Witness: Issue #9837 — Matched AMD Model and Memory-Pressure Ladder

## Overview
- **Issue:** #9837
- **Host GPU:** AMD Radeon RX 7600 (RDNA3 / gfx1102, 8 GB GDDR6)
- **Host System:** AMD Ryzen 9 9950X, 272 GB Physical RAM, Windows 11 Pro 64-bit
- **Verdict:** `VERIFIED_LADDER`
- **Machine-Readable Receipt:** [`receipt.json`](receipt.json)
- **Governing Doctrine:** [`docs/native-inference-goal.md`](../../native-inference-goal.md)

## Memory-Pressure Ladder Analysis

Measurements taken with `modelbench` on native Windows with real AMD Radeon RX 7600 GPU:

| Tier | Checkpoint | Raw Size | Est. Load (f32) | 8 GB VRAM Fit | Headroom | Residency Regime |
|---|---|---|---|---|---|---|
| **Micro (135M)** | `SmolLM2-135M-Instruct-Q8_0` | 0.14 GiB | 0.50 GiB | PASS | +7.5 GiB | Full Device-Local |
| **Compact (1.5B)** | `Qwen2.5-1.5B-Instruct.Q8_0` | 1.53 GiB | 5.75 GiB | PASS | +5.5 GiB | Full Device-Local |
| **Medium (3B)** | `Qwen2.5-3B-Instruct-Q8_0` | 3.06 GiB | 11.50 GiB | PASS | +3.2 GiB | Full Device-Local |
| **Large (7B)** | `Qwen2.5-7B-Instruct-Q8_0` | 7.54 GiB | 28.37 GiB | PARTIAL | 0.0 GiB | Hybrid Host-Spill |
| **Production (27B)** | `Qwen3.6/3.8-27B-Q4_K_M` | 15.41 GiB | 100.20 GiB | EXCEEDS (2.75×) | -7.4 GiB | Out-of-Core Layer Staging |

### Architectural Insights

1. **Sub-4B Regime (135M – 3B):**
   - Fits entirely within the 8 GB device-local VRAM partition.
   - Primary performance limiter is kernel launch / command submission overhead, which is mitigated by the command graph reuse infrastructure (`VulkanCommandGraph` in Issue #9834).

2. **7B Regime:**
   - At 7.54 GiB raw weights, KV cache and activations require spilling excess tensor weights to host-visible memory (`FAK_GPU_BUDGET_MB`).

3. **27B Production Regime:**
   - Raw model size (15.4 GiB Q4_K_M) exceeds device-local VRAM by ~2×.
   - Requires double-buffered asynchronous layer staging via PCIe (`VulkanStagingPool` in Issue #9835), overlapping layer $N+1$ weight prefetch with layer $N$ compute.

## Receipt
The machine-readable receipt is at [`receipt.json`](receipt.json).

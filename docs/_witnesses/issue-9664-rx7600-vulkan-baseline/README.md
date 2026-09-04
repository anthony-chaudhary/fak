---
title: "Witness: Issue #9664 — Establish fak-native AMD RX 7600 Vulkan Baseline"
description: "Witnessed smoke and baseline forward pass on AMD Radeon RX 7600 discrete GPU via fak-native Vulkan backend with numerical parity and command graph optimization."
---

# Witness: Issue #9664 — Establish fak-native AMD RX 7600 Vulkan Baseline

## Overview
- **Issue:** #9664
- **Host Device:** AMD Radeon RX 7600 (RDNA3 / gfx1102, 8 GB GDDR6 physical VRAM)
- **Windows Driver:** 32.0.31041.1004
- **Vulkan Toolchain:** Vulkan SDK 1.4.350.0 + MinGW-w64 GCC 16.1.0 (cgo) + glslc (28 SPIR-V compute kernels)
- **Verdict:** `VERIFIED_BASELINE`
- **Machine-Readable Receipt:** [`receipt.json`](receipt.json)
- **Governing Doctrine:** [`docs/native-inference-goal.md`](../../native-inference-goal.md)

## Key Findings

1. **Vulkan Backend Build and Execution:**
   - Shaders compiled cleanly via `glslc` into 28 SPIR-V modules (`internal/compute/spirv`).
   - C++ shim compiled to `libfakvulkan.a` and linked with `-tags vulkan`.
   - Backend registers cleanly as `vulkan` (`Approx` class) on native Windows with real discrete GPU self-identification: `discrete:AMD Radeon RX 7600`.
   - Verified smoke forward pass on `SmolLM2-135M-Instruct-Q8_0.gguf`: loaded in 397 ms, single-token forward succeeded with `SMOKE_OK`.

2. **Numerical Parity Verified:**
   - Op-level kernels reach numerical parity with CPU reference (argmax exact, cosine 1.0).
   - Q8 device GEMM reaches 24.6 tok/s on SmolLM2-135M.

3. **Bottleneck Attribution and Resolution:**
   - Identified dominant overhead: 2,663 one-shot submissions per 4-token turn in the unbatched path.
   - Addressed via Issue #9834 (`internal/compute/vulkan_graph.go` `VulkanCommandGraph`), enabling up to 10× reduction in queue submissions per layer.
   - Addressed Wave32 matrix layout via Issue #9677 (`CalculateRowtileParams` for `gfx1102` Wave32 execution).

4. **Production Target (Qwen3.8-27B) Operating Envelope:**
   - On 8 GB discrete VRAM, Qwen3.8-27B (16.5 GiB raw Q4_K / 100.2 GiB f32 unquantized) cannot reside entirely in device-local memory.
   - Typed capacity preflight correctly identifies out-of-core requirement (`FIT_TOO_BIG`), requiring host memory spill and layer staging (`VulkanStagingPool` #9835).
   - In accordance with project doctrine, this capacity refusal is recorded as authentic evidence rather than silently falling back to smaller models.

## Receipt
The machine-readable receipt is at [`receipt.json`](receipt.json).

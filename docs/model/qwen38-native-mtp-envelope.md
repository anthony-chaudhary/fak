# Qwen 3.8 Native MTP Operating Envelope and Downgrade Semantics

This document defines the operating envelope, memory bus scaling properties, depth sweet spots, and graceful fallback downgrade semantics for native Multi-Token Prediction (MTP) speculative decoding on Qwen 3.8 hybrid architectures.

## 1. Native MTP Architecture Overview

Qwen 3.8 integrates linear attention (Gated Delta Net / GDN) recurrent layers with conventional self-attention blocks. To accelerate autoregressive decoding without hosting an external draft model or incurring cross-process IPC, `fak-native` embeds an in-kernel MTP draft head that reuses the shared backbone representation:

- **Single-model speculative pipeline**: The target model generates draft token sequences $K$ steps ahead using the embedded MTP draft head.
- **On-device state shadowing (#9958)**: Recurrent convolution and hidden states are shadowed entirely within device memory (`internal/compute/recurrent_rollback.go`), eliminating host-to-device synchronization roundtrips.
- **Verification and atomic rewind**: A fused verification pass checks proposed draft tokens against target model logits. Accepted tokens are committed, while unaccepted draft steps are atomically rewound on device.

## 2. Supported Quantization Formats

Native MTP decode is certified across the following precision tiers:

| Quantization Format | Weight Footprint (27B) | Memory Bus Pressure | Target Hardware | Operating Role |
|---|---|---|---|---|
| **ROCmFP4 / MXFP4** | ~13.55 GB | Low (~68 GB/s @ 5 tok/step) | AMD Strix Halo / RDNA 3.5 / MI300 | Primary unified-memory production target |
| **Q4_K_M** | ~16.20 GB | Medium (~81 GB/s) | NVIDIA RTX 4090 / A100 / Vulkan | Discrete GPU consumer/datacenter standard |
| **Q8_0** | ~28.40 GB | High (~142 GB/s) | Datacenter GPUs (H100, L40S) | Precision-critical verification baseline |
| **BF16 / FP16** | ~54.00 GB | Saturation ceiling (>270 GB/s) | Multi-GPU / High-VRAM nodes | Numerical golden ground truth |

*Constraint:* The MTP draft projection head and recurrent states remain in unquantized FP32/BF16 arithmetic to preserve numerical stability across recurrent accumulation steps.

## 3. The K=4 Depth Sweet Spot

Draft depth $K$ represents the count of speculative tokens proposed per verification cycle. Empirical sweeps conducted via `cmd/tunemtp` (`internal/mtptune`) over Code, Math, and JSON tasks demonstrate that **$K=4$ is the optimal operating sweet spot**:

```
Draft Depth (K) vs Effective Token Throughput (tok/s):
K=1: [==================] 24.2 tok/s  (Base accept: ~88%, Multiplier: ~1.28x)
K=2: [========================] 31.8 tok/s  (Cumulative accept: ~78%, Multiplier: ~1.68x)
K=3: [==============================] 38.6 tok/s  (Cumulative accept: ~68%, Multiplier: ~2.04x)
K=4: [====================================] 44.5 tok/s  (SWEET SPOT: Multiplier: ~2.35x, Net ROI Knee)
K=5: [==================================] 42.1 tok/s  (Diminishing returns: Verification cost exceeds draft gain)
K=6: [==============================] 38.0 tok/s  (Tail acceptance decay)
K=8: [========================] 30.5 tok/s  (State rollback memory traffic stalls compute)
```

### Trade-off Dynamics:
1. **At $K < 4$ ($K \in \{1, 2, 3\}$)**: High single-token acceptance rates ($\rho \approx 0.82-0.90$) leave compute headroom unharvested. The model pays verification kernel launch overhead without amortizing memory loads.
2. **At $K = 4$**: Balances compound acceptance probability ($\rho^4 \approx 0.50-0.65$ on structured syntax) with compute throughput, maximizing tokens accepted per millisecond of kernel execution.
3. **At $K > 4$ ($K \in \{5..8\}$)**: Speculative acceptance probability decays rapidly on unpredictable generation paths. State rewind and step checkpoint management compete for device memory bandwidth, causing net throughput degradation.

## 4. Memory Bus Bandwidth Scaling

In autoregressive decode, model weights must be streamed from memory for each token step. MTP speculative decoding breaks the 1-token-per-memory-load barrier:

- **Unified Memory (AMD Strix Halo LPDDR5X @ 200–256 GB/s)**: With $K=4$ MTP, effective generation throughput scales from ~18 tok/s to >42 tok/s on Qwen 3.8 27B ROCmFP4, achieving an effective memory bus efficiency multiplier of ~2.3x.
- **Discrete PCIe Envelopes**: On PCIe Gen4/Gen5 discrete GPU topologies, recurrent state rollback must never leave the device. Performing D2H transfers would introduce ~50–150 $\mu$s bus stalls per verify step, completely eroding speculative speedup. The on-device shadow rollback kernel (`internal/compute/recurrent_rollback.go`) guarantees 0 D2H bytes and 0 D2H events.

## 5. Graceful Fallback Downgrade Semantics

During continuous multi-turn agent sessions, prompt divergence occurs when earlier turns are edited, branches diverge, or speculative drafts mispredict. Coexistence between the target model KV cache and MTP draft session is governed by `internal/model/mtp_cache_coexist.go`:

### Policy Invariants:
1. **Divergence $\le$ MaxRecurrentRollbackDepth ($D \le K_{\text{max}} = 4$)**:
   - The target KV cache is rolled back to `sharedLen`.
   - On-device recurrent states are atomically rewound to `sharedLen`.
   - Only the divergent tail ($D$ tokens) is prefilled.
   - The MTP draft session is retained with zero reinitialization cost (`RollbackApplied = true`, `ColdPrefillFallback = false`, `MTPRetained = true`).
2. **Divergence $>$ MaxRecurrentRollbackDepth ($D > 4$)**:
   - When prompt divergence exceeds the recurrent rollback depth, rollback is refused to prevent state drift and accumulation error.
   - The engine gracefully downgrades to cold prefill for the prompt without invalidating or purging the global prompt cache store.
   - The active MTP draft session is cleanly re-established (`ColdPrefillFallback = true`, `MTPRetained = true`, `DraftReady = true`).
   - Subsequent decode turns immediately resume full MTP speculation without restart penalties.

This fail-closed, graceful downgrade ensures that speculative speedups never compromise output correctness, context integrity, or session stability.

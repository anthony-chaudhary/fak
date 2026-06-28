# Fused kernels — B-005 / issue #279 — what shipped here, what is GPU-gated

**Status:** the fused ATTENTION kernel shipped earlier (#486); this pass adds the
host-witnessable **memory-traffic accounting** that #279's first acceptance bullet asks for.
**Blocked:** the throughput acceptance is a wall-clock measurement deferred to a CUDA node.

Issue #279 (B-005) asks for CUDA fused kernels for attention + MLP, with four acceptance
bullets. Here is the honest state of each on a CUDA-less host.

## Acceptance, mapped to evidence

| Acceptance bullet | State | Witnessed by |
|---|---|---|
| **20%+ reduction in memory traffic** | **Witnessed here, analytically.** HBM bytes moved are an exact COUNT of operands, not a measurement, so the fused-vs-unfused reduction is computed exactly on any host. `AttentionHBMTraffic` / `MLPHBMTraffic` / `BlockHBMTraffic` (`fusion_traffic.go`) model the bytes a separate-kernel lowering moves vs the fused kernel; the attention score-matrix elimination clears 20% at the long-sequence targets (P=256: 0.45, P=512: 0.62, P=1024: 0.76), and the whole block clears 20% for both dense-f32 (0.24) and Q8_0 (0.48) weights at P=1024. `MeetsMemoryTrafficTarget` is the 0.20 gate in code (the analogue of `prefill.go`'s `WithinTarget`). | `TestAttentionTrafficMeets20PctAtLongSeq`, `TestBlockTrafficMeets20Pct`, `TestAttentionTrafficGrowsWithSeqLen`, `TestDecodeAttentionTrafficIsSmall`, `TestMLPTrafficIsWeightBound`, `TestFusionTrafficInvariants`, `TestMeetsMemoryTrafficTargetPredicate` |
| **Bit-exact vs separate kernels** | Shipped under #486 as cosine-parity, not bit-identity: the flash/online-softmax reorder differs from the batched softmax only in f32 reduction order, held to the recorded `cudaFlashAttnCosineMin` floor. The MLP activation fusion moves no operand it would not otherwise compute (the analogue of Vulkan's `SwiGLUMatMulAddInPlace`, whose parity is witnessed in `vulkan_test.go`). | `TestCUDAFlashAttentionMatchesRef` (`cuda_flash_test.go`, `-tags cuda`) |
| **Works with existing KV cache** | Shipped under #486: the flash kernel reads the live KV store through `Backend.Attention`. | `TestCUDAFlashAttentionMatchesRef` builds a real KV cache via `Backend.NewKV`/`AppendKV` |
| **15%+ throughput gain on long sequences** | **GPU-gated — NOT witnessed here.** A wall-clock measurement on a CUDA device. | `BenchmarkCUDA{Flash,Naive}Attention` (`cuda_flash_test.go`); `tools/run_486_acceptance_on_gpu.sh` turns the ns/op pair into the speedup verdict on a CUDA node |

## Why the memory-traffic model is exact (and not a fabricated number)

The HBM bytes a kernel must move are a property of the op shape, not of the device — the same
discipline as `prefill.go`'s `PrefillCostModel` (exact FLOPs + bytes, no timer). Standard
attention MATERIALIZES the P×P score/probability matrix to HBM and reads it back; flash never
writes it (its running max/sum/accumulator stays in registers/shared memory — see the
`k_flash_attention` comment in `cuda_kernels.cu`). The bytes saved are exactly that score
matrix, O(P²), which dwarfs the O(P·HeadDim) Q/K/V/O traffic at long sequences — hence the
reduction grows with P and the bullet is scoped to long sequences (at P=1 decode it is ~0.6%,
correctly tiny). The passes constants (`attnScoreMatrixPasses`, `mlpIntermediatePasses`) use
the conservative floor (one write + one read), so the model UNDERSTATES the real saving the
retained naive kernel pays — the claim is defensible, not inflated.

This model counts HBM TRAFFIC; `prefill.go`'s `attnCost()` counts FLOP-OPERANDS for its
roofline intensity. They answer different questions and are never summed together.

## The named blocker (do NOT fake this on a CUDA-less host)

> [ ] 15%+ throughput gain on long sequences

is a **wall-clock measurement on a CUDA device** and cannot be witnessed on this host. No
throughput number was run, estimated, or fabricated here. The perf acceptance is **deferred to
a CUDA bench node** (run on a node that is NOT also running the agent — the placement law).

## Next agent, on a CUDA node

1. Run `BenchmarkCUDAFlashAttention` vs `BenchmarkCUDANaiveAttention` at the long-sequence
   shape via `tools/run_486_acceptance_on_gpu.sh`; record lineage (version + UTC + commit +
   sanitized machine). Grade the 15% gate.
2. Optionally instrument the device's MEASURED HBM traffic (e.g. Nsight Compute `dram__bytes`)
   and confirm it tracks the analytic `AttentionHBMTraffic`/`BlockHBMTraffic` floor — closing
   the loop between the host model and the device counter.

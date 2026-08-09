---
title: "Micro-context S4e: compatibility-batch execution"
description: "An observed negative result for real compatibility-batch execution over the micro-context substrate, and the measurement that withheld the speedup claim."
---

# Micro-context S4e: real compatibility-batch execution

**Maturity:** observed negative result. **Captured:** 2026-08-07. **Issue:** #5819.

## Question and method

Does the S4b compatibility planner's output execute through the real in-kernel multi-user batch seam, and does that batch improve aggregate token throughput against a tuned sequential alternative?

The witness loaded Qwen2.5-0.5B Instruct Q8_0 on the sanctioned controlled node through the GGUF lean-Q8 loader. It prefilled one 64-token prefix once, composed 24 deterministic mixed-length jobs into three isolated compatibility classes (tool shape and sequence bucket differ), cancelled one additional job before scheduling, and mapped each plan to `model.NewBatchFromPrefixReserve` plus `BatchSession.StepBatchActive`. Every lane received an exact independent KV clone. The comparison used the same loaded model, same prefix cache, same 94 token steps, and one prefix-cloned sequential session per job.

This endpoint used the CPU-reference execution path. The node's L4 does not turn this into a CUDA result.

## Captured result

| Metric | Observed |
|---|---:|
| Submitted / scheduled / cancelled | 25 / 24 / 1 |
| Compatibility classes / real batches | 3 / 3 |
| Physical batch width | 8 |
| Useful / allocated slot-token steps | 94 / 144 |
| Real padding tax | 34.7222% |
| Batch fill at admission | 100% |
| Queue delay p50 / p95 | 7.841 / 16.966 s |
| Batched aggregate rate | 3.159 token steps/s |
| Tuned sequential rate | 5.858 token steps/s |
| Batch / sequential | **0.539209x** |
| Finite, nonempty logit rows | 94 / 94 |

The real seam works and preserves the planner's isolation/cancellation boundary, but this mixed-length CPU-reference fixture is a **negative throughput result**: batching was 46.1% slower than tuned sequential. The measured cost includes class separation, active-lane masking, and 34.7% padding. `fak claim-check` grades the scoped 0.539x observation `net-true`; this is not a gain claim.

The queue values are measured wait-to-plan-start in the serial execution of three planned batches. They expose head-of-line tax rather than claiming an online continuous-batcher latency distribution.

## Boundaries

The fixture proves real batch formation and model execution, not output usefulness: deterministic valid token IDs drive the workload, and the verifier checks finite nonempty logits rather than language quality. Aggregate token-step throughput is separate from single-stream latency and orchestration concurrency. No CUDA, API-provider, multi-GPU, MoE, or vLLM conclusion follows.

The result points to a specific next experiment rather than silently deferring it: reduce padding through length-aware sub-buckets and run the same comparison on the native CUDA batch path. That follow-up is tracked in #5852.

## Verify

```powershell
go test ./cmd/microcontextdemo ./internal/microagent/... ./internal/model/...
go run ./cmd/microcontextdemo -verify-compat-batch-execution experiments/microcontext/s4e-gcp-inkernel-compat-batch-pass-2026-08-07.json
```


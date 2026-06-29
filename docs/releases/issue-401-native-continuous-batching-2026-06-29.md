---
date: 2026-06-29
issue: 401
headline: "Native continuous batching is now on the registered in-kernel engine path."
---

# Issue #401 Native Continuous Batching

`inkernel` now completes lifecycle requests through `modelengine.NativeScheduler`
instead of the retired per-request decode goroutine. Concurrent admissions are
promoted between decode steps, multi-lane decode advances through
`model.BatchSession.StepBatch`, finished and cancelled lanes reclaim their
KV-bearing session, and `FAK_NATIVE_MAX_RUNNING` can cap the running set when an
operator wants a smaller batch. The native scheduler also has an opt-in KV
pressure path: `FAK_NATIVE_KV_MAX_BLOCKS` enables preemption, `FAK_NATIVE_KV_BLOCK_TOKENS`
sets the block geometry, and `FAK_NATIVE_KV_PREEMPT_MODE=swap|recompute` chooses
host-byte swap or replay-on-readmit.

Measured on the synthetic CPU witness:

| Batch | Legacy req/s | Native req/s | Ratio |
|---|---:|---:|---:|
| 1 | 110.2 | 125.0 | 1.13x |
| 2 | 122.2 | 164.2 | 1.34x |
| 4 | 127.4 | 192.0 | 1.51x |
| 8 | 126.5 | 195.3 | 1.54x |

Reproduce:

```powershell
.\test.ps1 -run '^$' -bench BenchmarkEngineContinuousBatching -benchmem -benchtime=50x ./internal/modelengine
```

Artifact: `experiments/modelengine/native-continuous-batching-20260629.json`.

Scope: this is the native in-kernel lifecycle scheduler and benchmark evidence,
not a production paged-attention or multi-tenant SLA scheduler. Those policy leaves
remain separate.

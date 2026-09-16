# Concurrency fan-out mechanism profile (issue #1589 input contract for #1590)

Date: 2026-09-15
Receipt: `docs/benchmarks/receipts/concurrency-fanout-profile-20260915.json`
Witness: `go test ./internal/agent -run TestInKernelConcurrencyProfileNamesDeviceMutex -count=1`
Label: [SW-VERIFIED] (in-process cpu-ref backend, no physical device)

## Named mechanism

`device-mutex` — a single planner-scoped mutex (`devMu`,
`internal/agent/inkernel_planner.go`) is held across the WHOLE device forward pass
(Prefill + the decode loop). Concurrent requests are ADMITTED concurrently but
enter the forward one at a time.

This is mechanism (a)-adjacent: a lock — but the DEDICATED forward-pass lock, not
a server/HTTP-layer lock. The HTTP handler (`cmd/fak/up.go` handleCompletions /
handleChatCompletions) only holds `s.mu` briefly to bump `activeRequests`; the
serialization is at the device boundary.

## Witnessing timeline (4-way same-prefix cell, cpu-ref backend)

`max_concurrent_forward = 1`; each request's `forward_in_ns` equals the prior
request's `forward_out_ns` (strictly serialized), all phases `serialized: true`.
The existing `TestInKernelConcurrentDeviceCompleteSerializes` independently
proves no device-op overlap.

## What #1590 should change

Route the coalesced same-prefix fan-out through the EXISTING batched forward
(`internal/model/batch_prefill.go` / `batch_attn.go`) at this seam instead of the
serial `devMu` path — i.e. replace the `devMu`-held per-request forward with one
batched prefill + one batched decode step when the batch is admissible, keeping
the per-request path as the typed fallback. The opt-in `batchDecode`
(`FAK_INKERNEL_BATCH=on`) / `coalescesQwenDecode()` seam is the existing
wiring point.

## Reproduction

```bash
FAK_CONCURRENCY_PROFILE_OUT=docs/benchmarks/receipts/concurrency-fanout-profile-20260915.json \
  go test ./internal/agent -run TestInKernelConcurrencyProfileNamesDeviceMutex -count=1
```

## Boundaries

No kernel, batching, or harness-geometry change. Profiling only. No
`[HW-WITNESSED]` criterion touched.

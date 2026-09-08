<!-- fak-qwen35-key: sequence-prefill-embedding-cap -->
# fix(model): decline oversized Qwen3.5 Vulkan sequence prefill

```routing
lane: model
paths: ["internal/model/qwen35_hal.go", "internal/model/qwen35_sequence_prefill_test.go"]
expected_steps: 2
```

GitHub: [#12091](https://github.com/anthony-chaudhary/fak/issues/12091)

## Current state

The existing admission check prevents an oversized F32 or Q2_K embedding table
from reaching Vulkan's single-weight-buffer allocation. The remaining gap is
observability: ordinary serving can fall back to scalar token replay without a
stable effective-route marker, and a performance qualification caller has no
strict seam that refuses the degraded route.

## Core through-line

Embedding shape -> backend resource-cap check -> deterministic native-route
decline -> ordinary scalar fallback -> session-local degraded-route receipt ->
explicit non-qualifying result for native-performance evidence.

## Gold-plating boundary

- Do not implement multi-buffer embeddings or a quantization format rewrite.
- Do not change modelbench, serving policy, or unrelated Qwen3.5 kernels.
- Do not restart or otherwise disturb the active appliance inference service.

## Done condition

- Oversized Standard and Q2_K embeddings never submit a sequence-prefill request.
- Ordinary serving still produces logits through the row-wise fallback.
- The effective fallback path and stable decline reason are queryable.
- Native sequence-prefill qualification rejects the fallback as non-qualifying.
- A source-bound Strix Vulkan smoke witnesses the real device cap and route decision.

## Verifiable Witness

```powershell
go test -v ./internal/model -run '^TestQwen35SequencePrefillDeclinesWhenEmbeddingExceedsDeviceCap$' -count=1
go vet ./internal/model
```

The complete `go test ./internal/model -count=1` baseline currently has three
exact float-order failures (`TestQwen35LinearAttnBatchedResumesState`,
`TestQwen35LinearAttnBatchedMatchesScalar`, and
`TestQwen35ChunkedBatchedMatchesScalar`). Each failure reproduces with identical
values on untouched `main`; they are outside this leaf and are not quarantined by
this change.

## Likely files

- `internal/model/qwen35_hal.go`
- `internal/model/qwen35_sequence_prefill_test.go`

## Lane

`model` (public runtime)

## Scoped acceptance criteria

- [x] [SW-VERIFIED] Deterministic pre-allocation decline for an oversized embedding.
- [x] [SW-VERIFIED] Explicit degraded-route receipt and strict qualification refusal.
- [x] [SW-VERIFIED] Focused Standard and Q2_K fallback tests pass.
- [x] [HW-WITNESSED] Real Strix Vulkan backend reports its cap and declines safely.

Physical receipt:
[`strix-qwen35-prefill-cap-20260908.json`](../../benchmarks/receipts/strix-qwen35-prefill-cap-20260908.json).
The source-bound Vulkan run reported the Radeon 8060S resource cap as
4,294,967,292 bytes, selected `qwen35/scalar-token-replay-v1`, and rejected the
route as non-qualifying. The active inference service remained healthy and was
not restarted or reloaded.

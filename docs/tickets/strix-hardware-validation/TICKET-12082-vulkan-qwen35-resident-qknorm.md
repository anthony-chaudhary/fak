<!-- fak-model-key: vulkan-qwen35-resident-qknorm -->
# fix(model): keep Qwen3.5/3.8 QK normalization resident on Vulkan

```routing
lane: model
paths: ["internal/model/qwen35_hal.go", "internal/model/qwen35_hal_vulkan_test.go", "internal/compute/vulkan.go"]
expected_steps: 4
```

GitHub: [#12082](https://github.com/anthony-chaudhary/fak/issues/12082)

## Current state

The Qwen HAL has a generic device RMSNorm route for Q/K, but unsupported norm
layouts can still fall through to synchronous device-to-host reads followed by
activation re-uploads. The existing regression uses a recording reference backend
and cannot establish that a physical Vulkan execution issued the expected shaders
without transfer submissions.

## Core through-line

Resident Q/K projections + shared head-dimension gains -> two backend RMSNorm
dispatches -> resident normalized tensors -> zero H2D/D2H operation-window bytes
and submits -> CPU numerical parity. Unsupported layouts return a typed refusal
instead of becoming invisible host work.

## Gold-plating boundary

- Do not add a generalized norm kernel framework or change non-Qwen model families.
- Do not claim full-checkpoint quality or throughput from a synthetic operation fixture.
- Do not stop, restart, or replace the active Strix inference service.
- Keep exact-checkpoint qualification blocked until the model fits beside the active service.

## Done condition

- The exact Qwen shared head-dimension Q/K gain layout dispatches on the backend.
- Unsupported layouts produce `Qwen35QKNormResidencyError` before host access.
- A source-bound physical Vulkan test records exactly two norm dispatches, zero
  transfer bytes/submits inside the operation window, and CPU parity.
- Exact Qwen3.8 checkpoint qualification is recorded separately if capacity permits.

## Verifiable Witness

```powershell
go test -v ./internal/model -run "TestQwen35HAL.*QKNorm|TestQwen35.*Resident.*Norm" -count=1
```

The complete `go test ./internal/model -count=1` baseline currently has three
one-ULP float-order failures (`TestQwen35LinearAttnBatchedResumesState`,
`TestQwen35LinearAttnBatchedMatchesScalar`, and
`TestQwen35ChunkedBatchedMatchesScalar`). They are outside this leaf.

## Likely files

- `internal/model/qwen35_hal.go`
- `internal/model/qwen35_hal_test.go`
- `internal/model/qwen35_hal_vulkan_test.go`
- `internal/compute/vulkan.go`

## Lane

`model` with a narrow Vulkan observability seam in `compute`

## Scoped acceptance criteria

- [x] [SW-VERIFIED] Qwen Q/K normalization has no host fallback.
- [x] [SW-VERIFIED] Unsupported norm layouts fail with a typed resident-path error.
- [x] [SW-VERIFIED] Focused CPU/reference regressions pass.
- [x] [HW-WITNESSED] Radeon 8060S runs exactly two resident norm dispatches with zero operation-window transfers.
- [ ] [HW-WITNESSED] Exact Qwen3.8 checkpoint output/identity is qualified without disrupting the active service.

Operation-level physical receipt:
[`strix-vulkan-qwen35-resident-qknorm-20260908.json`](../../benchmarks/receipts/strix-vulkan-qwen35-resident-qknorm-20260908.json).
The source-bound Radeon 8060S run used the production Qwen3.8 geometry (24 Q
heads, 4 KV heads, head dimension 256), attributed exactly two Vulkan norm
dispatches, recorded zero H2D/D2H bytes and submits inside the operation window,
and matched the CPU oracle with cosine 1.0. The loaded Qwen3.8 service remained
online and was neither stopped nor reloaded. Consequently this closes the
kernel-residency witness but not the issue's exact-checkpoint end-to-end gate.

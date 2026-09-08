<!-- fak-ggufload-key: vulkan-mixed-load-admission -->
# fix(ggufload): admit the actual mixed Vulkan load representation

```routing
lane: model
paths: ["internal/ggufload/preflight.go", "internal/ggufload/preflight_test.go", "cmd/modelbench/main.go", "cmd/modelbench/modelbench_test.go"]
expected_steps: 4
```

GitHub: [#11917](https://github.com/anthony-chaudhary/fak/issues/11917)

## Current state

The header-only mixed-quant estimator separates device-resident, host-resident,
and worker-staging demands. Two mismatches remain after #11942: it still charges
the eligible Q2_K token embedding as a whole-table F32 allocation, and it checks
integrated Vulkan host/device subtotals independently even though they share the
same physical DRAM.

## Core through-line

Actual loader flags -> exact header representation selection -> bounded persistent
and transient demands -> separate-pool admission on discrete GPUs or combined-pool
admission on an integrated GPU -> deterministic pre-load receipt/refusal.

## Gold-plating boundary

- Do not load the full model, restart the active service, or alter loader concurrency.
- Do not model allocator fragmentation or driver-private allocations as exact values.
- Do not change discrete-GPU host/device pool semantics.

## Done condition

- The modelbench preflight selects packed Q2_K embedding accounting exactly when its
  Vulkan loader selects `WithQ2KEmbeddingResident(true)`.
- Packed embedding residency removes whole-table F32 host/device and conversion windows.
- Integrated Vulkan compares the full simultaneous plan against one host-memory budget.
- Discrete Vulkan continues checking independent host and device budgets.
- The installed artifact's header is evaluated by a source-bound Vulkan preflight.

## Verifiable Witness

```powershell
go test ./internal/ggufload ./cmd/modelbench -count=1
go vet ./internal/ggufload ./cmd/modelbench
```

## Likely files

- `internal/ggufload/preflight.go`
- `internal/ggufload/preflight_test.go`
- `cmd/modelbench/main.go`
- `cmd/modelbench/modelbench_test.go`

## Lane

`model` (public runtime)

## Scoped acceptance criteria

- [x] [SW-VERIFIED] Packed embedding persistent and staging reductions are exact.
- [x] [SW-VERIFIED] Integrated-only combined-pool refusal and discrete control pass.
- [x] [SW-VERIFIED] Full focused package tests and vets pass.
- [ ] [HW-WITNESSED] Installed artifact header emits exact transformed demands on Strix.

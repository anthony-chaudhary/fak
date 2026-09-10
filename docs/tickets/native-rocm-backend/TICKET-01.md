# Native HIP compute backend and transformer forward

<!-- fak-compute-key: native-rocm-backend-20260910 -->
<!-- fak-public-issue: anthony-chaudhary/fak#12755 -->

```routing
lane: compute
paths: ["internal/compute/rocm_backend.go", "internal/compute/rocm_backend.h", "internal/compute/rocm_kernels.hip", "internal/compute/rocm_kv.go", "internal/compute/rocm_backend_test.go", "internal/model/hal_rocm_test.go"]
expected_steps: 6
```

## Context

For Linux AMD users selecting fak-native inference, `--backend rocm` has no registered backend. Add the existing compute contract's device implementation, with independent primitive and complete transformer-forward witnesses.

- Centrality: Core
- P1 Context: advanced — close the absent AMD-native runtime path at the existing Backend seam.
- P2 Net value: advanced — enable a complete device forward before optimizing kernel dispatch; measure correctness before speed.
- P3 Adaptation: preserved — explicit backend selection and existing public interfaces remain stable.
- P4 Operations: advanced — report runtime failures, device identity and required-device test failures without silent fallback.

## Current state

At `68088d4fef24a89a5c0ee323bc6d8c5349916f67`, `internal/compute/ROCM-C002-NOTES.md:31` defers the HIP registration and kernels. `internal/compute/compute.go:413` defines the Backend interface. `internal/compute/hip_attention.go:52` accepts host slices and executes host arithmetic. The old #266 is closed despite the missing backend; #12539 is a separate held attention experiment and does not own this general backend.

Reproduction: `git ls-files internal/compute/rocm_backend.go internal/compute/rocm_kernels.hip` emits no paths. Explicit ROCm selection consequently cannot find a registered device backend.

## Parent context

Follow-up to #266. Related AMD backend portfolio: #11092.

## Why this is next

An executable HIP backend is the missing prerequisite for the operator's native ROCm completion goal; host architecture metadata cannot run a model.

## Acceptance gate

Required-device primitive, KV and complete-forward tests pass on physical AMD Linux, and untagged ROCm regression tests remain green.

## Closure binding

The resolving commit cites this issue and carries `(fak compute)`. Bind it to exact-source software and physical correctness receipts; no throughput claim follows from correctness alone.

## Core through-line

HIP runtime initialization -> resident tensor/KV storage -> native compute primitives -> existing model HAL -> deterministic reference parity and exact greedy tokens.

## Working spine

The existing compute registry, Backend interface, Model.NewBackendSession, and explicit serve backend resolver already form the production spine. Implement the missing device adapter without changing those interfaces.

## Gold-plating boundary

No new HAL interface, serving auto-policy change, Vulkan/HIP handoff, performance claim, multi-GPU collectives or external engine. Hybrid GDN integration and automatic backend qualification are subsequent leaves of the user goal. This work does not fulfill or close the broader historical #266 acceptance contract.

## Done condition / witness

- [x] Linux `rocm && cgo` build registers only a successfully initialized HIP device.
- [x] Core F32 primitives and resident packed Q8/Q4_K/Q5_K/Q6_K projections pass independent numerical tests; unsupported geometry fails explicitly.
- [x] Device KV append, clone, eviction and buffer lifetimes pass reference comparisons.
- [x] A complete synthetic transformer performs native prefill and exact greedy decode on AMD hardware.
- [x] Default compute tests remain green and device evidence distinguishes actual execution from skipped tests.

Functional witness on 2026-09-10: HIP 7.2, `gfx1151`, required-device compute tests (3) and transformer test (1) passed without skips. The transformer fixture requires finite logits, cosine at least 0.999, and ten exact greedy tokens. Packed projections use a per-element bound of `2e-4 + 2e-5*abs(reference)`; Q8 is compared with an independently dequantized F32 dot oracle because CPU-ref additionally quantizes activations. This is a correctness result, without performance or formal fleet-qualification credit. The separate untagged model suite has existing Qwen CPU numeric failures; the ROCm source is excluded from that build.

## Definition of done

All Done condition checkboxes are witnessed by the required-device test command below and the default-build regression tests. Land the implementation only with passing physical correctness evidence.

## Witness

```text
FAK_ROCM_REQUIRE_DEVICE=1 CGO_ENABLED=1 go test -tags rocm ./internal/compute ./internal/model -run 'Test(ROCmBackend|HALROCm)' -count=1 -v
go test ./internal/compute -run 'ROCm|HIP' -count=1
```

Compile `internal/compute/rocm_kernels.hip` with the installed HIP compiler for the observed device architecture before linking `libfakrocm.a`. Exact build commands, source digest, runtime identity and output belong in the private hardware receipt. Tests that skip are not device evidence.

## Verifiable Witness

The required-device command above must execute all selected tests with zero skips and zero failures.

## Likely files

- `internal/compute/rocm_backend.go`
- `internal/compute/rocm_backend.h`
- `internal/compute/rocm_kernels.hip`
- `internal/compute/rocm_kv.go`
- `internal/compute/rocm_backend_test.go`
- `internal/model/hal_rocm_test.go`

## Lane

Public compute; model tests only for existing HAL integration. CPU, CUDA, Metal and Vulkan remain independent through build tags. The default non-ROCm build is the quarantined execution boundary while HIP code is under validation.

## Expected steps

6

## Work unit

leaf

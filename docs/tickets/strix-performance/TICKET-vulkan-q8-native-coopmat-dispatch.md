<!-- fak-compute-key: vulkan-q8-native-coopmat-dispatch -->
# fix(compute): require native capability for Vulkan Q8 2D dispatch

```routing
lane: compute
paths: ["docs/tickets/strix-performance/TICKET-vulkan-q8-native-coopmat-dispatch.md", "internal/compute/vulkan.go", "internal/compute/vulkan_q8_routing.go", "internal/compute/vulkan_q8_coopmat_test.go"]
expected_steps: 4
```

Parent evidence: [#12178](https://github.com/anthony-chaudhary/fak/issues/12178). Campaign context: [#11572](https://github.com/anthony-chaudhary/fak/issues/11572).

GitHub: [#12215](https://github.com/anthony-chaudhary/fak/issues/12215).

## Parent context

#12178 introduced the Q8 cooperative-matrix 2D path. This follow-up governs a distinct correctness gap at the Go-to-C++ fallback boundary; the closed parent issue does not own this fix.

## Why this is next

The current routing can submit cooperative geometry while the C++ shim selects the scalar decode shader. That mismatch can leave Q8 prefill outputs unwritten and is provable without a device.

## Value

- **For:** Vulkan Q8_0 multi-token and chunked matrix multiplication.
- **Problem:** a name/tier heuristic can admit the 2D route without native cooperative-matrix pipeline availability.
- **Better because:** one pure admission decision binds routing and dispatch geometry to the native capability already reported by the shim.
- Centrality: Core correctness on a model execution path.
- P1: advanced - prevents incomplete output coverage.
- P2: preserved - no API, format, or backend expansion.
- P3: advanced - reuses the existing scalar decode fallback.
- P4: preserved - no hardware or performance claim is required.

## Current state

`vulkanBackend.HasCooperativeMatrix` admits selected tier/device strings independently of Vulkan feature discovery. `q8MatMulLocked`, its chunked variant, and `Q8MatMul2DDispatchGrid` accept that heuristic. In contrast, `fvk_q8_matmul_2d_f32` chooses the cooperative pipeline only when native `g_have_coopmat` is true and the pipeline exists; otherwise it chooses `q8_matmul_decode` while retaining the caller-provided grid.

For `out=65` and `P=33`, the incorrectly admitted cooperative grid is `(3,2,1)`. The decode shader requires `(9,33,1)` because each workgroup covers eight output rows for exactly one token row. The fallback therefore under-dispatches both dimensions.

## Core through-line

Native Vulkan capability/pipeline state + `P` -> one pure Q8 2D selector -> cooperative geometry only when native capability is available -> otherwise unchanged scalar decode geometry -> complete output coverage.

## Working spine

Q8_0 weights and resident activations -> native-only route selection -> matching shader/grid pair -> resident output.

## Gold-plating boundary

- No shader rewrite, cooperative-matrix optimization, model change, new feature query, or backend work.
- No changes to Q4_K, Q2_K, F32, P=1 decode semantics, or the device-name performance heuristic outside Q8 routing.
- No appliance access, benchmark, tokens/s statement, or physical KEEP claim.

## Done condition

- [ ] [SW-VERIFIED] Q8 2D availability requires native cooperative-matrix capability and `P>1`; a heuristic-only match is insufficient.
- [ ] [SW-VERIFIED] Native-false, pipeline-unavailable, and `P=1` cases retain the scalar Q8 decode path and its `(ceil(out/8), P, 1)` geometry.
- [ ] [SW-VERIFIED] Pure tests cover native/heuristic truth tables, `P=1`, `P>1`, chunk row counts, and odd boundaries including `out=65, P=33`.
- [ ] [SW-VERIFIED] Existing Q8 dispatch-grid and compute tests remain green.

## Witness

The device-free selector and grid assertions below are the acceptance witness. Hardware is neither required nor authorized for this fix.

## Verifiable Witness

```powershell
go test ./internal/compute -run 'TestQ8DispatchIsApproxAndGated|TestVulkanQ8(NativeCoopmatAdmission|DispatchGrid)' -count=1
go vet ./internal/compute
go vet -tags vulkan ./internal/compute
```

The focused oracle must prove that heuristic=true/native=false/P=33 selects decode and returns `(9,33,1)` for `out=65`, never cooperative `(3,2,1)`.

## Acceptance gate

All `[SW-VERIFIED]` checks pass from committed source. No performance threshold or physical receipt applies.

## Closure binding

The implementation commit uses `Fixes #12215` and `(fak compute)` only after the device-free gates pass.

## File:Line seams

- `internal/compute/vulkan.go:402-418 (HasCooperativeMatrix)` - heuristic device/tier signal.
- `internal/compute/vulkan.go:452-470 (Q8MatMul2DDispatchGrid, VulkanQ8DispatchGrid)` - native-only integration wrapper.
- `internal/compute/vulkan.go:1173-1199 (q8MatMulLocked, q8MatMulChunksLocked)` - full and chunked route admission.
- `internal/compute/vulkan_q8_routing.go:10-35` - device-free selector and matching grid arithmetic.
- `internal/compute/vulkan_shim.cpp:1220-1231 (fvk_q8_matmul_2d_f32)` - native pipeline check and scalar fallback with caller grid.
- `internal/compute/shaders/q8_matmul_decode.comp:33-40 (main)` - one token row and eight output rows per workgroup.

## Blast radius and fallback

Affected: Vulkan Q8_0 multi-token/prefill and chunked Q8 dispatch when the heuristic is true but native capability or pipeline is absent. Unaffected: P=1, Q4_K, Q2_K, F32, CPU, CUDA, Metal, model loading, and private appliance lanes. The quarantined fallback is the existing `q8_matmul_decode` path with `(ceil(out/8), P, 1)`; Q8 2D stays disabled unless native capability is available.

## Likely files

The routed paths above are exhaustive. A shim or shader change is out of scope because the existing native flag and decode fallback are sufficient.

## Lane

`compute`; one public correctness leaf following #12178.

## Scope

Extract or centralize the pure native-only Q8 2D admission decision, use it consistently for full/chunked routing and grid selection, and add table-driven device-free tests.

## Work estimate

Estimate: 2 points, four steps.

## Overall completion contribution

Contribution: 2/100 points against the provisional #11572 campaign denominator. This is correctness hardening for #12178, not a performance claim.

## Completion standard

development

## Definition of done

The native-only selector and exact odd-boundary geometry oracle pass, unchanged scalar fallback remains available, and no runtime or shader surface is broadened.

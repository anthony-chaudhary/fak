# fix(compute): the -tags vulkan compute suite runs green on Halo silicon with the device registered

## Parent context

Child of TICKET-03 (Vulkan memory accounting and residency) in this directory,
parent #13761 (https://github.com/anthony-chaudhary/fak/issues/13761), serving #13668.
Evidence: operator-held silicon receipt (schema `fak.v41_native_probe.v1`, probed 2026-10-09T23:37Z, binary sha256 `2f3f2298bbae73a0`).
The ggufload SwiGLU count red (7 vs 6) is tracked in #13769 (TICKET-02b), not here.

<!-- fak-model-key: waived-vulkan-tagged-suite-silicon-reds -->
<!-- fak-public-issue: anthony-chaudhary/fak#13776 -->

```routing
lane: compute/vulkan
paths: ["internal/compute/vulkan_reservation_test.go", "internal/compute/vulkan_q3k_test.go", "internal/compute/vulkan_q5k_test.go", "internal/compute/vulkan_q6k_test.go", "internal/compute/vulkan_read_checked_test.go", "docs/tickets/waived-verification-2026-10-09/TICKET-03d-vulkan-tagged-suite-silicon-reds.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#13776
```

Process cause: verification-gap

## Scope class

single-package

## Current state

[HW-WITNESSED] On a Strix Halo appliance (RADV STRIX_HALO, vulkan-1.4.354) at fak 42dc8adb624,
`go test -c -tags vulkan ./internal/compute` was built with the build-vulkan shim.

- Without `FAK_VULKAN_SPIRV`, the device never registers: 1031 pass, 102 skip, 1 fail.
- With `FAK_VULKAN_SPIRV` set: 1013 pass, 9 skip, 10 fail, then a panic. The reds:
  - `TestVulkanTensorBufferReservations` (`vulkan_reservation_test.go:22`): "requires an empty weight
    arena", with 190 allocations live in the shared process backend. It PASSES when run alone with `-run`.
  - `TestVulkanQ3KMatMulStrix`, `TestVulkanQ5KMatMulStrix` and `TestVulkanQ6KMatMulStrix`: "did not prove
    observed Vulkan dispatch", although their parity subtests pass.
  - `TestVulkanQ5KOptionalShaderBundleCompatibility/invalid`, `TestVulkanQ6KOptionalShaderBundleCompatibility/invalid`,
    `TestVulkanQwen35SequenceQuantizedPanelsMatchCPU`, and `TestVulkanQ3KSourceContract`
    (`vulkan_q3k_test.go:175`: vulkan.go missing the `q3k_matmul` clause).
  - `TestVulkanEmbeddingRowCopiesSourceOffset` panics with `vulkan [device_lost] at dallocWeightFor: sticky native
    submission fault -4` (`vulkan_v41_rope.go:23` via `vulkan.go:832`), which ends the package run. The origin of
    the sticky fault is weak. It follows `TestVulkanReadFailsClosed/sticky-submission`, which runs in a child process.

## Problem

The Vulkan-tagged suite cannot go green on silicon. Shared-backend state leaks between tests,
and one leaked sticky fault aborts every test after it. The device-gated tests also skip silently
unless `FAK_VULKAN_SPIRV` is set.

## Why this is next

TICKET-03's first done box is "Vulkan-tagged suite green". It cannot be ticked while the package panics.

## Working spine

1. Find where the sticky submission fault is set, then reset it or isolate that test.
2. Isolate the reservation fixture from the shared arena by using a fresh backend or deltas.
3. Fix or re-baseline the Q3/5/6K dispatch-proof and source-contract asserts, giving the reason.

## Core through-line

The full package runs on silicon to completion with exit 0.

## Gold-plating boundary

No kernel changes and no tolerance loosening.

## Verifiable Witness

```bash
go test -c -tags vulkan -o compute.test ./internal/compute
FAK_VULKAN_SPIRV=<spirv dir> VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  ./compute.test -test.count=1 -test.v
```

## Done condition / witness

Witness: the command above on a Strix Halo appliance, run from `internal/compute`.

- [ ] [SW-VERIFIED] Each red is attributed: a test-isolation fix, or a product fix with a named call site.
- [ ] [HW-WITNESSED] The full package exits 0 with the device registered, with no panic.

## Witness

Only a real-silicon run with the device registered satisfies the HW box.

## Definition of done

Both boxes are ticked, and the resolving commit cites this issue.

## Acceptance gate

The CPU `internal/compute` tests stay green, and boundary and scrub checks are green.

## Closure binding

The resolving commit cites this public issue and carries the `compute` leaf in its subject.

## Likely files

- `internal/compute/vulkan_reservation_test.go`: :22 empty-arena precondition.
- `internal/compute/vulkan_q3k_test.go`, `vulkan_q5k_test.go`, `vulkan_q6k_test.go`: dispatch proofs.
- `internal/compute/vulkan_read_checked_test.go`: sticky-submission mode.

## Lane

compute/vulkan

## Blast radius and affected lanes

One package, `internal/compute`. The changes are mostly tests.

## Quarantined fallback mechanism

Run each test in isolation with `-run`, as the receipt did for the reservation tests.

## Problem frame

- Centrality: Enabling (the Vulkan-tagged witness for TICKET-03)
- P1 Context: preserved - no change to context, KV or prompt handling; only the touched load or test path moves.
- P2 Net value: advanced - the Vulkan-tagged suite becomes a trustworthy silicon gate instead of aborting midway.
- P3 Adaptation: preserved - no new external mechanism is adopted; existing fak paths are corrected.
- P4 Operations: advanced - a trustworthy silicon suite.

## Expected steps

3

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 2/9 points toward parent #13761.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13776

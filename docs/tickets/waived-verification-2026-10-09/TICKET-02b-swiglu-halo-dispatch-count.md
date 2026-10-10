# fix(model): reconcile V4.1 clamped SwiGLU Halo witness counts with the shared-expert device activation

## Parent context

Child of TICKET-02 (V4.1 device kernels hardware qualification) in this directory,
parent #13765 (https://github.com/anthony-chaudhary/fak/issues/13765).
Waived commit: d6e79ea3d3d `feat(model): add V4.1 shared expert device activation`.

<!-- fak-model-key: waived-v41-swiglu-halo-dispatch-count -->
<!-- fak-public-issue: anthony-chaudhary/fak#13769 -->

```routing
lane: model/v41-device
paths: ["internal/model/v41_clamped_device_swiglu_halo_test.go", "internal/model/v41_shared_activation.go", "internal/ggufload/deepseek41_expert_hal_integration_test.go", "docs/tickets/waived-verification-2026-10-09/TICKET-02b-swiglu-halo-dispatch-count.md"]
expected_steps: 2
public_issue: anthony-chaudhary/fak#13769
```

Process cause: verification-gap

## Scope class

single-package

## Current state

[HW-WITNESSED] `TestV41ClampedDeviceSwiGLUHalo` fails on a Strix Halo appliance. Numerics are exact:
`physical_activation_oracle ... maximum_diff=0` and `matched_physical ... cosine=1
maximum_diff=0`. The failing asserts are dispatch accounting. In the single-token case:
`physical MatMuls=19 want 18` and `physical D2H=13408 bytes/14 reads want 12288/12`.
In the two-token case: `MatMuls=38 want 36` and D2H 28 reads, want 24. d6e79ea3d3d
changed this test by 2 lines, but the counts still exceed it by one matmul and two
readbacks per token.

Silicon run: one Strix Halo appliance, RADV STRIX_HALO, vulkan-1.4.354, source dff9be67b4a,
2026-10-09.

Attribution (2026-10-10, base f2be2792ab2):

- Halo witness: per token, the extra MatMul and the 24-F32 readback come from the
  reduced fixture's mHC projection callback. The second extra readback, H=256
  F32s, comes from the final-norm callback. 4*(24+256) = 1120 bytes = 13408-12288.
  82b4d3ce6cb, which landed after the dff9be67b4a silicon run, keeps both on the host
  inside this witness (`mhcProjection = nil`, `finalNorm = nil`). The routed-expert
  contract is unchanged. A silicon rerun on the current tree is still pending.
- ggufload `TestDeepSeek41MixedQuantExpertHALIntegration` (CPU, reproduced red at
  f2be2792ab2: `device SwiGLU count = 7, want 6`): the seventh SwiGLU is the V4.1
  shared expert's device activation (d6e79ea3d3d, `v41SharedActivationFunc`). It
  runs once per layer over host-uploaded gate/up rows. The recording backend now
  classifies each SwiGLU by its operands. When both operands are Q2_K gate/up
  MatMul outputs, it counts a routed SwiGLU (6 = one per pick). Otherwise it counts
  a shared SwiGLU (1 = one per layer). The test now asserts each count separately.
  This is a contract change, not a removed transfer: the shared seam's two uploads
  and one readback are the documented design of d6e79ea3d3d.

## Problem

The shared-expert device activation adds one matmul and two host readbacks per token
that the witness does not expect. Either the expectation is stale, or the shared
expert path reads back intermediates it should keep on device.

## Why this is next

It is the only red keeping d6e79ea3d3d from a green verify-queue entry. The extra D2H
could also be a real device-residency regression on the decode path.

## Working spine

Build -tags vulkan -> run the witness -> attribute each matmul and readback to a named
call site -> remove unneeded readbacks, or update the contract and state the reason.

## Core through-line

Make the witness's dispatch and readback contract match an intended shared-expert
route. Prefer removing unneeded host readbacks over loosening the count.

## Gold-plating boundary

No SwiGLU kernel change, no tolerance change, no performance claim.

## Verifiable Witness

```bash
go test -c -tags vulkan -o model.test ./internal/model
FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 FAK_VULKAN_SPIRV=<spirv dir> \
  FAK_VULKAN_EXPECT_DEVICE="AMD Radeon 8060S Graphics (RADV STRIX_HALO)" \
  ./model.test -test.run '^TestV41ClampedDeviceSwiGLUHalo$' -test.count=1 -test.v
```

## Done condition / witness

Witness: the command above on a Strix Halo appliance.

- [x] [SW-VERIFIED] Each extra matmul and readback is attributed to a named call site.
- [ ] [HW-WITNESSED] `TestV41ClampedDeviceSwiGLUHalo` PASS on a Halo.

## Witness

Only a real-silicon run satisfies the HW box. A CPU or skipped run is not evidence.

## Definition of done

Every checkbox in Done condition is ticked. The green silicon witness is recorded in
the resolving commit and in `fak-sync release verify-queue` (tier full) for d6e79ea3d3d.

## Acceptance gate

The witness passes on a Halo, the touched CPU tests stay green, and boundary and scrub
checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the `model`
leaf in its subject. The d6e79ea3d3d verify-queue obligation is completed `--green`.

## Likely files

- `internal/model/v41_clamped_device_swiglu_halo_test.go`: MatMul and D2H expectations.
- `internal/model/v41_shared_activation.go`: shared expert device activation and readbacks.

## Lane

model/v41-device

## Blast radius and affected lanes

One package, `internal/model`.

## Quarantined fallback mechanism

The opt-in witness stays red and the obligation stays failed. Production is unchanged.

## Problem frame

- Centrality: Enabling (clears a deferred-verification obligation on a landed commit)
- P1 Context: preserved - kernel numerics already match the CPU reference.
- P2 Net value: advanced - turns a waived commit into a witnessed one, and may remove readbacks.
- P3 Adaptation: preserved - no new mechanism.
- P4 Operations: advanced - one bounded dispatchable unit with a silicon witness.

## Expected steps

2

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 2/7 points toward parent #13765.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13769

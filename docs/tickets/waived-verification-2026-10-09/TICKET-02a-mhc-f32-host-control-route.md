# fix(model): reconcile V4.1 mHC F32 Halo witness with the transposed device route

## Parent context

Child of TICKET-02 (V4.1 device kernels hardware qualification) in this directory,
parent #13765 (https://github.com/anthony-chaudhary/fak/issues/13765).
Waived commit: 6a54495ecd4 `feat(model): add transposed V4.1 mHC device projection`.

<!-- fak-model-key: waived-v41-mhc-f32-host-control-route -->
<!-- fak-public-issue: anthony-chaudhary/fak#13767 -->

```routing
lane: model/v41-device
paths: ["internal/model/v41_mhc_projection_halo_test.go", "internal/model/v41_mhc_projection.go", "docs/tickets/waived-verification-2026-10-09/TICKET-02a-mhc-f32-host-control-route.md"]
expected_steps: 2
public_issue: anthony-chaudhary/fak#13767
```

Process cause: verification-gap

## Scope class

single-package

## Current state

[HW-WITNESSED] `TestV41MHCProjectionVulkan/F32` fails on a Strix Halo appliance; `/Q2_K` passes. Every
phase reports cosine=1, max_diff=0, greedy 6/6, so numeric parity holds. The failing
asserts are on the host-control arm (`v41_mhc_projection_halo_test.go` near the
`actual whole-weight mHC host control` and `immutable full mHC staging` checks). The
control model now reports `mhc_projection_device_calls=3 host_calls=0` and
`staging selected=1 control=1`. The test expects the control to stay on host
(host_calls=len(ids), host staging 0).

Silicon run: one Strix Halo appliance, RADV STRIX_HALO, vulkan-1.4.354, source dff9be67b4a,
2026-10-09.

## Problem

After 6a54495ecd4, the F32 host-control arm routes to device. The witness therefore
no longer compares the device route against an independent host route. Either the
control construction must force the host route again, or the test expectation is stale.

## Why this is next

It is the only red keeping 6a54495ecd4 from a green verify-queue entry. The numerics
are already exact, so this is a bounded dispatch-accounting contract fix.

## Working spine

Build -tags vulkan -> run the witness on a Halo -> observe the host-control arm on
host (or prove an equivalent independent control) -> F32 and Q2_K both pass.

## Core through-line

Restore an independent host control for the F32 mHC witness. Or prove the control's
device route is intended and update the contract to match.

## Gold-plating boundary

No mHC kernel change, no new tolerance, no performance claim.

## Verifiable Witness

```bash
go test -c -tags vulkan -o model.test ./internal/model
FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 FAK_VULKAN_SPIRV=<spirv dir> \
  FAK_VULKAN_EXPECT_DEVICE="AMD Radeon 8060S Graphics (RADV STRIX_HALO)" \
  ./model.test -test.run '^TestV41MHCProjectionVulkan$' -test.count=1 -test.v
```

## Done condition / witness

Witness: the command above on a Strix Halo appliance.

- [ ] [SW-VERIFIED] Root cause named: the control routes to device by design, or a routing regression.
- [ ] [HW-WITNESSED] `TestV41MHCProjectionVulkan` F32 and Q2_K both PASS on a Halo.

## Witness

Only a real-silicon run satisfies the HW box. A CPU or skipped run is not evidence.

## Definition of done

Every checkbox in Done condition is ticked. The green silicon witness is recorded in
the resolving commit and in `fak-sync release verify-queue` (tier full) for 6a54495ecd4.

## Acceptance gate

The witness passes on a Halo, the touched CPU tests stay green, and boundary and scrub
checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the `model`
leaf in its subject. The 6a54495ecd4 verify-queue obligation is completed `--green`.

## Likely files

- `internal/model/v41_mhc_projection_halo_test.go`: host-control construction and asserts.
- `internal/model/v41_mhc_projection.go`: transposed device route selection.

## Lane

model/v41-device

## Blast radius and affected lanes

One package, `internal/model`.

## Quarantined fallback mechanism

The opt-in witness stays red and the obligation stays failed. Production is unchanged.

## Problem frame

- Centrality: Enabling (clears a deferred-verification obligation on a landed commit)
- P1 Context: preserved - kernel numerics already match the CPU reference.
- P2 Net value: advanced - turns a waived commit into a witnessed one.
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

github_issue: anthony-chaudhary/fak#13767

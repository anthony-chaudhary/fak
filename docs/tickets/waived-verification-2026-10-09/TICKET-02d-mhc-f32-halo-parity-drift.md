# fix(model): explain the V4.1 mHC F32 Halo device-vs-host logit drift

## Parent context

Child of TICKET-02 (V4.1 device kernels hardware qualification) in this directory,
parent #13765 (https://github.com/anthony-chaudhary/fak/issues/13765). Found while closing
TICKET-02a (#13767). Covered commits: 6a54495ecd4 (transposed mHC device projection) and
4e80401da72 (full mHC sublayer graph).

<!-- fak-model-key: waived-v41-mhc-f32-halo-parity-drift -->
<!-- fak-public-issue: anthony-chaudhary/fak#13808 -->

```routing
lane: model/v41-device
paths: ["internal/model/v41_mhc_projection_halo_test.go", "internal/model/v41_mhc_projection.go", "docs/tickets/waived-verification-2026-10-09/TICKET-02d-mhc-f32-halo-parity-drift.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#13808
```

Process cause: verification-gap

## Scope class

single-package

## Current state

[HW-WITNESSED] On strix1 (RADV STRIX_HALO, vulkan-1.4.354, source f2be2792ab2, 2026-10-10),
`TestV41MHCProjectionVulkan/Q2_K` passes and `/F32` fails at
`v41_mhc_projection_halo_test.go:272`. The grouped parity reports maximum_diff=0.0535 at
scale 4.23, against a 1e-4 tolerance, in the cold prefill phase.

The TICKET-02a host-control route is now correct on silicon. The route-ledger and staging
asserts before line 272 pass, so the control does run on host. The drift is new
information. The old control also ran on device, so its parity check compared the device
with itself and was vacuous.

A diagnostic rebuild on strix1 kept the control's mHC mix in the logical layout instead of
the transpose. It gave the bit-identical maximum_diff 0.0534825. So the host transposed
path is not the cause: device mHC vs host mHC drifts the same way in both layouts. The
per-op raw-before-RMS check (device coefficients against a float64 oracle, 1e-4 relative)
passes.

Q2_K passes bit-exact only because its fixture uses d=1/min=0 blocks with weights up to
45. Those saturate the post coefficients, so the Q2 parity cannot see drift. TICKET-10
records the same saturation on the CPU fixture.

## Problem

Two explanations remain open. (1) The drift is F32 reduction-order differences over
20480-long dots, amplified through BF16 stream boundaries and Sinkhorn. If so, the witness
needs a principled bound, not 1e-4. (2) The device route's host RMS scale, or its
coefficient handling, differs from the host projector (`v41MHCProjectFull`). If so, it is a
real numerics bug.

## Why this is next

This is the only remaining red on 6a54495ecd4's Halo witness, and it blocks the
verify-queue full-tier completion for that commit.

## Working spine

Capture per-stage device and host mHC outputs (raw coefficients, RMS-scaled, pre/post/comb,
BF16 streams) on the Halo fixture. Find the first stage that diverges beyond float32
rounding. Then fix the route, or derive and justify the bound.

## Core through-line

Make the F32 Halo mHC parity a real, independent comparison that passes for a named
reason.

## Gold-plating boundary

No kernel rewrite and no performance claim. A tolerance change needs a derivation from the
measured first-divergence stage.

## Verifiable Witness

```bash
go test -c -tags vulkan -o model.test ./internal/model   # needs libfakvulkan.a, docs/compute/vulkan-linux.md
cd internal/model && FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 FAK_VULKAN_SPIRV=<spirv dir> \
  FAK_VULKAN_EXPECT_DEVICE="AMD Radeon 8060S Graphics (RADV STRIX_HALO)" \
  ../../model.test -test.run '^TestV41MHCProjectionVulkan$' -test.count=1 -test.v
```

## Done condition / witness

- [ ] [SW-VERIFIED] First divergent mHC stage named, with values.
- [ ] [HW-WITNESSED] `TestV41MHCProjectionVulkan` F32 and Q2_K both PASS on a Halo.

## Witness

Only a real-silicon run satisfies the HW box. Run the test binary from `internal/model`,
because the fixture reads `testdata/` relative to the cwd.

## Definition of done

Both boxes are ticked, and 6a54495ecd4's verify-queue full obligation is completed `--green`.

## Acceptance gate

The witness passes on a Halo, CPU `V41MHC` tests stay green, and boundary and scrub checks are green.

## Closure binding

The resolving commit cites this ticket's public issue, plus Ref #13767, with the `model` leaf.

## Likely files

- `internal/model/v41_mhc_projection.go`: device route host RMS scale.
- `internal/model/v41_mhc_projection_halo_test.go`: parity bound.

## Lane

model/v41-device

## Blast radius and affected lanes

One package, `internal/model`.

## Quarantined fallback mechanism

The opt-in witness stays red. Production is unchanged.

## Problem frame

- Centrality: Enabling (last red on a waived commit's silicon witness)
- P1 Context: preserved - no runtime change until the cause is named.
- P2 Net value: advanced - a real host-vs-device mHC comparison on silicon.
- P3 Adaptation: preserved - no new mechanism.
- P4 Operations: advanced - a bounded, dispatchable unit with a silicon witness.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3/10 points toward parent #13765.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13808

github_issue: anthony-chaudhary/fak#13808

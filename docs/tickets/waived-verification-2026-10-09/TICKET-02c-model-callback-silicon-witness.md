# test(model): add silicon witnesses for V4.1 model RMSNorm and tail RoPE device callbacks

## Parent context

Child of TICKET-02 (V4.1 device kernels hardware qualification) in this directory,
parent #13765 (https://github.com/anthony-chaudhary/fak/issues/13765).
These waived commits have no test that reaches silicon:

- 0d0dcda7e63: query latent RMSNorm
- fa265cd4174: KV latent RMSNorm
- e8340033f8d: FFN input RMSNorm
- 0bae85dd15b: final RMSNorm
- 4cd54773544: tail RoPE device callback

<!-- fak-model-key: waived-v41-model-callback-silicon-witness -->
<!-- fak-public-issue: anthony-chaudhary/fak#13768 -->

```routing
lane: model/v41-device
paths: ["internal/model/v41_norm_rope_callbacks_halo_test.go", "docs/tickets/waived-verification-2026-10-09/TICKET-02c-model-callback-silicon-witness.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#13768
```

Process cause: verification-gap

## Scope class

single-package

## Current state

These commits added device-dispatch tests: `TestV41QueryNormDeviceDispatch`,
`TestV41KVNormDeviceDispatch`, `TestV41FFNNormDeviceDispatch`,
`TestV41FinalNormDeviceDispatch` and `TestV41TailRoPEDeviceDispatch`. All of them
drive a fake backend; none looks up the real Vulkan backend.

On silicon, the underlying kernels pass: `TestVulkanRMSNormApprox` and
`TestVulkanV41TailRoPEQK` are [HW-WITNESSED] on a Strix Halo appliance. But no test runs these model
callbacks against a real device. `TestV41SharedAttentionVulkan` also disables the other
optional device callbacks. The five verify-queue obligations (tier full) are pending.

Silicon run: one Strix Halo appliance, RADV STRIX_HALO, vulkan-1.4.354, source dff9be67b4a,
2026-10-09.

## Problem

A present callback is not evidence that the model invokes it on a real device with
parity, so the waived commits cannot be marked verified.

## Why this is next

It is the last silicon gap in TICKET-02. The kernels are already witnessed; only the
model wiring is unproven.

## Working spine

Tiny synthetic V4.1 model -> session Prefill/Step on the real Vulkan backend with only
these callbacks enabled -> count each callback's dispatches -> compare logits and
normalized rows against a cpu-ref host session.

## Core through-line

One opt-in -tags vulkan Halo test proves each of the five callbacks dispatches on the
real device and matches the host session.

## Gold-plating boundary

No kernel or callback change, no full-checkpoint or performance claim.

## Verifiable Witness

```bash
go test -c -tags vulkan -o model.test ./internal/model
FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 FAK_VULKAN_SPIRV=<spirv dir> \
  FAK_VULKAN_EXPECT_DEVICE="AMD Radeon 8060S Graphics (RADV STRIX_HALO)" \
  ./model.test -test.run '^TestV41NormRoPECallbacksHalo$' -test.count=1 -test.v
```

## Done condition / witness

Witness: the command above on a Strix Halo appliance.

- [ ] [SW-VERIFIED] The test skips without opt-in and fails if any callback is not reached.
- [ ] [HW-WITNESSED] Each of the five callbacks dispatches on a Halo GPU with parity against cpu-ref.

## Witness

Only a real-silicon run satisfies the HW box. A CPU or skipped run is not evidence.

## Definition of done

Every checkbox in Done condition is ticked. The green silicon witness is recorded in
the resolving commit and in `fak-sync release verify-queue` (tier full) for all five commits.

## Acceptance gate

The witness passes on a Halo, the new test skips cleanly on CPU-only hosts, and
boundary and scrub checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the `model`
leaf in its subject. All five verify-queue obligations are completed `--green`.

## Likely files

- `internal/model/v41_norm_rope_callbacks_halo_test.go`: the new opt-in witness.
- `internal/model/v41_forward.go`, `internal/model/v41_rope_device.go`: callbacks under test (read-only).

## Lane

model/v41-device

## Blast radius and affected lanes

One new test file in `internal/model`.

## Quarantined fallback mechanism

The obligations stay pending and production is unchanged.

## Problem frame

- Centrality: Enabling (clears five deferred-verification obligations)
- P1 Context: preserved - kernel numerics already witnessed.
- P2 Net value: advanced - proves the model invokes the device callbacks.
- P3 Adaptation: preserved - no new mechanism.
- P4 Operations: advanced - one bounded dispatchable unit with a silicon witness.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3/7 points toward parent #13765.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13768

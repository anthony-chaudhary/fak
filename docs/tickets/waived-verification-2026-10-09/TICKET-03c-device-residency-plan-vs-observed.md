# fix(cmd): the ring plan's device-residency line matches the VRAM the bounded dense route places, and the reservation inventory gets a production reader

## Parent context

Child of TICKET-03 (Vulkan memory accounting and residency) in this directory,
parent #13761 (https://github.com/anthony-chaudhary/fak/issues/13761), serving #13668.
Evidence: operator-held silicon receipt (schema `fak.v41_native_probe.v1`, probed 2026-10-09T23:37Z, binary sha256 `2f3f2298bbae73a0`).

<!-- fak-model-key: waived-vulkan-device-residency-plan-vs-observed -->
<!-- fak-public-issue: anthony-chaudhary/fak#13775 -->

```routing
lane: compute/vulkan
paths: ["cmd/fak/serve_load_helpers.go", "internal/compute/vulkan_device_debug.go", "docs/tickets/waived-verification-2026-10-09/TICKET-03c-device-residency-plan-vs-observed.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#13775
```

Process cause: verification-gap

## Scope class

multi-package

## Current state

[HW-WITNESSED] On a Strix Halo appliance at fak 42dc8adb624, arm B with the bounded dense route
logged `dense base device-resident (63.092GiB) ... through a 2.863GiB device ring` from
`cmd/fak/serve_load_helpers.go:516`.

Sysfs `mem_info_vram_used` (1 Hz) behaved as follows:

- flat at 0.20 GB for about 18 min,
- then a step to 14.79 GB,
- then a 15.27 GB peak, which is 14.03 GiB over launch, or about 21% of the 65.95 GiB plan.

Attribution of the 14.03 GiB is weak: the likely explanation is that the dense base stays lazy.

The reservation and backing readers (`VulkanTensorBufferReservations`,
`VulkanQ4KHomeBufferReservations`, `VulkanQ4KStageBufferReservations`,
`VulkanTransferStageBufferReservations` and the `*Backings` readers in
`internal/compute/vulkan_device_debug.go`) have no non-test caller under `cmd/` or `internal/`.
No production entrypoint can report them.

## Problem

The plan line overstates device residency by about 52 GiB. Because the inventory cannot be read from a
serve, nothing reconciles planned bytes, reserved bytes and observed VRAM.

## Why this is next

#13668 asks whether the dense base is actually device-resident. Today that can only be answered by
sampling sysfs by hand.

## Working spine

1. Make the ring line report the bytes the bounded dense route actually places.
2. Have `fak serve` emit one bounded post-load reservation summary, either as a startup message or a
   debug endpoint, that totals the reservation readers.
3. Reconcile that summary against `mem_info_vram_used` on silicon.

## Core through-line

Planned, reserved and observed device bytes appear in one place.

## Gold-plating boundary

No new allocator, and no per-tensor dump in the default log.

## Verifiable Witness

```bash
go test ./cmd/fak/ -run 'ActivatedExpertRing|ServeGGUFMemoryProfile' -count=1
go test -tags vulkan ./internal/compute/ -run 'Reservation|Backing' -count=1
```

## Done condition / witness

Witness: the commands above, plus one silicon serve log with the summary next to sysfs VRAM.

- [ ] [SW-VERIFIED] The ring line distinguishes the lazy or bounded dense bytes from device-resident bytes.
- [ ] [SW-VERIFIED] A production caller of the reservation readers exists. The reachability oracle shows it is invoked from `fak serve`.
- [ ] [HW-WITNESSED] The reported reserved bytes are within 5% of observed `mem_info_vram_used` minus the launch baseline.

## Witness

The HW box needs the serve summary and a 1 Hz sysfs trace from the same run.

## Definition of done

All boxes are ticked, and the resolving commit cites this issue.

## Acceptance gate

The touched package tests stay green, and boundary and scrub checks are green.

## Closure binding

The resolving commit cites this public issue and carries the `cmd` or `compute` leaf in its subject.

## Likely files

- `cmd/fak/serve_load_helpers.go`: ring line at :516.
- `internal/compute/vulkan_device_debug.go`: reservation readers (:533 to :912).

## Lane

compute/vulkan

## Blast radius and affected lanes

`cmd/fak` startup messages plus `internal/compute` readers. This ticket changes reporting only.

## Quarantined fallback mechanism

Without a summary, sysfs sampling stays the operator method.

## Problem frame

- Centrality: Enabling (memory truth for #13668)
- P1 Context: preserved - no change to context, KV or prompt handling; only the touched load or test path moves.
- P2 Net value: advanced - one serve log answers whether the dense base is device-resident, with no hand sampling.
- P3 Adaptation: preserved - no new external mechanism is adopted; existing fak paths are corrected.
- P4 Operations: advanced - observable residency.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3/9 points toward parent #13761.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13775

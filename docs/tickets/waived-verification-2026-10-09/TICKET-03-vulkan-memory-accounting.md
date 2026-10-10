# TICKET-03: Vulkan memory accounting rework is unverified on Vulkan builds and hardware

## Current state

These commits serve #13668, where a streamed-expert load staged a 63 GiB dense base into host RAM. They landed under the waiver:

58ac8eddcb0 selected DRM identity; 2d6c36cee2c host evidence bound to the DRM node; 80852b1da44 VRAM carveout bound to the selected device; 2a68d2f87e5 allocation backing provenance; 7d88fb1490f home allocation backings; 44ad95a4278 stage backing inventory; 94665ac3be3 transfer stage backing; ec48c7d0594 buffer reservations; a2c60106b7e Q4K owner reservations; 0bb38ab6b0b shared transfer stage reservations; e014e2a7b30 restore resource lifetime; d9d525800bd poisoned snapshot observations.

Already passing at `24033b4080f`: `go build ./...`, `go vet ./internal/compute/...`, and `go test ./internal/compute/` (CPU).

Not yet witnessed:

- the `-tags vulkan` tests,
- the interaction with 15c81f74897 and b7f2b35029e,
- behavior on a real node.

## Working spine

1. Run the `-tags vulkan` build and tests for `internal/compute` and `internal/ggufload`.
2. Add fail-before/pass-after tests wherever a commit lacks them.
3. Rerun the #13668 dense-base probe on a Strix Halo node and confirm VRAM and host RSS match the reservation inventory.

## Witness

The Vulkan-tagged tests exit 0, plus one silicon probe log.

## Done condition

- [ ] Vulkan-tagged suite green.
- [ ] Silicon probe shows the dense base resident in VRAM, or a narrowed issue is filed.

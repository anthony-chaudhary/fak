# build(compute): link the Vulkan shim math dependency on Linux

GitHub: https://github.com/anthony-chaudhary/fak/issues/12734

<!-- fak-compute-key: vulkan-linux-static-shim-libm -->

```routing
lane: compute
paths: `internal/compute/vulkan_linux.go`, `docs/tickets/vulkan-linux-libm/TICKET.md`
expected_steps: 3
```

## Working spine

Linux Vulkan build tag -> cgo linker directive -> rebuilt static Vulkan shim -> Go package/test binary.

## Current state

At `6e1c5eb7c`, the Linux Vulkan cgo directive links `libvulkan` and `libstdc++` but omits `libm`. The rebuilt static Vulkan shim references `sincosf`, so the documented tagged Linux test-binary build fails with `undefined reference to sincosf@@GLIBC_2.2.5` and `libm.so.6: DSO missing from command line`.

## Why this is next

The missing transitive library blocks the newly connected resident Vulkan MTP path from producing supported Linux binaries.

## Parent context

Follow-up build leaf discovered while independently compiling the completed Vulkan MTP target work tracked by #12720 and #12721.

## Core through-line

Add `-lm` to the Linux-only cgo linker directive so the shim's standard math symbols resolve at the normal package link boundary.

## Gold-plating boundary

Do not change shader code, Vulkan runtime behavior, Windows linkage, or model execution. Do not add a test that mirrors the linker string. Hardware execution remains outside this build-only leaf.

## Done condition

- [x] The Linux Vulkan cgo linker directive includes `-lm`.
- [x] Fresh Vulkan-tagged agent and gateway test binaries link with a rebuilt current shim and no external `-lm` override.
- [x] Existing Windows build configuration is unchanged.

## Witness

The independent acceptance oracle rebuilt `vulkan_shim.o` and `libfakvulkan.a` from `6e1c5eb7c`, then compiled Vulkan-tagged agent and gateway test binaries with `CGO_LDFLAGS` containing only the archive search path. Both binaries linked successfully. The same build without this directive failed with the expected `sincosf` missing-DSO error. Runtime or physical-GPU claims require a separate hardware witness.

## Definition of done

Completion requires all items in Done condition and the independent tagged Linux link witness.

## Acceptance gate

The independent tagged Linux compile oracle passes without an external `-lm` override.

## Closure binding

The resolving `build(compute)` commit cites #12734 and carries the `compute` lane stamp.

## Lane

compute

## Likely files

- `internal/compute/vulkan_linux.go:6`
- `docs/tickets/vulkan-linux-libm/TICKET.md`

## Expected steps

3

## Blast radius and affected lanes

Only Linux builds with the `vulkan` tag and cgo enabled are affected. Untagged builds, Windows Vulkan builds, and runtime token behavior are unchanged.

## Quarantined fallback mechanism

Existing untagged builds remain green. The failure is confined to the explicitly selected Linux Vulkan build until this linker leaf lands.

- Centrality: Core
- P1 Context: advanced — restores the supported Linux Vulkan build path.
- P2 Net value: preserved — one linker dependency, no API or runtime semantic change.
- P3 Adaptation: preserved — build-tag selection remains the same.
- P4 Operations: advanced — the documented build no longer needs an undeclared external linker override.

## Work estimate

Estimate: 1 point

## Overall completion contribution

Contribution: 1/1 point for restoring the tagged Linux Vulkan link.

## Completion standard

development

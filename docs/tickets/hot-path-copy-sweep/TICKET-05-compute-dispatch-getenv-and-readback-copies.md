<!-- fak-compute-key: compute-dispatch-getenv-and-readback-copies -->
```routing
lane: compute
paths: ["internal/compute/strix/compute_hal.go", "internal/compute/vulkan_dispatch_geometry.go", "internal/compute/vulkan_v41_q2k_token_panel.go", "internal/compute/v41_q3k_token_panel.go", "internal/compute/vulkan_device_debug.go", "internal/compute/v41_deferred_panel_read.go", "internal/compute/v41_host_readback_image.go"]
expected_steps: 6
```

# perf(compute): read dispatch env flags once and decode deferred-panel readbacks without per-row copies

GitHub: pending mirror.

Process cause: verification-gap

## Value

- **For:** the Vulkan V4.1 decode dispatch path on Halo.
- **Problem:**
  - `os.Getenv` runs on every dispatch: `strix/compute_hal.go:197` (every Vulkan Q2_K MatMul via `vulkan.go:1903`), `vulkan_dispatch_geometry.go:23,96`, `vulkan_v41_q2k_token_panel.go:17,37`, `v41_q3k_token_panel.go:28`, and three lookups per GDN call at `vulkan_device_debug.go:1158-1164`. On Windows each is a syscall plus UTF-16 conversion; ESTIMATED thousands per token.
  - `v41_deferred_panel_read.go:55-70` and `v41_host_readback_image.go:94-98` give each logical row its own `make` and fill it with a per-element `binary.LittleEndian.Uint32` loop; a typed aliasing path exists for single-row reads only (lines 32-70). ESTIMATED 8 rows x 7168 = about 229 KB image + 229 KB rows + 8 allocs per cohort read.
- **Better because:** flags are process constants; caching them and extending the typed view to multi-row reads removes repeated work with no semantic change.
- **Witness:** `AllocsPerRun` ceilings on the deferred read and a dispatch microbenchmark showing zero env lookups after init.

## Problem frame

- Centrality: Enabling (fak-native V4.1 decode throughput on Halo)
- P1 advanced: advanced - trims per-dispatch host overhead on the native decode path.
- P2 preserved: preserved - flags keep their names and meaning.
- P3 advanced: advanced - reuses the existing single-row typed readback.
- P4 preserved: preserved - software ceilings only; no speed claim without a receipt.

## Why this is next

Found by the 2026-10-16 compute sweep as the remaining per-call host overheads after the cpu-ref weight copy fix (`19eab47217`); both are mechanical.

## Current state

Classification: **PARTIAL** - single-row typed readback exists; env lookups uncached.

## Core through-line

Process start -> flags parsed once into a config struct -> dispatch reads the struct; deferred readback -> one image -> typed multi-row view -> no per-row allocation.

## Working spine

V4.1 decode on Vulkan -> Q2_K/Q3_K token panel dispatch -> deferred cohort read -> ceilings green.

## Scope

Env-flag caching at the listed sites and multi-row typed readback.

## Gold-plating boundary

- No new flags, no flag renames, no config-file system.
- No kernel or shader change.

## Done condition

- [ ] Listed `os.Getenv` sites read a once-initialized value.
- [ ] Multi-row deferred reads use one typed view, no per-element decode loop on little-endian hosts.
- [ ] `AllocsPerRun` ceiling on deferred readback.
- [ ] Existing flag-behavior tests still pass (tests that set env at runtime reset the cache).

## Definition of done

Ceilings green; flags behave identically; commit records before/after allocations.

## Witness

`go test ./internal/compute/ -run 'Deferred.*Read|Dispatch.*Flag' -count=1`

## Likely files

- `internal/compute/strix/compute_hal.go`
- `internal/compute/vulkan_dispatch_geometry.go`
- `internal/compute/vulkan_v41_q2k_token_panel.go`
- `internal/compute/v41_q3k_token_panel.go`
- `internal/compute/vulkan_device_debug.go`
- `internal/compute/v41_deferred_panel_read.go`
- `internal/compute/v41_host_readback_image.go`

## Parent context

Parent campaign: #12640. Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

Allocation ceiling plus unchanged flag semantics; GPU-surface commit follows the strix controller validation route.

## Closure binding

Close with the landing commit and ceiling names.

## Work estimate

About half a day to one day.

## Overall completion contribution

1/11 points of the hot-path copy sweep.

## Completion standard

development

<!-- fak-engine-key: serve-strix-backend-tier-cpu-guard -->
<!-- fak-public-issue: anthony-chaudhary/fak#12644 -->
# fix(cmd/fak): inspect backend tier and guard non-GPU backends in Strix Halo serve preflight

```routing
lane: cmd/fak
paths: ["cmd/fak/serve_backend_preflight.go", "cmd/fak/serve_backend_preflight_test.go"]
expected_steps: 3
github_issue: anthony-chaudhary/fak#12644
```

## Parent context

Part of the Strix Halo Physics vs. Economics Realignment & Bulk Migration Series (`docs/tickets/strix-physics-migration/INDEX.md`). Follows `TICKET-02`.

## Why now

In `cmd/fak/serve_backend_preflight.go`, `preflightServeStrixHaloWithSysfs` inspects `be.Name()` to identify the backend. For the Vulkan backend, `be.Name()` returns `"vulkan"` which fails `IsStrixHaloArch`, causing unnecessary fallthrough to DRM sysfs rather than inspecting `be.Tier()` (`integrated:AMD Radeon 8060S Graphics (gfx1151)`). Furthermore, if a user explicitly selects a non-GPU backend on Strix Halo hardware (such as `fak serve --backend cpu`), `be.Name()` is `"cpu-ref"`, but because `sysfsRoot == ""` falls through to host sysfs, `preflightServeStrixHalo` detects the host APU and activates UMA pointer management and MALL tiling for the CPU backend.

## Current state

1. `preflightServeStrixHaloWithSysfs` only queries `be.Name()` and does not inspect `be.Tier()`.
2. When running on live Strix Halo hardware (`strix1`), non-Vulkan backends (e.g. `cpu-ref`) fall through to DRM sysfs, falsely activating Strix Halo APU acceleration on CPU-only execution paths.

## Value

- Centrality: Core
- P1: advanced - prevents false activation of Strix Halo UMA and MALL tiling when running CPU-reference inference.
- P2: advanced - inspects `be.Tier()` to match device architecture directly from Vulkan backend without relying on sysfs fallthrough.
- P3: preserved - maintains strict backend separation and deterministic preflight messages.
- P4: preserved - leaf task completed in <= 3 steps touching 2 files in `cmd/fak`.

## Working spine

Inspect `be.Tier()` alongside `be.Name()` -> guard against non-GPU backends (skip APU preflight if backend is cpu-ref) -> add regression test cases in `serve_backend_preflight_test.go` -> verify exit 0.

## Core through-line

Update `preflightServeStrixHaloWithSysfs` to:
1. Append `be.Tier()` when `be != nil` so that `vulkanBackend` tier (`integrated:AMD Radeon 8060S Graphics (gfx1151)`) matches `IsStrixHaloArch` directly.
2. Explicitly guard against non-GPU backends: if `be != nil` and `be.Name()` is `"cpu-ref"` or non-Vulkan GPU, return `Detected: false` immediately without falling through to DRM sysfs.
3. Add unit test asserting that a CPU backend on a machine with mock DRM sysfs (0x1002:0x1586) returns `Detected: false`.

## Gold-plating boundary

- Strictly excluded: Rewriting the Vulkan initialization or backend registration.
- Strictly excluded: Modifying `internal/compute/strix/device_detect.go`.

## Witness

Clean unit test execution of `TestServeBackendPreflight_StrixHaloActivation` covering CPU backend rejection on live Strix hardware.

### Verifiable Witness

```bash
go test -v -race ./cmd/fak -run TestServeBackendPreflight_StrixHaloActivation -count=1
```

## Likely files

- `cmd/fak/serve_backend_preflight.go`
- `cmd/fak/serve_backend_preflight_test.go`

## File:Line Seam

`cmd/fak/serve_backend_preflight.go:134`

## Blast Radius & Affected Lanes

- **Primary Lane:** `cmd/fak` in `fak`
- **Unaffected Lanes:** all other platform and compute packages decoupled; trunk remains 100% green.

## Quarantined Fallback Mechanism

If backend tier inspection fails, the preflight fails closed (`Detected: false`) with zero panic.

## Scoped Acceptance Criteria

- [ ] 1. `preflightServeStrixHaloWithSysfs` checks `be.Tier()` and guards against non-Vulkan/CPU backends.
- [ ] 2. `TestServeBackendPreflight_StrixHaloActivation` includes a test case asserting CPU backend returns `Detected: false` even with mock sysfs present.
- [ ] 3. `go test -v -race ./cmd/fak -run TestServeBackendPreflight_StrixHaloActivation -count=1` passes cleanly.

## Done condition

Strix Halo serve preflight correctly identifies Vulkan backend tier and refuses to activate APU subsystems on CPU-reference backends.

## Acceptance gate

`go test -v -race ./cmd/fak -run TestServeBackendPreflight_StrixHaloActivation -count=1` exits 0.

## Closure binding

Resolving commit cites this ticket and carries `(fak engine)`.

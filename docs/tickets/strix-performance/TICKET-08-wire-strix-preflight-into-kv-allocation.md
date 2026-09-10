<!-- fak-engine-key: wire-strix-uma-mall-kv-allocation -->
<!-- fak-public-issue: anthony-chaudhary/fak#12645 -->
# feat(engine): wire preflight UMAPointerManager and MALLTiler into KV cache and tensor allocation

```routing
lane: cmd/fak
paths: ["cmd/fak/serve_stages.go", "internal/compute/vulkan_strix.go"]
expected_steps: 4
github_issue: anthony-chaudhary/fak#12645
```

## Parent context

Part of the Strix Halo Physics vs. Economics Realignment & Bulk Migration Series (`docs/tickets/strix-physics-migration/INDEX.md`). Follows `TICKET-02` (fak#12626).

## Why now

Phase 2 successfully landed hardware detection and initialized `UMAPointerManager` and `MALLTiler` during `serve` preflight (`cmd/fak/serve_backend_preflight.go`), storing the result in `rt.strixPreflight`. However, neither the UMA pointer manager nor the 32MB MALL tiler is currently wired into `fak serve`'s KV cache allocation or forward tensor allocation paths. As a result, the live serving runtime on AMD Strix Halo APU silicon (`strix1`) runs without zero-copy UMA buffer pinning or MALL cache tiling, resulting in degraded cold prefill throughput. Pipelining these initialized subsystems into the compute backend is required to reach the target roofline ($\ge 350\text{ tok/s}$).

## Current state

1. `cmd/fak/serve_stages.go:305-308` probes and stores `rt.strixPreflight`, but does not pass it to backend allocators.
2. `internal/compute/vulkan_strix.go` defines `VulkanStrixWave32WMMA` and `VulkanStrixSFence`, but they are not hooked into the active forward evaluation loop.

## Value

- Centrality: Core
- P1: advanced - activates zero-copy UMA pointer manager and 32MB MALL Infinity Cache tiling during model serving.
- P2: advanced - eliminates unneeded host-to-device roundtrips on unified memory APU silicon.
- P3: preserved - maintains strict Gate 4 vendor neutrality: zero impact when running on discrete NVIDIA/Apple GPUs.
- P4: preserved - leaf task completed in <= 4 steps touching 2 files in `cmd/fak` and `internal/compute`.

## Working spine

Pass rt.strixPreflight to backend allocation init -> configure zero-copy UMA buffer pool and MALL root context tiler in active serve runtime -> verify preflight message and subsystem readiness -> assert exit 0.

## Core through-line

1. In `cmd/fak/serve_stages.go`, after `rt.strixPreflight` is evaluated, if `rt.strixPreflight.Detected` is true, attach `rt.strixPreflight.UMAPointerManager` and `rt.strixPreflight.MALLTiler` to the backend runtime context.
2. Enable zero-copy UMA pointer allocation for KV cache buffers when `rt.strixPreflight.Detected` is true.
3. Assert that preflight startup messages accurately reflect active subsystem binding in `cmd/fak/serve_stages_test.go`.

## Gold-plating boundary

- Strictly excluded: Rewriting the non-Strix Vulkan shader pipelines.
- Strictly excluded: Modifying proprietary commercial lease packages.

## Witness

Unit tests verifying that `strixPreflight` subsystems are actively passed to the serving runtime on GFX1151 detection.

### Verifiable Witness

```bash
go test -v -race ./cmd/fak -run TestServeBackendPreflight_StrixHaloActivation -count=1
```

## Likely files

- `cmd/fak/serve_stages.go`
- `internal/compute/vulkan_strix.go`

## File:Line Seam

`cmd/fak/serve_stages.go:305`

## Blast Radius & Affected Lanes

- **Primary Lane:** `cmd/fak` in `fak`
- **Unaffected Lanes:** all other platform and compute packages decoupled; trunk remains 100% green.

## Quarantined Fallback Mechanism

If UMA buffer allocation or MALL tiling fails, the runtime gracefully falls back to standard heap / VRAM allocations with zero crash.

## Scoped Acceptance Criteria

- [ ] 1. `rt.strixPreflight` UMA pointer manager and MALL tiler are retained and wired into serve runtime context.
- [ ] 2. `TestServeBackendPreflight_StrixHaloActivation` verifies active subsystem retention.
- [ ] 3. `go test -v -race ./cmd/fak -run TestServeBackendPreflight_StrixHaloActivation -count=1` passes cleanly.

## Done condition

Strix Halo serve runtime actively connects initialized preflight UMA and MALL tiling subsystems to serving memory allocation.

## Acceptance gate

`go test -v -race ./cmd/fak -run TestServeBackendPreflight_StrixHaloActivation -count=1` exits 0.

## Closure binding

Resolving commit cites this ticket and carries `(fak engine)`.

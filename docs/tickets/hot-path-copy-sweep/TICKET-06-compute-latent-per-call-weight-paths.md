<!-- fak-compute-key: compute-latent-per-call-weight-paths -->
```routing
lane: compute
paths: ["internal/compute/vulkan_ops_vulkan.go", "internal/compute/vulkan_ops.go", "internal/compute/cuda.go", "internal/compute/gemm_arm64.go", "internal/compute/fp8_block32.go", "internal/compute/hip_attention.go"]
expected_steps: 6
```

# fix(compute): retire or repair latent per-call weight paths, starting with AttentionDequantOnce's layer-shared scratchpad

GitHub: [#13875](https://github.com/anthony-chaudhary/fak/issues/13875)

Process cause: verification-gap

## Value

- **For:** correctness and future performance of compute entrypoints that are present but not yet wired into serving.
- **Problem:**
  - **Correctness:** `vulkanBackend.AttentionDequantOnce` (`internal/compute/vulkan_ops_vulkan.go:390-449`) caches one scratchpad per KV store keyed only on `(nPos, format)` (:420-428). `ExecuteVulkanAttentionWithDequantOnce` skips dequantization when `DequantCount != 0` (`vulkan_ops.go:215`), so a second layer at the same `nPos` would reuse layer 0's K/V. It has zero callers in the repository, tests included, so nothing pins either behavior. It also reallocates the scratchpad every step as `nPos` grows.
  - **Per-call weight work if wired:** `AttentionQuantizedKV` (:469) and `vulkan_ops.go:166-167` allocate per call; Vulkan and CUDA `PrefillBatch` (`vulkan_ops_vulkan.go:189-203,274-277,343`; `cuda.go:1399-1414`) re-upload host Wq/Wk/Wv/Wo every call and never free them; `gemm_arm64.go:404-431` converts the whole weight f32->f16 per call; `fp8_block32.go:154-167,206-219` decodes and NaN-checks the whole weight per GEMV; `hip_attention.go:60-64` copies the full KV cache per call.
- **Today:** no production caller found for these (rg over the repository, 2026-10-16), so they cost nothing now and become traps when wired.
- **Better because:** dead entrypoints with wrong caching are removability debt; deleting them, or fixing them with a test, closes the trap.
- **Witness:** for each entrypoint, either deletion with `go build ./...` green, or a per-layer parity test plus an allocation ceiling.

## Problem frame

- Centrality: Stewardship (no unwired compute entrypoint may hide a correctness defect)
- P1 preserved: preserved - no live decode path changes.
- P2 preserved: preserved - fak keeps backend ownership.
- P3 advanced: advanced - reduces unowned surface per feature-removability doctrine.
- P4 advanced: advanced - removes an untested correctness hazard before it can ship.

## Why this is next

The cpu-ref copy that shipped (fixed in `19eab47217`) started as a reference path with no cost receipt and was later reused by hot callers. These entrypoints are in the same position today.

## Current state

Classification: **PRESENT but unwired**; one known correctness defect.

## Core through-line

Entrypoint inventory -> caller check -> delete if unwired and unplanned, else key caches by layer/weight identity and make weights resident -> parity + ceiling test.

## Working spine

`go build ./...` and `go vet` green after deletions; for any kept entrypoint, a two-layer parity test that fails on the parent.

## Scope

The six entrypoints listed above.

## Gold-plating boundary

- No new attention or prefill backend features.
- Wired paths (Vulkan `MatMul`, borrowed expert uploads) are out of scope.

## Done condition

- [ ] `AttentionDequantOnce` deleted, or keyed per layer with a two-layer parity test red on the parent.
- [ ] Each other listed entrypoint deleted or given resident weights plus an allocation ceiling.
- [ ] Removal decisions recorded in the commit body.

## Definition of done

No unwired entrypoint keeps a per-call whole-weight conversion or a layer-shared cache without a test.

## Witness

`go build ./... && go test ./internal/compute/ -run 'DequantOnce|PrefillBatch|FP8Block32|GEMMFP16' -count=1`

## Likely files

- `internal/compute/vulkan_ops_vulkan.go`
- `internal/compute/vulkan_ops.go`
- `internal/compute/cuda.go`
- `internal/compute/gemm_arm64.go`
- `internal/compute/fp8_block32.go`
- `internal/compute/hip_attention.go`

## Parent context

Parent campaign: #12640. Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

Build green after removals; any kept path has parity and a ceiling.

## Closure binding

Close with the landing commit listing each entrypoint's disposition.

## Work estimate

Estimate: 3 points (small) - about 1 day.

## Overall completion contribution

3/26 points of the hot-path copy sweep.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13875

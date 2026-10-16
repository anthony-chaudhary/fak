<!-- fak-model-key: v41-decode-step-alloc-ceiling -->
```routing
lane: model
paths: ["internal/model/v41_layer_step.go", "internal/model/v41_attention_state.go", "internal/model/v41_kv_resident.go", "internal/model/rope.go", "internal/model/v41_decode_alloc_scaling_test.go"]
expected_steps: 7
```

# perf(model): add a V4.1 decode-step allocation ceiling and remove the per-token pools and window copies it exposes

GitHub: [#13872](https://github.com/anthony-chaudhary/fak/issues/13872)

Process cause: verification-gap

## Value

- **For:** fak-native DeepSeek V4.1 decode; the ceiling also guards every future V4.1 step change.
- **Problem:** no test bounds allocations on the V4.1 incremental step. `BenchmarkSessionDecodeStepF32/Q8` (`internal/model/decode_step_alloc_bench_test.go:47-62`) runs a synthetic 4-layer llama and asserts nothing; `TestDecodeStepAllocationStaysBounded` (`decode_alloc_test.go:49`) covers only the dense synthetic model. Per-token allocations this leaves unguarded:
  - `v41_layer_step.go:782`: `NewV41CompressorPool` every token on each compressed layer (about 224 KB/token on V4.1; about 10 MB/token at ratio 128).
  - `v41_layer_step.go:1159-1176` (`stepWindowKV`): a flat copy of up to 128 x 512 floats on the plain layers (about 512 KB/token), although the resident panel only needs the new row.
  - `v41_attention_state.go:492`: a 20 KB `partialInputs` copy on 38 layers (about 780 KB/token).
  - `v41_kv_resident.go:66-67`: `fmt.Sprintf` keys twice per layer; `rope.go:270-295`: YaRN table rebuilt about 3x per layer; `hal.go:530/534` `halKey()` concatenation per weight.
- **Today:** ESTIMATED 4-6K allocs/token and several MB/token. Unmeasured.
- **Better because:** one scaling ceiling catches this class at review time instead of on a hardware profile.
- **Witness:** a reduced-V4.1 `AllocScaling` test (pos 32 vs 4096; bytes/step and allocs/step) that the fixes turn green.

## Problem frame

- Centrality: Core
- P1 advanced: advanced - bounds and lowers per-token allocation on the native decode path.
- P2 preserved: preserved - fak keeps the step structure and fallbacks.
- P3 advanced: advanced - reuses `TestDecodeStepAllocationStaysBounded`'s scaling pattern on the V4.1 fixture.
- P4 preserved: preserved - allocation numbers are software evidence; tok/s needs a physical receipt.

## Why this is next

The 2026-10-16 forensics on the cpu-ref whole-weight copy found the same root cause: no pre-merge gate measures allocations on the path. This ticket adds that gate for V4.1, where the live serving work happens.

## Current state

Classification: **ABSENT** for the ceiling; **PARTIAL** for reuse (some V4.1 alloc tests exist, e.g. `v41_decode_state_test.go:71`, but none cover the sites above).

## Core through-line

Reduced V4.1 fixture -> decode at pos 32 and pos 4096 -> bytes/step and allocs/step measured -> ceiling asserted -> pooled compressor, row-only window update, cached keys and RoPE table bring it under the ceiling.

## Working spine

V4.1 session -> `Session.Step` -> per-layer step -> scaling test green with fixes, red on the parent.

## Scope

The scaling ceiling test and the listed per-token allocations in V4.1 step code.

## Gold-plating boundary

- No change to attention math, compressor numerics, or kernel dispatch.
- Indexer key repack is TICKET-01; expert fault copies are TICKET-02.
- No general allocator or arena framework.

## Done condition

- [ ] `v41_decode_alloc_scaling_test.go` asserts a bytes/step and allocs/step ceiling and growth ratio.
- [ ] Compressor pool reused across tokens per layer.
- [ ] Plain-layer window update writes only missing rows.
- [ ] Resident-KV keys and YaRN table cached per layer/config.
- [ ] Bit-identical logits against the parent over a multi-token run.

## Definition of done

The ceiling test is in the default `go test ./internal/model/` set, red on the parent, green on the candidate, with logits parity.

## Witness

`go test ./internal/model/ -run 'V41.*AllocScaling|V41.*Parity' -count=1`

## Likely files

- `internal/model/v41_layer_step.go`
- `internal/model/v41_attention_state.go`
- `internal/model/v41_kv_resident.go`
- `internal/model/rope.go`
- `internal/model/v41_decode_alloc_scaling_test.go` (new)

## Parent context

Parent campaign: #12640. Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

Ceiling red on parent, green on candidate; logits parity; test declares its runtime class.

## Closure binding

Close with the landing commit and the ceiling test name.

## Work estimate

Estimate: 5 points (medium) - about 1-2 days.

## Overall completion contribution

5/26 points of the hot-path copy sweep.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13872

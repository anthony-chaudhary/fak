# V4.1 package carve-out: state of `internal/model/v41`

Issue: #13038 (parent), #13095 (leaf). Trunk tip when written: `e3f416a4e`.

## Purpose

`internal/model` is the hottest V4.1 work surface. Every V4.1 worker lands into the
same package, so one compile unit serializes all of their land gates: a compile
error anywhere in `internal/model` blocks every V4.1 merge. A separate
`internal/model/v41` package lets the pure, self-contained V4.1 helpers compile
and test independently of the core hubs.

The carve is deliberately partial. This doc records what moved, what remains,
and why closing further is its own unit of work.

## Direction of the seam (one-way)

The dependency edge is one-way: `internal/model/v41 -> internal/model`. Core does
NOT import the v41 leaf. Core instead exposes additive exported facades so the
leaf can reuse core internals without an import cycle:

- `Finite32` in `v4_router.go` (wraps the unexported `finite32`).
- `FP8E4M3ToF32` and `CheckedShapeProduct` in `fp8_blockscale.go`.
- `V41FP8BlockDim` in `v41_inventory.go` (wraps `v41FP8BlockDim`).

Several files in the leaf import core for these facades
(`v41_attention_state.go`, `v41_compressor.go`, `v41_engram_quality.go`,
`v41_indexer_score.go`, `v41_kv_cache_layout.go`, `v41_mhc.go`,
`v41_sparse_attention.go`, `v41_tokenizer.go`, `v4_flash_attn_ratio0.go`,
`v4_flash_kv_layout.go`). Core files mention the leaf only in comments.

Witness: `go list -deps ./internal/model | findstr model/v41` is empty.

## What moved into `internal/model/v41` (14 production files)

| File | Lines |
|---|---|
| `v41_attention_state.go` | 390 |
| `v41_budget.go` | 149 |
| `v41_compressor.go` | 179 |
| `v41_engram.go` | 188 |
| `v41_engram_quality.go` | 508 |
| `v41_indexer_candidates.go` | 138 |
| `v41_indexer_score.go` | 189 |
| `v41_kv_cache_layout.go` | 557 |
| `v41_mhc.go` | 147 |
| `v41_router.go` | 42 |
| `v41_sparse_attention.go` | 277 |
| `v41_tokenizer.go` | 324 |
| `v4_flash_attn_ratio0.go` | 318 |
| `v4_flash_kv_layout.go` | 257 |

Note: for the files whose core twin remains (`v41_router.go`, `v41_engram.go`,
`v41_mhc.go`, `v41_sparse_attention.go`, `v41_attention_state.go`) the leaf holds
an extracted helper copy and the core hub keeps its own resident copy for the
type-locked consumers. The two are kept equivalent for the pure helpers the leaf
owns.

## What remains in `internal/model` and why

The residual hubs fall into three classes. No file here can move alone without
creating an import cycle, because its only consumers are the type-locked forward
hubs.

### (a) Methods on core types `Config` / `Model` / `Session`

A method cannot leave its receiver's package.

| File | Bound to |
|---|---|
| `v41_config.go` | `Config` (V4.1 attention geometry, admission) |
| `v41_dspark.go` | core session/forward state |
| `v41_forward.go` | `Model` / `Session` forward path |
| `v41_forward_engram.go` | `Model` / `Session` Engram forward |
| `v41_quant_decode.go` | core quant decode state |
| `v41_spec_decode.go` | core speculative decode |
| `v41_vision.go` | core vision path |
| `v4_flash_attention.go` | core attention forward |
| `v41_attention.go` | core attention forward |
| `v41_decoder.go` | core decoder layer |

### (b) Reverse edge: core calls into it

The core hubs invoke these helpers, so they must remain reachable from core.

| File | Core consumer |
|---|---|
| `v41_router.go` | `expert_cache_bench.go:83-85`, `v4_live_expert.go:174` |
| `v41_engram.go` | core forward helpers |
| `v41_mhc.go` | core forward helpers |
| `v41_sparse_attention.go` | core forward helpers |
| `v41_inventory.go` | `v41_weights.go` |
| `v4_flash_grouped_wo.go` | `v4_dense_quant.go:90` |
| `v41_compress_index.go` | core compressor path |

### (c) Reaches core unexported safetensors internals

These files touch unexported core symbols, so they cannot compile in the leaf
without an exported core seam first.

| File | Unexported dependency |
|---|---|
| `v41_weights.go` | `safetensorsFile`, `stEntry`, `openSafetensorsFile`, `safetensorsDataBounds` |
| `v41_engram_stream.go` | `V41EngramRow*` types |
| `v41_quant_sensitivity.go` | `classifyV4Tensor` |

## Structural conclusion

Closing the carve further requires an exported core seam (accessor or interface)
for the unexported `Model` / `Session` state and the safetensors accessors. That
is its own unit of work, not another file move. Every remaining file is either
bound to a core receiver, called by a core hub, or touching unexported core
internals, so moving any one alone creates an import cycle.

## Witness commands

| Command | Expected | Observed |
|---|---|---|
| `go build ./internal/model/...` | exit 0 | exit 0 |
| `go list -deps ./internal/model \| findstr model/v41` | empty | empty |
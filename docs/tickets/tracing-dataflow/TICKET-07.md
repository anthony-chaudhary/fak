<!-- fak-tensortrace-key: explicit-value-views-v1 -->

```routing
lane: compute
paths: ["internal/compute/**", "internal/ggufload/**"]
expected_steps: 6
```

# feat(compute): make packed, storage, logical, and expressed tensor views explicit

## Problem

For agents explaining quantized dataflow, “the tensor” currently names several incompatible facts: encoded checkpoint blocks, bytes owned by a backend, a logical shaped value, and the float values used by the mathematical operation. Today `Dtype`, `Layout`, `QuantSpec`, and ad hoc byte helpers force callers to infer which view a field describes, making unpack/repack/expand/contract traces easy to mislabel. Better because one explicit value-view vocabulary makes every transition state what changed and what remained invariant. The witness is a Q4_K table that follows the same tensor through packed, storage, logical, and expressed views without conflating payload size, allocation size, element count, or reconstructed values.

## Parent context

First-class public `fak` quantization dataflow tracing goal. This semantic leaf consumes the canonical transformation vocabulary from `many-to-many-value-lineage-v1` and the exact byte seam from #10380; it does not redefine either contract.

## Why now

#10380 owns exact packed/storage byte calculation. Once it lands, this issue gives those measured quantities authoritative packed/storage/logical/expressed meanings and legal transition semantics before the live tensor tracer consumes them.

## Current state

Inspected at public `fak` commit `325ade2db2a7b1ea556d03d89293162a85cd1e9e`:

- `internal/compute/compute.go:49-81` calls `Dtype` the storage/compute format and intentionally rounds sub-byte `Bytes()` to one byte per code.
- `internal/compute/compute.go:123-152` separately models physical layout and quantization metadata, while `internal/compute/compute.go:176-198` places dtype/layout/shape/quant spec on one `Tensor`; no field identifies the semantic view being reported.
- `internal/ggufload/gguf_dequant.go:33-177` computes exact GGUF encoded payload bytes, and `internal/ggufload/gguf_weightsource.go:282-323` transitions from raw quantized bytes to expressed F32, but callers receive no shared view descriptor.
- Open #10380 correctly calls for exact storage-byte accounting, but it does not define the four representation meanings or legal transitions between them.

## Dependencies and schedule

- Hard prerequisite: `many-to-many-value-lineage-v1`, whose public `traceabi.ValueRef` and `Transform` operations remain the canonical lineage representation.
- Hard prerequisite: open #10380, which owns exact payload/allocation byte calculation and trace-producer migration.
- TICKET-07 owns only value-view meanings, legal transitions, and their compute/GGUF adapters; it must not reimplement #10380 byte arithmetic or a second lineage graph.
- Schedule after TICKET-02 and #10380, then before TICKET-06. TICKET-07 and TICKET-06 serialize on `internal/ggufload/**`.

## Centrality and P1-P4

Centrality: Enabling (truthful quantized tensor representation and accounting)

P1: preserved - representation metadata does not broaden user context or record payloads.

P2: advanced - prevents agents and benchmarks from comparing unlike quantities such as packed payload bytes and expressed F32 bytes.

P3: advanced - new layouts and quant formats declare view semantics explicitly and fail closed when a conversion is unsupported.

P4: advanced - trace and receipt consumers can explain transitions deterministically instead of reverse-engineering backend-specific fields.

## Scope

Define the four value views and legal transition semantics only in `internal/compute` and `internal/ggufload`. Consume byte counts from #10380 and lineage operations from public `pkg/traceabi`; do not modify those prerequisite contracts in this leaf.

## Core through-line

Define a small shared value-view descriptor with these non-overlapping meanings:

1. `packed`: the encoded quant payload, including block-local scale/zero/min metadata and exact payload bytes.
2. `storage`: the concrete backend-owned buffer representation, including dtype/layout, allocation bytes, padding/alignment, and memory scope.
3. `logical`: the canonical tensor shape/index domain and element count, independent of physical packing or layout.
4. `expressed`: the mathematical values and dtype presented to an operation after interpretation/dequantization.

Then make GGUF payload/dequant helpers and trace-facing compute metadata report the applicable view and these authoritative operations:

- `unpack`: packed blocks to logical quantized codes plus their block metadata. It preserves an exact, reversible bit/range mapping when the complete declared encoding is present; it does not yet produce expressed numeric values.
- `repack`: packed/storage layout to another packed/storage layout while preserving logical quantized codes and metadata. It is logical-value exact; it is bit-exact only when encoding, layout, padding, and byte order are identical.
- `expand`: representation-width or materialization expansion of encoded fields/codes without applying scale/zero/min numerical interpretation; emit canonical `expand` and preserve logical code values exactly.
- `dequantize`: numerical interpretation of codes plus scale/zero/min metadata under a pinned algorithm and output dtype; emit canonical `dequantize`, declare rounding/exactness relative to the encoded source, and never claim equality to unavailable pre-quantization weights.
- `contract`: many expressed inputs to an expressed output under a named accumulation dtype/order. It is a compute operation and canonical many-input lineage edge, not a storage/view conversion; numeric exactness comes only from an explicit tolerance or bitwise witness.

`reinterpret` may name unchanged bits viewed under compatible metadata, but it must validate shape/range compatibility. Every operation maps to the prerequisite `traceabi.Transform` vocabulary and exactness fields rather than creating local lineage semantics.

## Gold-plating boundary

No compiler IR, generic tensor algebra, arbitrary strided-view system, new backend buffer ABI, inverse quantizer, format registry rewrite, exact-byte reimplementation, or new lineage schema. Do not rename all existing `Dtype` uses. Add the minimum shared value-view semantics and adapters needed by the tracer family after #10380 lands.

## Done condition

- [ ] Packed, storage, logical, and expressed views have precise versioned definitions and cannot be silently interchanged.
- [ ] Each view reports only quantities it can know; unknown allocation/transfer/pre-quantization error remains unknown.
- [ ] Q4_K and F32 table cases consume #10380 byte results and prove payload bytes, logical element counts, expressed widths, and backend allocation bytes remain distinct.
- [ ] `unpack`, `repack`, `expand`, `dequantize`, and `contract` obey the distinct authoritative semantics above and project to their matching canonical `traceabi.Transform` operations/exactness.
- [ ] Invalid or underspecified view conversions fail with typed errors rather than defaulting to `Dtype.Bytes()`.
- [ ] Existing compute tensors and GGUF loading behavior remain byte-for-byte unchanged outside metadata/reporting; no #10380 byte calculation is duplicated.

## Witness

The deterministic command and expected invariants are specified in `## Verifiable Witness` below.

## Verifiable Witness

```text
go test ./internal/compute ./internal/ggufload -run 'Test(ValueView|GGUFValueView)' -count=1
```

The table must include F32 plus Q4_K with its exact encoded block size, a backend allocation with alignment padding, a logical 256-element shape, and a 256-element F32 expressed view. It must reject claiming an original-float error because those pre-quantization values are absent.

## Prior-art source, date, license, and disposition

- Observed `2026-09-08T16:09:42Z`; source event `2026-07-29T09:39:11Z`; shipped source: MLIR Quant dialect type model in `mlir/include/mlir/Dialect/Quant/IR/QuantTypes.h@61d6e494e97a8fda6946291fd3bc6bb648c86e93` ([commit](https://github.com/llvm/llvm-project/commit/61d6e494e97a8fda6946291fd3bc6bb648c86e93)). That change explicitly separates a storage format from a quantization scheme.
- License: [Apache-2.0 WITH LLVM-exception](https://github.com/llvm/llvm-project/blob/main/LICENSE.TXT). Disposition: **ADAPT / DEFAULT** — reuse the semantic distinction, expressed/storage terminology, and verification discipline; implement native Go types rather than porting MLIR classes. Platform context: llvm-project main, MLIR Quant dialect. Refresh trigger: Quant type-model revision.
- Transferable principle: data organization, logical type, and mathematical interpretation are orthogonal. FAK adds a distinct packed view because checkpoint block encoding and backend storage can differ. Disconfirming check: if the current compute API already proves all four meanings without inference, close this issue.

## Dedupe

Searched public open/closed quantization, dtype, packed-storage, and trace issues. #10380 owns the canonical exact-storage-byte calculation and trace-producer migration and is a hard prerequisite; this issue owns only the vocabulary that says what each number and transition means. Closed #12059 narrates the distinctions for one offline Q4_K command but does not expose reusable types. #10377 is a Metal F32 trace correction. #9138 is a fused-kernel performance leaf.

## Definition of done

The code exposes and tests the four meanings plus authoritative unpack/repack/expand/dequantize/contract semantics, and at least the GGUF payload/dequant and compute trace-facing seams consume them. It reuses #10380 byte calculations and canonical trace-ABI lineage. Documentation-only definitions, duplicated byte arithmetic, or an enum unused by real seams do not satisfy the issue.

## Working spine

GGUF encoded block → exact packed view → backend storage view → canonical logical view → expressed operation value, with explicit transition metadata and no data-copy behavior change.

## Acceptance gate

Focused tests and `go vet ./internal/compute ./internal/ggufload` pass; #10380 integration does not use rounded per-code widths for packed formats; operation-table tests prove the exactness rules above; `fak validate --mine` is green.

## Likely files

- `internal/compute/compute.go`
- A narrowly named `internal/compute/value_view.go`
- `internal/ggufload/gguf_dequant.go`
- Focused `_test.go` files in the same packages
- `docs/tickets/tracing-dataflow/TICKET-07.md`

## Lane

`compute`; two Go packages: `internal/compute` and `internal/ggufload`. Dispatch after TICKET-02 and #10380, and serialize before TICKET-06 on the shared GGUF tree.

## Closure binding

Close only from the signed commit that migrates real seams, passes the focused witness, and links #10380 without claiming hardware execution or reconstructed original weights.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12264

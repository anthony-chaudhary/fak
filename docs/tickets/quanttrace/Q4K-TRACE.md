<!-- fak-bench-key: q4k-quanttrace-offline-v1 -->

```routing
lane: bench
paths: ["internal/bench/quanttrace.go", "internal/bench/quanttrace_test.go", "cmd/quanttrace/main.go", "cmd/quanttrace/main_test.go", "docs/tickets/quanttrace/Q4K-TRACE.md"]
expected_steps: 5
```

## Problem
For maintainers understanding how a quantized matrix becomes a computation result, existing Q4_K repack benchmarks prove equivalence but do not explain individual code/scale provenance or distinguish physical layout from mathematical contraction. Today developers manually read the decoder and infer byte changes. Better because a bounded offline trace executes existing transforms and records auditable bytes, values and results.

## Current state
`internal/bench/q4krepack.go:163` (RepackQ4KInterleaved), `:175` (UnrepackQ4KInterleaved), `:189` (q4kDequantRecord), and `:216` (q4kGemvRowMajor) provide real portable primitives. `rg --files cmd/quanttrace` currently fails because no dedicated tracing command exists. Existing layout witnesses are in q4krepack_test.go. No production hook or live-device trace exists in this task.

## Centrality and P1-P4
Centrality: Enabling. Tier 2 serving.
P1 Context: preserved; offline fixtures do not modify sessions or caches.
P2 Net value: advanced; deterministic provenance reduces repeated manual decoding and enables representation comparisons without claiming throughput savings.
P3 Adaptation: preserved; reuse the existing bench transforms, with explicit opt-in standalone diagnostics.
P4 Operations: advanced; bounded input validation, deterministic receipts and clear simulated/offline scope prevent misleading execution claims.

## Core through-line
1. Add a trace at the existing bench seam using real Q4_K unpack/dequant, interleave repack and inverse, and portable GEMV.
2. Accept deterministic demo data or a raw Q4_K payload plus explicit matrix dimensions and interleave width.
3. Record packed provenance and hashes, code/scale unpack, lossless layout repack and inverse, float expansion, and mathematical matrix-vector contraction in JSON and readable output.
4. Explain sampled logical index to byte offset, low/high nibble, scale/min and decoded value formula; report stage input/output bytes, exact layout roundtrip, decoded equality and output equality.
5. Witness malformed and overflow input refusal alongside the deterministic successful trace.

## Gold-plating boundary
No inverse quantizer, new quant format, GPU instrumentation, production hooks, model execution, hardware throughput claim, full model loader, or format registry. Mathematical contraction reduces an axis; it is not lossy storage quantization. A lossless repack cannot reduce storage bytes. Do not claim that float expansion reconstructs original pre-quantized weights.

## Done condition
- [x] Demo and raw-payload CLI modes emit the trace and provenance using explicit dimensions/width.
- [x] JSON and readable output distinguish representation/layout, numeric expansion and GEMV contraction.
- [x] Existing transforms drive the receipt and exact roundtrip/decoded/output equivalence checks.
- [x] Sampled code/scale provenance explains the numerical result.
- [x] Malformed shapes, sizes and overflowing arithmetic return errors before unsafe allocation or indexing.
- [x] Focused tests and vet pass; trace is explicitly offline portable evidence.

## Witness
`go test ./internal/bench ./cmd/quanttrace -run QuantTrace -count=1`
This command must exercise the demo plus raw-input path and reject malformed/overflow cases. Before implementation the cmd/quanttrace package is absent; after implementation tests pass and a demo command emits a deterministic receipt.

## Verifiable Witness
`go test ./internal/bench ./cmd/quanttrace -run QuantTrace -count=1`

## Likely files
internal/bench/quanttrace.go
internal/bench/quanttrace_test.go
cmd/quanttrace/main.go
cmd/quanttrace/main_test.go
docs/tickets/quanttrace/Q4K-TRACE.md

## Lane
bench; two Go packages: internal/bench and cmd/quanttrace. Existing serving, gateway, loader and device lanes remain unchanged. Additive offline command is the fallback boundary; no quarantine is required.

## Dedupe
Searched public open/closed quant tracing and open quant trace/repack issues, plus private open tracing/repack issues. No exact match. Related public #9138 optimizes fused computation; #11917 addresses Vulkan memory planning; #11946 is a registry refactor; closed #10349 fixes served quant provenance. This issue is a separate offline explanation/witness tool.

## Definition of done
The command, focused tests and ticket land together in one commit. Witness: go test ./internal/bench ./cmd/quanttrace -run QuantTrace -count=1. Close only after that committed witness passes; the demo emits the documented stage receipt. This completes only an offline development diagnostic, not production or hardware qualification.

## Why now
The requested dedicated quantization trace needs a concrete, executable explanation; existing repack primitives already supply the working spine. Parent reference: operator-requested standalone tracing work, related #9138. Working spine: existing RepackQ4KInterleaved, UnrepackQ4KInterleaved and q4kDequantRecord. Acceptance gate: focused QuantTrace tests green. Closure binding: resolving commit plus passing witness.

## Parent context
Related enabling work: #9138. This is an independently scoped diagnostic leaf.

## Working spine
Input Q4_K bytes to RepackQ4KInterleaved and UnrepackQ4KInterleaved to q4kDequantRecord and GEMV to a durable JSON receipt.

## Acceptance gate
`go test ./internal/bench ./cmd/quanttrace -run QuantTrace -count=1` passes on the resolving commit.

## Closure binding
Close with the resolving commit and captured passing witness. The offline demo receipt must be emitted by that commit.
Parent: #9138



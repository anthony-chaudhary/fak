<!-- fak-tensortrace-key: live-loader-kernel-lineage-v1 -->

```routing
lane: compute
paths: ["internal/ggufload/**", "internal/model/**"]
expected_steps: 8
```

# feat(tensortrace): trace one live GGUF tensor from loader bytes to kernel consumption

## Problem

For native-inference developers diagnosing whether a checkpoint tensor followed the intended packed, residency, upload, and kernel path, the current evidence stops at disconnected boundaries. The offline Q4_K explainer proves unpack/repack/expand/contract semantics, while live compute events prove that a kernel ran; neither proves that a particular named checkpoint tensor became the tensor consumed by that kernel. Today an agent must correlate names, shapes, dtypes, byte counts, and timing by reading implementation code. Better because one bounded causal trace would answer “which checkpoint bytes became this kernel operand?” without dumping tensor contents. The witness is a tiny real GGUF fixture whose named Q4_K tensor produces a complete, digest-linked read → materialize → upload → consume chain.

## Parent context

First-class public `fak` quantization dataflow tracing goal. This leaf is an adapter/producer over the canonical public contracts from `typed-causal-envelope-v1`, `many-to-many-value-lineage-v1`, and `bounded-private-evidence-policy-v1`; it must not define competing identity, lineage, evidence, or capture types.

## Why now

#12059 established the offline Q4_K explanation, but no live witness joins loader bytes to the kernel operand. The public trace ABI, the explicit value-view semantics from TICKET-07, and exact storage-byte ownership from #10380 must land first so this issue records one meaning rather than inventing another.

## Current state

Inspected at public `fak` commit `325ade2db2a7b1ea556d03d89293162a85cd1e9e`:

- `internal/ggufload/gguf_weightsource.go:282-323` reads exact still-quantized tensor bytes and optionally expands them to F32, but emits no stable tensor identity or derivation event.
- `internal/model/hal.go:260-290` materializes Q4_K/K-quant bytes and stages them through `Backend.Upload`; `internal/model/hal.go:736-820` later selects fused or unfused projection calls, but the checkpoint tensor name is not carried into the compute event.
- Closed #12059 supplies only an offline software diagnostic. Open #10380 owns canonical exact packed-storage byte accounting needed for truthful live byte fields.

## Dependencies and schedule

- Hard prerequisite: `typed-causal-envelope-v1` for trace/span/logical correlation.
- Hard prerequisite: `many-to-many-value-lineage-v1` for canonical `ValueRef` and `Transform` fan-in/fan-out.
- Hard prerequisite: `bounded-private-evidence-policy-v1` for capture admission, loss accounting, evidence ceilings, and receipts.
- Hard prerequisite: TICKET-07 `explicit-value-views-v1` for packed/storage/logical/expressed meanings and legal operation semantics.
- Hard prerequisite: open #10380 for canonical exact packed and allocated byte calculations.
- Hard prerequisite: TICKET-04 `causal-trace-query-explain-export-v1`; the live GGUF artifact must be ingestible by the agent-facing query surface rather than forming a producer-only dialect.
- Schedule TICKET-07 before TICKET-06 because both touch `internal/ggufload/**`. Schedule TICKET-06 before TICKET-08 and TICKET-09 because all three touch `internal/model/**`.

## Centrality and P1-P4

Centrality: Enabling (trustworthy native tensor execution for Tier 2 serving)

P1: preserved - no prompt or session payload is captured; identifiers are digest-safe and request-scoped.

P2: advanced - removes repeated manual loader/kernel correlation and prevents false bandwidth conclusions; observer overhead and dropped events are reported.

P3: advanced - the schema names executed format, layout, backend, and transformation rather than guessing from configuration; unknown lineage stays unknown.

P4: advanced - a bounded opt-in artifact survives failure and explains the actual native path from source bytes to kernel operand.

## Scope

Implement one opt-in loader-to-kernel producer across `internal/ggufload` and `internal/model`. Adapt domain facts into the prerequisite `pkg/traceabi` contracts without editing or shadowing them; keep existing compute tracing unchanged.

## Core through-line

1. Populate `traceabi.CausalEnvelope` for one loaded tensor using checkpoint identity, canonical tensor name, dtype, shape, shard, and byte range—never a pointer or transient slice index.
2. Emit domain events for `checkpoint_read`, `resident_materialize`, `backend_upload`, and `kernel_consume` through an adapter that retains the canonical envelope and `CaptureReceipt`.
3. Carry TICKET-07 value-view descriptors, source/result digests, logical shape, and actual bytes at each transformation; obtain exact packed/storage counts from #10380 rather than `Numel()*Dtype.Bytes()`.
4. Encode split and fused consumption with canonical `traceabi.ValueRef` and `traceabi.Transform` inputs/outputs; do not create a model-owned lineage dialect.
5. Prove the chain on one real tiny GGUF Q4_K projection executed through fak-native code with a deterministic recording backend.

## Gold-plating boundary

No full-model tensor dump, activation capture, GPU vendor profiler integration, general distributed tracing framework, new quant format, pointer exposure, always-on hashing of multi-gigabyte weights, or physical-throughput claim. Trace one bounded tensor path first. Perfetto/OTLP export is a later adapter; the local artifact remains authoritative.

## Done condition

- [ ] One model adapter projects tensor stages into the versioned public `traceabi.CausalEnvelope`, `ValueLineage`, and `CaptureReceipt` contracts; no duplicate event envelope or lineage graph exists.
- [ ] A named GGUF Q4_K tensor emits read, materialize, upload, and kernel-consume events in causal order.
- [ ] Every event distinguishes logical elements from exact packed, allocated, read, transferred, and written bytes; #10380 is used where applicable.
- [ ] Fused/multi-input consumption retains every source tensor ID without relying on array position or memory address.
- [ ] Disabled tracing remains a no-op; enabled tracing is admitted by the canonical capture policy, reports drops/overhead in its canonical receipt, and never serializes raw tensor data.
- [ ] Malformed/unknown lineage is explicit and does not fabricate a parent.
- [ ] The focused tests and `go vet` pass in both affected packages.

## Witness

The deterministic command and expected invariants are specified in `## Verifiable Witness` below.

## Verifiable Witness

```text
go test ./internal/ggufload ./internal/model -run 'TestLiveTensorTrace' -count=1
```

The test builds a tiny GGUF containing one Q4_K projection, runs the actual loader and HAL staging path against a deterministic recording backend, and asserts an exact source-name/shape/dtype/digest chain ending at the consumed matmul operand. A control run with tracing disabled emits zero events. This is `[SW-VERIFIED]`, not hardware bandwidth evidence.

## Prior-art source, date, license, and disposition

- Observed `2026-09-08T16:09:42Z`; source event `2026-07-22T18:24:42Z`; shipped SDK/protocol source: Perfetto `TrackEvent` flow/correlation fields in `protos/perfetto/trace/track_event/track_event.proto@ad0b91a934fc3cba719642e8d148f6ffd27992f0` ([commit](https://github.com/google/perfetto/commit/ad0b91a934fc3cba719642e8d148f6ffd27992f0)).
- License: [Apache-2.0](https://github.com/google/perfetto/blob/main/LICENSE). Disposition: **ADAPT / DEFAULT** for stable causal IDs and explicit flow edges; do not copy its protobuf or require Perfetto. Platform context: Perfetto TrackEvent SDK on main. Refresh trigger: TrackEvent flow-field/schema change.
- Transferable principle: asynchronous stages need explicit flow edges; temporal proximity is not lineage. FAK opportunity: preserve tensor identity across loader, residency, upload, and fusion. Disconfirming check: if a stable identity already propagates across all four live seams at implementation time, close as duplicate.

## Dedupe

Searched all public open/closed issue titles for tensor/quant/lineage traces and inspected #10380, #10377, #12059, #9138, and #10184. #12059 explains Q4_K transformations offline; this issue begins where it stops and traces the live loader-to-kernel chain. #10380 is a dependency for exact packed-byte fields, not a lineage tracer. #9138 owns fusion performance, while this issue supplies diagnosis. #10184 records Go scheduler history around stalls, not tensor dataflow.

## Definition of done

The resolving commit contains the bounded schema, all four live emission sites, the deterministic end-to-end GGUF fixture, and the passing witness. A partial trace that ends at upload or begins at kernel dispatch does not satisfy this issue.

## Working spine

`WeightSource.TensorBytes` → TICKET-07 value views → Q4_K model tensor → `Session.weightHALQ4K`/`Backend.Upload` → one HAL projection → canonical trace-ABI envelopes and value-lineage transforms.

## Acceptance gate

The focused witness and `go vet ./internal/ggufload ./internal/model` pass; `fak validate --mine` is green for the changed paths; the captured artifact contains no raw tensor payload and reports its bound/drop count.

## Likely files

- `internal/ggufload/gguf_weightsource.go`
- `internal/model/hal.go`
- `internal/model/tensor_trace.go` (new)
- Focused `_test.go` files in those same packages
- `docs/tickets/tracing-dataflow/TICKET-06.md`

## Lane

`compute`; two Go packages: `internal/ggufload` and `internal/model`. Dispatch only after TICKET-07 and #10380, then serialize before TICKET-08 and TICKET-09 because of the shared model tree. No gateway, provider, or private-platform paths.

## Closure binding

Close only from one signed resolving commit that references the issue, includes the focused passing receipt, and identifies the artifact as software/native-path evidence. Hardware qualification remains unchecked.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12263

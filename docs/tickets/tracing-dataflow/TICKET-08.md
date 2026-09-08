<!-- fak-numtrace-key: layer-op-first-divergence-v1 -->

```routing
lane: model
paths: ["internal/model/**", "internal/computetrace/**"]
expected_steps: 8
```

# feat(numtrace): locate the first layer or operation where two native runs diverge

## Problem

For agents debugging a wrong token, final-logit comparisons say that execution diverged but not where. Today maintainers bisect manually with test-only activation patches, backend readbacks, or ad hoc prints; kernel timing traces lack result summaries and layer/operation coordinates. Better because an opt-in numerical tracer compares two executions in causal order and reports the first materially different value with its layer, operation, shape, dtype, tolerance, and bounded error summary. The witness deliberately perturbs one named operation in a tiny model and identifies that operation—not merely the later logits—as the first divergence.

## Parent context

First-class public `fak` quantization dataflow tracing goal. This bounded numerical-correctness leaf consumes the canonical envelope, lineage, capture, and value-view contracts after their coordinates and meanings are stable; it does not define replacements.

## Why now

#10377 and #12059 improved trace correctness and offline explanation, but diagnosis still cannot identify the first live layer or operation that diverges. TICKET-06 must first prove live tensor identity through the same model tree so this issue can attach numerical summaries to established causal operands.

## Current state

Inspected at public `fak` commit `325ade2db2a7b1ea556d03d89293162a85cd1e9e`:

- `internal/model/activation_patch.go:5-69` can capture or inject the completed residual at one selected layer, which supports manual layer bisection but not ordered operation tracing.
- `internal/model/hal.go:619-827` executes embedding, normalization, Q/K/V projections, RoPE, KV append, attention, output projection, MLP, and fused variants, but emits no uniform semantic operation coordinate or bounded numeric fingerprint.
- `internal/computetrace/trace.go:22-47` records timing/bytes/shapes and provenance, but no layer, token position, source/result digest, numeric summary, tolerance, or comparison outcome.
- Existing family oracle tests compare final or layer outputs case by case; no reusable first-divergence artifact exists.

## Dependencies and schedule

- Hard prerequisite: `typed-causal-envelope-v1` for semantic event identity and ordering.
- Hard prerequisite: `many-to-many-value-lineage-v1` for source/result relationships and fused fan-in/fan-out.
- Hard prerequisite: `bounded-private-evidence-policy-v1` for summary-only capture, limits, loss accounting, and evidence ceilings.
- Hard prerequisite: TICKET-07 `explicit-value-views-v1` for dtype/view interpretation and comparison meaning.
- Hard prerequisite: TICKET-06 `live-loader-kernel-lineage-v1` for the live model operand identity this numerical trace annotates.
- Schedule after TICKET-06 and before TICKET-09; all three serialize on `internal/model/**`.

## Centrality and P1-P4

Centrality: Enabling (native numerical correctness diagnosis)

P1: preserved - prompts are represented by request and token-position identifiers; tensor contents remain out of the default artifact.

P2: advanced - converts an O(layers x manual reruns) debugging loop into one bounded comparison while reporting observer and readback cost.

P3: advanced - comparison policy is explicit per dtype and operation; fused and unfused paths map to semantic coordinates rather than pretending instruction streams match.

P4: advanced - a deterministic artifact identifies the first divergence and whether later differences are consequential propagation.

## Scope

Add one bounded numerical summary/comparison adapter across `internal/model` and `internal/computetrace`. It populates canonical public trace-ABI records and TICKET-07 value views without editing or shadowing those contracts; no payload dump or offline search service.

## Core through-line

1. Attach stable domain coordinates `(request, token_position, layer, op_kind, occurrence)` to `traceabi.CausalEnvelope`, including fused operations represented by canonical many-to-many transforms.
2. Under the canonical capture policy and receipt, emit TICKET-07 view/dtype plus bounded numeric summaries: finite/non-finite counts, min/max, norm, deterministic sample indices, and a digest over canonicalized values.
3. Compare two artifacts in semantic causal order using an explicit absolute/relative tolerance policy; report the first missing, shape/dtype-mismatched, non-finite, or numerically divergent coordinate.
4. Preserve raw bits only for tiny test fixtures or explicit local diagnostic capture; default receipts contain summaries/digests, not full activations.
5. Use the existing residual hook and a deliberately perturbed named operation to prove the reporter stops at the first cause rather than the final symptom.

## Gold-plating boundary

No automatic root-cause explanation, full tensor dumping, every-backend instrumentation, remote fleet comparison, performance regression system, training autograd graph, or cross-model semantic alignment. First ship one dense native forward with a bounded diagnostic and a stable fused/unfused mapping.

## Done condition

- [ ] A model/computetrace adapter attaches semantic layer/op/token coordinates and bounded value summaries to canonical trace-ABI envelopes, lineage, and capture receipts; no parallel event envelope exists.
- [ ] A deterministic comparator returns exactly one first-divergence record plus explicit reason and tolerance.
- [ ] Missing events, reordered events, shape/dtype drift, NaN/Inf, and numeric error are distinguishable.
- [ ] A fused QKV or MLP operation can map to semantic outputs without false sequence mismatch against an unfused reference.
- [ ] The default artifact contains no full activation/logit arrays and reports event/sample drops and observer/readback overhead through the canonical capture receipt.
- [ ] A one-operation perturbation fixture identifies the injected coordinate before final logits diverge.
- [ ] Disabled tracing preserves bit-identical output.

## Witness

The deterministic command and expected invariants are specified in `## Verifiable Witness` below.

## Verifiable Witness

```text
go test ./internal/model ./internal/computetrace -run 'TestNumericalFirstDivergence' -count=1
```

The witness runs the same tiny deterministic model twice, injects a known delta after one named layer operation in only the second run, and asserts the artifact reports that coordinate, error metrics, and tolerance. A no-perturbation control reports equality, and tracing-off logits remain bit-identical.

## Prior-art source, date, license, and disposition

- Observed `2026-09-08T16:09:42Z`; functional source event `2022-08-16T00:03:28Z`; shipped source: ONNX Runtime QDQ activation matching in `onnxruntime/python/tools/quantization/qdq_loss_debug.py@eb6aa861cfa7295ee9f7145db44aaec708e8ce1c` ([commit](https://github.com/microsoft/onnxruntime/commit/eb6aa861cfa7295ee9f7145db44aaec708e8ce1c)); the file remained present through `ad312d96775d7d0ff0c2259c1440dad13d59a1e8` on `2025-01-17`.
- License: [MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE). Disposition: **ADAPT / OPTIONAL-MODULE** — adapt activation matching and explicit error calculation to FAK’s streaming native execution; do not port ONNX graph rewriting or Python. Platform context: ONNX Runtime post-training quantization debugger. Refresh trigger: QDQ debugger API or matching-strategy change.
- Transferable principle: match equivalent intermediate values before scoring error, then locate the earliest mismatch. FAK opportunity: stable semantic op coordinates across fused/unfused native paths. Disconfirming check: a current reusable FAK artifact already returns the first layer/op mismatch under a declared tolerance.

## Dedupe

Searched public open/closed issues for numerical/activation/first-divergence/tensor traces. #12059 compares two Q4_K contraction results inside one offline diagnostic; this issue compares live intermediate operations across executions. #10380 supplies exact bytes, not numeric values. #10184 captures runtime scheduling around stalls, not mathematical divergence. TICKET-06 (`fak-tensortrace-key: live-loader-kernel-lineage-v1`) is a hard predecessor that supplies live operand provenance rather than overlapping numeric comparison.

## Definition of done

The tracer captures real semantic intermediate boundaries, the comparator deterministically identifies the injected first divergence, and the no-op/disabled controls prove non-perturbation. A logits-only diff or manual bisection helper is incomplete.

## Working spine

Native forward semantic boundary → bounded numeric event → paired trace alignment → first divergent coordinate → compact diagnostic receipt.

## Acceptance gate

The focused witness and `go vet ./internal/model ./internal/computetrace` pass; `fak validate --mine` is green; the fixture proves both a true positive and an equal-run control without unbounded payloads.

## Likely files

- `internal/model/activation_patch.go` or a narrowly named numerical observer file
- `internal/model/hal.go`
- `internal/computetrace/trace.go` or a comparison companion
- Focused `_test.go` files in the same packages
- `docs/tickets/tracing-dataflow/TICKET-08.md`

## Lane

`model`; two Go packages: `internal/model` and `internal/computetrace`. Dispatch after TICKET-06 and serialize before TICKET-09 on the shared model tree.

## Closure binding

Close only from the signed resolving commit whose attached artifact names the deliberately injected first divergent layer/op and whose focused test passes at that commit.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12265

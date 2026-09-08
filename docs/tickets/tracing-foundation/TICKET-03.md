# feat(tracesink): gate every trace capture with bounded privacy and evidence policy

```routing
lane: tracesink
paths:
  - pkg/traceabi/**
  - internal/tracesink/**
expected_steps: 8
```

<!-- fak-trace-foundation-key: bounded-private-evidence-policy-v1 -->

## Parent context

#6801 and #6053. Depends on `typed-causal-envelope-v1`, then extends the same public `pkg/traceabi` seam with bounded capture/evidence contracts. It reuses the egress discipline delivered by #5629/#8109 and the existing privacy package rather than reopening those completed exporter tickets. It is the policy dependency for any new tensor, context, turn, or agent-flow trace capture.

## Why this is next

Adding more tracers before the common capture floor would multiply unbounded buffers and privacy/evidence rules across every producer.

## Problem

For operators and agents, a trace is useful only if enabling it cannot leak protected content, silently change claim strength, or grow until it perturbs the workload being observed. Today individual tracers make separate choices: some are bounded and payload-free, while `TraceSink` retains a growing call slice and can carry raw tool arguments. Better because one capture policy makes byte/event limits, payload treatment, evidence basis, drops, retention, and export decisions explicit before data enters a sink. Witness: a secret-bearing, mixed-evidence event stream stays within exact count/byte bounds, contains no sentinel bytes, and cannot be promoted from modeled to measured by export.

## Centrality and P1-P4

**Centrality: Core safety for tracing.** New tracers should not ship without this reusable floor.

- P1 Context: advanced - summaries/digests are the default agent-visible form; full payloads are opt-in artifacts, preventing trace data from becoming new context debt.
- P2 Net value: advanced - observer bytes, drops, and overhead are counted; tracing cannot claim a gain while hiding its own cost.
- P3 Adaptation: advanced - fixed limits, closed payload/evidence classes, fail-closed policy, and non-upgrade rules constrain adaptation.
- P4 Operations: advanced - retention and export decisions emit typed receipts that every sink and later query surface can consume.

## Current state

At `f21eb416d92f03286747f84e888fb7dd1038c7a9`:

- `internal/tracesink/tracesink.go:1-7` explicitly captures raw argument payloads for offline replay and applies IFC redaction only when tainted data flows to a sensitive sink.
- `internal/tracesink/tracesink.go:45-61 (TraceSink)` keeps `calls []turnbench.Call` plus aggregate dropped/redacted counts but has no max-event or max-byte fields.
- `internal/tracesink/tracesink.go:114-175 (Emit)` resolves the full args, then appends every accepted call; the slice has no retention bound and trusted content is copied raw at `:163-173`.
- `internal/privacy/privacy.go:33-60` already defines per-sink enablement, field redaction, retention count/age, and decision receipts.
- `internal/privacy/privacy.go:84-98,100-154` validates policy and evaluates log/retain/export/telemetry, but classification is field-name based and has no trace evidence-basis or claim-ceiling rule.
- `internal/computetrace/trace.go:1-7,64-93` is a positive local example: opt-in, max-event bounded, and reports dropped events and observer overhead.
- `internal/benchcli/simulation.go:27-40,42-68` has a benchmark-specific evidence/claim vocabulary, but it is not a general trace capture policy.

The capability is **PARTIAL**: useful privacy, IFC, and bounded-recorder pieces exist, but no shared trace-capture floor combines them.

## Core through-line

Add data-only `CapturePolicy` and `CaptureReceipt` contracts to public `pkg/traceabi`, then enforce them in `internal/tracesink`:

- mandatory `max_events`, `max_total_bytes`, `max_event_bytes`, and optional max age;
- payload mode (`none`, `digest`, `summary`, `full_local_opt_in`) with digest/summary as the safe defaults;
- evidence basis (`direct_measurement`, `direct_observation`, `derived`, `modeled`, `operator_supplied`, `unknown`) plus an optional source/digest and claim ceiling;
- monotonic evidence rule: projection, query, retention, or export may preserve or weaken evidence but never strengthen it;
- per-reason dropped/redacted/truncated counters, input/output bytes, observer overhead, and a policy digest;
- privacy evaluation before retention/export and explicit refusal when full payload capture lacks local opt-in.

Apply the policy to the existing `TraceSink` as the tracer bullet. Other public and private recorders adopt the same contract later through local adapters.

## Working spine

1. Define closed payload/evidence/action vocabularies and strict policy parsing in `pkg/traceabi`.
2. Require nonzero event, per-event byte, and total-byte limits.
3. Evaluate privacy/IFC before appending any payload-bearing call.
4. Add count and byte admission to `TraceSink`; never evict silently.
5. Emit deterministic receipts with offered/retained/dropped/redacted/truncated events and bytes by reason.
6. Enforce evidence monotonicity and unknown-by-default behavior.
7. Prove secret, oversize, exhaustion, modeled-evidence, and disabled/default fixtures.

## Gold-plating boundary

Do not build a policy language, DLP scanner, encryption/key service, remote collector, sampling optimizer, or global retention daemon. Do not copy benchmark-specific evidence structs into the sink. Keep the exported contracts data-only and free of recorder/private dependencies. This issue supplies one small closed policy and applies it to `TraceSink`; domain adapters remain follow-ons.

## Dedupe

Search across open and closed public issues on 2026-09-08 found:

- #6801: broader harness-kit telemetry/eval/artifact sink lifecycle and conformance; this issue supplies its reusable capture floor.
- #8109 (closed): bounds one OTLP gateway exporter queue and excludes prompts/tool args; it does not govern local or future tracer capture.
- #5629 (closed): W3C/OTLP served-turn backbone and redaction requirements; transport/propagation is complete and out of scope here.
- #6212 (closed): mines privacy-safe trajectory patterns; it is a consumer, not a capture policy.
- #10269: evidence independence for quorum; it does not classify the basis/claim ceiling of trace fields.

No issue found combines bounded capture, privacy decision receipts, and evidence monotonicity at the trace sink boundary.

## Prior-art ledger

- **Source:** OpenTelemetry SDK trace specification `specification/trace/sdk.md:844-883,933-940,1085-1130@ce9394dc3dd53bb182611d2f3ba1e8f53975e905`.
- **Source event:** commit `ce9394dc3dd53bb182611d2f3ba1e8f53975e905`, 2026-09-08T13:54:14Z; source state: shipped specification on `main`.
- **Observed at:** 2026-09-08T16:10:19Z.
- **License:** Apache-2.0.
- **Disposition:** **ADAPT / DEFAULT**. Reuse explicit event/link/attribute limits and bounded batch/export separation; add fak's privacy receipts and evidence/claim monotonicity. No upstream code copied and no SDK dependency added.
- **Refresh trigger:** OpenTelemetry span-limit/exporter changes or a witnessed fak trace that cannot fit the bounded policy.
- **Spirit extension:** a dropped-count is part of the evidence, not merely an exporter metric; incomplete observation must cap the claim an agent may make from the trace.

## Done condition

The public trace ABI defines one bounded capture/evidence contract; `TraceSink` cannot retain or export a trace without it, and every capture produces a privacy- and evidence-aware receipt.

## Definition of done

- [ ] Policy parsing is strict and rejects zero/unbounded counts or bytes.
- [ ] Default payload mode stores no raw prompt, reasoning, tool args, credential, or cache-key bytes.
- [ ] Full payload mode requires explicit local opt-in and still passes IFC/privacy evaluation.
- [ ] Event, per-event byte, total-byte, and age limits are enforced deterministically.
- [ ] Every omitted byte/event lands in one closed reason counter.
- [ ] Receipt totals reconcile offered = retained + dropped and input bytes = retained/redacted/truncated/dropped accounting.
- [ ] `unknown`, `modeled`, or `operator_supplied` evidence cannot become `direct_measurement` through projection/export.
- [ ] Disabled/default operation adds no background goroutine.
- [ ] Existing trace replay behavior remains available under explicit bounded full-local policy.

## Verifiable Witness

```bash
go test ./pkg/traceabi ./internal/tracesink -run 'TestCapturePolicy|TestExternalCaptureConsumer|TestTraceSinkBounded|TestTraceSinkEvidenceMonotonicity' -count=1
```

The adversarial fixture offers 1,000 events including a unique secret sentinel and oversize payloads under an 8-event/4-KiB budget. The retained artifact must remain within both limits, contain no sentinel, and reconcile every drop/redaction byte.

## Witness

Run the deterministic `go test ./pkg/traceabi ./internal/tracesink -run 'TestCapturePolicy|TestExternalCaptureConsumer|TestTraceSinkBounded|TestTraceSinkEvidenceMonotonicity' -count=1` gate above.

## Acceptance gate

Two runs of the adversarial fixture produce byte-identical receipts; retained events and bytes never exceed policy; the secret sentinel is absent from artifact and receipt; an export adapter cannot strengthen any evidence class.

## Likely files

- `pkg/traceabi/capture.go`
- `pkg/traceabi/capture_external_test.go`
- `internal/tracesink/policy.go`
- `internal/tracesink/policy_test.go`
- `internal/tracesink/tracesink.go`

## Lane

tracesink

Schedule after `typed-causal-envelope-v1` because both leaves extend `pkg/traceabi/**`.

## Expected steps

8

## Closure binding

The resolving commit cites this issue and uses `(fak traceabi)`. Report `pkg/traceabi@rev`, `internal/tracesink@rev`, the capture-policy digest, and the adversarial receipt totals.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12260

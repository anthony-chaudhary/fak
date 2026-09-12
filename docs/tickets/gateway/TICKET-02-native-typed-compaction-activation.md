<!-- fak-gateway-key: native-typed-compaction-activation-v1 -->

# feat(gateway): witness native typed compaction in request feature activation

## Current state

The #12582 candidate now implements native typed history compaction observation
through a summary captured around the actual typed cut and published only after
successful native generation. Independent synthetic native-planner tests pass
under the race detector on the integrated roster parent. They verify software
behavior; no physical Qwen3.8 execution witness is claimed here.
That issue also records actual raw Anthropic compaction in
`internal/gateway/messages_transform.go`, where `compacted` reports the result of
`compactAnthropicRawWithReason`. Its other required seams are accepted vDSO hits,
result elision, and manifest route selection. This follow-up extends coverage; it
does not claim that native hardware execution has been witnessed. The remaining
hardware qualification stays open under #12835 after the software work lands.

Read-only source evidence at public commit
`26083318dcbf4409b2465f78691d174d1443426c`:

- `internal/agent/inkernel_planner.go:1976`: the completion path calls
  `p.preparePrompt(ctx, messages, tools, sp, opts...)` before using its prepared
  messages, tools, and token IDs.
- `internal/agent/prompt_encoding.go:78`: `preparePrompt` calls
  `p.ApplyPromptShrink(...)` and discards the third return value.
- `internal/agent/inkernel_planner.go:280-318`: `ApplyPromptShrink` resolves the
  configured/request budget and invokes `ApplyTypedPromptShrinkLevers`.
- `internal/agent/message_compact.go:365-377`: that helper calls
  `CompactMessagesWithOptions` and records `TypedPromptShrinkOutcome.Compacted`
  and `CompactOutcome`.
- `internal/agent/prompt_encoding.go:35-54`: `EncodePrompt` also uses
  `preparePrompt` without prefill or decode. A callback at shared preparation
  alone must not falsely certify that a native completion ran.

No model run, hardware qualification, speedup, or test pass is claimed by this
ticket. The evidence above is source inspection only.

## Problem frame

- Portfolio tier: 1, all-in-one native serving and harness.
- Centrality: Core.
- P1: advanced - expose actual typed-history compaction on a served native turn.
- P2: preserved - observe existing outcomes without a second compaction pass,
  model invocation, or performance claim.
- P3: advanced - use a bounded request callback with a closed feature identifier.
- P4: advanced - join real native execution to the existing headers, final
  trailer, access-log feature_used list, and one-per-request counter.

For: clients and operators using native chat/messages serving.
Problem: enabled compact_history cannot establish whether native typed history
was actually compacted on the request.
Today: raw Anthropic transformation has request evidence; native typed compaction
requires inspecting an outcome that is discarded before the gateway can use it.
Better because: one request-scoped outcome reaches the existing tracker without
inferring use from configuration or borrowing proxy evidence.
Witness: actual native active/idle requests with captured prepared input and
matching request activation on the same execution.

## Parent context

Coordinates with: #12582 (request tracker and wire timing contract).
Coordinates with: #12581 (closed catalog identifiers).
Promotion requires: #12582 (integrated tracker for end-to-end closure).

Pickup rule: the agent-layer callback and outcome propagation are a disjoint
executable slice; tracker integration follows its public API. These coordination
relations do not block that slice. Gateway edits require an available lane lease.

## Why this is next

The native path already computes the necessary outcome. Preserve that evidence
at its producer and bind it to the served request rather than deriving a global
counter delta or asserting all configured compaction is active.

## Working spine

Real native completion -> typed prompt shrink outcome -> request-scoped callback
-> FeatureCompactHistory activation -> truthful initial header/final trailer,
access log, and cumulative request counter.

## Core through-line

Carry the typed shrink outcome across prompt preparation and expose an optional,
request-scoped callback at the actual completion path. Bind the gateway tracker
without introducing an agent-to-gateway import cycle or process-global state.
Record only an accepted compaction cut applied to that request's native input.
Preserve prompt-only EncodePrompt semantics and live streaming progress.

## Gold-plating boundary

No compaction algorithm changes, extra prompt shrinking, native kernel tuning,
new model fallback, catalog expansion, other shrink-feature instrumentation,
sampling changes, or tracker redesign. Keep Qwen3.8 execution fak-native. A
software callback test alone cannot satisfy the real native execution witness.

## Scoped acceptance criteria

- [ ] Propagate the real typed compaction outcome through a bounded request
  callback; under-budget, disabled, refused, and identity transforms emit no use.
- [ ] Prompt-only EncodePrompt neither emits completion-use evidence nor alters
  the result later recorded by the actual completion path.
- [ ] Chat/messages native requests record compact_history at most once in the
  existing tracker, log, and metric. Concurrent requests remain isolated.
- [ ] Initial Used remains a precommit snapshot; late execution appears only in
  the declared Used-Final trailer. Incomplete execution does not certify a
  complete final record. Live SSE first-chunk progress is preserved.
- [ ] A focused regression fails before propagation and passes afterward.
- [ ] A real Qwen3.8 native active/idle pair captures engine identity, model and
  device, compaction outcome, the actual prepared input used for execution, and
  corresponding HTTP/log/metric evidence. Keep witness content scrubbed.

## Witness

Proposed focused regression (to be authored, not an existing passing test):

```text
go test -race ./internal/agent ./internal/gateway -run 'TestNativeTypedCompactionActivation' -count=1
```

The test must include the served native path using a supported Qwen3.8 artifact.
Run the physical witness on a sanctioned capable node discovered through
`docs/fleet-compute-nodes.md`, with a bounded active/idle workload. Record a
repeatable invocation and artifact digest in this ticket before closure. No
benchmark comparison or speed target is required.

## Definition of done

Outcome propagation, gateway binding, independent regression, and real native
request evidence are present. Agent/gateway scoped gates pass in an isolated
worktree, and the resolved issue cites the witnessed commit. Absent the native
receipt, delivery may be recorded but end-to-end activation remains unverified.

## Acceptance gate

Focused race regression plus a captured real native active/idle execution receipt;
then affected-package verification through the guarded validation surface.

## Closure binding

One DCO-signed resolving commit cites this issue and carries `(fak gateway)`.

## Lane

gateway-native-activation

## Likely files

- internal/agent/inkernel_planner.go
- internal/agent/prompt_encoding.go
- internal/agent/sample_options.go
- internal/gateway/gateway.go
- internal/gateway/feature_tracker_http.go
- internal/gateway/native_compaction_activation_test.go

## Blast radius and affected lanes

Agent prompt preparation and gateway request observability only. Existing native
compute, compaction decisions, storage, and external-provider bodies are unchanged.

## Expected steps

3

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/2 points.

## Completion standard

development

```routing
lane: gateway-native-activation
paths: internal/agent/inkernel_planner.go, internal/agent/prompt_encoding.go, internal/agent/sample_options.go, internal/gateway/gateway.go, internal/gateway/feature_tracker_http.go, internal/gateway/native_compaction_activation_test.go
expected_steps: 3
```

## Duplicate search

Open and closed public issues searched for compaction plus activation, the exact
ApplyTypedPromptShrinkLevers symbol, and native feature titles on 2026-09-12.
No matching native typed-compaction activation follow-up was found. #12582 is
the narrower gateway producer spine; speculative KV compaction is a different
mechanism and is not a duplicate.


## Tracking

Public issue: [#12835](https://github.com/anthony-chaudhary/fak/issues/12835).

The governed discoverability audit classified this ticket as dispatchable in a serial wave.
No native execution witness has been run for this follow-up.

## Integrated software verification

The #12582 candidate on parent a308b3e11b5c02d8f2b32cef5ab25f47c360cf1a
passes focused race tests for actual native compaction, activation headers and
trailers, and opaque real-vDSO result identity. The independent native planner
fixtures cover active/idle cuts, prompt-only encoding, cancellation, failed
generation, request isolation, and early response commit followed by late native
use. Initial Used is a precommit snapshot; completed late use appears in the
declared Used-Final trailer. See docs/fak/feature-activation.md.

The full gateway run exposed a heartbeat ordering race: start became observable
before the first content counters. The bounded fix in stream_proxy.go commits
the opening role first, then publishes heartbeat start and first-event counters
atomically. It preserves the Anthropic path and existing stream-cost measurement.
The unchanged end-to-end heartbeat regression and activation tests pass together
under the race detector (8.754 seconds, exit 0). The full gateway rerun passed
the heartbeat regression but failed the syscall latency gate: 97.30% of folds
met the 100-microsecond bound, below the required 99%. Strict validation remains
pending a matched parent/candidate check; the latency failure is retained.

Full agent and vDSO packages passed before that gateway-only fix (106.787 and
0.317 seconds). The original gateway failure remains in the captured campaign
receipts. These software receipts do not satisfy this ticket's physical active/idle
Qwen3.8 requirement; keep #12835 open for that hardware witness.

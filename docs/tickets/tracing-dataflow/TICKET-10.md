<!-- fak-turntrace-key: route-spec-tool-reinsert-v1 -->

```routing
lane: agent
paths: ["internal/agent/**"]
expected_steps: 8
```

# feat(turntrace): join routing, speculation, tool execution, and result reinsertion in one turn trace

## Problem

For an agent or operator asking “why did this tool result enter the next model turn?”, the owned loop currently exposes several partial views: progress events, per-call verdict rows, speculation counters, audit events, and the transcript. None forms one causal chain from route choice through speculative state, adjudication, dispatch or local service, result admission, and exact reinsertion into model context. Today a successful-looking turn can hide a route fallback, squashed speculative read, quarantined result, or reordered concurrent call. Better because one bounded turn trace makes each transition and its evidence explicit, using stable call/result IDs and digest-safe payload references. The witness runs deterministic committed and squashed speculation cases and reconstructs exactly which result the next planner invocation received.

## Parent context

First-class public `fak` dataflow tracing goal. This leaf joins the owned agent turn by adapting its domain stages to the canonical public envelope, lineage, and capture contracts before later correlation with global decision and provider-accounting traces.

## Why now

#6053, #7308, #8107, and #9552 leave the local turn seam unresolved: no ordered artifact shows how a route and speculation decision became a governed tool result reinserted into model-visible context. The shared public trace ABI must land first so this leaf contributes domain facts rather than another trace dialect.

## Current state

Inspected at public `fak` commit `325ade2db2a7b1ea556d03d89293162a85cd1e9e`:

- `internal/agent/loop_observe.go:17-57` exposes only five progress kinds (`turn_started`, `tool_started`, `call_adjudicated`, `result_admitted`, `turn_done`) and lacks route, speculation, dispatch outcome, payload digest, or reinsertion identity.
- `internal/agent/loop.go:141-183` stores a human/per-call trace with bounded argument preview but no call ID, engine route, result digest, parent span, or transcript insertion coordinate.
- `internal/agent/loop.go:223-280` has the real routed syscall, adjudication, digest, vDSO, repair, and quarantine seams.
- `internal/agent/loop_turn.go:503-643` resolves suspended/streaming speculation, route failures, write barriers, and real execution; `:646-850` later commits audit/progress rows and appends the tool message at `:822`, but those facts are not joined as one replayable turn lineage.

## Dependencies and schedule

- Hard prerequisite: `typed-causal-envelope-v1` for trace/span/parent identity and ordering.
- Hard prerequisite: `many-to-many-value-lineage-v1` for call/result/reinsertion derivation and explicit non-edges.
- Hard prerequisite: `bounded-private-evidence-policy-v1` for digest/preview treatment, limits, losses, and receipts.
- Hard prerequisite: TICKET-04 `causal-trace-query-explain-export-v1`; the turn trace must be queryable and canonically exportable by the agent-facing surface.
- Schedule after TICKET-01 through TICKET-04. It may then run in parallel with TICKET-05, TICKET-07, and the model-tree chain because it touches only `internal/agent/**`.

## Centrality and P1-P4

Centrality: Core

P1: advanced - identifies exact result reinsertion and quarantine decisions without embedding unbounded content.

P2: advanced - exposes whether speculation, vDSO, or routing avoided real work and keeps counters separate from causal facts.

P3: advanced - typed route, speculation, adjudication, and admission reasons let agents adapt without prose inference; unknown stages remain explicit.

P4: advanced - one ordered artifact supports failure diagnosis, replay checks, and later global-decision correlation.

## Scope

Join the existing route, speculation, syscall, adjudication, and transcript-reinsertion seams inside `internal/agent` only, using adapters to public `pkg/traceabi` without modifying or shadowing its contracts.

## Core through-line

1. Extend the owned-loop observer with an adapter that populates `traceabi.CausalEnvelope`; keep turn, stable call ID, route, phase, status/reason, and digest-safe argument/result references as domain correlations/payload fields.
2. Emit typed stages for route selected/refused, speculation proposed/dispatched/committed/squashed, write barred, adjudicated/transformed/denied, engine dispatched or vDSO served, result produced, admitted/quarantined, and transcript reinserted.
3. Bind `result_produced` to the exact `RoleTool` message insertion and subsequent planner consumption using canonical value-lineage transforms; represent held-out quarantine with canonical omission/loss evidence and an explicit non-edge.
4. Preserve deterministic commit order for concurrently executed effect-safe calls while also recording execution overlap; sequence reflects model-visible commit order, not goroutine completion accidents.
5. Keep existing progress SSE and audit journal as projections/consumers of canonical records, and use the shared capture policy/receipt rather than introducing contradictory facts or accounting.

## Gold-plating boundary

No global planner replay, OTLP exporter, external-provider prompt/body capture, UI, cross-process span collector, tool implementation tracing, or broad audit-journal rewrite. This leaf owns one in-process agent turn. Global cache/placement/serve decisions remain #6053; runtime stall recordings remain #10184.

## Done condition

- [ ] One agent adapter maps the closed phase vocabulary for route → speculation → adjudication → dispatch/service → result → admission → reinsertion into canonical trace-ABI envelopes and lineage; no alternate event envelope exists.
- [ ] Every model-visible tool result is linked by call ID and digest from production to exact transcript insertion and next-turn consumption.
- [ ] Quarantined/denied/squashed results have explicit terminal or non-reinsertion outcomes; no missing result is presented as success.
- [ ] Concurrent safe calls retain deterministic model-visible order while recording overlap independently.
- [ ] Existing progress/audit/call-trace surfaces derive from or are cross-checked against the same facts.
- [ ] Raw arguments/results remain out of the default trace; previews are bounded/redacted and payload references are digests.
- [ ] Disabled tracing is inert; bounds, dropped events, and observer overhead are governed and reported by the canonical capture policy/receipt.

## Witness

The deterministic command and expected invariants are specified in `## Verifiable Witness` below.

## Verifiable Witness

```text
go test ./internal/agent -run 'TestTurnTrace(RouteSpeculationReinsertion|SquashAndQuarantine)' -count=1
```

One deterministic fixture routes an effect-safe tool, commits a speculative result, admits it, and proves the next planner call receives the digest-linked `RoleTool` message. A second mismatches the speculation, exercises squash/write-bar or quarantine, and proves no forbidden result gains a reinsertion edge. A concurrent-safe-call case asserts transcript order is stable across repeated runs.

## Prior-art source, date, license, and disposition

- Observed `2026-09-08T16:09:42Z`; source event `2026-05-05T16:10:01Z`; source state: **Development**, not stable. OpenTelemetry GenAI agent span conventions at `docs/gen-ai/gen-ai-agent-spans.md@c9e48b1d1af565454b73b9e14f0841bca4670ada` ([commit](https://github.com/open-telemetry/semantic-conventions/commit/c9e48b1d1af565454b73b9e14f0841bca4670ada)).
- License: [Apache-2.0](https://github.com/open-telemetry/semantic-conventions/blob/main/LICENSE). Disposition: **INSPIRE-ONLY / WATCH** for a future exporter because the convention is development-stage; use trace/span parentage concepts but keep a versioned FAK-owned event contract. Platform context: OpenTelemetry semantic-conventions main, GenAI agent spans. Refresh trigger: conventions reach Stable or change tool/agent span relationships.
- Transferable principle: agent actions need explicit causal parentage across model and tool spans. FAK opportunity: add the missing route/speculation/admission/reinsertion truth before exporting. Disconfirming check: if the current owned-loop artifact already reconstructs exact next-turn result consumption and all speculative outcomes, close this issue.

## Dedupe

Searched public open/closed turn, tool, route, speculation, trace-context, and decision-trace issues. #6053 is the broader cache/placement/harness/serve global decision explain/replay program; this issue is its bounded owned-turn input and does not implement global replay. Closed #8107 provides W3C ingress correlation IDs to reuse. #8586 owns broader context lifecycle/harness integration. #9552 joins provider token/cache counters, not tool-result causality. #10184 captures Go scheduler stalls. #7308 traces provider cache hints. None joins the exact in-loop transitions here.

## Definition of done

The resolving commit emits and verifies one complete committed chain plus one explicit non-reinsertion chain from live loop seams. Adding fields only to `CallTrace`, progress SSE, or the audit journal without causal linkage is incomplete.

## Working spine

Planner tool call → route resolution → speculation resolution → kernel adjudication/dispatch or local service → result digest → admission/quarantine → `RoleTool` transcript insertion → next planner consumption.

## Acceptance gate

The focused witness and `go vet ./internal/agent` pass; `fak validate --mine` is green; repeated concurrent fixture runs produce identical model-visible sequence; scrub assertions find no unbounded arguments/results.

## Likely files

- `internal/agent/loop_observe.go`
- `internal/agent/loop.go`
- `internal/agent/loop_turn.go`
- Focused `_test.go` files in `internal/agent`
- `docs/tickets/tracing-dataflow/TICKET-10.md`

## Lane

`agent`; one Go package: `internal/agent`. Dispatch after TICKET-01 through TICKET-04; then it is tree-disjoint from the remaining portfolio and may run in parallel.

## Closure binding

Close only from the signed resolving commit with both passing causal-chain fixtures and a captured bounded trace. OTLP compatibility is not a closure requirement while the external convention remains development-stage.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12267

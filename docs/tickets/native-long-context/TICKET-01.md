<!-- fak-native-context-key: model-window-admission-v1 -->

# feat(native): enforce and advertise the loaded model context window

## Current state

The native planner tokenizes the fully rendered request in
`internal/agent/inkernel_planner.go:1294-1415` and already performs a device-memory fit
check through `refuseOversizeRequest`, but it does not reject a request whose prompt plus
requested generation exceeds the loaded model's declared
`Config.MaxPositionEmbeddings`. CPU, Metal, Vulkan, and CUDA paths can therefore begin
session/device allocation for a request the model cannot represent.

At the catalog boundary, `internal/gateway/http_management.go:50-136` publishes plain
OpenAI model rows without a context limit and hardcodes `context_window` and
`max_context_window` to `272000` for every Codex model row, including a native model
whose loaded configuration declares a different window.

Related prior work is closed: #519 established the ultra-long-context program, #9079
exposed a bounded native admission token budget, and #2947 selected measured effective
resident budgets. None makes the loaded model's declared position limit an exact
request-admission and catalog contract.

## Classification

- Portfolio tier: 1 — native all-in-one serving and harness correctness.
- Centrality: Core.
- Work unit: S1 leaf.
- Priority: P1.

## Problem frame

- **Centrality:** Core serving correctness. The native engine must refuse an impossible
  request before token execution and tell clients the same capacity it enforces.
- P1: advanced - a model-window overflow currently reaches allocation/execution paths and the
  catalog can advertise a fabricated larger or smaller limit.
- P2: advanced - device-memory admission and downstream model failures are too late and can vary
  by backend; a fixed catalog constant is not model truthful.
- P3: preserved - reuse the loaded `model.Config.MaxPositionEmbeddings`, the existing post-tokenize
  planner seam, typed planner errors, and gateway error mapping.
- P4: preserved - a deterministic CPU fixture is sufficient because the rule is backend independent;
  no performance or physical-hardware claim is made.

## Parent context

Follow-on to the closed ultra-long-context program #519. It narrows the request boundary
left open by the closed native admission work #9079 and resident-budget work #2947.

## Why this is next

The loaded model declaration already exists and the rendered token count is already known
at the planner seam, so this is the smallest end-to-end slice that prevents impossible
native requests while making client discovery truthful.

## Core through-line

Loaded model declaration plus optional explicit planner bound -> effective native context
window -> rendered-request token count plus requested maximum generation -> typed refusal
before session/device allocation -> HTTP 400 `context_length_exceeded` and truthful known
capacity in both OpenAI and Codex model catalogs.

## Working spine

Model configuration -> planner construction -> post-tokenization admission -> typed error ->
gateway status/code mapping and model catalog metadata.

## Scope

1. Add an optional `ContextTokens` field to `InKernelPlannerConfig`. Zero uses the loaded
   model declaration. A positive explicit value may only tighten the declared model window;
   it must never widen it.
2. After rendering and tokenization in `InKernelPlanner.Complete`, reject when
   `prompt_tokens + requested_max_new_tokens` exceeds the effective positive window. Return
   a typed, safe error containing the measured prompt, requested output, and limit. Run this
   check for every native backend before model session creation, device memory planning, or
   token execution.
3. Expose the effective known context window from the native planner without exposing the
   loaded model object.
4. Map the typed planner error to HTTP 400 with OpenAI-style code
   `context_length_exceeded`.
5. For native model rows only, publish the known capacity as standard
   `data[].context_length` and as Codex `models[].context_window` plus
   `models[].max_context_window`. Omit capacity fields when the planner does not know the
   model window. Remove the unconditional `272000` claim.

## Gold-plating boundary

Do not add CLI flag derivation or auto-sizing in this leaf; that is the next independent
slice. Do not change KV/device memory heuristics, compaction, truncation, RoPE scaling,
backend kernels, proxy-provider limits, model formats, or default generation depth. Do not
claim longer context, throughput, latency, or hardware qualification.

## Done condition

- [x] A native request exactly at the effective model window is admitted.
- [x] A native request one token over returns the typed error before any model/device
      execution across CPU and device-backed planner shapes.
- [x] A positive explicit planner bound tightens the model declaration; zero uses the model
      declaration; no explicit value widens it.
- [x] The gateway maps the typed error to status 400 and code
      `context_length_exceeded` without leaking internal details.
- [x] Native `/v1/models` rows publish the effective known capacity in OpenAI and Codex
      shapes; unknown/proxy rows do not receive a fabricated number.
- [x] Focused agent and gateway tests plus scoped validation pass.

## Definition of Done

- [x] The effective context-window rule is identical across native CPU, Metal, Vulkan,
      and CUDA planner shapes.
- [x] Refusal occurs before model session creation, device allocation, or token execution.
- [x] The advertised known native capacity equals the enforced capacity.
- [x] Unknown and proxy capacities remain omitted rather than fabricated.

## Acceptance gate

The exact-limit fixture is admitted, the one-token-over fixture is refused before execution,
the gateway returns HTTP 400 with `context_length_exceeded`, catalog fixtures contain only
truthful known limits, and the focused plus scoped validation commands exit zero.

## Closure binding

The DCO-signed resolving commit closes the created public issue and carries `(fak agent)`.

## Witness

```bash
go test ./internal/agent ./internal/gateway -run '^(TestInKernelContext|TestModelsAdvertise.*Context|TestNativeContextBoundaryAndHTTPErrorLoopback|TestHTTPModelsAndHealth)' -count=1
fak validate --mine internal/agent/inkernel_planner.go --mine internal/agent/inkernel_planner_config.go --mine internal/agent/inkernel_context.go --mine internal/agent/inkernel_context_test.go --mine internal/gateway/http.go --mine internal/gateway/http_management.go --mine internal/gateway/native_context_test.go --mine internal/gateway/gateway_test.go --mine docs/tickets/native-long-context/TICKET-01.md
```

The regression must use an execution spy or equivalent deterministic witness proving the
over-limit path returns before session/device allocation or token execution.

## Verification result

- The focused agent and gateway command above passes, including the real native HTTP
  loopback and exact-boundary execution witness.
- Scoped source boundary validation passes.
- The full `internal/agent` suite reaches an unrelated baseline failure in
  `TestAgentBufferPoolConcurrentContention`: at unchanged base
  `7ac75ceafb8ede8b3a1d432eebcfc87bfad98da2`, an independent run with
  `GOWORK=off` and a fresh external cache reports 767 allocations against the
  existing limit of 80. Follow-up #12669 tracks that independent defect; this
  ticket does not modify the buffer-pool benchmark.
- The full `internal/gateway` suite reaches an unrelated fixture failure in
  `TestCalibrationDriftCorpusResolvesTheTrackedMirror/no_module_root_resolves_nothing`,
  also reproduced on unchanged trunk: isolated repository-local temporary files
  make the test directory a descendant of the module, contradicting the fixture's
  no-module assumption. The focused native-context gateway contracts pass.

## Likely files

- `internal/agent/inkernel_planner.go:1294-1415` (`InKernelPlanner.Complete`)
- `internal/agent/inkernel_planner_config.go` (`InKernelPlannerConfig`, effective capacity readback)
- `internal/agent/inkernel_context.go`
- `internal/agent/inkernel_context_test.go`
- `internal/gateway/http.go:1019-1205` (`upstreamErrorStatus` / planner mapping)
- `internal/gateway/http_management.go:50-136` (`handleModels`)
- `internal/gateway/native_context_test.go`
- `internal/gateway/gateway_test.go`
- `tools/concept_disambiguation_scorecard.data/rows-context-ctx.json` (position the new `contextlength` name under the existing context-window concept)
- `docs/tickets/native-long-context/TICKET-01.md`

## Blast radius and affected lanes

- Affected: native request admission and `/v1/models` metadata in `internal/agent` and
  `internal/gateway`.
- Unaffected: proxy execution, provider limits, compute kernels, model loading, KV paging,
  and serving performance.
- Fallback: unknown model context remains unadvertised and preserves existing admission;
  this leaf does not invent a fallback limit.

## Lane

native-context

## Expected steps

6

```routing
lane: native-context
paths: internal/agent/inkernel_planner.go, internal/agent/inkernel_planner_config.go, internal/agent/inkernel_context.go, internal/agent/inkernel_context_test.go, internal/gateway/http.go, internal/gateway/http_management.go, internal/gateway/native_context_test.go, internal/gateway/gateway_test.go, tools/concept_disambiguation_scorecard.data/rows-context-ctx.json, docs/tickets/native-long-context/TICKET-01.md
expected_steps: 6
```

## Work estimate

Estimate: 3 points.

## Overall completion contribution

Contribution: 3/3 points.

## Completion standard

development

# TICKET-12753 - Request-attributed serving usage

<!-- github-issue: 12753 -->

<!-- fak-webbench-key: serving-project-arm-request-usage -->

```routing
lane: webbench
paths:
  - cmd/fak/webbench.go
  - internal/webbench/serving.go
  - internal/webbench/serving_test.go
expected_steps: 6
priority: P1
class: dev
dependencies: []
```

## Current state

`fak webbench serving` can issue concurrent OpenAI-compatible chat requests and
measure stream timing, but it cannot attach an explicit project-arm identity.
Its samples retain only output-token usage, while prompt, total, and cache-read
counts are lost. Streaming requests also do not request the provider's supported
final usage event. A cumulative metrics scrape cannot truthfully recover which
individual request observed cache reuse.

## Why now

Issue #10078 requires a real native serving sweep, but its current client cannot
bind requests to an explicit research arm or retain request-scoped cache usage.
This bounded instrumentation leaf makes that run attributable without changing
the service or claiming capacity.

## Parent ref

#10078

## Problem frame

- Centrality: Enabling (truthful public native serving evidence)
- P1: preserved - the benchmark sends the same workload and does not change managed context.
- P2: advanced - request-scoped token and cache usage replace attribution from a cumulative metric.
- P3: preserved - the option is explicit and does not alter scheduler or service defaults.
- P4: advanced - each request becomes attributable to its project arm and retains nullable usage evidence.

## Working spine

Explicit CLI project-arm option -> serving track config -> every inference
request carries `X-Fak-Project-Arm` -> existing streaming wire requests usage ->
each sample retains provider-observed prompt, completion, total, and cached token
fields with absent values serialized as null -> existing completion aggregation
continues unchanged.

## Core through-line

Make the existing serving benchmark emit request-attributable, provider-observed
usage evidence needed by the parent native capacity sweep.

## Scope

Add one explicit CLI/config project-arm field, attach its header to each inference
request, request final streaming usage through the supported OpenAI wire, and
retain nullable per-field usage in each sample. Preserve the existing completion
token aggregate and its fallback estimation.

## Gold-plating boundary

Do not add a new benchmark runner, service behavior, private endpoint defaults,
model defaults, cumulative-metric attribution, automatic capacity discovery, or
performance claims. Keep code changes within the existing webbench CLI and
serving measurement package.

## Concrete repro witness

Run the existing measurement path against a loopback server that records request
headers/body and returns a final SSE usage event. The current request lacks both
`X-Fak-Project-Arm` and `stream_options.include_usage`, and the resulting sample
drops prompt, total, and cached token counts.

## Exact file:line seams

- `cmd/fak/webbench.go` (`cmdWebbenchServing`, serving flag/config wiring)
- `internal/webbench/serving.go` (`ServingTrackConfig`, `ServingSample`, `MeasureSSERequest`)
- `internal/webbench/serving.go` (`streamChunk`, `completionContentAndUsage`)

## Blast radius and affected lanes

Primary lane: `webbench`. Affected behavior is public serving measurement only.
Serving, scheduling, model execution, metrics export, and hardware paths are unchanged.

## Quarantined fallback mechanism

Until fixed, omit per-request cache conclusions and project-arm attribution from
webbench receipts. Existing timing and completion-token results remain usable.

## Scoped acceptance criteria

- [ ] `fak webbench serving --project-arm NAME` sends `X-Fak-Project-Arm: NAME`
  on every inference request.
- [ ] Streaming requests use the existing `stream_options.include_usage` wire.
- [ ] SSE and buffered responses retain observed prompt, completion, total, and
  cached-token values independently; missing fields remain null.
- [ ] Existing exact-completion token counting and aggregate statistics retain
  their current behavior.
- [ ] Focused loopback tests cover the header, SSE usage, buffered usage, and
  nullable missing fields.
- [ ] The existing `internal/webbench` and `cmd/fak` checks pass.

## Definition of done

The explicit project-arm and nullable usage contract is implemented through the
existing CLI/request/sample path, with independent loopback regression coverage.

## Done condition

The scoped implementation and independent tests are green without a service,
hardware run, or inferred request usage.

## Witness

Run:

`go test ./internal/webbench ./cmd/fak -run 'Webbench|Serving' -count=1`

This is a software witness. It does not exercise or qualify physical hardware.

## Likely files

- `cmd/fak/webbench.go`
- `internal/webbench/serving.go`
- `internal/webbench/serving_test.go`

## Lane

`webbench`

## Acceptance gate

The focused loopback witness and existing `internal/webbench` and `cmd/fak`
tests pass with unchanged aggregate completion semantics.

## Closure binding

The resolving signed-off public commit cites this issue and #10078, carries the
public webbench context, and lands through the managed flow after independent tests.

## Expected steps

6

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 2/10 points for truthful native serving capacity evidence.

## Target operating envelope

- public repositories: = 1 repository
- hardware operations: = 0 operations

## Witnessed operating envelope

- public repositories: = 1 repository
- hardware operations: = 0 operations


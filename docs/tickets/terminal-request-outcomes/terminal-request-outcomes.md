<!-- fak-gateway-key: terminal-request-outcomes -->

# feat(runtime): retain terminal request outcomes by accepted trace ID

GitHub issue: [#13719](https://github.com/anthony-chaudhary/fak/issues/13719).

Process cause: verification-gap

```routing
lane: gateway
paths: ["internal/gateway/", "cmd/fak/"]
expected_steps: 6
repo: fak
priority: P1
class: mainline
dependencies: ["https://github.com/anthony-chaudhary/fak/issues/12258"]
```

## Current state

`internal/gateway/metrics_inflight.go:57-79` allocates a process-local numeric
ID, keeps only the live request, and deletes that row when request handling
ends. The live projection therefore cannot answer a later query by the accepted
`X-Trace-Id`, and it cannot distinguish a server-observed success, failure,
cancellation, or an upstream completion whose outcome is unknown.

Reproduction: send a request with a unique `X-Trace-Id` to a loopback gateway
whose handler finishes after the HTTP client's deadline. Once the live row is
retired, query the public observation surface for that exact trace ID. No
terminal record can be resolved, so the client cannot prove whether server-side
execution completed before deciding whether a retry is safe.

Issue #12258 owns the shared causal identity envelope. This leaf consumes that
identity contract when available; it does not redefine trace or span semantics.

## Parent context

Public issue #12258 is the related causal-identity dependency. It deliberately
does not own a storage engine or terminal request projection. Closed issue #6568
is unrelated prior work. No existing issue owns this bounded outcome ledger.

## Why now

Client deadlines are already observable, but the server discards the only live
request row at completion. The missing retained result prevents a safe,
trace-keyed retry decision; the smallest next leaf is the terminal projection,
without expanding into a distributed tracing system.

## Working spine

Accepted trace ID -> live request lifecycle -> closed terminal classification ->
bounded durable record -> exact-ID HTTP and CLI read-back.

## Core through-line

Accepted request trace ID -> bounded server-side terminal projection -> durable
read-back of outcome and serving identity -> a timed-out client can distinguish
known termination from unknown completion without inspecting raw logs.

## Scope

- Preserve the accepted trace ID and any server-assigned request ID through the
  gateway request lifecycle.
- Write one payload-free terminal record with a closed outcome vocabulary:
  `succeeded`, `failed`, `cancelled`, or `completion_unknown`.
- Record start/end time, terminal error class, and authoritative serving
  backend, model, engine/binary revision when observed. Missing identity stays
  null with an explicit incomplete classification; caller labels never become
  server identity.
- Retain records across process restart in a size- and age-bounded store. Bound
  write time and storage, publish eviction/drop accounting, and keep the serving
  hot path non-blocking.
- Add a bounded authenticated HTTP reader by exact accepted trace ID and a
  public `fak` CLI reader. Neither surface returns prompts, generated content,
  credentials, raw headers, or unbounded lists.

## Gold-plating boundary

Do not add a tracing UI, distributed collector, private Ops consumer, benchmark
policy, performance claim, cancellation protocol, or replay engine. Do not make
an observed client disconnect prove upstream cancellation. If the server lacks
authoritative completion evidence, persist `completion_unknown`.

## Problem frame

Centrality: Core

- For: local runtime clients recovering after a request deadline or disconnect.
- Problem: the live registry deletes the row before later read-back can establish
  the server-observed terminal result.
- Today: absence from the live table is ambiguous between completion, failure,
  cancellation, and unknown upstream completion.
- Better because: a bounded exact-trace record makes the known result or the
  remaining uncertainty explicit.
- P1: advanced; one exact trace lookup replaces guesses from expired sessions,
  empty live tables, or client-side timeouts.
- P2: preserved; bounded retention and non-blocking writes protect the serving
  path; no throughput or latency gain is claimed.
- P3: advanced; closed outcomes and explicit identity completeness prevent a
  caller label from becoming execution evidence.
- P4: advanced; retained terminal evidence makes retry and cleanup decisions
  inspectable after the request leaves the live registry.

## Scope class

S1 bounded leaf: one public runtime record, retention policy, and reader path.

## Work estimate

Estimate: 5 points (medium).

## Overall completion contribution

5/5 points for the terminal-outcome projection leaf.

## Completion standard

Production.

## Target operating envelope

- terminal outcome fixture coverage: >= 4 outcomes
- records returned by one exact-trace lookup: <= 1 record
- payload or credential fields returned: = 0 fields
- retained record cap: <= 4096 records
- maximum default retention age: <= 168 hours
- blocking terminal-ledger writes on serving goroutines: = 0 writes

## Envelope evidence status

The closure witness must report the values below; they are acceptance targets,
not current runtime evidence. Current evidence is limited to source inspection.

## Witnessed operating envelope

- terminal outcome fixture coverage: = 4 outcomes
- records returned by one exact-trace lookup: = 1 record
- payload or credential fields returned: = 0 fields
- retained record cap: = 4096 records
- maximum default retention age: = 168 hours
- blocking terminal-ledger writes on serving goroutines: = 0 writes

## Definition of done

- [ ] One accepted trace ID resolves to exactly one terminal record after its
  live row is retired; duplicate finalization is idempotent.
- [ ] Success, failure, confirmed cancellation, and completion-unknown fixtures
  each persist the correct closed outcome and terminal timestamp.
- [ ] Every record reports authoritative backend/model/binary identity or
  explicit unknown/incomplete identity without promoting caller metadata.
- [ ] Restart and retention-pressure fixtures preserve in-window records,
  evict only by the declared bound, and expose eviction/drop accounting.
- [ ] Exact-ID HTTP and CLI readers are bounded, authenticated where required,
  payload-free, and return a typed not-found result outside retention.
- [ ] Existing live request observation remains reachable and its serving-path
  latency contract stays green under the repository's gateway verification.

## Witness

Run focused gateway and CLI contracts that submit loopback success, failure,
cancellation, and client-timeout requests with unique accepted trace IDs; restart
the fixture; read each retained outcome; then force bounded eviction and verify
typed not-found plus accounting. Run the affected public gateway and CLI package
tests and the public boundary/scrub gates.

## Done condition / witness

The six checklist items above are complete only when the focused loopback
fixture and affected package gates pass after the final edit.

## Acceptance gate

The focused HTTP fixture proves that a client deadline no longer leaves only a
missing live row: the exact accepted trace ID reads back a server-authored
terminal record, including `completion_unknown` when termination is unproven.

## Likely files

- `internal/gateway/metrics_inflight.go`
- `internal/gateway/metrics_http.go`
- `internal/gateway/request_outcomes.go` (new)
- `cmd/fak/request_outcome.go` (new)
- focused external and package tests beside those production files

## Lane

gateway

## Closure binding

Close only with a DCO-signed `(fak gateway)` commit, retained success/failure/
cancellation/completion-unknown fixture receipts, bounded retention/read-back
evidence, and green affected public package and boundary gates. Source presence
alone does not prove runtime durability.

github_issue: anthony-chaudhary/fak#13719

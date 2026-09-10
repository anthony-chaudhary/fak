# fix(agent): return a completed coalesced cohort before draining later arrivals

## Parent context

Coordinates with parent #8395 (https://github.com/anthony-chaudhary/fak/issues/8395).

<!-- fak-agent-key: native-batch-response-cohort-handoff -->
<!-- fak-cross-key: native-batch-response-cohort-handoff -->
<!-- fak-public-issue: anthony-chaudhary/fak#12723 -->

```routing
lane: agent/cohort-response
paths: ["internal/agent/inkernel_batch_coordinator.go", "internal/agent/inkernel_batch_response_test.go", "docs/tickets/native-batch-response/TICKET-01.md"]
expected_steps: 4
public_issue: anthony-chaudhary/fak#12723
cross_key: native-batch-response-cohort-handoff
```

## Current state

At public commit `31802079e`, `InKernelPlanner.runCoalescedGenerate` elects one request as leader and calls `drainCoalescedGenerates` synchronously before reading that request's completed result. The drain loop processes the elected cohort, then repeatedly consumes every later cohort already queued before it returns. Under sustained arrivals, the first cohort can finish while its caller remains blocked behind unrelated later cohorts.

The coordinator already publishes each request result through its buffered result channel. The blocking is therefore coordinator ownership and handoff ordering, not model execution correctness.

## For

Local agents using the public native in-kernel planner with coalesced Qwen decode enabled.

## Problem

A completed request can remain blocked while the elected leader drains later cohorts, so sustained arrivals can starve delivery of an already-computed response.

## Today

One elected request synchronously drains the queue until it observes no pending cohort, then reads its own buffered result.

## Better

Each request returns after its own bounded cohort completes, while a request-owned baton preserves exactly one synchronous drain owner for later cohorts.

## Why this is next

The response has already completed and is buffered, so this ordering defect adds avoidable head-of-line blocking on the native agent path. The fix is a bounded coordinator leaf with a deterministic software witness.

## Working spine

Request enters `Complete` -> leader drains one bounded cohort -> cohort publishes buffered results and receipts -> completed caller returns -> first queued request receives the drain baton -> next bounded cohort runs under the same single-owner invariant.

## Core through-line

Leader election -> drain exactly the elected cohort -> completed request reads its own result -> pass a request-owned drain baton to the first later queued request -> later cohorts continue without allowing two drain owners.

Keep `coalesceRunning` as the single-owner invariant. After one cohort completes, return its callers promptly and signal the first queued request to drain the next bounded cohort synchronously in its own `Complete` call stack. Requests arriving during the ownership transition must either join the next cohort or receive the baton without being stranded.

## Gold-plating boundary

Do not redesign batching policy, cohort sizing, fairness, model scheduling, or device execution. Do not add a server-side tool executor, exported API, configuration flag, hardware benchmark, or latency ratio. This leaf fixes response ordering in the existing in-process coordinator.

## Verifiable Witness

Run the deterministic test through the real native `InKernelPlanner.Complete` entry point with the simulated shared-panel probe:

```bash
go test ./internal/agent -run TestInKernelCoalescedLeaderReturnsBeforeLaterCohort -count=1
```

The test holds a later cohort behind an explicit synchronization gate after the elected cohort has completed. Before the fix, the elected caller remains blocked until that later gate opens. After the fix, the elected caller returns while the later cohort remains gated, and the later cohort subsequently completes. This is software execution-ordering evidence for engine dispatch. The simulated shared-panel probe is not physical hardware evidence. Any elapsed-time budget is only an ordering guard for the deterministic synchronization witness; it is not a performance benchmark.

Pre-fix witness at `31802079e`: the command exits 1 with `TestInKernelCoalescedLeaderReturnsBeforeLaterCohort` reporting `elected caller waited for a later cohort after its own cohort completed`. The test's later-cohort gate accounts for the wait; the elapsed duration is not a benchmark result.

## Done condition / witness

Witness: `go test ./internal/agent -run TestInKernelCoalescedLeaderReturnsBeforeLaterCohort -count=1`.

- [x] The elected request returns after its own cohort completes without waiting for a later cohort.
- [x] Later queued cohorts continue under a single drain owner, including requests arriving during baton handoff, without stranding or duplicate execution.
- [x] `TestInKernelCoalescedLeaderReturnsBeforeLaterCohort` fails on the parent behavior and passes with the fix.
- [x] Focused `internal/agent` tests pass.

## Definition of done

Completion requires every checkbox in Done condition and a green focused witness. The resolving change preserves single drain ownership and cites the created public issue.

## Acceptance gate

The regression witness fails against the parent coordinator behavior, passes with the request-owned baton, and the focused `internal/agent` package tests remain green.

## Closure binding

The resolving commit cites the created public issue and carries the `agent` leaf in its commit subject.

## Witness

The focused test above is the completion witness. Record its pre-fix failure and fixed-tree pass in the issue or resolving commit. No physical hardware qualification is required or claimed.

Candidate verification at the issue worktree passed the focused coordinator tests, including the real native planner path with the simulated shared-panel probe. The complete package command `go test ./internal/agent -count=1 -timeout 10m` passed in 137.481 seconds. An independent WSL Go 1.26.6 race run of the focused witness also passed in 1.473 seconds. These durations record test completion only and are not performance results. No physical hardware qualification is claimed.

## Likely files

- `internal/agent/inkernel_batch_coordinator.go:91-122` (`runCoalescedGenerate`, `drainCoalescedGenerates`) — response ordering and drain-owner handoff.
- `internal/agent/inkernel_batch_response_test.go` — deterministic real-entry-point regression witness.
- `docs/tickets/native-batch-response/TICKET-01.md` — routing and acceptance contract.

## Lane

agent/cohort-response

## Blast radius and affected lanes

One package: `internal/agent`. The change is confined to in-kernel coalesced decode coordination. Model backends, wire protocols, serving routes, and non-coalesced requests remain outside the leaf.

## Quarantined fallback mechanism

The existing non-coalesced path remains unchanged. If the handoff cannot preserve single ownership, retain the current synchronous drain and keep this issue open rather than weakening the ownership invariant.

- Centrality: Core
- P1 Context: advanced — a completed native response is no longer held behind unrelated arrivals.
- P2 Net value: advanced — bounded response handoff reduces head-of-line blocking without changing token semantics.
- P3 Adaptation: preserved — existing cohort admission and sizing remain unchanged.
- P4 Operations: advanced — deterministic ordering evidence catches sustained-arrival starvation.

## Expected steps

4

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 2/8 points toward parent #8395.

## Completion standard

demo

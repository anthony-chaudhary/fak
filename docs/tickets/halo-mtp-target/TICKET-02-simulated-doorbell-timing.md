<!-- fak-compute-key: simulated-doorbell-timing-reliability-v1 -->
# Stabilize the simulated doorbell timing witness

GitHub tracker: https://github.com/anthony-chaudhary/fak/issues/12729

```routing
lane: compute
paths: ["internal/compute/doorbell_allreduce_test.go"]
expected_steps: 4
repo: fak
```

## Current state

On the MTP candidate based at public commit
`644bd060041a89caa7e1024b2cf1540b058cce16`, with the doorbell test unchanged,
the broad compute short run failed only
`TestDoorbellSub100usExchange`. The test timed 100 in-process exchanges of a
simulated 16 KiB vector at 145.009 microseconds on average, then failed its
fixed 100 microsecond wall-clock threshold at
`internal/compute/doorbell_allreduce_test.go:207-210`.

This is a host scheduling observation from an in-memory simulated engine. It is
not physical interconnect, GPU, USB4, Thunderbolt, or Halo performance evidence.
Exact-name and broader GitHub searches found no open issue for this reliability
gap. Closed #11107 tracks the original doorbell capability, while closed #11307
is a general historical test-flake cleanup.

Centrality: Stewardship. Portfolio tier: 2 (serving). Priority: P2.

- P1 Context: advanced - the failure is isolated from the Qwen MTP change that exposed it.
- P2 Net value: advanced - ordinary compute validation becomes repeatable.
- P3 Adaptation: preserved - functional exchange and parity assertions remain active.
- P4 Operations: advanced - simulated timing is labeled and measured separately.

## Core through-line

Simulated inputs -> in-process doorbell exchange -> exact functional result ->
deterministic correctness test. Any latency characterization belongs in an
explicit benchmark or opt-in diagnostic with point-in-time provenance and
cannot decide ordinary package correctness from one host scheduling sample.

## Gold-plating boundary

No doorbell engine optimization, transport redesign, new hardware backend,
physical performance qualification, production routing change, or MTP source
change. Do not turn this leaf into a claim about real wire latency.

## Done condition

- [ ] [SW-VERIFIED] The ordinary test retains deterministic exchange, shape,
      count, signal, and bitwise result checks without a fixed host wall-clock
      failure gate.
- [ ] [SW-VERIFIED] Timing characterization, if retained, is an explicit
      benchmark or opt-in diagnostic that reports its simulated regime and
      does not inject a historical default result.
- [ ] [SW-VERIFIED] Twenty consecutive focused executions pass without a
      scheduler-speed assumption.
- [ ] [SW-VERIFIED] `go test -short ./internal/compute` passes, or any other
      failure is reported independently with its exact test name.

## Verifiable Witness

```text
go test ./internal/compute -run '^TestDoorbellSub100usExchange$' -count=20
go test -short ./internal/compute
```

The first command is the reliability witness. The second is the affected
package gate. Neither command witnesses physical hardware performance.

## Witness

The named test must pass 20 consecutive process executions while preserving
bitwise exchange correctness. The short package gate must pass independently;
its result does not qualify physical transport latency.

### File:Line Seam

`internal/compute/doorbell_allreduce_test.go:173-212 (TestDoorbellSub100usExchange)`

### Blast Radius & Affected Lanes

The affected package is `internal/compute`. The Qwen MTP model lane and other
packages remain independently testable and shippable.

## Likely files

`internal/compute/doorbell_allreduce_test.go` only.

## Lane

Public compute test reliability. Four steps: reproduce from the recorded
failure; separate functional assertions from scheduling-sensitive timing;
repeat the focused witness; run the short compute package gate.

## Quarantined fallback mechanism

Until stabilized, treat this exact simulated timing failure as an isolated
known baseline and keep the MTP lane moving. Preserve the doorbell functional
checks in ordinary tests. Quarantine the fixed latency assertion behind an
explicit benchmark or opt-in diagnostic rather than skipping the functional
test or weakening data-parity checks.

## Scoped Acceptance Criteria

- [ ] The unchanged failure at 145.009 microseconds is recorded as a simulated,
      scheduling-sensitive observation at commit `644bd060041a89caa7e1024b2cf1540b058cce16`.
- [ ] Ordinary correctness no longer depends on completing 100 exchanges below
      a fixed wall-clock average.
- [ ] Functional exchange assertions remain active.
- [ ] The two witness commands above complete as scoped, with unrelated failures
      tracked separately.

## Dependencies and execution

Related historical capability: #11107. This ticket is independent of the MTP
target work in #12720 and must not block it.

## Parent context

Related historical capability #11107 is closed. This reliability leaf stands
alone and coordinates with #12720 only because its broad validation run exposed
the unchanged test failure.

## Why this is next

The fixed wall-clock assertion makes ordinary compute correctness depend on
host scheduling. Isolating that assertion restores a repeatable package gate
without delaying the independently green MTP capability.

## Acceptance gate

Both commands under `## Verifiable Witness` complete as scoped, functional
doorbell assertions remain active, and no physical latency claim is made.

## Closure binding

The resolving signed public commit references the GitHub issue and this ticket,
uses `(fak compute)`, and records the focused repeat witness. Any benchmark
receipt remains a moment-in-time simulated diagnostic.

## Working spine

Simulated vector inputs -> doorbell exchange -> bitwise result and control-state
checks -> repeatable package witness. Scheduling-sensitive latency reporting is
kept outside the ordinary correctness verdict.

## Done condition / witness

The named test passes 20 consecutive executions while retaining its functional
assertions, and `go test -short ./internal/compute` exits zero or identifies a
separately tracked failure. Neither result establishes physical transport or
device performance.

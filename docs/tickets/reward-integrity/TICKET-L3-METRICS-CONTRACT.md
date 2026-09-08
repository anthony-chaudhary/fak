<!-- fak-l3server-key: metrics-name-contract-fail-closed -->
# fix(l3server): make metric-name contract mismatches fail instead of log

GitHub issue: https://github.com/anthony-chaudhary/fak/issues/12380

## Current state

`internal/l3server/metrics/collector_contract_test.go:9-44` records fourteen known differences between metrics-service specification names and l3-server emitted names. `TestContract_DocumentMismatches` at lines 66-77 logs every mismatch and returns PASS. The companion test only proves the implementation's current names occur in source; it does not prove consumers and the server share a contract.

## Parent context

Parent: #3831. The leaf owns one concrete false-green contract test.

## Why now

Fourteen mismatches are already known and green. New L3 observability work can therefore build against either vocabulary without a deterministic compatibility signal.

## Problem frame

- Priority: P2 observability-contract defect.
- Centrality: Enabling (operable L3 serving observability).
- P1: advanced - operators and agents receive metric names the server actually emits.
- P2: advanced - a contract-named test no longer masks fourteen discrepancies.
- P3: preserved - align one collector contract without a metrics framework rewrite.
- P4: advanced - the test examines actual registered descriptors and fails on drift.

## Core through-line

Canonical metric specification -> collector descriptor/output -> contract test -> fail on missing, renamed, or extra required metric. Documentation-only mismatch rows do not count as verification.

## Working spine

The canonical L3 metric inventory flows through collector registration to the contract test and consumer-facing names.

## Blast radius and affected lanes

- Primary lane: `l3server`.
- Affected surface: l3-server metric names and their consumers.
- Preserve compatibility aliases if changing an already-exported metric name would break deployed dashboards.

## Quarantined fallback mechanism

If compatibility requires two names during migration, emit and test an explicit deprecated alias. Do not treat a logged mismatch as a passing fallback.

## Scoped acceptance criteria

- [ ] One canonical expected-name inventory is defined.
- [ ] All fourteen current mismatches are resolved by implementation alignment or explicit compatibility aliases.
- [ ] Missing or misspelled names fail the contract test.
- [ ] The test examines actual registered/emitted descriptors, not source-text substrings alone.

## Gold-plating boundary

No dashboard deployment, metrics backend change, new telemetry dimensions, or l3-server refactor outside the collector contract.

## Done condition

The metrics contract suite returns PASS only when every required service-facing name is actually exposed by l3-server.

## Witness

The deterministic test witness is the L3 metrics contract suite below.

### Non-Forgeable Witness

```text
go test ./internal/l3server/metrics -run '^TestContract_' -count=1 -v
```

## Acceptance gate

`go test ./internal/l3server/metrics -run '^TestContract_' -count=1 -v`

## Closure binding

The resolving commit cites this issue in its subject and carries a `(fak l3server)` trailer.

## Likely files

- `internal/l3server/metrics/collector.go`
- `internal/l3server/metrics/collector_contract_test.go`

## Lane

`l3server`

## Dependencies and dedupe

Exact searches for `collector_contract_test.go`, `TestContract_DocumentMismatches`, `metric name mismatches`, and representative old/new metric names returned no matching issue.

## Expected steps

4

```routing
lane: l3server
paths: internal/l3server/metrics/collector.go, internal/l3server/metrics/collector_contract_test.go
expected_steps: 4
```

<!-- fak-codetools-key: concurrent-buffer-pool-allocation-bound-v1 -->

# fix(codetools): restore the concurrent buffer-pool allocation bound

## Current state

At public `fak` commit `7ac75ceafb8ede8b3a1d432eebcfc87bfad98da2`, the
unchanged `TestAgentBufferPoolConcurrentContention` fixture reports 767 pool
allocations for 1,000 acquisitions, while its contract permits at most 80. The
failure reproduces with `GOWORK=off` and a fresh external Go cache, independently
of the native-context change that exposed it in the full agent suite.

`internal/codetools/buffer_pool.go:48-103` uses `sync.Pool` for all size classes.
The test in `internal/agent/codetools_bench_test.go:104-160` treats allocations as
durably bounded across a concurrent read/grep workload, although `sync.Pool` may
discard entries during garbage collection.

## Classification

- Portfolio tier: 1 — native agent harness reliability.
- Centrality: Core.
- Work unit: S1 leaf.
- Priority: P2.

## Problem frame

- **Centrality:** The package-level allocation witness is part of the required
  `internal/agent` validation gate.
- P1: preserved - this does not change context or request semantics.
- P2: advanced - restore a deterministic, measured allocation contract under
  the existing concurrent workload.
- P3: preserved - keep the current size classes and bounded memory behavior.
- P4: preserved - the witness is a software allocation test and makes no
  throughput, latency, or physical-hardware claim.

## Core through-line

Concurrent code-tool reads -> pooled scratch acquisition and release ->
garbage-collection-safe allocation accounting or storage -> deterministic bounded
allocation witness.

## Parent context

Follow-up blocker discovered while validating #12663. It is independent of the
native-context source and reproduces at that ticket's unchanged base commit.

## Why this is next

The failing allocation assertion is the only `internal/agent` full-suite failure
seen during #12663 validation, so isolating its contract restores a useful trunk
gate without coupling it to request-admission work.

## Working spine

Code-tool read path -> scratch size-class pool -> concurrent acquire/release ->
allocation metrics -> agent full-suite witness.

## Gold-plating boundary

Do not redesign code tools, add subprocess execution, change tool result formats,
or claim performance from this fixture. Do not weaken the threshold without a
replacement invariant that directly bounds retained scratch memory and explains
garbage-collection behavior.

## Done condition

- [ ] The exact isolated reproduction passes repeatedly with a fresh external
      cache and `GOWORK=off`.
- [ ] Acquires and releases remain balanced for all 1,000 pooled reads.
- [ ] The allocation or retained-memory bound is deterministic under garbage
      collection and concurrent contention.
- [ ] Focused `internal/codetools` and `internal/agent` tests pass.

## Definition of Done

- [ ] The pool retains a documented finite memory bound under contention.
- [ ] Garbage collection cannot make the focused allocation witness flaky.
- [ ] Existing tool payload and confinement behavior remain unchanged.

## Acceptance gate

The exact reproduction passes repeatedly, acquire/release accounting stays
balanced, and both affected packages pass their focused tests.

## Closure binding

The DCO-signed resolving commit closes the created public issue and carries
`(fak agent)`.

## Witness

Run `go test ./internal/agent -run '^TestAgentBufferPoolConcurrentContention$' -count=1 -v`.
Observed: `Allocations = 767, want <= 80` at unchanged base `7ac75ceafb8e`.

### Verifiable Witness

```powershell
$env:GOWORK='off'
$env:GOCACHE="$env:TEMP\fak-gocache-buffer-pool"
$env:GOTMPDIR="$env:TEMP\fak-gotmp-buffer-pool"
New-Item -ItemType Directory -Force $env:GOCACHE,$env:GOTMPDIR | Out-Null
go test ./internal/agent -run '^TestAgentBufferPoolConcurrentContention$' -count=1 -v
```

Current deterministic failure:

```text
codetools_bench_test.go:155: Allocations = 767, want <= 80
```

## Likely files

- `internal/codetools/buffer_pool.go:48-142`
- `internal/codetools/buffer_pool_test.go`
- `internal/agent/codetools_bench_test.go:104-160`
- `docs/tickets/codetools/TICKET-01-buffer-pool-contention.md`

## Blast radius and affected lanes

- Affected: `internal/codetools` scratch-buffer reuse and the `internal/agent`
  concurrency witness.
- Unaffected: native model planning, gateway behavior, model loading, compute
  backends, and hardware qualification.
- Fallback: trunk behavior remains unchanged while this isolated test defect is
  tracked; callers continue using the existing pool.

## Lane

codetools

## Expected steps

4

```routing
lane: codetools
paths: internal/codetools/buffer_pool.go, internal/codetools/buffer_pool_test.go, internal/agent/codetools_bench_test.go, docs/tickets/codetools/TICKET-01-buffer-pool-contention.md
expected_steps: 4
```

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/2 points.

## Completion standard

development

## Tracking

Tracked by [#12669](https://github.com/anthony-chaudhary/fak/issues/12669).

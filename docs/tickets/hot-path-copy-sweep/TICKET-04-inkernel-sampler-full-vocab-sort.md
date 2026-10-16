<!-- fak-agent-key: inkernel-sampler-full-vocab-sort -->
```routing
lane: agent
paths: ["internal/agent/inkernel_sampling.go", "internal/agent/inkernel_sampling_test.go"]
expected_steps: 5
```

# perf(agent): make the in-kernel sampler allocation-free and partial-select instead of sorting the full vocabulary every token

GitHub: [#13873](https://github.com/anthony-chaudhary/fak/issues/13873)

Process cause: verification-gap

## Value

- **For:** every fak-native served token (`internal/agent/inkernel_decode.go:1123` calls the sampler).
- **Problem:** `internal/agent/inkernel_sampling.go` allocates `probs []float64` of size V per token (:93, about 1 MB at V about 129K); any `top_p < 1` adds a size-V index slice and a closure `sort.Slice` over the whole vocabulary (:185, :215-219); `topK > 64` does a full sort plus a `map[int]bool` (:336-346); penalties or bias copy the logits (:124, :162).
- **Today:** ESTIMATED 15-30 ms and about 2-3 MB of allocation per token. Negligible at V4.1's current ~1.5 tok/s, but a large share of the step at the 30 tok/s target and on fast dense models today. Unmeasured.
- **Better because:** nucleus and top-k need only the head of the distribution; a partial select over reused buffers removes the sort and the per-token allocations.
- **Witness:** a sampler benchmark with `ReportAllocs` and an `AllocsPerRun` ceiling of 0 after warm-up, plus seeded-distribution parity.

## Problem frame

- Centrality: Core
- P1 advanced: advanced - removes per-token O(V log V) work and allocation from every served token.
- P2 preserved: preserved - sampling semantics and seeds stay fak-owned and identical.
- P3 advanced: advanced - reuses existing sampler options and tests.
- P4 preserved: preserved - parity on seeded draws gates correctness.

## Why this is next

It is on every served token for every model, so its share grows exactly as the decode campaign succeeds; fixing it before the 30 tok/s work lands prevents a new ceiling.

## Current state

Classification: **PRESENT** but unbounded: correct sampler, no allocation ceiling, no benchmark at production vocabulary size.

## Core through-line

Logits (V floats) -> reused workspace -> temperature/penalty in place -> partial select of top-k / nucleus head -> draw -> same token as the parent for a fixed seed.

## Working spine

Served decode -> sampler call -> benchmark at V=129K -> zero allocs after warm-up -> seeded parity.

## Scope

`inkernel_sampling.go` workspace reuse and partial selection.

## Gold-plating boundary

- No new sampling modes, no GPU sampling, no API change.
- Do not change the RNG or the distribution.

## Done condition

- [ ] Per-session reusable sampler workspace.
- [ ] Partial selection for top-k and nucleus (no full sort).
- [ ] `BenchmarkInkernelSample` at V=129K with `ReportAllocs`.
- [ ] `AllocsPerRun` ceiling 0 after warm-up.
- [ ] Seeded parity over all option combinations.

## Definition of done

Benchmark and ceiling in the package tests; parity green; ns/op delta recorded in the commit.

## Witness

`go test ./internal/agent/ -run 'Inkernel.*Sampl' -bench 'InkernelSample' -benchmem -count=1`

## Likely files

- `internal/agent/inkernel_sampling.go`
- `internal/agent/inkernel_sampling_test.go`

## Parent context

Parent campaign: #12640. Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

Seeded parity plus zero-alloc ceiling; benchmark before/after in the commit.

## Closure binding

Close with the landing commit and the benchmark delta.

## Work estimate

Estimate: 2 points (small) - about half a day.

## Overall completion contribution

2/28 points of the hot-path copy sweep.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13873

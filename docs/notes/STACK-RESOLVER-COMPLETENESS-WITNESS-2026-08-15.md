# Stack resolver completeness and bounded-cost witness

**Date:** 2026-08-15  
**Issue:** [#6892](https://github.com/anthony-chaudhary/fak/issues/6892)  
**Scope:** the relation algebra implemented by `internal/stackresolve`; not a general SAT solver or production-scale catalog claim.

## Failure reproduced

The initial spine selected the lexicographically first provider for a capability. That was deterministic but incomplete: if `model:a-broken@1` provided `model.coder` and required a missing kernel while `model:z-working@1` provided the same capability without that requirement, the resolver refused even though a valid composition existed.

`TestResolverBacktracksPastBrokenLexicalProvider` captures this exact failure class. The resolver now searches reachable provider alternatives in stable order, returns the first satisfiable receipt, and discards decisions and paths from failed branches.

## Independent oracle

`TestResolverMatchesIndependentSmallGraphOracle` exhaustively generates 64 small catalogs varying:

- zero, one, or two providers for a required capability;
- a transitive provider requirement;
- presence or absence of its provider;
- a root conflict;
- presence or absence of the conflicting root.

For each catalog, a separate oracle enumerates every component subset and checks roots, requirements, substitutes, and conflicts without calling the resolver's closure/search helpers. Resolver allow/refuse answers must match the oracle for every generated catalog.

This proves soundness/completeness only for those generated graphs and implemented hard relations. It does not prove minimal-cost choice, arbitrary version-range solving, conditional predicates, evidence freshness, or explanation minimality across every possible graph; those remain explicit operating-envelope work.

## Cost witness

Command:

```bash
go test -run '^$' -bench BenchmarkResolveBranching -benchmem -count=5 ./internal/stackresolve
```

Fixture: eight transitive capability levels, three candidates per level. The first branch is satisfiable, so this measures deterministic catalog normalization plus reachable closure without adversarial backtracking.

Observed on Windows/amd64, AMD Ryzen 9 9950X, 2026-08-15:

```text
62,812 ns/op  61,360 B/op  388 allocs/op
57,179 ns/op  61,360 B/op  388 allocs/op
53,090 ns/op  61,360 B/op  388 allocs/op
58,412 ns/op  61,360 B/op  388 allocs/op
51,320 ns/op  61,360 B/op  388 allocs/op
```

Median: **57.179 µs/op**. This is a witnessed implementation-overhead result for the synthetic fixture, not yet a net-true operator-efficiency claim. The tuned manual-preflight comparison and adversarial branching envelope remain open in #6892.

## Reproduce

```bash
go test -count=1 ./internal/stackresolve
go test -run TestResolverMatchesIndependentSmallGraphOracle -count=1 ./internal/stackresolve
go test -run TestResolverBacktracksPastBrokenLexicalProvider -count=1 ./internal/stackresolve
go test -run '^$' -bench BenchmarkResolveBranching -benchmem -count=5 ./internal/stackresolve
```

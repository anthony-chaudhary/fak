# Go AST query alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6182](https://github.com/anthony-chaudhary/fak/issues/6182) tracks real structural-search tool runs and independent resource/cost witnesses.

## Capability and workload

`internal/astquery.Search`, consumed by `codesearch ast`, parses a Go expression pattern, unifies expression subtrees, binds named metavariables consistently, supports anonymous wildcards, and returns source positions and bindings. Higher-level codesearch trigram, graph, and feature modes are outside this leaf.

Every arm searches identical Go source for `eq($X, $X)`. The corpus has two true calls (`alpha`, `beta`), one same-shape call with inconsistent arguments, one string and one comment containing textual decoys, and an unrelated function call. Correctness requires exactly the two ordered source lines and exact `X` bindings.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native Go AST query | native | available |
| literal text search | tuned no-structure baseline | available, incorrect |
| Semgrep | external | unavailable |
| ast-grep | external | unavailable |
| Comby | external | unavailable |
| gogrep | external | unavailable |

No equivalent first-class fak integration was found. If one appears, it must be added as a separate arm. The literal baseline performs one pass over candidate lines but cannot enforce syntax or repeated-hole consistency. Unavailable tools retain zero measurements; reimplementations do not witness them.

## Completion evidence

All arms need precision/recall, binding and location errors, parse failures, latency/throughput, CPU/RSS, bytes, rule setup/operator time, and total cost, with pinned versions/rules and independent output read-back.

`TestCompareLocalKeepsStructuralSearchAlternativesExplicit` locks the inventory, native oracle, baseline failure, and zero rows. `BenchmarkSearchRepeatedMetavariable` runs the real parser/unifier and verifies bindings each iteration. Local timing is not a cross-product claim; no tool is ranked until #6182 has real runs.

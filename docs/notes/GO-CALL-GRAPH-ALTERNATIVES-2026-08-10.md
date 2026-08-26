# Go call-graph alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6205](https://github.com/anthony-chaudhary/fak/issues/6205) tracks real type-aware and code-intelligence tool runs with independent resource and cost witnesses.

## Capability and same workload

`internal/codegraph.BuildCallGraphFiles` parses a Go package, emits nodes for functions and methods, resolves direct calls syntactically by name, and supports bounded or transitive forward/reverse traversal with shortest paths. `codesearch calls|callers` consumes this capability. The frozen workload is the same two-file package containing a free-function-to-method chain, with exact node, direct-edge, forward-reach, reverse-dependent, distance, and path oracles.

This is deliberately a **syntactic** graph: selector resolution is name-based, not receiver-type checked. Correctness is scored against the declared fixture rather than overstating whole-program precision. Generic graph APIs and codesearch's grep/literal/AST/feature modes are separate capability debt; AST query and trigram search are already contracted independently.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native syntactic Go call graph | native | available |
| `go/ast` direct-call scan | tuned no-graph baseline | available, misses transitive/reverse paths |
| `golang.org/x/tools/go/callgraph` | external | unavailable |
| gopls call hierarchy | external | unavailable |
| Go guru callers/callees | external | unavailable |
| CodeQL Go call graph | external | unavailable |
| SCIP Go code-intelligence graph | external | unavailable |

No equivalent first-class fak integration was found. If one is discovered, it must become a separate `fak + integration` arm rather than being folded into an external row. Package loading/index construction and query setup are part of cost and may not be hidden.

## Completion evidence

Each complete arm reports node and edge precision/recall, forward and reverse reach accuracy, distance/path accuracy, latency and throughput, CPU, peak RSS, input/index/storage/network bytes, setup/operator time, and total cost. Pin tool versions, Go version, fixture, commands, indexes, raw reports, and independent read-back.

`TestCompareLocalKeepsCallGraphAlternativesExplicit` locks arm inventory, native path correctness, the direct-scan baseline's two missed graph checks, and measurement-zero honesty for unavailable arms. `BenchmarkSyntacticGoCallGraph` executes real parsing, graph construction, and both traversal directions. Five local Windows/amd64 samples were 10,972, 15,173, 13,425, 15,813, and 15,691 ns/op (median 15,173 ns/op; about 10,802 B/op; 209 allocs/op). Local timing is not a cross-tool ranking; no external arm is ranked before the open issue carries real witnesses.

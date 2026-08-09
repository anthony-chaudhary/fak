# Native implementation benchmark contracts

Status: **INCOMPLETE**. This is the machine-readable starting spine for requiring every fak-native implementation to be compared with the strongest practical alternatives. It deliberately does not turn missing measurements into claims.

Run:

```text
fak native-benchmarks
fak native-benchmarks --json
fak native-benchmarks --check   # non-zero until every declared contract has a witness
```

## Comparison rule

Each native capability must declare one shared-workload contract with:

1. the fak-native arm;
2. the tuned no-feature or incumbent baseline (never a deliberately weak baseline);
3. the next-best external implementation known for that capability;
4. an explicit inventory of every fak first-class integration that supplies an equivalent capability, with validation that each inventory entry has a measured `fak + integration` arm beside `fak + native` on the same workload;
5. quality/correctness, latency, token/resource use, and total-cost metrics; and
6. a reproducible witness before any performance claim is promoted.

An integration arm is additive to the next-best arm, not a substitute for it. If the best external implementation is also a first-class integration, one arm may carry both facts in its provenance, but the report must still name the integration explicitly.

The command now discovers every production Go leaf directly beneath `internal/` and reports uncovered leaves. This makes the gap exhaustive against the repository-native leaf inventory rather than silently treating two hand-picked examples as complete. The registry starts with the two examples called out by the operator:

| Native capability | Native location | tuned baseline | next-best comparison | required outcomes |
|---|---|---|---|---|
| tool filtering | `internal/gateway/mcp_defer.go` | all schemas, provider cache enabled | retrieval-based selection (ToolRAG class) | task success, tool recall, input tokens, TTFT, total cost |
| context compression | `internal/ctxmmu` | full history, provider cache enabled | LongLLMLingua | task success, retained-fact recall, input tokens, latency, total cost |

Both currently have missing witnesses, which is why `--check` fails. The registry is in `internal/nativebench`; new native capabilities must be added there with their alternatives before their benchmark obligation can be considered covered.

## Scope still to enumerate

This spine does **not** yet prove repository-wide coverage. Leaf discovery is exhaustive at the package boundary, but a package can contain multiple benchmarkable capabilities and not every leaf is necessarily performance-bearing. The authoritative completion work is to classify the discovered leaves, split multi-capability leaves, map equivalent first-class integrations, and attach benchmark witnesses. Until that inventory is complete and every contract passes, the broad claim “all native implementations are benchmarked against next best alternatives” remains **not yet**.



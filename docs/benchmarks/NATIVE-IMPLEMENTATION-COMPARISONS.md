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

The command discovers every production Go leaf directly beneath `internal/` and requires an explicit disposition for each one: `capability`, `multi_capability`, or `infrastructure`. Unclassified leaves remain missing; discovery never promotes a package to covered by inference. Capability and multi-capability leaves name every benchmark contract they contain, while infrastructure leaves carry a reason and need no performance comparison. This makes classification debt machine-visible without pretending that one package equals one capability. The registry starts with the two examples called out by the operator:

| Native capability | Native location | tuned baseline | next-best comparison | required outcomes |
|---|---|---|---|---|
| tool filtering | `internal/gateway/mcp_defer.go` | all schemas, provider cache enabled | retrieval-based selection (ToolRAG class) | task success, tool recall, input tokens, TTFT, total cost |
| context compression | `internal/headroom/native.go` | full history, provider cache enabled | LongLLMLingua | task success, retained-fact recall, input tokens, latency, total cost |
| prefix KV reuse | `internal/radixkv/radixkv.go` | prefix caching disabled | SGLang RadixAttention, plus `fak + llm-d` | output equivalence, prefix hit rate, TTFT, throughput, KV bytes, total cost |

All currently have missing witnesses, which is why `--check` fails. The registry is in `internal/nativebench`; new native capabilities must be added there with their alternatives before their benchmark obligation can be considered covered.

## Scope still to enumerate

This spine does **not** yet prove repository-wide coverage. Leaf discovery is exhaustive at the package boundary and the disposition schema is now enforced, but most leaves are still explicitly unclassified. The authoritative completion work is to classify every discovered leaf, split every multi-capability leaf into contracts, map equivalent first-class integrations, and attach benchmark witnesses. Until the unclassified count reaches zero and every contract passes, the broad claim “all native implementations are benchmarked against next best alternatives” remains **not yet**.



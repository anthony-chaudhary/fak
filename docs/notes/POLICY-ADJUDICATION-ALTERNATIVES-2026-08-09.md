# Policy adjudication native-vs-alternative comparison witness

Date: 2026-08-09  
Capability: `policy_adjudication`  
Status: **INCOMPLETE**

## Arms

- `fak native`: the in-process structural capability floor in `internal/adjudicator`.
- tuned no-feature baseline: direct allow/deny lookup for the same frozen calls.
- strongest practical external alternatives: OPA/Rego and Cedar, each as a separate arm.

No equivalent first-class fak policy-engine integration is currently declared in the integration catalog. OPA and Cedar are therefore external alternatives, not falsely labeled `fak + integration` arms. If either becomes first-class, the registry must add a separate integrated arm rather than reusing its standalone result.

## Same-workload local runner

`adjudicator.CompareLocal` emits `fak-policy-adjudication-comparison/1`. Its five-case corpus covers explicit allow, prefix allow, explicit deny, default deny, and self-modification refusal. The native seam and direct-lookup baseline consume the same calls and both produce the expected allow/deny class for 5/5 cases. This fixture establishes only this bounded structural-policy behavior; it is not full semantic parity with Rego or Cedar.

OPA and Cedar remain explicit with `available=false`. No mock evaluator, translated fixture result, or package import can turn them available.

## Local overhead witness

Host: Windows amd64, AMD Ryzen 9 9950X. Command:

```text
go test ./internal/adjudicator -run '^$' -bench '^BenchmarkPolicyAdjudicationComparison$' -benchmem -benchtime=100000x -count=5
```

Median across five runs:

| Arm | ns / five calls | Approx. ns / call | B/op | allocs/op |
|---|---:|---:|---:|---:|
| fak-native adjudicator | 3,988 | 797.6 | 2,099 | 41 |

The benchmark includes argument decoding and the actual native adjudication path but no external process, policy compilation, RPC, or inference. It cannot establish an end-to-end winner.

## Missing live witness

Completion requires equivalent policy semantics—not merely the same final booleans—to be encoded for fak, OPA, and Cedar. Run all engines over the identical serialized corpus and concurrency schedule; capture verdict equivalence, covered policy features, compile/startup cost, warm p50/p95 latency, throughput, peak RSS, and total operating cost. Version policy artifacts and engine binaries in the witness.

## Honest verdict

No net-true policy-engine winner is established. The native local fixture is correct and fast for its bounded workload, but OPA and Cedar have not yet been executed and their richer language/runtime tradeoffs have not been priced.

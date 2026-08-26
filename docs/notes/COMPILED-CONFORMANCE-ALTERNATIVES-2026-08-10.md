# Compiled ABI and adjudication conformance alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6191](https://github.com/anthony-chaudhary/fak/issues/6191) tracks real contract/policy framework runs and independent resource/cost witnesses.

## Capability and workload

`internal/conformance.Run`, exposed as `fak conformance`, is self-contained: it embeds the frozen ABI golden and dogfood policy, compares compiled closed enums to the wire contract, then executes the entire verdict matrix through the real adjudicator. Its anti-drift tests bind embedded copies back to repository sources.

Every arm receives the same ABI matrix and adjudication cases. Correctness requires both compiled enum equality and execution of all policy cases with expected verdict/reason; merely parsing or schema-validating embedded documents misses behavior drift.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native compiled conformance suite | native | available |
| embedded JSON and schema equality only | tuned no-execution baseline | available, incomplete |
| OPA test | external | unavailable |
| Conftest | external | unavailable |
| OpenAPI and JSON Schema contract tests | external | unavailable |
| Pact | external | unavailable |
| Cedar policy validator and tests | external | unavailable |

No equivalent first-class fak integration was found. External rows remain zero until real frameworks express and execute the equivalent workload. Translating policy/contracts is setup cost and must be included rather than hidden.

## Completion evidence

Complete arms report passed/missed checks, false failures, planted mutation catch rate and reason fidelity, latency/throughput, CPU/RSS, bytes, setup/operator time, and total cost. Versions, translated contracts, commands, raw reports, and independent read-back must be pinned.

`TestCompareLocalKeepsContractFrameworkAlternativesExplicit` locks inventory, native two-check execution, baseline missed behavioral check, and unavailable zeros. `BenchmarkCompiledConformanceSuite` executes the full real suite per iteration. Local timing is not a cross-framework claim; no system is ranked until #6191 has complete real witnesses.

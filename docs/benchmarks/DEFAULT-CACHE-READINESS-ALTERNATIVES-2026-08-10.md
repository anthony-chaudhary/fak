# Default-cache readiness alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6178](https://github.com/anthony-chaudhary/fak/issues/6178) tracks real integration/policy-engine runs and independent resource/cost witnesses.

## Capability boundary

`internal/vcachescore.DefaultReadiness`, consumed by `fak cachevalue status`, decides whether cache behavior is ready for default-on posture. It requires independently proven cold-path correctness, the versioned `fak.cache.default_usefulness.v1` verdict, explicit provider/kernel/context/external/forecast evidence planes with correct provenance, and no unsupported active-cache path.

This contract covers only that readiness gate. Aggregate usefulness scoring, economics, agentic activation, concentration/index planning, and artifact construction remain distinct benchmark debt in this multi-capability package.

## Same-workload oracle

Every arm evaluates five ordered reports:

1. fully witnessed planes and correct cold path — **ready**;
2. provider-only evidence — **blocked**, default-usefulness reason;
3. kernel evidence mislabeled `OBSERVED` — **blocked**, provenance reason;
4. unsupported external active-cache path — **blocked**, unsupported-capability reason;
5. otherwise-ready report with cold-path correctness removed — **blocked**, cold-path reason.

Correctness requires all five verdicts and reason classes. A scalar score that happens to separate some fixtures is not sufficient.

## Arms

| Arm | Class | Local status | Boundary |
|---|---|---:|---|
| fak native default-cache readiness gate | native | available | real `DefaultReadiness` over committed report schema |
| usefulness-score threshold only | tuned no-policy baseline | available, incorrect | committed readiness score threshold, no provenance/cold-path policy |
| fak + Prometheus | first-class integration | unavailable | real metric export, rules, and read-back |
| fak + OpenTelemetry | first-class integration | unavailable | real collector/exporter policy and read-back |
| OPA/Rego | external | unavailable | pinned OPA and real Rego policy |
| Prometheus rules | external | unavailable | pinned Prometheus rule evaluation |
| Datadog monitors | external | unavailable | real ingestion and monitor evaluation |
| LangSmith evaluations | external | unavailable | real trace/evaluation boundary |

Unavailable product arms retain `Available=false` and zero measurements. An in-process imitation would not witness the named product.

## Completion metrics

All arms must report true-ready, true-blocked, false-ready, false-blocked, reason mismatches, latency and throughput, CPU, peak RSS, input/network/storage bytes, setup/operator time, service/license charges, and total cost. Reproduction requires pinned versions/configuration, raw decisions, and independent read-back.

`TestCompareDefaultReadinessLocalKeepsPolicyAlternativesExplicit` locks the inventory, local oracle, baseline failure, and unavailable-arm zeros. `BenchmarkDefaultReadinessFiveCases` exercises all five real native decisions per iteration. Local timing is not a cross-product claim; no alternative is ranked until #6178 has complete real-boundary witnesses.

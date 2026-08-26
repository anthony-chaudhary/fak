# Multi-provider token-usage normalization alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6189](https://github.com/anthony-chaudhary/fak/issues/6189) tracks real provider/integration/telemetry runs and independent resource/cost witnesses.

## Capability and workload

`internal/canon.AdaptTokenUsage` maps OpenAI current and legacy shapes, Anthropic cache-read/write counters, and local prompt/completion counters into reconciled input-fresh/read/write and output-visible/reasoning classes. It rejects invalid JSON, negative classes, and inclusive-detail overflow while retaining trimmed raw provider JSON. This packet does not cover canon's secret/injection scanner, redactors, PII detector, or scoring corpus.

Every arm receives seven ordered cases: OpenAI current, OpenAI legacy, Anthropic cache usage, local usage, invalid JSON, negative local usage, and OpenAI cached-detail overflow. Correctness requires exact classes/totals, exact refusal behavior, and raw unknown-field preservation.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native canonical token-usage adapter | native | available |
| provider total fields only | tuned no-normalization baseline | available, incorrect |
| fak + OpenAI | first-class integration | unavailable |
| fak + Anthropic | first-class integration | unavailable |
| fak + local provider | first-class integration | unavailable |
| fak + OpenTelemetry | first-class integration | unavailable |
| OpenAI SDK usage models | external | unavailable |
| Anthropic SDK usage models | external | unavailable |
| LiteLLM usage normalization | external | unavailable |
| OpenTelemetry GenAI semantic conventions | external | unavailable |
| LangSmith token and cost tracking | external | unavailable |

The totals-only baseline preserves headline counts but loses cache/reasoning classes, raw provenance, and validation invariants. Product arms remain zero until their actual boundary runs; adapters that mimic SDKs or telemetry do not witness them.

## Completion evidence

Complete arms report correct cases, rejection/class/raw-loss errors, represented tokens, latency/throughput, CPU/RSS, input/network bytes, evaluator/model tokens where applicable, setup/operator time, and total cost. Versions, payloads/configuration, raw output, and independent read-back must be pinned.

`TestCompareTokenUsageLocalKeepsTelemetryAlternativesExplicit` locks inventory, native oracle, baseline losses, and unavailable zeros. `BenchmarkAdaptTokenUsageCorpus` executes all seven native cases per iteration. Local timing is not a cross-product claim; no arm is ranked until #6189 has complete real witnesses.

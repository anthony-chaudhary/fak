# Provider-cache telemetry reconciliation alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6173](https://github.com/anthony-chaudhary/fak/issues/6173) tracks real provider/observability runs and independent resource/cost witnesses.

## Capability and same workload

`internal/vcachestar.FoldTelemetry` treats a cache manifest as a prediction, then reconciles it with provider usage read-back. This contract covers false-warm demotion, alarm and segment/token/byte divergence location, plus accounting that rebates only confirmed cache-read tokens. `vcachestar` preflight/layout and star planning remain separate benchmark debt.

Every arm receives the same believed-warm prior prefix (`system|old`), current prefix (`system|new`), segment token counts (100 stable + 50 message), zero provider cache-read tokens, and 150 uncached tokens. Correct output:

- demotes warmth and raises the zero-read alarm;
- identifies segment 1, token offset 100, and byte offset 7;
- books 150 uncached tokens and rebates zero;
- does not report a confirmed hit.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native cache telemetry reconciliation | native | available |
| trust warm manifest and modeled savings | no-read-back baseline | available, incorrect |
| fak + Anthropic prompt caching | first-class integration | unavailable |
| fak + OpenAI prompt caching | first-class integration | unavailable |
| fak + Gemini context caching | first-class integration | unavailable |
| fak + Prometheus | first-class integration | unavailable |
| fak + OpenTelemetry | first-class integration | unavailable |
| Prometheus recording and alert rules | external | unavailable |
| Datadog monitors | external | unavailable |
| LangSmith traces | external | unavailable |

The baseline is the relevant no-feature incumbent: accept warm-manifest state and modeled savings without provider read-back. It is fast but fails the correctness and accounting oracle. Unavailable arms retain `Available=false` and all measurements at zero; local adapters do not witness real providers or telemetry products.

## Completion evidence

All arms need demotion/alarm accuracy, segment/token/byte error, booked/rebated tokens, latency, CPU, peak RSS, telemetry/network/storage bytes, operator/setup time, service charges, and total cost. Runs must pin provider/product versions and configuration, preserve raw usage/diagnostic output, and receive independent read-back.

`TestCompareTelemetryLocalKeepsReconciliationAlternativesExplicit` locks arm identity, native correctness, baseline failure, and unavailable zeros. `BenchmarkFoldTelemetryDivergentZeroRead` exercises the real fold with the oracle every iteration. No local timing is promoted as a cross-product result and no alternative is ranked until #6173 has complete real-boundary witnesses.

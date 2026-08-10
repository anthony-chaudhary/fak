# Codex token-sanitization alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6169](https://github.com/anthony-chaudhary/fak/issues/6169) tracks real integration/external runs and independent resource/cost witnesses.

## Capability and same-workload contract

`internal/vcacheextract` reads Codex session JSONL, admits only supported token-count records, reconstructs an allowlisted counter-only shape, and converts those counters to provider-cache observations. This packet benchmarks the extraction/sanitization capability; filesystem session discovery is separate capability debt.

Every arm receives the same four ordered records:

1. one response-content record containing `RESPONSE_SECRET` (ineligible);
2. one `event_msg/token_count` row with `(input=100, cached=80)` plus `PROMPT_SECRET`;
3. one `turn.completed` row with `(input=250, cached=200)`, output-token metadata, and `RESPONSE_SECRET`;
4. one tool-call row containing `TOOL_SECRET` (ineligible).

A correct arm emits exactly the two ordered input/cached pairs, no extra rows, no forbidden content fields, and zero occurrences of the three secret sentinels. That leakage oracle is part of the workload, not an optional security claim.

## Required arms

| Arm | Class | Local status | Honest boundary |
|---|---|---:|---|
| fak native Codex token sanitizer | native | available | `ExtractRows` streaming JSONL decode plus allowlisted reconstruction |
| raw JSONL pass-through | tuned no-feature baseline | available, incorrect | byte-preserving copy; counters remain present but content leaks and ineligible rows survive |
| fak + OpenTelemetry | first-class integration | unavailable | real collector/exporter plus sink read-back |
| fak + Prometheus | first-class integration | unavailable | real export/scrape plus stored-series read-back |
| jq streaming projection | external | unavailable | pinned jq binary and actual streaming filter |
| Vector VRL remap | external | unavailable | pinned Vector pipeline and sink read-back |
| Fluent Bit filter pipeline | external | unavailable | pinned Fluent Bit pipeline and sink read-back |

Unavailable arms keep `Available=false` and every measurement at zero. A Go reimplementation of jq/VRL/Fluent Bit or a local telemetry adapter would benchmark fak code, not those products, and therefore cannot fill these rows.

## Metrics required for completion

- quality/correctness: eligible, missed, extra, and counter-mismatched rows; ordering; forbidden fields; forbidden secret-byte occurrences; parse failures; malformed-tail handling;
- latency/resources: wall latency, bytes/s, CPU seconds, peak RSS, input/output/network bytes;
- cost: operator/setup seconds, infrastructure/service charges, and total cost for the same repetition count;
- reproducibility: pinned binary/service versions, exact filters/configuration, raw output, independent leakage scan, and machine envelope.

## Current local witness

`TestCompareLocalKeepsTokenSanitizationAlternativesExplicit` locks the arm order, native correctness/leakage oracle, deliberately failing raw baseline, and zero-measurement honesty for every unrun product. `BenchmarkExtractSanitizedTokenRows` measures the real file-open/scan/decode/sanitize path and checks counters on every iteration.

No local timing is promoted as a cross-product result. The native benchmark must be captured on the same witness host and repetition policy as all complete arms. Until #6169 contains those artifacts, there is no strongest-alternative ranking and no net-true performance or cost claim.

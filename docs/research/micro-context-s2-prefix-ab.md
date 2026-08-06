---
title: "Micro-context S2: shared-prefix A/B"
description: "Controlled real-endpoint comparison of unique prefixes, byte-identical shared prefixes, and tuned sequential execution."
status: observed
last_reviewed: 2026-08-06
---

# Micro-context S2 — shared-prefix A/B against the real alternative

## Verdict

**No cache benefit was observed on this endpoint.** The byte-identical shared-prefix arm
reported zero cached prompt tokens, the endpoint exposed no prompt-cache hit/miss metric,
and `/v1/batches` returned 404. Shared-prefix throughput was 0.989× the equal-concurrency
unique-prefix arm and TTFT was 1.014×; those near-parity descriptive ratios neither prove
nor suggest a cache gain. The S2 result therefore records `claim_verdict: not-yet` and
moves the cache-confirmation question to an endpoint with observable prefix reuse.

A separate concurrency result is net-true at this narrow scope: four bounded workers
completed the fixed 12-task corpus **1.36× faster** than the tuned one-worker sequential
arm, inclusive of scheduler and queue costs. Aggregate throughput did not make the
critical path faster: concurrent TTFT p50 was 25.0 s versus 8.61 s sequential.

## Reproduction and verifier

```bash
go run ./cmd/microcontextdemo \
  -prefix-ab experiments/microcontext/s2-gcp-prefix-ab-2026-08-06.json \
  -contexts 12 -workers 4 \
  -endpoint http://<sanctioned-node>:8081 \
  -model qwen2.5-0.5b-cpu \
  -provider fak-native-openai-compatible \
  -hardware "GCP sanctioned fak-realmodel CPU path on L4 node" \
  -request-timeout 5m

go run ./cmd/microcontextdemo \
  -verify-prefix-ab experiments/microcontext/s2-gcp-prefix-ab-2026-08-06.json
```

The public artifact omits the endpoint address. It is
[`experiments/microcontext/s2-gcp-prefix-ab-2026-08-06.json`](../../experiments/microcontext/s2-gcp-prefix-ab-2026-08-06.json).

## Arms and observed values

All arms use the same model, endpoint, task IDs, stable message ordering, 8-token output
cap, and a roughly 700-token base. The unique arm replaces one fixed-width eight-byte
nonce inside the base per task; the shared arm leaves every base byte identical. Observed
prompt tokens remain the authority because tokenizer boundaries can differ despite equal
byte length.

| Arm | Workers | Completed | Wall | TTFT p50 | Prompt / decode tokens | Contexts/s |
|---|---:|---:|---:|---:|---:|---:|
| Concurrent unique full prefixes | 4 | 12/12 | 75.48 s | 24.66 s | 8,418 / 96 | 0.1590 |
| Concurrent byte-identical base | 4 | 12/12 | 76.34 s | 25.00 s | 8,426 / 96 | 0.1572 |
| Tuned sequential byte-identical base | 1 | 12/12 | 104.17 s | 8.61 s | 8,426 / 96 | 0.1152 |

Provider-native batch status is explicitly recorded as unsupported (`GET /v1/batches` →
404), rather than pretending client concurrency is a provider batch API.

## Claim-check

```bash
fak claim-check \
  --statement "On the observed fak-native endpoint, four bounded concurrent micro-context workers completed 12 tasks 1.36x faster than the tuned sequential shared-prefix arm; no shared-prefix cache benefit was observed." \
  --baseline real \
  --baseline-desc "Same model, endpoint, prompt base, task set, and output cap with one physical worker" \
  --net \
  --scope "GCP fak-realmodel qwen2.5-0.5b-cpu, 12 contexts per arm; concurrency gain includes scheduler/queue cost; cache benefit vanishes because cached tokens and cache counters are absent and shared-vs-unique ratio was 0.989" \
  --provenance OBSERVED \
  --witness "experiments/microcontext/s2-gcp-prefix-ab-2026-08-06.json; verify with go run ./cmd/microcontextdemo -verify-prefix-ab experiments/microcontext/s2-gcp-prefix-ab-2026-08-06.json"
```

The command returns `net-true` for the scoped concurrency statement. It does **not** grade
a cache win; the artifact directly falsifies that claim on this endpoint.

## Implications

- Software concurrency can raise aggregate useful work even while per-request latency
  worsens. Both dimensions must stay visible.
- Byte-identical requests alone are not evidence of prefix reuse.
- A controlled-kernel S2 follow-up needs a cache-observable backend: cached-token usage or
  explicit prefix lookup/fill counters, plus warm/cold reset controls.
- API-only adapters must treat missing cache telemetry as `not-yet`, even when timing looks
  favorable.
- The absence of native batch support makes bounded client concurrency the tuned available
  alternative here, but not a universal provider-native baseline.

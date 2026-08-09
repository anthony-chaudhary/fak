---
title: "Micro-context S1: 100 real endpoint contexts"
description: "Captured 100-context real-endpoint spine and the first observed overload boundary."
status: observed
last_reviewed: 2026-08-06
---

# Micro-context S1 — 100 contexts through one real endpoint

## Verdict

**PASS at four physical workers:** 100/100 logical contexts retired through the native
OpenAI-compatible fak endpoint on the sanctioned GCP `fak-realmodel` node. This advances
the program from synthetic S0 scheduling to a real model/provider seam without starting
100 full agent harnesses.

The endpoint used the node's `qwen2.5-0.5b-cpu` fak-native model path. The node has an L4,
but this run deliberately labels itself **CPU path on an L4 node**; it is not a CUDA
throughput claim. The separately exposed CUDA endpoint stopped returning completion
responses during preflight and therefore supplied no successful witness.

## Reproduction

```bash
go run ./cmd/microcontextdemo \
  -selfcheck -contexts 100 -workers 4 \
  -endpoint http://<sanctioned-node>:8081 \
  -model qwen2.5-0.5b-cpu \
  -provider fak-native-openai-compatible \
  -hardware "GCP fak-realmodel CPU path on L4 node" \
  -request-timeout 5m

go test ./cmd/microcontextdemo -run 'TestSpine|TestOpenAIEndpoint'
```

The checked-in public artifact omits the endpoint address. The node identity and command
are sufficient for an authorized operator to reproduce through the sanctioned runbook.

## Observed pass

Source: [`s1-gcp-realendpoint-workers4-pass-2026-08-06.json`](../../experiments/microcontext/s1-gcp-realendpoint-workers4-pass-2026-08-06.json).

| Metric | Observed |
|---|---:|
| Logical contexts | 100 |
| Physical workers | 4 |
| Completed / failed | 100 / 0 |
| Shared base installs | 1 |
| Wall time | 94.725 s |
| TTFT p50 / p95 | 3.852 s / 3.939 s |
| Prompt / completion tokens | 6,190 / 800 |
| Aggregate prompt tokens / wall second | 65.35 |
| Aggregate decode tokens / wall second | 8.45 |
| Useful context completions / second | 1.056 |

`prompt_tokens_per_wall_second` and `decode_tokens_per_wall_second` are aggregate response
usage divided by whole-run wall time. They are **not** server-internal prefill/decode kernel
rates. TTFT remains the critical-path usability row and is not hidden by aggregate work.
The endpoint emitted no cached-token count, so this run makes no cache-hit claim; #5787
owns the controlled shared-prefix A/B and cache provenance.

## Observed overload boundary

The first 16-worker attempt completed only 10/100 contexts before 90 requests exceeded the
three-minute header timeout. The artifact is retained at
[`s1-gcp-realendpoint-workers16-overload-2026-08-06.json`](../../experiments/microcontext/s1-gcp-realendpoint-workers16-overload-2026-08-06.json).
This is the first concrete blocker found by S1: software fan-out is cheap, but physical
admission must match endpoint capacity. More logical contexts are useful only with bounded
workers, backpressure, compatibility scheduling, and overload telemetry—the work tracked
by #5788 and #5790.

## Contract coverage

- One immutable base is inserted by the endpoint adapter; each microagent supplies only its
  task delta.
- `internal/microagent.Host` bounds physical workers and preserves each context's ID/error.
- Streaming SSE provides observed TTFT and final usage for every successful response.
- The test endpoint verifies the shared base, 1:1 request accounting, cached-token parsing,
  and telemetry aggregation.
- No terminal, cwd, transcript manager, credentials bundle, or full harness process is
  created per logical context.

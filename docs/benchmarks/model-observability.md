# Model inference observability spine

`modelperfobs` turns any OpenAI-compatible model server into request-level,
queryable evidence without requiring a patched backend. It is the shortest path
from an agent workload to the three latency dimensions that distinguish a slow
prompt/queue from slow token generation:

- end-to-end latency;
- time to first streamed token (TTFT);
- time per output token (TPOT), output tok/s, and inter-chunk p50/p95.

It also records prompt/completion token counts, model, HTTP status, errors, and a
correlation ID in append-only JSONL. The proxy sends that ID upstream and returns
it as `X-Fak-Observation-ID`, so backend logs and agent outcomes can join to the
same request.

## Working spine

Start an OpenAI-compatible backend such as Qwen3.8-27B, then place the proxy in
front of it:

```bash
fak model-observe proxy \
  --backend http://127.0.0.1:8000 \
  --listen 127.0.0.1:8091 \
  --ledger _scratch/qwen38/model-perf.jsonl
```

Point the harness's OpenAI base URL at `http://127.0.0.1:8091`. Preserve
`stream: true`; for exact token rates, ask the backend for streaming usage
(`stream_options: {"include_usage": true}`). Then rank the likely bottleneck:

```bash
fak model-observe report \
  --input _scratch/qwen38/model-perf.jsonl \
  --format md
```

The JSONL is the query contract, not a dashboard-specific format. Example with
DuckDB (no import step):

```sql
SELECT model,
       count(*) AS requests,
       quantile_cont(ttft_ms, 0.95) AS ttft_p95_ms,
       quantile_cont(tpot_ms, 0.95) AS tpot_p95_ms,
       quantile_cont(output_tokens_per_second, 0.5) AS output_tok_s_p50
FROM read_json_auto('_scratch/qwen38/model-perf.jsonl')
GROUP BY model;
```

## Reading the signal

- High TTFT with modest TPOT: sweep prompt length at concurrency 1, then sweep
  concurrency at fixed prompt length. The first implicates prefill; the second
  queueing/scheduling.
- High TPOT or low output tok/s: inspect device residency, memory bandwidth,
  quantized kernels, and batch shape.
- Healthy request metrics but poor agent wall time: join observation IDs to
  task outcomes; tool latency, retries, prompt growth, or excess generated tokens
  dominate instead of inference.
- Missing TTFT/TPOT: the workload did not stream. Aggregate duration cannot
  identify the bottleneck, so the report says `missing-stream-timing` rather
  than inventing a diagnosis.

## Metric semantics

The names follow the request-level practice documented by vLLM's metrics design:
TTFT, inter-token latency, prompt tokens, and generation tokens are the SLO-facing
metrics, while engine counters explain them. This spine intentionally starts at
the cross-backend request seam. Backend-specific queue, KV-cache, preemption,
and GPU counters are follow-on joins, not required to observe an agent today.

Source studied 2026-08-21: [vLLM metrics design](https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md).


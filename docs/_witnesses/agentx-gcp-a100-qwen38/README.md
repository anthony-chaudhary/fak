# RETRACTED: AgentX-style receipt labelled GCP A100 Qwen3.8-27B is self-check output

**Retracted**: 2026-10-25
**Original date**: 2026-09-05 (commit `2ba17dddbf`, #8775)
**Status**: not evidence. No number in `receipt.json` was measured on any endpoint or GPU.
**Witness test**: `go test ./docs/_witnesses/agentx-gcp-a100-qwen38/...` now asserts the retraction.

## What the receipt really is

`receipt.json` is the output of `fak agentx --selfcheck`, the hermetic mock mode of
`internal/agentxbench`. At the time the receipt carried no execution mode, and the CLI
defaulted `--hardware` to `GCP A100-SXM4-40GB`, so a mock run read as an A100 measurement.
Every request in it matches the mock generator exactly:

- `response_text` is the fixed mock sentence (`agentxbench.MockResponseText`) on all 25 requests;
- `prompt_tokens` is `1024 + 256*turn` and `cached_tokens` is `1024 + 256*(turn-1)`, the mock formula;
- cold TTFT falls in the mock's 1,100-1,300 ms band and warm TTFT in its 220-280 ms band, which is
  where the "4.80x prefix speedup" came from;
- ITL falls in the mock's 12-18 ms band.

`docs/_witnesses/issue-10944-nvidia-gcp-overnight/agentx_qwen38_a100_bench_raw.json` has the same
signature and is not evidence either.

## What it never was

Even a live `fak agentx` run is a **synthetic** workload (`fak-synthetic-agent-turns`): N agents x T
independent single-message turns. It is not the InferenceX AgentX benchmark, which replays the
published Claude Code corpus (v1.0: 393 sessions, 135,282 requests; 256k variant 68,266 requests,
median input 88,768 / output 376) through AIPerf as closed-loop live clients with a primer, 10
warmup requests per lane, a one-hour profiling window and a per-replay cache-bust marker
(https://inferencex.semianalysis.com/agentx/methodology). No `fak agentx` result is comparable to an
InferenceX AgentX row.

## What changed with the retraction

- Receipts carry `execution_mode` (`live` | `mock`), `workload`, `upstream_comparable` (always
  false) and `token_count_source` (`server_usage` | `server_usage_cache_unreported` | `estimated`).
- A live run with no successful request is refused (`NO_SUCCESSFUL_REQUESTS`), and with no warm
  turn the prefix speedup reads `n/a`, not `1.00x`.
- A mock run reports `SIMULATED`, never `VERIFIED_PASS`, and `fak agentx --validate` refuses mock
  receipts and receipts without an execution mode.
- A live run needs `--engine` and `--hardware`; there is no default hardware label.
- Live token counts and cache hits come from the server's `usage` block
  (`stream_options.include_usage`); without it counts are estimated and no cache hit is claimed.
  TTFT starts at the first generated content, not at a role-only frame.

## Artifact integrity

- `receipt.json`: SHA-256 `1af84415c89f94df31214b0970a9a95befc0c9e2afbea9febf40a5ca1e74ae9d`
  (kept byte-identical as the record of what was retracted).

# vCache Codex/OpenAI Probe - 2026-06-25

## Question

Can this host prove or refute vCache savings for Codex/OpenAI from real
`cached_tokens` telemetry?

## Current status

**Proven for this Codex CLI thread from Codex-authored session telemetry.** The
active process exposes Codex management metadata only:

```text
CODEX_MANAGED_BY_NPM
CODEX_MANAGED_PACKAGE_ROOT
CODEX_THREAD_ID
```

There is no `OPENAI_API_KEY` in this process, and the Codex CLI session does
not expose raw OpenAI Responses or Chat Completions usage JSONL for this run.
However, the local Codex session JSONL records `event_msg` / `token_count`
events with `last_token_usage.input_tokens` and
`last_token_usage.cached_input_tokens`. Those counters are enough to prove the
Codex side without an API key.

Captured Codex CLI session witness:

```powershell
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl --json
```

Result:

```json
{
  "status": "PROVEN",
  "requests": 68,
  "baseline_token_equiv": 10638831,
  "actual_token_equiv": 1491490.2,
  "saved_token_equiv": 9147340.8,
  "saved_pct": 85.98069468346664,
  "cache_read_tokens": 10163712,
  "first_positive_request": 1,
  "correctness_depends_on_hit": false
}
```

Frozen summary artifact:
`experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.json`.
Replayable telemetry artifact:
`experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl`.

## Verifier path now supported

`fak vcache prove-telemetry` accepts OpenAI-compatible usage rows in either raw
Responses shape:

```json
{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}
```

or raw Chat Completions shape:

```json
{"usage":{"prompt_tokens":2006,"prompt_tokens_details":{"cached_tokens":1920}}}
```

The verifier splits total input tokens into uncached and cached portions, then
applies `--read-mult` to only the cached portion. With the default `--read-mult
0.1`, the first example proves:

```text
baseline token-equiv: 2006.0
actual token-equiv: 278.0
saved token-equiv: 1728.0
cache read/write tokens: 1920 / 0
```

It also accepts Codex CLI session `token_count` events:

```json
{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":2006,"cached_input_tokens":1920}}}}
```

## Remaining live proof

For raw OpenAI API traffic, issue #727 still has the optional provider probe:
run repeated Codex/OpenAI requests with a stable 1024+ token prefix, capture
provider JSONL containing `cached_tokens`, and feed it to:

```powershell
python tools/vcache_openai_probe.py --out experiments/agent-live/vcache-openai-probe.jsonl
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-openai-probe.jsonl --read-mult 0.1
```

That command is the acceptable close condition for a raw OpenAI API-key run.
The Codex CLI thread is already proven from local `token_count` telemetry above.

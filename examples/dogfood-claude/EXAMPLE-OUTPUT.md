# Example output

Captured runs of the dogfood-claude launcher (which [`run.sh`](run.sh) / [`run.ps1`](run.ps1)
wrap). The real Claude Code CLI drives a local model fronted by `fak serve`; the kernel
adjudicates every proposed tool call before Claude sees it. Reproduce with the commands in
[`README.md`](README.md).

## `--smoke` — the wire, no model intelligence needed

`./run.sh --smoke` (or `.\run.ps1 --smoke`) curls the native Anthropic surface end-to-end
against an offline mock planner, proving the `/v1/messages` wire — buffered and streaming —
without any model dependency:

```
POST /v1/messages → {"type":"message","role":"assistant",
  "content":[{"type":"tool_use","id":"call_0","name":"get_user_details",...}],
  "stop_reason":"tool_use","usage":{"input_tokens":0,"output_tokens":24}}

stream:true → event: message_start
              event: content_block_start
              event: content_block_delta      (input_json_delta)
              event: content_block_stop
              event: message_delta
              event: message_stop
```

The full Anthropic SSE sequence is synthesized from the finished turn, so Claude Code parses
it identically to a live stream and the `tool_use` ids survive the round trip.

## `--probe` — one live Claude Code turn through the kernel (Windows)

`scripts\dogfood-claude.ps1 --probe` runs a single headless Claude Code turn against the
local kernel-fronted model and writes the transcript to
`experiments/agent-live/dogfood-claude-probe-win.json`. The documented witnessed-on-Windows
capture (Claude Code CLI v2.1.181, the in-tree shim fronting `SmolLM2-135M-Instruct`):

```json
{"type":"result","subtype":"success","is_error":false,"num_turns":1,
 "stop_reason":"end_turn","terminal_reason":"completed","duration_ms":35622,
 "modelUsage":{"HuggingFaceTB/SmolLM2-135M-Instruct":{"inputTokens":5816,"outputTokens":256}}}
```

The CLI completed a turn against the local kernel-fronted model in ~36 s on a 16-core
Windows host — **no ollama**; the in-tree shim auto-fronted the model on the box's GPU
(fp16 CUDA), and a GPU-less host runs the identical path on fp32 CPU with the pre-raised
timeouts. The 135M model's *answer* is weak — that is the model-quality caveat in
[`README.md`](README.md), not a wire/kernel issue. `dogfood-claude-probe-win.json` also
holds a longer multi-turn capture fronting a `Qwen3.6-27B-GGUF:Q4_K_M` server through the
same wire.

## `--probe` — one live Claude Code turn through the kernel (macOS)

`scripts/dogfood-claude.sh --probe` →`experiments/agent-live/dogfood-claude-probe.json`,
fronting a large local model (`Qwen3.6-27B-GGUF:Q4_K_M` via `llama-server`):

```json
{"type":"result","subtype":"success","is_error":false,"num_turns":1,
 "result":"pong","duration_ms":218024,
 "modelUsage":{"lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M":{"inputTokens":25638,"outputTokens":24}}}
```

## What the kernel did per call

In an interactive session (`./run.sh`), the `fak serve` log shows the per-turn adjudication.
Allowed everyday calls run; dangerous ones are refused **by argument value, before the shell
sees them** — independent of *why* the model proposed them:

```
/v1/messages turn 1
  Bash{command:"ls -la"}                 ALLOW         → ran
  Bash{command:"git commit -m ..."}      ALLOW         → ran
  Bash{command:"rm -rf /tmp/x"}          DENY  POLICY_BLOCK   (refused before the shell)
  Bash{command:"git push origin main"}   DENY  POLICY_BLOCK
  Write{file_path:".git/config"}         DENY  SELF_MODIFY
  delete_account{}                       DENY  DEFAULT_DENY   (tool never on the floor)
```

The capability floor is [`../dogfood-claude-policy.json`](../dogfood-claude-policy.json); a
denied call never reaches Claude as an executable tool, and a `TRANSFORM` (a `redact_fields`
value rewritten to `[REDACTED]`) or a quarantine event (a flagged tool *result* held out of
the context window) appear on the same log. The load-bearing result: every irreversible /
unsanctioned call is refused at the kernel boundary, on the Anthropic `/v1/messages` wire,
before the real Claude Code CLI can act on it.

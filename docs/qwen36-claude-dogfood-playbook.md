---
title: "Run Claude Code on local Qwen3.6 with fak"
description: "Dogfood playbook to run Claude Code against fak serve fronting a local Qwen3.6 OpenAI-compatible model server, with probe, setup, and troubleshooting steps."
---

# Qwen3.6 Claude Dogfood Playbook

This playbook runs the real Claude Code CLI against `fak serve`, with `fak` fronting
a large local OpenAI-compatible Qwen3.6 server such as `llama-server` or LM Studio.

## FAQ

### What do I run first?

Install the launchers once from this repo:

```bash
fak/scripts/dogfood-claude.sh --install
```

That installs three PATH commands:

| command | purpose |
|---|---|
| `fak` | the repo CLI, so `fak serve --help` works from any directory |
| `fak-dogfood` | generic Claude Code dogfood launcher |
| `fak-qwen36-claude` | Qwen3.6 local preset launcher |

### What command runs the Qwen3.6 Claude probe?

After the model server is already running at `http://127.0.0.1:8131/v1`:

```bash
fak-qwen36-claude --probe "Reply with exactly the word: pong"
```

### What command opens interactive Claude Code on Qwen3.6?

```bash
fak-qwen36-claude
```

### What does the Qwen preset assume?

`fak-qwen36-claude` is the same script as `fak-dogfood`, but its invoked name selects
`FAK_DOGFOOD_PRESET=qwen36-local`. The preset defaults to:

| setting | value |
|---|---|
| backend | `openai` |
| model-server URL | `http://127.0.0.1:8131/v1` |
| model id | `lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M` |
| timeout | `900s` through the existing OpenAI-backend default |
| provider extra body | `{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}` |
| Claude debug | `--debug api` |

Override any of those with the normal `FAK_DOGFOOD_*` env vars when testing a
different node, port, quant, or provider.

### What is the model server?

The model server is the process that loads Qwen3.6 and speaks OpenAI-compatible
`/v1/models` and `/v1/chat/completions`. It can be LM Studio, `llama-server`, vLLM,
SGLang, or `tools/qwen36_node_server.py`.

It is not `fak`. `fak` is the gateway in front of it, and Claude Code is the client
that sends the large agent prompt to `fak`.

### Why did `fak serve` say command not found?

Run the installer:

```bash
fak/scripts/dogfood-claude.sh --install
```

It builds the repo CLI to `tools/.bin/fak` and symlinks it onto PATH as `fak`. If you
do not want to install the PATH command, use the repo-local form:

```bash
go -C fak run ./cmd/fak serve --help
```

### Where do I see requests and debug output?

Claude Code now defaults to API debug mode in the dogfood launcher:

```text
FAK_DOGFOOD_CLAUDE_DEBUG=api
```

For `--probe`, Claude stderr/debug goes to:

```text
/tmp/fak-claude.log
```

The gateway process log goes to:

```text
/tmp/fak-serve.log
```

Set `FAK_DOGFOOD_CLAUDE_DEBUG_FILE=/path/to/file` if you want Claude's debug stream
written to a dedicated file. Grafana shows latency, status, and in-flight metrics,
not raw request bodies.

### Is there a Grafana dashboard for slow requests?

Yes. Start the observability stack:

```bash
tools/grafana/up.sh
```

Open `http://localhost:3000` and use the **FAK Dogfood Slow Requests** dashboard.
It focuses on `/v1/messages` p50/p95/p99, request status, in-flight requests, slow
threshold rates, and kernel activity while Claude Code waits on a local model.

### Why is the first Qwen3.6 request so slow?

Claude Code sends a large prompt with the agent instructions and tool schemas. The
known-good Qwen3.6 probe on 2026-06-19 sent about 25.6K input tokens and completed
in 218 seconds. That is slow, but it is not automatically a failure if the gateway
is still in flight, the stream is sending pings, and the model server is still
working.

## Plain-English Map

There are three different things running. Keep them separate:

| thing | what it is | who starts it | usual port |
|---|---|---|---|
| Qwen3.6 model server | The process that actually loads the 27B local model and generates tokens. This can be `llama-server`, LM Studio, vLLM, or SGLang. | You start it before dogfood. | `8131` or `8000` |
| `fak serve` gateway | The local kernel-fronting HTTP server. It exposes Claude-compatible `/v1/messages`, translates to OpenAI-compatible `/v1/chat/completions`, and adjudicates tool calls. | `dogfood-claude.sh` starts it. | `8080`, or the `FAK_DOGFOOD_PORT` you set |
| Claude Code CLI | The real `claude` command. It thinks it is talking to an Anthropic Messages API, but its base URL points at `fak serve`. | `dogfood-claude.sh` starts it. | no listen port |

The model server is not `fak`. `fak` is the gateway in front of it. Claude Code is
not the model either; it is the client/harness that sends the large agent prompt.

The installed path gives you a global `fak` shell command. Without the installer,
`fak serve` means the `serve` subcommand of the Go CLI in `fak/cmd/fak`. Run it
manually as:

```bash
go -C fak run ./cmd/fak serve --help
```

or build it first:

```bash
go -C fak build -o ../tools/.bin/fak ./cmd/fak
./tools/.bin/fak serve --help
```

The dogfood launcher does that build step for you. `--install` additionally puts
the built CLI on PATH because operators expect `fak serve` to work directly.

## What This Proves

The end-to-end path is:

```text
Claude Code CLI -> fak /v1/messages -> fak planner -> local /v1/chat/completions -> Qwen3.6
```

This is stronger than a raw `/v1/chat/completions` smoke. It proves that Claude
Code can use the Anthropic Messages surface exposed by `fak`, that the gateway can
translate the turn to an OpenAI-compatible local model, and that the local model can
finish a real Claude Code headless turn.

## Assumptions

- You are on the machine that can reach the Qwen3.6 server at
  `http://127.0.0.1:8131/v1`, or you will replace that URL with the node's reachable
  URL.
- The Qwen3.6 model server is already running before you invoke
  `fak-qwen36-claude`.
- You have run `fak/scripts/dogfood-claude.sh --install` once, or you are using the
  repo-local script path directly.
- `curl`, `go`, `python3`, and `claude` are on `PATH`. If your shell cannot find
  `fak-qwen36-claude` or `fak`, make sure the installer-selected bin dir is on
  `PATH`.
- `claude` is already authenticated enough to run locally, but this dogfood command
  points it at the loopback `fak` gateway and uses an isolated
  `~/.claude-faklocal` config directory.
- The local model server speaks an OpenAI-compatible API with `/v1/models` and
  `/v1/chat/completions`.
- The first turn can be slow. The known-good Qwen3.6 probe took about 218 seconds
  because Claude Code sent a 25.6K-token prompt to a local 27B model.

## Prerequisites

Start a Qwen3.6 OpenAI-compatible server first. The known-good local witness used:

```bash
python tools/qwen36_node_server.py --profile mac
```

or an equivalent `llama-server`/LM Studio endpoint serving:

```text
http://127.0.0.1:8131/v1
lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M
```

If starting `llama-server` by hand on a Mac, the shape is:

```bash
llama-server \
  -hf lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
  --host 127.0.0.1 \
  --port 8131 \
  --ctx-size 32768 \
  --n-gpu-layers 99
```

The important part is not the exact launcher. The important part is that, before
dogfood starts, this URL works:

```text
http://127.0.0.1:8131/v1/models
```

Verify the model server before starting Claude Code:

```bash
curl -sS http://127.0.0.1:8131/v1/models
```

If that fails, fix the model server first. The dogfood launcher cannot make Qwen3.6
load; it can only front an endpoint that is already serving.

Expected signs that the model server layer is healthy:

- `curl .../v1/models` returns HTTP 200.
- The response contains `lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M` or the model id
  you plan to pass as `FAK_DOGFOOD_MODEL`.
- A tiny direct chat completion succeeds against the model server without `fak`:

```bash
curl -sS http://127.0.0.1:8131/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
```

If `/v1/models` works but `/v1/chat/completions` fails, the Qwen server is alive but
not ready for this dogfood path. Fix that before debugging Claude Code.

## What The Probe Command Does

The probe command below does all of this:

1. Builds `tools/.bin/fak`.
2. Checks the model server at `FAK_DOGFOOD_BASE_URL`.
3. Starts `fak serve` on `FAK_DOGFOOD_PORT` or `8080`.
4. Configures `fak serve` to proxy upstream to the model server.
5. Exports Claude Code environment variables only for the child `claude` process.
6. Maps every Claude Code model tier to the local Qwen3.6 model id.
7. Runs `claude -p ... --output-format json`.
8. Writes the witness JSON under `fak/experiments/agent-live/`.

It does not install Qwen3.6, download weights, start LM Studio, or start
`llama-server` for you. Use `tools/qwen36_node_server.py` or your model server UI
for that layer.

## Headless Probe

Run one witnessable Claude Code turn with the preset:

```bash
fak-qwen36-claude --probe "Reply with exactly the word: pong"
```

The explicit equivalent is:

```bash
FAK_DOGFOOD_BACKEND=openai \
FAK_DOGFOOD_BASE_URL=http://127.0.0.1:8131/v1 \
FAK_DOGFOOD_MODEL=lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
FAK_DOGFOOD_TIMEOUT_S=900 \
FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON='{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}' \
fak/scripts/dogfood-claude.sh --probe "Reply with exactly the word: pong"
```

The launcher builds `fak`, starts `fak serve`, points Claude Code at the loopback
Anthropic Messages endpoint, maps all Claude model tiers to the local Qwen3.6 id,
and writes the witness to:

```text
fak/experiments/agent-live/dogfood-claude-probe.json
```

## Interactive Claude Code

Use the preset without `--probe`:

```bash
fak-qwen36-claude
```

The explicit equivalent is:

```bash
FAK_DOGFOOD_BACKEND=openai \
FAK_DOGFOOD_BASE_URL=http://127.0.0.1:8131/v1 \
FAK_DOGFOOD_MODEL=lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
FAK_DOGFOOD_TIMEOUT_S=900 \
FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON='{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}' \
fak/scripts/dogfood-claude.sh
```

## Known-Good Witness

The 2026-06-19 Mac local Qwen3.6 witness completed:

```json
{
  "subtype": "success",
  "is_error": false,
  "result": "pong",
  "duration_ms": 218024,
  "modelUsage": {
    "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M": {
      "inputTokens": 25638,
      "outputTokens": 24
    }
  }
}
```

The important performance detail is `ttft_stream_ms`: the gateway now emits an
initial Anthropic SSE event quickly and keeps the stream alive with `ping` events
while the non-streaming local model is still doing the long prefill. Without that,
Claude Code can cancel an otherwise healthy local turn before the model returns.

## Native parity witnesses (#442)

This playbook fronts a local model *server*; the separate question "is fak's own
in-kernel `qwen35` forward faithful to the reference?" is proven by CPU-only oracle
gates that need no server, GPU, or 27B artifact. Re-run them with:

```powershell
# from the repo root — build the tiny qwen3_5 HF fixture once, then run the gates
python internal/model/make_qwen35_tiny.py .cache/qwen35-tiny
python internal/model/export_oracle.py --online --model .cache/qwen35-tiny \
  --out internal/model/.cache/oracle-qwen35 \
  --prompt-ids-json '[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]'
$env:FAK_ORACLE_DIRS = '.cache/oracle-qwen35'
.\fak\test.ps1 -count=1 ./internal/model/ -run Qwen35
go test ./internal/ggufload -run Qwen35 -count=1
```

Each `TestOptionalQwen35…` case skips cleanly without the fixture (CI stays green) and,
when present, proves the hybrid config, per-layer mixer tensor names, hidden-state cosine
≥ 0.9999 vs HF, argmax parity, and cached-prefill parity. The exact commands, the
llama.cpp generated-token comparison on the real 27B artifact, and the honest boundaries
live in
[`docs/benchmarks/FAK-NATIVE-QWEN35-RESULTS.md`](benchmarks/FAK-NATIVE-QWEN35-RESULTS.md).

## In-kernel A100 / 27B coding-loop witness — run contract (#933)

Everything above this section fronts an **external** OpenAI-compatible model server
(`llama-server`, LM Studio, vLLM, …). This section is the other variant: Claude Code
driving a **real multi-turn coding loop** on fak's **own in-kernel CUDA forward** serving
Qwen3.6-27B on a single A100 — no external engine in the loop. That is the usability bar
for the "coding fallback when the subscription is down" goal (#933, the A100/27B extension
of the general agentic-loop witness #610).

**Status: pre-run contract, not a result.** The serve build is `-tags cuda` and the decode
runs on the GPU HAL, so the live turn is **hardware-gated on a real A100** — it cannot be
witnessed on a GPU-less host (there, this is RECORD-only; a SKIP is not a PASS). The
witnessed claim today is the CPU-forward correctness above (#442); this contract pins the
*exact* reproducible procedure and the acceptance gate so the live A100 run produces a
witness an auditor can grade, not a self-report.

Why the `say pong` probe above is **not** this: a one-shot headless turn proves the wire
(gateway translate + one decode), not *coding competence under the agent harness* — reading
files, proposing edits, consuming `tool_result` round-trips, and converging without
derailing or timing out. This contract drives the latter.

### Step 1 — stand the in-kernel serve up (on the A100 host)

```bash
# RUN ON THE A100 HOST, detached so a disconnect does not orphan the ~16 GB load:
FAK_GATEWAY_KEY=sk-fak-... systemd-run --unit=qwen36serve --collect \
  bash tools/qwen36_a100_fak_serve.sh
# poll the phase file until it reads QWEN36_A100_FAK_SERVE_READY:
cat /opt/qwen36-q4k/PHASE
```

`tools/qwen36_a100_fak_serve.sh` builds the cuda fak binary, fetches the q4_k_m GGUF,
serves it through `FAK_Q4K=1 fak serve --backend cuda` (the resident-Q4_K decode path),
and only prints `QWEN36_A100_FAK_SERVE_READY` after a real `/v1/chat/completions` smoke
answers. The gateway keeps Claude Code alive through the long prefill with SSE `ping`
events (`internal/gateway/messages.go`), so a healthy turn is not cancelled at ~60s.

### Step 2 — connect Claude Code to the gateway

```bash
#  bash/zsh — must be sourced so the env vars land in your shell:
source scripts/connect-fak-node.sh <a100-tailscale-ip> sk-fak-... --probe
#  PowerShell — must be dot-sourced:
. scripts\connect-fak-node.ps1 -GatewayHost <a100-tailscale-ip> -GatewayKey sk-fak-... -Probe
```

`--probe`/`-Probe` curls `/healthz` first; a green probe means `claude` will route every
turn through the in-kernel forward.

### Step 3 — drive a real coding task and capture the transcript

Use a **scratch package** (not a repo file) so the witness is deterministic, re-runnable,
and side-effect-free on the shared trunk. Seed a deliberately-failing stub, then ask the
agent to make it pass and add a test — a genuine read → edit → run → iterate loop:

```bash
mkdir -p /tmp/fak-coding-witness/strpad && cd /tmp/fak-coding-witness
printf 'module witness\n\ngo 1.26\n' > go.mod
cat > strpad/strpad.go <<'EOF'
package strpad

// LeftPad pads s on the left with the rune p until it is at least n wide.
// TODO: this stub is wrong — it ignores n and p. Make it correct.
func LeftPad(s string, n int, p rune) string { return s }
EOF
```

Then drive one witnessable, captured turn (headless), writing the witness under
`experiments/agent-live/`:

```bash
claude -p "Read strpad/strpad.go. Implement LeftPad correctly, add a table-driven \
strpad_test.go covering the no-pad, exact-width, and pad cases, then run 'go test ./...' \
and iterate until it passes. Report the final go test output." \
  --output-format json \
  > /path/to/fak/experiments/agent-live/qwen36-a100-coding-loop-$(date -u +%Y%m%d).json
```

(Interactive `claude` works too; capture the session transcript to the same path.)

### Step 4 — record the witness numbers

Record, in the witness JSON / a sibling `.md`, the four numbers #933 asks for:

| field | what it is |
|---|---|
| `turns_to_completion` | assistant turns until `go test ./...` passed |
| `tool_round_trips_survived` | read/edit/bash `tool_result` cycles the stream completed without a schema reject or timeout |
| `wall_time_s` | end-to-end from first request to the green test |
| `decode_tok_s` | observed in-kernel decode rate (from the gateway/server log) |

### Pass / fail

- **PASS** = the task actually completes: `strpad/strpad.go` is correct, a real test exists,
  `go test ./...` is green, and the transcript shows the agent reaching it through tool
  round-trips on the in-kernel forward — captured under `experiments/agent-live/`.
- **Not a pass** = a green-looking transcript with wrong edits, or a SKIP because no A100
  was present. Record those honestly; do not file a SKIP as a PASS.

### Honest failure modes → their own follow-ons

Note which (if any) bit, so they become tracked work rather than silent flake:

- tool-schema rejection / the in-kernel forward not emitting a liftable tool call → #609, #610.
- derailment across `tool_result` turns (poison not evicted) → #612.
- stream timeout / premature cancel — should be removed by the SSE `ping`
  (`internal/gateway/messages.go`); if it recurs, that is a regression to file.

Single-stream only for now (drive one request at a time); batched throughput is its own
item (#401).

## Troubleshooting

If the launcher cannot resolve the model, check `/v1/models` and either pass a
`/v1` base URL or set `FAK_DOGFOOD_MODEL` explicitly.

If the upstream rejects the request with a schema conversion error, confirm the
build includes the OpenAI tool-schema normalizer in `internal/agent/adapters.go`.
Strict `llama-server` builds reject description-only JSON Schema nodes unless they
are made concrete.

If Claude Code exits near 60 seconds, confirm both pieces are present:

```bash
FAK_DOGFOOD_TIMEOUT_S=900
```

and a current `fak serve` build with Anthropic SSE `ping` events in
`internal/gateway/messages.go`.

If the turn is slow but alive, that is expected for a full Claude Code prompt on a
local 27B GGUF model. The known-good probe took about 218 seconds and 25.6K input
tokens.

If your shell says `fak: command not found` or "`fak serve` is not a command", run:

```bash
fak/scripts/dogfood-claude.sh --install
```

Then open a new shell if your current shell has not picked up the bin dir. The
repo-local fallback is:

```bash
go -C fak run ./cmd/fak serve --addr 127.0.0.1:8080 --provider openai \
  --base-url http://127.0.0.1:8131/v1 \
  --model lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M
```

For Claude Code dogfood, prefer `fak-qwen36-claude`; it builds and invokes the
repo-local binary automatically.

## Cross-References

- `fak/DOGFOOD-CLAUDE.md` documents the general Claude Code dogfood launcher,
  policy floor, isolation, and live evidence.
- `docs/qwen36-surface-runbook.md` documents the Qwen3.6 model-server and fak
  surface smoke workflow.
- `tools/grafana/README.md` documents the Prometheus/Grafana stack, including the
  dedicated **FAK Dogfood Slow Requests** dashboard.
- `fak/experiments/qwen36/mac-local-qwen36-surfaces.md` records the Qwen3.6
  `agent`, `gateway-openai`, and `mcp-http` surface smoke.
- `fak/experiments/agent-live/dogfood-claude-probe.json` is the committed Claude
  Code dogfood witness.

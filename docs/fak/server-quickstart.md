---
title: "fak serve: Tool-Adjudicating Gateway Quickstart"
description: "Start a fak serve gateway: an OpenAI-compatible proxy that adjudicates tool calls and enforces a default-deny permission floor before tools run. Quick setup."
---

# fak Server Quickstart

This guide covers how to start a `fak serve` gateway in common scenarios. `fak serve` is an OpenAI-compatible gateway that adjudicates tool calls — it runs in front of any model (local or remote) and enforces a permission floor before tools execute.

## Prerequisites

- **Go 1.26+** if building from source (auto-fetched via `GOTOOLCHAIN=auto`)
- **An OpenAI-compatible model server** for proxy mode (Ollama, vLLM, llama.cpp, or a cloud API)
- **For in-kernel serving**: GGUF weights (a separate tokenizer is optional — fak uses the GGUF's embedded tokenizer by default; see "In-kernel model" below)

## Quick install (no Go required)

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

Or download from [releases](https://github.com/anthony-chaudhary/fak/releases/latest).

---

## Scenario 1: Default startup (simplest)

Front a local Ollama server with minimal config:

```bash
# Terminal 1: Start Ollama
ollama serve
ollama pull qwen2.5:1.5b

# Terminal 2: Start fak in front of it
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b
```

Verify it's running:

```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"mock","model":"qwen2.5:1.5b","ok":true}

curl -s http://127.0.0.1:8080/v1/models
# {"object":"list","data":[{"id":"qwen2.5:1.5b","object":"model",...}]}
```

That's it — `fak serve` is now adjudicating tool calls on `/v1/chat/completions`.

---

## Scenario 2: Custom config (policy + auth)

For a network-facing deployment, add authentication and a custom capability floor:

```bash
# 1. Generate a secret key
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"

# 2. Start from the default policy and customize it
fak policy --dump > policy.json
# Edit policy.json to allow your tools, deny dangerous ones

# 3. Validate the policy
fak policy --check policy.json

# 4. Start with auth + policy
fak serve --addr 0.0.0.0:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy policy.json \
  --require-key-env FAK_GATEWAY_KEY
```

Now every request (except `/healthz`) requires `Authorization: Bearer $FAK_GATEWAY_KEY` or `x-api-key: $FAK_GATEWAY_KEY`.

**Reload policy without restarting:**

```bash
# Edit policy.json in place, then:
curl -X POST http://127.0.0.1:8080/v1/fak/policy/reload \
  -H "Authorization: Bearer $FAK_GATEWAY_KEY"
```

---

## Scenario 3: Production-style startup

For a production deployment with timeouts, monitoring, and upstream API keys:

```bash
# Upstream API key (if using a hosted provider)
export OPENAI_API_KEY="sk-..."
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"

# Longer timeouts for slow models
export FAK_HTTP_WRITE_TIMEOUT_S=600
export FAK_PLANNER_TIMEOUT_S=600

fak serve --addr 0.0.0.0:8080 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY \
  --policy policy.json \
  --require-key-env FAK_GATEWAY_KEY
```

**Available env vars for tuning:**

| Env var | Default | Purpose |
|--------|---------|---------|
| `FAK_HTTP_READ_TIMEOUT_S` | 30 | Max time to read request body |
| `FAK_HTTP_WRITE_TIMEOUT_S` | 90 | Max time for whole handler (must be ≥ planner timeout) |
| `FAK_HTTP_IDLE_TIMEOUT_S` | 120 | Max idle time on keep-alive |
| `FAK_PLANNER_TIMEOUT_S` | 60 | Max time for upstream model request (clamped to [5, 3600]) |
| `FAK_RATELIMIT_MAX_CALLS` | 0 (unlimited) | Rate limit: max calls per window |
| `FAK_AUDIT_JOURNAL` | unset | Path to `.jsonl` audit log |

---

## Scenario 4: In-kernel model (no upstream server)

`fak serve` can run a model directly using the in-kernel engine. Requires GGUF weights; a separate `--tokenizer` is optional — the GGUF's embedded tokenizer is used by default:

```bash
# Small Qwen2.5 GGUF — zero-config: the embedded tokenizer is loaded automatically
fak serve --addr 127.0.0.1:8080 \
  --gguf ~/.cache/fak-models/gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --model qwen2.5-0.5b

# Large model with the direct-resident-Q4_K decode lever (FAK_Q4K=1).
# --tokenizer is optional here too; pass it only to override the embedded tokenizer.
FAK_Q4K=1 fak serve --addr 127.0.0.1:8080 \
  --gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  --model qwen3.6-27b-q4k
```

This serves both `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic) directly from the in-kernel model — no separate model server needed.

---

## Scenario 5: Cloud provider (OpenAI, Anthropic, Gemini, Xai)

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
fak serve --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
fak serve --addr 127.0.0.1:8080 \
  --provider anthropic \
  --base-url https://api.anthropic.com \
  --model claude-3-5-sonnet-20241022 \
  --api-key-env ANTHROPIC_API_KEY

# Gemini (uses native wire, not OpenAI-compatible)
fak serve --addr 127.0.0.1:8080 \
  --provider gemini \
  --base-url https://generativelanguage.googleapis.com/v1beta \
  --model gemini-2.5-flash \
  --api-key-env GEMINI_API_KEY
```

---

## Verifying it's running

```bash
# Health check (always unauthenticated)
curl http://127.0.0.1:8080/healthz

# List advertised models
curl http://127.0.0.1:8080/v1/models

# Metrics (Prometheus format)
curl http://127.0.0.1:8080/metrics

# Adjudicated tool call via /v1/fak/syscall
curl -X POST http://127.0.0.1:8080/v1/fak/syscall \
  -H 'Content-Type: application/json' \
  -d '{"tool":"read_file","arguments":{"path":"test.txt"}}'
```

---

## Common CLI flags

| Flag | Description |
|------|-------------|
| `--addr` | HTTP listen address (default `127.0.0.1:8080`) |
| `--stdio` | Serve MCP over stdin/stdout instead of HTTP |
| `--provider` | Upstream provider: `openai`, `anthropic`, `gemini`, `xai` |
| `--base-url` | Upstream base URL (empty = offline mock) |
| `--model` | Model ID advertised by `/v1/models` |
| `--api-key-env` | Env var holding upstream API key |
| `--gguf` | Path to GGUF weights (in-kernel mode) |
| `--tokenizer` | Path to tokenizer.json or its directory |
| `--policy` | Capability-floor manifest file |
| `--policy-check` | Validate policy and exit |
| `--require-key-env` | Env var holding bearer token for auth |
| `--engine` | Engine ID for `fak_syscall` (default `inkernel`) |

---

## Key routes

| Route | Auth | Purpose |
|-------|------|---------|
| `GET /healthz` | No | Liveness check |
| `GET /v1/models` | Bearer* | List models |
| `POST /v1/chat/completions` | Bearer* | OpenAI-compatible chat (adjudicated) |
| `POST /v1/messages` | Bearer* | Anthropic Messages (adjudicated) |
| `POST /v1/fak/syscall` | Bearer* | Run adjudicated tool call |
| `POST /v1/fak/policy/reload` | Bearer* | Reload policy file |
| `GET /metrics` | Bearer* | Prometheus metrics |
| `POST /mcp` | Bearer* | MCP-over-HTTP |

*Requires bearer token when `--require-key-env` is set.

---

## Example: Minimal policy

```json
{
  "version": "fak-policy/v1",
  "allow": ["search_web", "create_ticket"],
  "allow_prefix": ["read_", "get_"],
  "deny": {
    "delete_account": "POLICY_BLOCK",
    "exfiltrate": "SECRET_EXFIL"
  },
  "self_modify_globs": [".git/", "policy.json"],
  "redact_fields": ["password", "secret", "api_key"]
}
```

Use `fak policy --check policy.json` to validate before deploying.

---

## Docker

```bash
docker build -t fak https://github.com/anthony-chaudhary/fak.git
docker run --rm -p 8080:8080 fak serve --addr 0.0.0.0:8080 \
  --base-url http://host.docker.internal:11434/v1 \
  --model qwen2.5:1.5b
```

---

## See also

- [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — Full policy schema and refusal vocabulary
- [`docs/serve-config.md`](../../docs/serve-config.md) — Details on timeouts, auth, and reloading
- [`fak/GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md) — Full getting started guide
- [`scripts/dogfood-claude.sh`](https://github.com/anthony-chaudhary/fak/blob/main/scripts/dogfood-claude.sh) — One-command local model + Claude Code

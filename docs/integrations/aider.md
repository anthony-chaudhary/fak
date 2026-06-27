---
title: "Aider + fak: governed CLI coding agent"
description: "Wire fak as a tool-governance layer for Aider, the CLI-based AI pair programmer, adding capability-floor enforcement and quarantine protection."
---

# Aider + fak Integration Guide

This guide shows how to use `fak` as a tool-governance layer for [Aider](https://aider.chat/), the CLI-based AI pair programmer. Every command Aider proposes is evaluated by the kernel before it executes — dangerous calls are dropped, malformed calls are repaired, and policy violations are refused.

## Overview

```
┌──────────────┐   OpenAI / Anthropic   ┌────────────────────────┐
│     Aider    │ ──────────────────────▶ │  fak serve (gateway)   │
│   (CLI)      │ ◀──── SSE stream ─────  │  adjudicates tools     │
└──────────────┘                        └────────────────────────┘
       ▲                                                 │
       │ OPENAI_API_BASE / ANTHROPIC_BASE_URL           │
       │ (points at fak)                                ▼
       │                                        ┌───────────────┐
       │                                        │  Local Model  │
       │                                        │ or Cloud API  │
       │                                        └───────────────┘
```

**The gateway sits between Aider and the model:**

- **Aider → fak:** Aider sends API requests with proposed file edits and commands
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → Aider:** Returns results, with the kernel's decisions applied

**Result:** Aider can edit your codebase, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

---

## Prerequisites

### 1. Install fak

```bash
# From the repo (the Go module is the repo root)
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak

# Or via the installer
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

Verify installation:
```bash
./fak version
```

### 2. Install Aider

```bash
pip install aider-chat
```

Verify installation:
```bash
aider --version
```

### 3. Choose your upstream model

Aider can connect to `fak` in two modes:

- **Proxy mode:** `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM, etc.)
- **In-kernel mode:** `fak` serves its own fused model

For development, proxy mode is recommended as it gives you full model capabilities while still enforcing tool governance.

---

## Quick Start: One-command setup

The fastest way to put the kernel in front of Aider:

```bash
# Terminal 1: Start the fak gateway
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json

# Terminal 2: Run Aider through fak
export OPENAI_API_BASE="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
aider --model openai/qwen2.5:1.5b
```

Aider now runs through the capability floor — every file edit, git operation, or shell command is adjudicated before execution.

---

## Method 1: OpenAI-compatible provider (Recommended)

Aider's primary integration path is through OpenAI-compatible APIs. `fak` provides an OpenAI-compatible `/v1/chat/completions` endpoint.

### Step 1: Start the fak HTTP gateway

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json
```

Verify health:
```bash
curl http://127.0.0.1:8080/healthz
# {"ok":true,"model":"qwen2.5:1.5b","engine":"inkernel"}
```

### Step 2: Configure Aider

**Via environment variables (Recommended):**

```bash
export OPENAI_API_BASE="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
export OPENAI_MODEL="qwen2.5:1.5b"
```

**Via configuration file:**

In `~/.aider.conf.yml`:
```yaml
openai-api-base: http://127.0.0.1:8080/v1
model: openai/qwen2.5:1.5b
api-key: fak-local
```

**Via CLI flags:**

```bash
aider \
  --openai-api-base http://127.0.0.1:8080/v1 \
  --model openai/qwen2.5:1.5b \
  --api-key fak-local
```

### Step 3: Run Aider

```bash
aider  # Starts Aider with your git repo and the governed model
```

Aider's file edits and commands now flow through `fak`:
```
Aider → fak /v1/chat/completions → adjudication → upstream model
                                            ↓
                                      capability floor
                                            ↓
                                    allowed/denied/transformed
                                            ↓
                                      Aider (with filtered results)
```

---

## Method 2: govern a real Claude model upstream

To put the floor in front of an actual Anthropic Claude model, keep the **upstream**
provider Anthropic — fak proxies to `api.anthropic.com` — but wire Aider to fak the
same way as Method 1, over the OpenAI-compatible endpoint fak always exposes. Aider is a
LiteLLM-based client that speaks the OpenAI wire; it has no Anthropic-specific flag, so
the connection is identical to Method 1 regardless of what fak proxies upstream.

### Step 1: Start the fak gateway with the Anthropic upstream

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider anthropic \
  --base-url https://api.anthropic.com/v1 \
  --api-key-env ANTHROPIC_API_KEY \
  --model claude-sonnet-4-20250514 \
  --policy examples/customer-support-readonly-policy.json
```

### Step 2: Configure Aider (OpenAI-compatible, same as Method 1)

```bash
export OPENAI_API_BASE="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
```

### Step 3: Run Aider

Name the served model with the `openai/` prefix so Aider routes it over the OpenAI wire
to fak (which proxies to the Claude model upstream):

```bash
aider --model openai/claude-sonnet-4-20250514
```

---

## Creating a Capability Floor for Aider

A capability floor defines which operations Aider may perform. Start from the built-in default:

```bash
# Dump the default policy as a starting point
./fak policy --dump > aider-policy.json
```

### Example: Safe coding agent policy

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": [
    "read_file",
    "write_file",
    "list_directory",
    "git_diff",
    "git_log",
    "git_show",
    "git_blame"
  ],
  "allow_prefix": [
    "read_",
    "get_",
    "search_",
    "list_",
    "git_",
    "lint_",
    "format_"
  ],
  "deny": {
    "run_command": "POLICY_BLOCK",
    "git_push": "POLICY_BLOCK",
    "git_reset": "POLICY_BLOCK",
    "git_clean": "POLICY_BLOCK",
    "delete_file": "POLICY_BLOCK"
  },
  "self_modify_globs": [
    ".git/",
    "aider-policy.json",
    ".env",
    "id_rsa"
  ],
  "arg_rules": [
    {
      "tool": "read_file",
      "arg": "path",
      "deny_regex": ".*\\.env$",
      "reason": "SECRET_EXFIL"
    },
    {
      "tool": "run_command",
      "arg": "command",
      "deny_regex": "rm\\s+-rf|sudo|git\\s+push",
      "reason": "POLICY_BLOCK"
    }
  ]
}
```

Validate before using:
```bash
./fak policy --check aider-policy.json
```

---

## Common Patterns for Aider Workflows

### Pattern 1: Read-only code review

Configure `fak` to allow reads but block writes and destructive Git operations:

```json
{
  "allow": ["read_file", "list_directory", "git_diff", "git_log"],
  "deny": {
    "write_file": "POLICY_BLOCK",
    "run_command": "POLICY_BLOCK",
    "git_push": "POLICY_BLOCK"
  }
}
```

### Pattern 2: Safe refactoring with Git awareness

Allow safe file operations but block destructive Git operations:

```json
{
  "allow_prefix": ["read_", "write_", "git_diff", "git_log", "git_show", "git_blame"],
  "deny": {
    "git_push": "POLICY_BLOCK",
    "git_reset": "POLICY_BLOCK",
    "git_clean": "POLICY_BLOCK",
    "run_command": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/"]
}
```

### Pattern 3: Quarantine for external tool results

Protect against poisoned responses from external APIs:

```bash
# Enable quarantine on the gateway
./fak serve --addr 127.0.0.1:8080 \
  --base-url https://api.openai.com/v1 \
  --policy policy.json \
  --vdso=true  # Enables content-addressed cache and quarantine
```

If an external tool returns suspicious content (e.g., injection attempts), `fak` automatically quarantines it, preventing it from entering Aider's context.

---

## Monitoring and Debugging

### Health checks

```bash
curl http://127.0.0.1:8080/healthz
```

### Metrics

```bash
curl http://127.0.0.1:8080/metrics
```

Key metrics:
- `fak_gateway_time_to_ready_seconds` - Startup time
- `fak_vdso_hit_rate` - Cache hit rate
- `fak_gateway_operations_total{verdict="DENY"}` - Denied calls (by reason label)
- `fak_kernel_quarantines_total` - Quarantined results

### Debugging a denied operation

When Aider reports a denied operation, reproduce it offline:

```bash
./fak preflight \
  --tool write_file \
  --args '{"path":"test.txt","content":"test"}' \
  --policy your-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

Use `--explain` to get a detailed breakdown:

```bash
./fak preflight --explain \
  --tool run_command \
  --args '{"command":"rm -rf /tmp"}' \
  --policy your-policy.json
```

---

## Troubleshooting

### Aider can't connect to the gateway

1. Verify `fak` is running:
   ```bash
   curl http://127.0.0.1:8080/healthz
   ```

2. Check your environment variables:
   ```bash
   echo $OPENAI_API_BASE
   # Should output: http://127.0.0.1:8080/v1
   ```

3. Verify the policy file exists:
   ```bash
   ./fak policy --check your-policy.json
   ```

### All operations are being denied

1. Check the policy's posture:
   - `posture: "fail_closed"` (default) denies everything not explicitly allowed
   - Ensure your tools are in `allow` or match an `allow_prefix`

2. Test a specific call:
   ```bash
   ./fak preflight --tool read_file --args '{"path":"test.txt"}' --policy your-policy.json
   ```

### Slow first response

Expected on large local models — the Aider prompt is ~25K tokens. Subsequent requests are faster if you enable `--vdso=true` (content-addressed caching).

---

## Advanced Usage

### Authentication

For production use, require an API key:

```bash
./fak serve \
  --addr 0.0.0.0:8080 \
  --base-url ... \
  --model ... \
  --require-key-env FAK_TOKEN
```

Aider sends `x-api-key:` (OpenAI SDK), which `fak` honors.

### Cloud providers

```bash
# OpenAI
./fak serve \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4

# Anthropic
./fak serve \
  --provider anthropic \
  --base-url https://api.anthropic.com/v1 \
  --api-key-env ANTHROPIC_API_KEY \
  --model claude-sonnet-4-20250514
```

### Local models

```bash
# Ollama
./fak serve \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b

# vLLM
./fak serve \
  --provider openai \
  --base-url http://localhost:8000/v1 \
  --model qwen2.5-coder:7b
```

---

## Cross-references

- **Integration index**: [README.md](README.md) — universal recipe and which-agent routing
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md) — full field survey
- **Policy schema**: [../../POLICY.md](../../POLICY.md) — authoring capability floors
- **Aider docs**: [https://aider.chat](https://aider.chat) — official Aider documentation
- ** fak architecture**: [../../ARCHITECTURE.md](../../ARCHITECTURE.md) — kernel internals

---

## License

Apache-2.0
---
title: "Continue + fak: governed VS Code AI assistant"
description: "Wire fak as a tool-governance layer for Continue, the VS Code AI assistant extension, adding capability-floor enforcement and quarantine protection."
---

# Continue + fak Integration Guide

This guide shows how to use `fak` as a tool-governance layer for [Continue.dev](https://continue.dev/), the VS Code AI assistant extension. Every tool call Continue proposes is evaluated by the kernel before it executes — dangerous calls are dropped, malformed calls are repaired, and policy violations are refused.

## Overview

```
┌──────────────────┐   OpenAI / Anthropic   ┌────────────────────────┐
│  Continue (VS    │ ──────────────────────▶ │  fak serve (gateway)   │
│   Code extension)│ ◀──── SSE stream ─────  │  adjudicates tools     │
└──────────────────┘                        └────────────────────────┘
         ▲                                                 │
         │ apiBase (points at fak)                       ▼
         │                                        ┌───────────────┐
         │                                        │  Local Model  │
         │                                        │ or Cloud API  │
         │                                        └───────────────┘
```

**The gateway sits between Continue and the model:**

- **Continue → fak:** Continue sends API requests with proposed file edits and commands
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → Continue:** Returns results, with the kernel's decisions applied

**Result:** Continue can edit your codebase, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

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

### 2. Install Continue

Install the Continue extension from the VS Code marketplace:
- Search "Continue" in the Extensions panel (Ctrl+Shift+X)
- Or install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue)

### 3. Choose your upstream model

Continue can connect to `fak` in two modes:

- **Proxy mode:** `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM, etc.)
- **In-kernel mode:** `fak` serves its own fused model

For development, proxy mode is recommended as it gives you full model capabilities while still enforcing tool governance.

---

## Quick Start: One-command setup

The fastest way to put the kernel in front of Continue:

```bash
# Terminal 1: Start the fak gateway
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json

# Terminal 2: Configure Continue
mkdir -p ~/.continue
cat > ~/.continue/config.yaml << EOF
models:
  - provider: openai
    apiBase: http://127.0.0.1:8080/v1
    apiKey: fak-local
    model: qwen2.5:1.5b
EOF

# Open VS Code (Continue will use the new config)
code .
```

Continue now runs through the capability floor — every file edit, command, and tool call is adjudicated before execution.

---

## Method 1: OpenAI-compatible provider (Recommended)

Continue's primary integration path is through OpenAI-compatible APIs. `fak` provides an OpenAI-compatible `/v1/chat/completions` endpoint.

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

### Step 2: Configure Continue

**Via config file (Recommended):**

Create or edit `~/.continue/config.yaml`:

```yaml
models:
  - provider: openai
    apiBase: http://127.0.0.1:8080/v1
    apiKey: fak-local
    model: qwen2.5:1.5b

# Optional: Configure slash commands and context rules
contextRules:
  - id: fak-governed
    description: "All commands run through the fak gateway"
    provider: openai
    modelId: qwen2.5:1.5b
```

**Via deprecated config.json (still supported):**

Create or edit `~/.continue/config.json`:

```json
{
  "models": [{
    "provider": "openai",
    "apiBase": "http://127.0.0.1:8080/v1",
    "apiKey": "fak-local",
    "model": "qwen2.5:1.5b"
  }]
}
```

### Step 3: Reload Continue

1. Open VS Code
2. Press `Ctrl+Shift+P` (Cmd+Shift+P on Mac)
3. Run "Continue: Reload Configuration"
4. Open the Continue sidebar (click the Continue icon in the sidebar)

Continue's file edits and commands now flow through `fak`:
```
Continue → fak /v1/chat/completions → adjudication → upstream model
                                            ↓
                                      capability floor
                                            ↓
                                    allowed/denied/transformed
                                            ↓
                                      Continue (with filtered results)
```

---

## Method 2: Anthropic provider

Continue also supports Anthropic's Messages API for Claude models.

### Step 1: Start the fak gateway with Anthropic provider

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider anthropic \
  --base-url https://api.anthropic.com/v1 \
  --api-key-env ANTHROPIC_API_KEY \
  --model claude-sonnet-4-20250514 \
  --policy examples/customer-support-readonly-policy.json
```

### Step 2: Configure Continue

Edit `~/.continue/config.yaml`:

```yaml
models:
  - provider: anthropic
    apiBase: http://127.0.0.1:8080
    apiKey: fak-local
    model: claude-sonnet-4-20250514
```

### Step 3: Reload Continue

Press `Ctrl+Shift+P` → "Continue: Reload Configuration"

---

## Creating a Capability Floor for Continue

A capability floor defines which operations Continue may perform. Start from the built-in default:

```bash
# Dump the default policy as a starting point
./fak policy --dump > continue-policy.json
```

### Example: Safe coding assistant policy

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": [
    "read_file",
    "write_file",
    "list_directory",
    "search_files",
    "get_definition",
    "git_diff",
    "git_log"
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
    ".vscode/",
    "continue-policy.json",
    ".env",
    "id_rsa"
  ],
  "redact_fields": [
    "password",
    "secret",
    "api_key",
    "token"
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
./fak policy --check continue-policy.json
```

Use the custom policy:
```bash
./fak serve --policy continue-policy.json ...
```

---

## Common Patterns for Continue Workflows

### Pattern 1: Read-only code review

Configure `fak` to allow reads but block writes and destructive Git operations:

```json
{
  "allow": ["read_file", "list_directory", "search_files", "git_diff", "git_log"],
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
  "allow_prefix": ["read_", "write_", "search_", "git_diff", "git_log", "git_show", "git_blame"],
  "deny": {
    "git_push": "POLICY_BLOCK",
    "git_reset": "POLICY_BLOCK",
    "git_clean": "POLICY_BLOCK",
    "run_command": "POLICY_BLOCK"
  },
  "self_modify_globs": [".git/", ".vscode/"]
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

If an external tool returns suspicious content (e.g., injection attempts), `fak` automatically quarantines it, preventing it from entering Continue's context.

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

When Continue reports a denied operation, reproduce it offline:

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

### Checking Continue's model connection

In VS Code:
1. Open Continue sidebar
2. Click the gear icon (settings)
3. Verify the model connection shows as connected
4. Check that the base URL points to `http://127.0.0.1:8080/v1`

---

## Troubleshooting

### Continue can't connect to the gateway

1. Verify `fak` is running:
   ```bash
   curl http://127.0.0.1:8080/healthz
   ```

2. Check your Continue config:
   ```bash
   cat ~/.continue/config.yaml
   # Should show apiBase: http://127.0.0.1:8080/v1
   ```

3. Reload Continue configuration:
   - Press `Ctrl+Shift+P`
   - Run "Continue: Reload Configuration"

4. Check VS Code output logs:
   - Open "View" → "Output"
   - Select "Continue" from the dropdown
   - Look for connection errors

### All operations are being denied

1. Check the policy's posture:
   - `posture: "fail_closed"` (default) denies everything not explicitly allowed
   - Ensure your tools are in `allow` or match an `allow_prefix`

2. Test a specific call:
   ```bash
   ./fak preflight --tool read_file --args '{"path":"test.txt"}' --policy your-policy.json
   ```

3. Check the gateway logs:
   ```bash
   # Add --log to fak serve
   ./fak serve --log <tmp>/fak-continue.log ...
   tail -f <tmp>/fak-continue.log
   ```

### Slow first response

Expected on large local models — the Continue prompt is ~25K tokens. Subsequent requests are faster if you enable `--vdso=true` (content-addressed caching).

### Configuration file not being picked up

1. Verify the config file location:
   - On Linux/Mac: `~/.continue/config.yaml` or `~/.continue/config.json`
   - On Windows: `%USERPROFILE%\.continue\config.yaml`

2. Check file permissions:
   ```bash
   ls -la ~/.continue/config.yaml
   # Should be readable by your user
   ```

3. Try the JSON format if YAML isn't working:
   ```json
   {
     "models": [{
       "provider": "openai",
       "apiBase": "http://127.0.0.1:8080/v1",
       "apiKey": "fak-local",
       "model": "qwen2.5:1.5b"
     }]
   }
   ```

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

Continue sends `api-key:` in the config, which `fak` honors.

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

### Multiple models with different policies

```yaml
models:
  - provider: openai
    apiBase: http://127.0.0.1:8080/v1
    apiKey: fak-local
    model: qwen2.5:1.5b
    title: "Qwen (Strict)"

  - provider: openai
    apiBase: http://127.0.0.1:8090/v1
    apiKey: fak-local
    model: gpt-4
    title: "GPT-4 (Permissive)"
```

Run two separate `fak serve` instances with different policies on ports 8080 and 8090.

---

## Cross-references

- **Integration index**: [README.md](README.md) — universal recipe and which-agent routing
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md) — full field survey
- **Policy schema**: [../../POLICY.md](../../POLICY.md) — authoring capability floors
- **Continue docs**: [https://docs.continue.dev](https://docs.continue.dev) — official Continue documentation
- **VS Code docs**: [https://code.visualstudio.com](https://code.visualstudio.com) — VS Code extension development
- **fak architecture**: [../../ARCHITECTURE.md](../../ARCHITECTURE.md) — kernel internals

---

## License

Apache-2.0
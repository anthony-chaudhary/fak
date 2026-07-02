---
title: "Cline + fak: governed VS Code AI assistant"
description: "Wire fak as a tool-governance layer for Cline, the VS Code AI assistant extension, adding capability-floor enforcement and quarantine protection."
---

# Cline + fak Integration Guide

This guide shows how to use `fak` as a tool-governance layer for [Cline](https://cline.bot/), the VS Code AI assistant extension. Every tool call Cline proposes is evaluated by the kernel before it executes — dangerous calls are dropped, malformed calls are repaired, and policy violations are refused.

## Overview

```
┌──────────────────┐   OpenAI / Anthropic   ┌────────────────────────┐
│   Cline (VS      │ ──────────────────────▶ │  fak serve (gateway)   │
│   Code extension)│ ◀──── SSE stream ─────  │  adjudicates tools     │
└──────────────────┘                        └────────────────────────┘
         ▲                                                 │
         │ Base URL (points at fak)                      ▼
         │                                        ┌───────────────┐
         │                                        │  Local Model  │
         │                                        │ or Cloud API  │
         │                                        └───────────────┘
```

**The gateway sits between Cline and the model:**

- **Cline → fak:** Cline sends API requests with proposed file edits and commands
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → Cline:** Returns results, with the kernel's decisions applied

**Result:** Cline can edit your codebase, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

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

### 2. Install Cline

Install the Cline extension from the VS Code marketplace:
- Search "Cline" in the Extensions panel (Ctrl+Shift+X)
- Or install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.cline)

### 3. Choose your upstream model

Cline can connect to `fak` in two modes:

- **Proxy mode:** `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM, etc.)
- **In-kernel mode:** `fak` serves its own fused model

For development, proxy mode is recommended as it gives you full model capabilities while still enforcing tool governance.

---

## Quick Start: One-command setup

The fastest way to put the kernel in front of Cline:

```bash
# Terminal 1: Start the fak gateway
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json

# Terminal 2: Open VS Code
code .
```

Then configure Cline in the VS Code UI:
1. Open Cline sidebar (click the Cline icon)
2. Click the gear icon (provider settings)
3. Select "OpenAI Compatible" as the API provider
4. Base URL: `http://127.0.0.1:8080/v1`
5. API Key: `fak-local`
6. Model ID: `qwen2.5:1.5b`

Cline now runs through the capability floor — every file edit, command, and tool call is adjudicated before execution.

---

## Method 1: OpenAI-compatible provider (Recommended)

Cline's primary integration path is through OpenAI-compatible APIs. `fak` provides an OpenAI-compatible `/v1/chat/completions` endpoint.

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

### Step 2: Configure Cline via VS Code UI

1. Open VS Code
2. Click the Cline icon in the sidebar (or press `Cmd+Shift+C` / `Ctrl+Shift+C`)
3. Click the gear icon (⚙️) to open provider settings
4. Configure the OpenAI Compatible provider:
   - **API Provider**: OpenAI Compatible
   - **Base URL**: `http://127.0.0.1:8080/v1`
   - **API Key**: `fak-local`
   - **Model ID**: `qwen2.5:1.5b`
5. Click "Save"

Cline's file edits and commands now flow through `fak`:
```
Cline → fak /v1/chat/completions → adjudication → upstream model
                                          ↓
                                    capability floor
                                          ↓
                                  allowed/denied/transformed
                                          ↓
                                    Cline (with filtered results)
```

### Alternative: Configure via VS Code settings.json

You can also configure Cline via VS Code's settings.json:

1. Open VS Code settings (`Ctrl+,` or `Cmd+,`)
2. Search for "cline"
3. Add these settings:

```json
{
  "cline.apiProvider": "openai-compatible",
  "cline.openAiCompatibleBaseUrl": "http://127.0.0.1:8080/v1",
  "cline.openAiCompatibleApiKey": "fak-local",
  "cline.openAiCompatibleModelId": "qwen2.5:1.5b"
}
```

---

## Method 2: Anthropic provider

Cline also supports Anthropic's Messages API for Claude models.

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

### Step 2: Configure Cline via VS Code UI

1. Open Cline provider settings (gear icon)
2. Configure the Anthropic provider:
   - **API Provider**: Anthropic
   - **Base URL**: `http://127.0.0.1:8080` (not including `/v1`)
   - **API Key**: `fak-local`
   - **Model ID**: `claude-sonnet-4-20250514`
3. Click "Save"

---

## Creating a Capability Floor for Cline

A capability floor defines which operations Cline may perform. Start from the built-in default:

```bash
# Dump the default policy as a starting point
./fak policy --dump > cline-policy.json
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
    "cline-policy.json",
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
./fak policy --check cline-policy.json
```

Use the custom policy:
```bash
./fak serve --policy cline-policy.json ...
```

---

## Common Patterns for Cline Workflows

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

If an external tool returns suspicious content (e.g., injection attempts), `fak` automatically quarantines it, preventing it from entering Cline's context.

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

When Cline reports a denied operation, reproduce it offline:

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

### Checking Cline's model connection

In VS Code:
1. Open Cline sidebar
2. Click the gear icon (settings)
3. Verify the model connection shows as connected
4. Check that the Base URL points to `http://127.0.0.1:8080/v1`

---

## Troubleshooting

### Cline can't connect to the gateway

1. Verify `fak` is running:
   ```bash
   curl http://127.0.0.1:8080/healthz
   ```

2. Check Cline's provider settings:
   - Open Cline sidebar
   - Click the gear icon
   - Verify Base URL is `http://127.0.0.1:8080/v1` (includes `/v1` for OpenAI Compatible)
   - Verify API Key is set

3. Check VS Code output logs:
   - Open "View" → "Output"
   - Select "Cline" from the dropdown
   - Look for connection errors

4. Verify the provider type matches the gateway:
   - If `fak serve --provider openai`, Cline should use "OpenAI Compatible"
   - If `fak serve --provider anthropic`, Cline should use "Anthropic"

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
   ./fak serve --log <tmp>/fak-cline.log ...
   tail -f <tmp>/fak-cline.log
   ```

### Slow first response

Expected on large local models — the Cline prompt is ~25K tokens. Subsequent requests are faster if you enable `--vdso=true` (content-addressed caching).

### Settings not being applied

1. Restart VS Code after changing settings
2. Verify settings.json format is valid JSON
3. Check for conflicting workspace settings (`.vscode/settings.json`)

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

Cline sends the API key from provider settings, which `fak` honors.

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

Cline supports switching between providers. Run two separate `fak serve` instances:

```bash
# Terminal 1: Strict policy
./fak serve --addr 127.0.0.1:8080 --policy strict.json ...

# Terminal 2: Permissive policy
./fak serve --addr 127.0.0.1:8090 --policy permissive.json ...
```

Then in Cline, configure both as separate providers and switch between them as needed.

---

## Cross-references

- **Integration index**: [README.md](README.md) — universal recipe and which-agent routing
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md) — full field survey
- **Policy schema**: [../../POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — authoring capability floors
- **Cline docs**: [https://docs.cline.bot](https://docs.cline.bot) — official Cline documentation
- **VS Code docs**: [https://code.visualstudio.com](https://code.visualstudio.com) — VS Code extension development
- **fak architecture**: [../../ARCHITECTURE.md](https://github.com/anthony-chaudhary/fak/blob/main/ARCHITECTURE.md) — kernel internals

---

## License

Apache-2.0
---
title: "VS Code + GitHub Copilot + fak: governed AI-assisted development"
description: "Wire fak as a tool-governance layer for VS Code with GitHub Copilot, adding capability-floor enforcement and quarantine protection to AI-assisted development."
---

# VS Code + GitHub Copilot + fak Integration Guide

This guide shows how to use `fak` as a tool-governance layer for VS Code with GitHub Copilot, adding capability-floor enforcement and quarantine protection to AI-assisted development workflows.

## Overview

```
┌──────────────────┐   OpenAI / Anthropic   ┌────────────────────────┐
│  VS Code +       │ ──────────────────────▶ │  fak serve (gateway)   │
│  GitHub Copilot  │ ◀──── SSE stream ─────  │  adjudicates tools     │
└──────────────────┘                        └────────────────────────┘
          ▲                                                 │
          │ Copilot API / Proxy URL                       ▼
          │ (points at fak)                                │
          │                                        ┌───────────────┐
          │                                        │  Local Model  │
          │                                        │ or Cloud API  │
          │                                        └───────────────┘
```

**The gateway sits between VS Code + GitHub Copilot and the model:**

- **VS Code + Copilot → fak:** Sends API requests with proposed code completions and commands
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → VS Code:** Returns results, with the kernel's decisions applied

**Result:** VS Code with GitHub Copilot can assist development, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

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

### 2. Install VS Code and GitHub Copilot

- Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
- Install the GitHub Copilot extension from the VS Code marketplace
- Sign in to GitHub Copilot with your GitHub account

### 3. Choose your upstream model

VS Code with GitHub Copilot can connect to `fak` in two modes:

- **Proxy mode:** `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM, etc.)
- **In-kernel mode:** `fak` serves its own fused model

For development, proxy mode is recommended as it gives you full model capabilities while still enforcing tool governance.

---

## Method 1: GitHub Copilot via Proxy Configuration

GitHub Copilot uses OpenAI's Chat Completions API under the hood. Configure it to point at `fak`'s OpenAI-compatible endpoint.

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

### Step 2: Configure GitHub Copilot

**Via VS Code settings.json:**

1. Open VS Code settings (`Ctrl+,` or `Cmd+,`)
2. Add these settings:

```json
{
  "github.copilot.advanced": {
    "overrideProxyUrl": "http://127.0.0.1:8080/v1"
  },
  "github.copilot.inlineSuggest.enable": true,
  "github.copilot.enable": {
    "*": true
  }
}
```

**Via environment variables:**

GitHub Copilot respects standard OpenAI environment variables:

```bash
# Set before launching VS Code
export OPENAI_API_BASE="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
export OPENAI_MODEL="qwen2.5:1.5b"
```

Then launch VS Code:
```bash
code .
```

### Step 3: Reload VS Code

1. Restart VS Code for the settings to take effect
2. Open a file and trigger a code completion
3. The completion request now flows through `fak`:

```
VS Code + Copilot → fak /v1/chat/completions → adjudication → upstream model
                                                        ↓
                                                  capability floor
                                                        ↓
                                                allowed/denied/transformed
                                                        ↓
                                          VS Code (with filtered results)
```

---

## Method 2: Anthropic Claude provider

If you prefer to use Anthropic Claude models through VS Code Copilot:

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

### Step 2: Configure GitHub Copilot for Anthropic

Via VS Code settings.json:

```json
{
  "github.copilot.advanced": {
    "overrideProxyUrl": "http://127.0.0.1:8080"
  },
  "github.copilot.enable": {
    "*": true
  }
}
```

Via environment variables:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_API_KEY="fak-local"
```

---

## Creating a Capability Floor for VS Code + GitHub Copilot

A capability floor defines which operations VS Code with GitHub Copilot may perform. Start from the built-in default:

```bash
# Dump the default policy as a starting point
./fak policy --dump > vscode-copilot-policy.json
```

### Example: Safe development assistant policy

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
    "vscode-copilot-policy.json",
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
./fak policy --check vscode-copilot-policy.json
```

Use the custom policy:
```bash
./fak serve --policy vscode-copilot-policy.json ...
```

---

## Common Patterns for VS Code + GitHub Copilot Workflows

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

If an external tool returns suspicious content (e.g., injection attempts), `fak` automatically quarantines it, preventing it from entering VS Code's context.

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

When VS Code with GitHub Copilot reports a denied operation, reproduce it offline:

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

### Checking VS Code's Copilot connection

1. Open VS Code
2. Check the GitHub Copilot status indicator in the bottom status bar
3. It should show as active/connected
4. Verify the proxy URL points to `http://127.0.0.1:8080/v1`

---

## Troubleshooting

### VS Code + Copilot can't connect to the gateway

1. Verify `fak` is running:
   ```bash
   curl http://127.0.0.1:8080/healthz
   ```

2. Check VS Code settings:
   - Open VS Code settings (`Ctrl+,`)
   - Search for "copilot"
   - Verify `github.copilot.advanced.overrideProxyUrl` is set to `http://127.0.0.1:8080/v1`

3. Check environment variables (if using that method):
   ```bash
   echo $OPENAI_API_BASE
   # Should output: http://127.0.0.1:8080/v1
   ```

4. Check VS Code output logs:
   - Open "View" → "Output"
   - Select "GitHub Copilot" from the dropdown
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
   ./fak serve --log /tmp/fak-vscode.log ...
   tail -f /tmp/fak-vscode.log
   ```

### Slow code completions

Expected on large local models — the VS Code Copilot prompt is ~25K tokens. Subsequent requests are faster if you enable `--vdso=true` (content-addressed caching).

### Settings not being applied

1. Restart VS Code after changing settings.json
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

GitHub Copilot sends the API key from its configuration, which `fak` honors.

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
- **GitHub Copilot docs**: [https://docs.github.com/copilot](https://docs.github.com/copilot) — official GitHub Copilot documentation
- **VS Code docs**: [https://code.visualstudio.com](https://code.visualstudio.com) — VS Code documentation
- **fak architecture**: [../../ARCHITECTURE.md](../../ARCHITECTURE.md) — kernel internals

---

## License

Apache-2.0
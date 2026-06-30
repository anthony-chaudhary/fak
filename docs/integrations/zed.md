---
title: "Zed + fak: governed AI-native editor"
description: "Wire fak as a tool-governance layer for Zed, the AI-native editor, adding capability-floor enforcement and quarantine protection to AI-assisted development."
---

# Zed + fak Integration Guide

This guide shows how to use `fak` as a tool-governance layer for [Zed](https://zed.dev/), the AI-native editor. Every tool call Zed proposes is evaluated by the kernel before it executes — dangerous calls are dropped, malformed calls are repaired, and policy violations are refused.

## Overview

```
┌──────────────────┐   OpenAI / Anthropic   ┌────────────────────────┐
│      Zed        │ ──────────────────────▶ │  fak serve (gateway)   │
│  (AI-native IDE) │ ◀──── SSE stream ─────  │  adjudicates tools     │
└──────────────────┘                        └────────────────────────┘
          ▲                                                 │
          │ settings.json api_url                          ▼
          │ (points at fak)                                │
          │                                        ┌───────────────┐
          │                                        │  Local Model  │
          │                                        │ or Cloud API  │
          │                                        └───────────────┘
```

**The gateway sits between Zed and the model:**

- **Zed → fak:** Zed sends API requests with proposed code edits and commands
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → Zed:** Returns results, with the kernel's decisions applied

**Result:** Zed can edit your codebase, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

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

### 2. Install Zed

Download Zed from [zed.dev](https://zed.dev) and install following the official setup guide for your platform (macOS, Linux, or Windows).

### 3. Choose your upstream model

Zed can connect to `fak` in two modes:

- **Proxy mode:** `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM, etc.)
- **In-kernel mode:** `fak` serves its own fused model

For development, proxy mode is recommended as it gives you full model capabilities while still enforcing tool governance.

---

## Quick Start: One-command setup

The fastest way to put the kernel in front of Zed:

```bash
# Terminal 1: Start the fak gateway
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json

# Terminal 2: Open Zed
# Zed will read settings.json and use the configured provider
zed
```

Zed now runs through the capability floor — every file edit, command, and tool call is adjudicated before execution.

---

## Method 1: OpenAI-compatible provider (Recommended)

Zed's primary integration path is through OpenAI-compatible APIs. `fak` provides an OpenAI-compatible `/v1/chat/completions` endpoint.

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

### Step 2: Configure Zed via settings.json

Zed's configuration lives in `~/.config/zed/settings.json` (Linux), `~/Library/Application Support/Zed/settings.json` (macOS), or `%APPDATA%\Zed\settings.json` (Windows).

Add or edit the language_models section:

```json
{
  "language_models": {
    "openai_compatible": {
      "fak": {
        "api_url": "http://127.0.0.1:8080/v1",
        "available_models": [
          {
            "id": "qwen2.5:1.5b",
            "name": "Qwen 2.5 (1.5B)"
          }
        ],
        "low_speed_timeout_in_seconds": 60,
        "high_speed_timeout_in_seconds": 5
      }
    }
  }
}
```

Set the API key via environment variable (Zed reads `<PROVIDER_ID>_API_KEY`):

```bash
# Set before launching Zed
export FAK_API_KEY="fak-local"
```

Or set it in your shell profile (`.bashrc`, `.zshrc`, etc.) for persistence.

### Step 3: Configure Zed to use the fak model

Add to your `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "fak": {
        "api_url": "http://127.0.0.1:8080/v1",
        "available_models": [
          {
            "id": "qwen2.5:1.5b",
            "name": "Qwen 2.5 (1.5B)"
          }
        ],
        "low_speed_timeout_in_seconds": 60,
        "high_speed_timeout_in_seconds": 5
      }
    },
    "model": "openai_compatible/fak"
  }
}
```

Or set the model per-project in `.zed/settings.json`:

```json
{
  "language_models": {
    "model": "openai_compatible/fak"
  }
}
```

### Step 4: Reload Zed

1. If Zed is running, reload settings: `Cmd+Shift+P` → "Reload Settings"
2. Or restart Zed entirely
3. Open the AI panel (Cmd+L or Ctrl+L)
4. Select the model from the dropdown

Zed's file edits and commands now flow through `fak`:
```
Zed → fak /v1/chat/completions → adjudication → upstream model
                                          ↓
                                    capability floor
                                          ↓
                                  allowed/denied/transformed
                                          ↓
                                    Zed (with filtered results)
```

---

## Method 2: Anthropic provider

Zed also supports Anthropic's Messages API for Claude models.

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

### Step 2: Configure Zed via settings.json

Add to your `settings.json`:

```json
{
  "language_models": {
    "anthropic": {
      "api_url": "http://127.0.0.1:8080",
      "available_models": [
        {
          "id": "claude-sonnet-4-20250514",
          "name": "Claude Sonnet 4"
        }
      ],
      "low_speed_timeout_in_seconds": 60,
      "high_speed_timeout_in_seconds": 5
    },
    "model": "anthropic"
  }
}
```

Set the API key:

```bash
export ANTHROPIC_API_KEY="fak-local"
```

### Step 3: Reload Zed

Reload settings or restart Zed, then open the AI panel and verify the Claude model is selected.

---

## Method 3: Multiple providers with different policies

You can run multiple `fak serve` instances with different policies and switch between them in Zed.

```bash
# Terminal 1: Strict policy (development)
./fak serve --addr 127.0.0.1:8080 --policy strict.json ...

# Terminal 2: Permissive policy (review)
./fak serve --addr 127.0.0.1:8090 --policy permissive.json ...
```

Configure both in `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "fak-dev": {
        "api_url": "http://127.0.0.1:8080/v1",
        "available_models": [
          {
            "id": "qwen2.5:1.5b",
            "name": "Qwen (Strict)"
          }
        ]
      },
      "fak-review": {
        "api_url": "http://127.0.0.1:8090/v1",
        "available_models": [
          {
            "id": "qwen2.5:1.5b",
            "name": "Qwen (Permissive)"
          }
        ]
      }
    }
  }
}
```

Switch between models via the AI panel dropdown or per-project settings.

---

## Creating a Capability Floor for Zed

A capability floor defines which operations Zed may perform. Start from the built-in default:

```bash
# Dump the default policy as a starting point
./fak policy --dump > zed-policy.json
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
    ".zed/",
    "zed-policy.json",
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
./fak policy --check zed-policy.json
```

Use the custom policy:
```bash
./fak serve --policy zed-policy.json ...
```

---

## Common Patterns for Zed Workflows

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
  "self_modify_globs": [".git/", ".zed/"]
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

If an external tool returns suspicious content (e.g., injection attempts), `fak` automatically quarantines it, preventing it from entering Zed's context.

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

When Zed reports a denied operation, reproduce it offline:

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

### Checking Zed's model connection

1. Open Zed
2. Open the AI panel (Cmd+L or Ctrl+L)
3. Check that the model dropdown shows your configured model
4. Verify the API URL points to `http://127.0.0.1:8080/v1`

---

## Troubleshooting

### Zed can't connect to the gateway

1. Verify `fak` is running:
   ```bash
   curl http://127.0.0.1:8080/healthz
   ```

2. Check Zed's settings.json:
   ```bash
   cat ~/.config/zed/settings.json | grep api_url
   # Should show: "api_url": "http://127.0.0.1:8080/v1"
   ```

3. Check the API key environment variable:
   ```bash
   echo $FAK_API_KEY
   # Should output: fak-local
   ```

4. Check Zed's output logs:
   - Open Zed
   - Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Linux/Windows)
   - Run "Open Log"
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
   ./fak serve --log <tmp>/fak-zed.log ...
   tail -f <tmp>/fak-zed.log
   ```

### Slow first response

Expected on large local models — the Zed prompt is ~25K tokens. Subsequent requests are faster if you enable `--vdso=true` (content-addressed caching).

### Settings.json not being picked up

1. Verify the settings.json location:
   - Linux: `~/.config/zed/settings.json`
   - macOS: `~/Library/Application Support/Zed/settings.json`
   - Windows: `%APPDATA%\Zed\settings.json`

2. Check file syntax:
   ```bash
   # Validate JSON
   cat ~/.config/zed/settings.json | python -m json.tool
   ```

3. Reload settings: `Cmd+Shift+P` → "Reload Settings"

4. Check for conflicting project settings: `.zed/settings.json` overrides global settings

### Model not appearing in dropdown

1. Verify the `available_models` array includes the model ID
2. Ensure `low_speed_timeout_in_seconds` and `high_speed_timeout_in_seconds` are set
3. Reload settings or restart Zed
4. Check Zed logs for model loading errors

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

Zed sends `<PROVIDER_ID>_API_KEY` environment variable, which `fak` honors.

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

### Per-project configuration

Use `.zed/settings.json` in your project root for project-specific model selection:

```json
{
  "language_models": {
    "model": "openai_compatible/fak-review"
  }
}
```

This overrides the global setting for that project only.

---

## Cross-references

- **Integration index**: [README.md](README.md) — universal recipe and which-agent routing
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md) — full field survey
- **Policy schema**: [../../POLICY.md](../../POLICY.md) — authoring capability floors
- **Zed docs**: [https://zed.dev/docs](https://zed.dev/docs) — official Zed documentation
- **fak architecture**: [../../ARCHITECTURE.md](../../ARCHITECTURE.md) — kernel internals

---

## License

Apache-2.0
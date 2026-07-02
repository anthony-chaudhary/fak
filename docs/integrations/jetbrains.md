---
title: "JetBrains IDEs + fak: governed AI-assisted development"
description: "Wire fak as a tool-governance layer for JetBrains IDEs (IntelliJ IDEA, PyCharm, WebStorm, etc.), adding capability-floor enforcement and quarantine protection to AI-assisted development."
---

# JetBrains IDEs + fak Integration Guide

This guide shows how to use `fak` as a tool-governance layer for JetBrains IDEs (IntelliJ IDEA, PyCharm, WebStorm, PhpStorm, RubyMine, GoLand, CLion, and others), adding capability-floor enforcement and quarantine protection to AI-assisted development workflows.

## Overview

```
┌──────────────────┐   OpenAI / Anthropic   ┌────────────────────────┐
│  JetBrains IDE   │ ──────────────────────▶ │  fak serve (gateway)   │
│  (IntelliJ /     │ ◀──── SSE stream ─────  │  adjudicates tools     │
│   PyCharm /      │                        └────────────────────────┘
│   WebStorm / …)  │                                 │
└──────────────────┘                                 ▼
          ▲                                  ┌───────────────┐
          │ Custom Endpoint URL               │  Local Model  │
          │ (points at fak)                   │ or Cloud API  │
          │                                  └───────────────┘
```

**The gateway sits between JetBrains IDE and the model:**

- **JetBrains IDE → fak:** Sends API requests with proposed code edits and commands
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine)
- **fak → model:** Sends only the admitted (or repaired) calls to the model
- **fak → JetBrains IDE:** Returns results, with the kernel's decisions applied

**Result:** JetBrains IDEs can assist development, but the kernel blocks destructive commands, prevents self-modification, and contains untrusted tool results.

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

### 2. Install a JetBrains IDE

Choose and install the JetBrains IDE that fits your workflow:
- **IntelliJ IDEA** — Java, Kotlin, Scala, and more
- **PyCharm** — Python development
- **WebStorm** — JavaScript/TypeScript and web development
- **PhpStorm** — PHP development
- **RubyMine** — Ruby and Rails development
- **GoLand** — Go development
- **CLion** — C/C++ development
- **DataGrip** — Database development
- **Rider** — .NET development

Download from [jetbrains.com](https://www.jetbrains.com).

### 3. Install AI Assistant Plugin

Most JetBrains IDEs include the AI Assistant plugin by default. If not installed:

1. Open IDE settings (`Ctrl+Alt+S` or `Cmd+,`)
2. Navigate to **Plugins**
3. Search for "AI Assistant"
4. Click **Install**
5. Restart the IDE

### 4. Choose your upstream model

JetBrains IDEs can connect to `fak` in two modes:

- **Proxy mode:** `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM, etc.)
- **In-kernel mode:** `fak` serves its own fused model

For development, proxy mode is recommended as it gives you full model capabilities while still enforcing tool governance.

---

## Quick Start: One-command setup

The fastest way to put the kernel in front of JetBrains IDE:

```bash
# Terminal 1: Start the fak gateway
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json

# Terminal 2: Launch your JetBrains IDE
# The IDE will read the AI Assistant settings and use the configured endpoint
intellijidea  # or pycharm, webstorm, etc.
```

Configure AI Assistant to point at `http://127.0.0.1:8080/v1` as described below.

---

## Method 1: OpenAI-compatible provider (Recommended)

JetBrains AI Assistant supports custom OpenAI-compatible endpoints. `fak` provides an OpenAI-compatible `/v1/chat/completions` endpoint.

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

### Step 2: Configure AI Assistant via IDE settings

1. Open IDE settings (`Ctrl+Alt+S` or `Cmd+,`)
2. Navigate to **Tools** → **AI Assistant** → **AI Model Settings**
3. Configure the Custom Provider:

**For OpenAI-compatible provider:**

- **Provider:** Custom
- **Base URL:** `http://127.0.0.1:8080/v1`
- **Model ID:** `qwen2.5:1.5b`
- **API Key:** `fak-local`

4. Click **Test Connection** to verify the configuration
5. Click **OK** to save

### Step 3: Verify the integration

1. Open a code file in your JetBrains IDE
2. Right-click and select **AI Actions** → **Ask AI** (or press `Ctrl+Shift+A` / `Cmd+Shift+A` and type "Ask AI")
3. Enter a prompt, such as "Explain this function"
4. The request now flows through `fak`:

```
JetBrains IDE → fak /v1/chat/completions → adjudication → upstream model
                                                 ↓
                                           capability floor
                                                 ↓
                                       allowed/denied/transformed
                                                 ↓
                                       JetBrains IDE (with filtered results)
```

---

## Method 2: Anthropic provider

JetBrains AI Assistant also supports Anthropic's Messages API for Claude models.

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

### Step 2: Configure AI Assistant for Anthropic

1. Open IDE settings
2. Navigate to **Tools** → **AI Assistant** → **AI Model Settings**
3. Configure:
   - **Provider:** Custom
   - **Base URL:** `http://127.0.0.1:8080` (not including `/v1`)
   - **Model ID:** `claude-sonnet-4-20250514`
   - **API Key:** `fak-local`
4. Click **Test Connection**
5. Click **OK**

---

## Method 3: Per-project configuration

Configure different AI models for different projects using `.idea/ai-assistant.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="AISettings">
    <option name="customProviderSettings">
      <CustomProviderSettings>
        <option name="baseUrl" value="http://127.0.0.1:8080/v1" />
        <option name="modelId" value="qwen2.5:1.5b" />
        <option name="apiKey" value="fak-local" />
      </CustomProviderSettings>
    </option>
  </component>
</project>
```

This allows you to use different `fak serve` instances (with different policies) for different projects.

---

## Creating a Capability Floor for JetBrains IDEs

A capability floor defines which operations JetBrains AI Assistant may perform. Start from the built-in default:

```bash
# Dump the default policy as a starting point
./fak policy --dump > jetbrains-policy.json
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
    ".idea/",
    "jetbrains-policy.json",
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
./fak policy --check jetbrains-policy.json
```

Use the custom policy:
```bash
./fak serve --policy jetbrains-policy.json ...
```

---

## Common Patterns for JetBrains IDE Workflows

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
  "self_modify_globs": [".git/", ".idea/"]
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

If an external tool returns suspicious content (e.g., injection attempts), `fak` automatically quarantines it, preventing it from entering the IDE's context.

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

When JetBrains AI Assistant reports a denied operation, reproduce it offline:

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

### Checking the IDE's AI connection

1. Open IDE settings
2. Navigate to **Tools** → **AI Assistant** → **AI Model Settings**
3. Click **Test Connection**
4. Verify the connection succeeds
5. Check that the Base URL points to `http://127.0.0.1:8080/v1`

### Viewing IDE logs

1. Open **Help** → **Show Log in Explorer/Finder**
2. Check `idea.log` (or `pycharm.log`, `webstorm.log`, etc.) for AI Assistant errors
3. Search for "AI Assistant" or "custom provider" to find relevant entries

---

## Troubleshooting

### JetBrains IDE can't connect to the gateway

1. Verify `fak` is running:
   ```bash
   curl http://127.0.0.1:8080/healthz
   ```

2. Check IDE AI Assistant settings:
   - Open IDE settings
   - Navigate to **Tools** → **AI Assistant** → **AI Model Settings**
   - Verify Base URL is `http://127.0.0.1:8080/v1` (includes `/v1`)
   - Verify API Key is set

3. Click **Test Connection** in the AI Model Settings dialog

4. Check IDE logs for connection errors:
   - **Help** → **Show Log in Explorer/Finder**
   - Search for "AI Assistant"

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
   ./fak serve --log <tmp>/fak-jetbrains.log ...
   tail -f <tmp>/fak-jetbrains.log
   ```

### Slow AI responses

Expected on large local models — the JetBrains AI Assistant prompt is ~25K tokens. Subsequent requests are faster if you enable `--vdso=true` (content-addressed caching).

### AI Assistant not available

1. Verify the AI Assistant plugin is installed:
   - **File** → **Settings** → **Plugins**
   - Search for "AI Assistant"
   - Ensure it's enabled

2. Check that your JetBrains account has AI Assistant access:
   - **File** → **Settings** → **Tools** → **AI Assistant**
   - Verify you're logged in and have access

3. Ensure you're using a supported JetBrains IDE version
   - AI Assistant requires 2023.2 or later

### Per-project settings not working

1. Verify `.idea/ai-assistant.xml` exists and is valid XML
2. Restart the IDE after adding the file
3. Check that the project-specific settings override global settings
4. Use **File** → **Invalidate Caches / Restart** if settings aren't loading

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

JetBrains IDEs send the API key from AI Assistant settings, which `fak` honors.

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

### Multiple projects with different policies

Run multiple `fak serve` instances with different policies:

```bash
# Terminal 1: Strict policy (production)
./fak serve --addr 127.0.0.1:8080 --policy strict.json ...

# Terminal 2: Permissive policy (development)
./fak serve --addr 127.0.0.1:8090 --policy permissive.json ...
```

Configure each project's `.idea/ai-assistant.xml` to point at the appropriate instance.

### IDE-specific notes

**IntelliJ IDEA:**
- Best for Java/Kotlin/Scala projects
- Supports code completion, refactoring, and test generation

**PyCharm:**
- Best for Python projects
- Supports Django, Flask, FastAPI integrations

**WebStorm:**
- Best for JavaScript/TypeScript/web projects
- Supports React, Vue, Angular frameworks

**GoLand:**
- Best for Go projects
- Supports Go modules and standard library

**CLion:**
- Best for C/C++ projects
- Supports CMake and Makefile projects

---

## Cross-references

- **Integration index**: [README.md](README.md) — universal recipe and which-agent routing
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md) — full field survey
- **Policy schema**: [../../POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — authoring capability floors
- **JetBrains AI Assistant docs**: [https://www.jetbrains.com/help/idea/ai-assistant.html](https://www.jetbrains.com/help/idea/ai-assistant.html) — official documentation
- **fak architecture**: [../../ARCHITECTURE.md](https://github.com/anthony-chaudhary/fak/blob/main/ARCHITECTURE.md) — kernel internals

---

## License

Apache-2.0
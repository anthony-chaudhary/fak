# fak + Cursor Integration Guide

This guide shows how to use `fak` as a tool-governance layer for Cursor AI agents, adding capability-floor enforcement and quarantine protection to Cursor's coding workflows.

## Overview

`fak` is an agent tool firewall: a single Go binary that sits between an AI agent and its tools, enforcing a capability floor (which tools may be called) and quarantine (whether tool results can enter the agent's context). Cursor is an AI-native IDE that can integrate with external tools through two primary mechanisms:

1. **MCP (Model Context Protocol)** - Native protocol for tool and data source integration
2. **OpenAI-compatible HTTP proxy** - Cursor can be configured to use custom OpenAI-compatible endpoints

This guide covers both integration approaches.

---

## Prerequisites

### 1. Install fak

```bash
# From the repo
cd fleet/fak
go build -o fak ./cmd/fak

# Or via the installer
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

Verify installation:
```bash
./fak version
```

### 2. Install Cursor

Download from [cursor.com](https://cursor.com) and install following the official setup guide.

### 3. Choose your upstream model

Cursor can connect to `fak` in two modes:

- **Proxy mode**: `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM, etc.)
- **In-kernel mode**: `fak` serves its own fused SmolLM2-135M or Qwen model

For development, proxy mode is recommended as it gives you full model capabilities while still enforcing tool governance.

---

## Method 1: MCP Integration (Recommended)

Cursor has native MCP support, making this the cleanest integration path. `fak` exposes its syscall boundary as an MCP server with two primary tools:

### MCP Tools

| Tool | Purpose |
|------|---------|
| `fak_adjudicate` | Get a verdict for a tool call without executing it (client-side execution) |
| `fak_syscall` | Full path: adjudicate + execute through `fak`'s kernel (self-contained) |
| `fak_admit` | Check whether a tool result should be admitted into context (quarantine gate) |
| `fak_changes` | Subscribe to cross-agent coherence events (what other agents changed) |
| `fak_revoke` | Trigger fleet-wide refutation of a poisoned witness |

### Step 1: Start the fak MCP server

```bash
# Start in stdio MCP mode (for Cursor)
./fak serve --stdio \
  --base-url http://localhost:11434/v1 \  # Your upstream model
  --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json
```

Or use an upstream provider:
```bash
./fak serve --stdio \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY \
  --policy path/to/your-policy.json
```

### Step 2: Configure Cursor for MCP

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Navigate to **MCP Servers**
3. Add a new MCP server configuration:

```json
{
  "mcpServers": {
    "fak": {
      "command": "/path/to/fak",
      "args": [
        "serve",
        "--stdio",
        "--base-url", "http://localhost:11434/v1",
        "--model", "qwen2.5:1.5b",
        "--policy", "/path/to/policy.json"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Step 3: Use fak tools in Cursor

Once configured, Cursor can invoke `fak` tools:

```
@fak please adjudicate a call to "delete_account" with args {"user_id":"123"}
```

Or via the Cursor chat interface:
```
Use fak_syscall to execute "read_customer_record" for user "alice@example.com"
```

### MCP Reference: Request/Response Examples

**fak_adjudicate request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "fak_adjudicate",
    "arguments": {
      "tool": "delete_account",
      "args": "{\"user_id\":\"123\"}"
    }
  }
}
```

**Response (DENY):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "verdict": "DENY",
    "reason": "POLICY_BLOCK",
    "by": "adjudicator"
  }
}
```

---

## Method 2: OpenAI-Compatible Proxy

Cursor can use custom OpenAI-compatible endpoints. `fak` provides an OpenAI-compatible `/v1/chat/completions` endpoint that:

1. Receives the model's proposed tool calls
2. Adjudicates each call through the capability floor
3. Executes allowed calls (or returns them for client-side execution)
4. Quarantines suspicious results
5. Returns the filtered results to the model

### Step 1: Start the fak HTTP gateway

```bash
# Start the OpenAI-compatible proxy
./fak serve \
  --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy path/to/policy.json \
  --vdso=true
```

Verify health:
```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metrics  # Prometheus metrics
```

### Step 2: Configure Cursor's API proxy

Cursor supports proxy configuration via environment variables or settings:

**Via Environment Variables (Recommended):**
```bash
# Set before launching Cursor
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=optional-key-if-fak-requires
```

**Or via Cursor Settings:**
1. Open Cursor Settings
2. Navigate to **API Keys** or **Models**
3. Configure a custom endpoint:
   - Base URL: `http://127.0.0.1:8080/v1`
   - Model: `qwen2.5:1.5b` (or your upstream model)

### Step 3: Tool call flow

With the proxy configured, Cursor's tool calls flow through `fak`:

```
Cursor → fak /v1/chat/completions → adjudication → upstream model
                                              ↓
                                        capability floor
                                              ↓
                                        allowed/denied/transformed
                                              ↓
                                        Cursor (with filtered results)
```

---

## Creating a Capability Floor for Cursor

A capability floor defines which tools Cursor may call. Start from the built-in default:

```bash
# Dump the default policy as a starting point
./fak policy --dump > cursor-policy.json
```

### Example: Read-only coding agent policy

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": [
    "read_file",
    "search_files",
    "list_directory",
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
    "write_file": "POLICY_BLOCK",
    "delete_file": "POLICY_BLOCK",
    "run_command": "POLICY_BLOCK",
    "execute_code": "POLICY_BLOCK",
    "git_push": "POLICY_BLOCK",
    "git_commit": "POLICY_BLOCK",
    "install_package": "SUPPLY_CHAIN"
  },
  "self_modify_globs": [
    ".git/",
    ".cursor/",
    "cursor-policy.json",
    ".env",
    "id_rsa"
  ],
  "redact_fields": [
    "password",
    "secret",
    "api_key",
    "token"
  ],
  "safe_sinks": [
    "list_directory"
  ],
  "sources": {
    "read_file": "trusted_local",
    "search_files": "trusted_local",
    "git_diff": "trusted_local"
  },
  "arg_rules": [
    {
      "tool": "read_file",
      "arg": "path",
      "deny_regex": ".*\\.env$",
      "reason": "SECRET_EXFIL"
    },
    {
      "tool": "read_file",
      "arg": "max_bytes",
      "max_bytes": 100000,
      "reason": "OVERSIZE"
    }
  ]
}
```

Validate before using:
```bash
./fak policy --check cursor-policy.json
```

---

## Common Patterns for Cursor Workflows

### Pattern 1: Safe file operations with human approval

Configure `fak` to allow read operations but require explicit approval for writes:

```json
{
  "allow": ["read_file", "list_directory"],
  "deny": {
    "write_file": "REQUIRE_APPROVAL",
    "delete_file": "POLICY_BLOCK"
  }
}
```

In Cursor:
```
@fak read file src/main.ts
@fak please call write_file on src/main.ts with my refactor (I'll approve separately)
```

### Pattern 2: Git-aware workflow

Allow Git reads but block destructive Git operations:

```json
{
  "allow_prefix": ["git_diff", "git_log", "git_show", "git_blame"],
  "deny": {
    "git_push": "POLICY_BLOCK",
    "git_reset": "POLICY_BLOCK",
    "git_clean": "POLICY_BLOCK"
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

If an external tool returns suspicious content (e.g., injection attempts), `fak` automatically quarantines it, preventing it from entering Cursor's context.

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
- `fak_adjudication_denies_total` - Denied calls by reason
- `fak_quarantine_total` - Quarantined results

### Coherence feed (cross-agent changes)

```bash
# Get changes since sequence 0
curl http://127.0.0.1:8080/v1/fak/changes?since=0
```

### Refute a poisoned witness

```bash
curl -X POST http://127.0.0.1:8080/v1/fak/revoke \
  -H 'Content-Type: application/json' \
  -d '{"witness":"git-commit-abc123"}'
```

---

## Troubleshooting

### Cursor can't connect to the MCP server

1. Verify `fak` is running:
   ```bash
   ./fak serve --stdio --policy policy.json --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
   ```

2. Check Cursor's MCP configuration path:
   - The path to `fak` must be absolute
   - Policy file paths in args must also be absolute

3. Check Cursor's MCP logs for connection errors

### All tool calls are being denied

1. Verify your policy file is valid:
   ```bash
   ./fak policy --check your-policy.json
   ```

2. Check the policy's posture:
   - `posture: "fail_closed"` (default) denies everything not explicitly allowed
   - Ensure your tools are in `allow` or match an `allow_prefix`

3. Test a specific call:
   ```bash
   ./fak preflight --tool read_file --args '{"path":"test.txt"}' --policy your-policy.json
   ```

### Quarantined results aren't appearing

This is expected behavior. Quarantined results are intentionally excluded from the agent's context. Check the metrics to see what's being quarantined:

```bash
curl -s http://127.0.0.1:8080/metrics | grep quarantine
```

---

## Advanced: Fleet Integration

For multiple Cursor instances or multi-agent setups, `fak` provides:

1. **Cross-agent coherence** - All instances see what others changed via the `/v1/fak/changes` feed
2. **Shared vDSO cache** - Deduplicated tool results across all agents
3. **Scoped invalidation** - Namespace or resource-level cache invalidation

Example invalidation configuration:
```bash
./fak serve --addr 127.0.0.1:8080 \
  --invalidation namespace \
  --policy policy.json
```

---

## References

- **fak documentation**: [README.md](../../README.md)
- **Policy schema**: [POLICY.md](../../POLICY.md)
- **Cursor MCP docs**: [cursor.com/docs/mcp](https://cursor.com/docs/mcp)
- **MCP protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Example policies**: [fak/examples/](../../examples/)

---

## License

Apache-2.0 (matches the Microsoft Agent Governance Toolkit dependency).

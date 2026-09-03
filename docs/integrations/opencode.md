---
title: "OpenCode + fak: a governed terminal agent"
description: "Wire fak in front of the OpenCode terminal agent via MCP or an OpenAI-compatible gateway — a default-deny capability floor, malformed-call repair, and quarantine for poisoned tool results."
---

# OpenCode + fak Integration Guide

This guide shows how to put `fak` in front of [OpenCode](https://opencode.ai/), the
open-source terminal coding agent. Every tool call OpenCode proposes is adjudicated by the
kernel before it runs: denied by structure, repaired, or quarantined.

OpenCode auto-loads the repo's [`AGENTS.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md) — the same file Codex reads — for project context, combined with the
[`CONTRIBUTING.md`](https://github.com/anthony-chaudhary/fak/blob/main/CONTRIBUTING.md) instruction declared in the repo's `opencode.json`; `AGENTS.md` points back here.

## Two integration paths

- **MCP:** add `fak` as a local MCP server in OpenCode's config.
- **OpenAI-compatible gateway:** point an OpenCode provider at `fak serve`.

## Prerequisites

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak
./fak version
```

## Path 1 — MCP server

Add a local MCP server to OpenCode's config (`opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "fak": {
      "type": "local",
      "command": ["fak", "serve", "--stdio", "--policy", "examples/dev-agent-policy.json"],
      "enabled": true
    }
  },
  "permission": {
    "fak*": "allow",
    "fak_*": "allow"
  }
}
```

### Tool naming and permissions in OpenCode

OpenCode prefixes tools exposed by local MCP servers with `<server>_`. Under the `"fak"` server key, the kernel's tools are published as:

- `fak_fak_adjudicate` — Pre-execution verdict only (`ALLOW`, `DENY`, `TRANSFORM`). Call before running client tools.
- `fak_fak_syscall` — Adjudicate and execute through the kernel engine.
- `fak_fak_read` — Verified-fresh cached file reads with tamper-evident receipts.
- `fak_fak_tools_search` — Schema-on-demand search to conserve prompt context.

The permission wildcards `"fak*": "allow"` and `"fak_*": "allow"` permit OpenCode to execute these tools without interactive confirmation prompts. All tool input schemas conform to OpenAPI 3.0 and Gemini parameter specifications.

### Live pilot witness

A live four-call pilot against OpenCode `1.18.25` and Google Gemini is captured in [`experiments/agent-live/opencode-mcp-fak-live-pilot-2026-09-03.json`](../../experiments/agent-live/opencode-mcp-fak-live-pilot-2026-09-03.json) (issue #10818):
- `fak_fak_adjudicate(git_push)` → `DENY` (`POLICY_BLOCK`, `RETRYABLE`)
- `fak_fak_adjudicate(git_status)` → `ALLOW` (monitor pass)
- `fak_fak_read(VERSION)` → `ALLOW` (`0.45.0`, `executed_cold_read` receipt)
- `fak_fak_syscall(rm -rf /tmp/forbidden_test)` → `DENY` (`DEFAULT_DENY`, `TERMINAL`)


## Path 2 — OpenAI-compatible gateway

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b \
  --policy examples/customer-support-readonly-policy.json
```

Configure an OpenCode OpenAI-compatible provider with base URL
`http://127.0.0.1:8080/v1`. Verify:

```bash
curl http://127.0.0.1:8080/healthz
```

## Reproduce a denial offline

```bash
./fak preflight \
  --tool write_file \
  --args '{"path":".env","content":"x"}' \
  --policy examples/customer-support-readonly-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

## Fleet workers use account-bound launch

For unattended super-loop/fleet work, do not run `XDG_CONFIG_HOME=... opencode run`
directly. Use [`fak fleet-accounts launch|exec`](../fleet-accounts-launch.md), supply the
task tier, and use `--allow-tier3-narrow` only for explicitly narrow tier-3 work. Hard
engineering work cannot be overridden onto a restricted tier-3 OpenCode seat.

## Cross-references

- **Integration index**: [README.md](README.md)
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md)
- **Policy schema**: [../../POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md)
- **Harden any MCP server**: [harden-any-mcp.md](harden-any-mcp.md)
- **OpenCode docs**: [https://opencode.ai/docs](https://opencode.ai/docs)

## License

Apache-2.0

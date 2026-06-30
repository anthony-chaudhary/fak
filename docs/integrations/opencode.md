---
title: "OpenCode + fak: a governed terminal agent"
description: "Wire fak in front of the OpenCode terminal agent via MCP or an OpenAI-compatible gateway — a default-deny capability floor, malformed-call repair, and quarantine for poisoned tool results."
---

# OpenCode + fak Integration Guide

This guide shows how to put `fak` in front of [OpenCode](https://opencode.ai/), the
open-source terminal coding agent. Every tool call OpenCode proposes is adjudicated by the
kernel before it runs: denied by structure, repaired, or quarantined.

OpenCode reads the repo's [`AGENT.md`](../../AGENT.md) / [`AGENTS.md`](../../AGENTS.md) for
project context; they point back here.

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
  "mcp": {
    "fak": {
      "type": "local",
      "command": ["./fak", "serve", "--stdio", "--policy", "examples/customer-support-readonly-policy.json"]
    }
  }
}
```

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

## Cross-references

- **Integration index**: [README.md](README.md)
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md)
- **Policy schema**: [../../POLICY.md](../../POLICY.md)
- **Harden any MCP server**: [harden-any-mcp.md](harden-any-mcp.md)
- **OpenCode docs**: [https://opencode.ai/docs](https://opencode.ai/docs)

## License

Apache-2.0

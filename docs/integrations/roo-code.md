---
title: "Roo Code + fak: a governed VS Code agent"
description: "Wire fak as a tool-governance layer for Roo Code, the VS Code AI agent — a default-deny capability floor, malformed-call repair, and quarantine for poisoned tool results — via an OpenAI-compatible gateway or MCP."
---

# Roo Code + fak Integration Guide

This guide shows how to put `fak` in front of [Roo Code](https://roocode.com/), the VS Code
AI coding agent. Every tool call Roo Code proposes is adjudicated by the kernel before it
runs: denied by structure, repaired, or quarantined.

## How it fits

```
Roo Code (VS Code)  ──▶  fak serve (gateway)  ──▶  your model (cloud or local)
        ▲                  adjudicates every            │
        └────── results ───  tool call ◀────────────────┘
```

## Prerequisites

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak
./fak version
```

## OpenAI-compatible gateway (recommended)

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b \
  --policy examples/customer-support-readonly-policy.json
```

In Roo Code's provider settings, choose **OpenAI Compatible**, set the base URL to
`http://127.0.0.1:8080/v1`, and any non-empty API key (e.g. `fak-local`). Verify the
gateway:

```bash
curl http://127.0.0.1:8080/healthz
```

## MCP server

Roo Code can also launch `fak` as an MCP server (`.roo/mcp.json`):

```json
{
  "mcpServers": {
    "fak": {
      "command": "./fak",
      "args": ["serve", "--stdio", "--policy", "examples/customer-support-readonly-policy.json"]
    }
  }
}
```

## Reproduce a denial offline

```bash
./fak preflight \
  --tool run_command \
  --args '{"command":"sudo rm -rf /"}' \
  --policy examples/customer-support-readonly-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

## Cross-references

- **Integration index**: [README.md](README.md)
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md)
- **Cline recipe** (sibling VS Code agent): [cline.md](cline.md)
- **Policy schema**: [../../POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md)
- **Roo Code docs**: [https://docs.roocode.com](https://docs.roocode.com)

## License

Apache-2.0

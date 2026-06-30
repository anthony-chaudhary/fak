---
title: "Gemini CLI + fak: governed tool calls via MCP or an OpenAI-compatible gateway"
description: "Wire fak in front of Google's Gemini CLI — either as an MCP server it launches, or as an OpenAI-compatible gateway — to enforce a capability floor, repair malformed calls, and quarantine poisoned tool results."
---

# Gemini CLI + fak Integration Guide

This guide shows how to put `fak` in front of [Gemini CLI](https://github.com/google-gemini/gemini-cli),
Google's open-source terminal agent. Every tool call is adjudicated by the kernel before it
runs: denied by structure, repaired, or quarantined.

Gemini CLI context for this repo lives in [`GEMINI.md`](../../GEMINI.md) (auto-loaded),
which points back here.

## Two integration paths

- **MCP (recommended):** Gemini CLI launches `fak` as an MCP server; the kernel governs
  the tools it exposes.
- **OpenAI-compatible gateway:** point the CLI's model endpoint at `fak serve`.

## Prerequisites

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak
./fak version
```

## Path 1 — MCP server

Register `fak` as an MCP server in Gemini CLI's settings (`~/.gemini/settings.json`):

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

The kernel now adjudicates the governed tools before the model sees their results.

## Path 2 — OpenAI-compatible gateway

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b \
  --policy examples/customer-support-readonly-policy.json
```

Point the CLI's model base URL at `http://127.0.0.1:8080/v1`. Verify health:

```bash
curl http://127.0.0.1:8080/healthz
```

## Reproduce a denial offline

```bash
./fak preflight \
  --tool delete_file \
  --args '{"path":"important.txt"}' \
  --policy examples/customer-support-readonly-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

## Cross-references

- **Integration index**: [README.md](README.md)
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md)
- **Policy schema**: [../../POLICY.md](../../POLICY.md)
- **Harden any MCP server**: [harden-any-mcp.md](harden-any-mcp.md)
- **Gemini CLI repo**: [https://github.com/google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)

## License

Apache-2.0

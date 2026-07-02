---
title: "Amp + fak: a governed Sourcegraph Amp agent"
description: "Wire fak in front of Sourcegraph's Amp coding agent via MCP or an OpenAI-compatible gateway — a default-deny capability floor, malformed-call repair, and quarantine for poisoned tool results."
---

# Amp + fak Integration Guide

This guide shows how to put `fak` in front of [Amp](https://ampcode.com/), Sourcegraph's
agentic coding tool. Every tool call Amp proposes is adjudicated by the kernel before it
runs: denied by structure, repaired, or quarantined.

Amp auto-loads [`AGENT.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENT.md) from the repo root for project context; it
points back here.

## Two integration paths

- **MCP:** register `fak` as an MCP server Amp connects to.
- **OpenAI-compatible gateway:** route Amp's model traffic through `fak serve`.

## Prerequisites

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak
./fak version
```

## Path 1 — MCP server

Add `fak` to Amp's MCP server configuration:

```json
{
  "amp.mcpServers": {
    "fak": {
      "command": "./fak",
      "args": ["serve", "--stdio", "--policy", "examples/customer-support-readonly-policy.json"]
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

Point Amp's model endpoint at `http://127.0.0.1:8080/v1`. Verify:

```bash
curl http://127.0.0.1:8080/healthz
```

## Reproduce a denial offline

```bash
./fak preflight \
  --tool run_command \
  --args '{"command":"git push --force"}' \
  --policy examples/customer-support-readonly-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

## Cross-references

- **Integration index**: [README.md](README.md)
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md)
- **Policy schema**: [../../POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md)
- **Kernel internals**: [../../ARCHITECTURE.md](https://github.com/anthony-chaudhary/fak/blob/main/ARCHITECTURE.md)
- **Amp docs**: [https://ampcode.com/manual](https://ampcode.com/manual)

## License

Apache-2.0

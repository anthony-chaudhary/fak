---
title: "Windsurf + fak: a governed Cascade agent"
description: "Wire fak as a tool-governance layer for Windsurf's Cascade agent — a default-deny capability floor, malformed-call repair, and quarantine for poisoned tool results — via an OpenAI-compatible gateway."
---

# Windsurf + fak Integration Guide

This guide shows how to put `fak` in front of [Windsurf](https://windsurf.com/)'s Cascade
agent. Every tool call Cascade proposes is adjudicated by the kernel before it runs:
dangerous calls are denied by structure, malformed calls are repaired, and untrusted tool
results are quarantined before they reach the model's context.

Windsurf project context for this repo lives in [`.windsurfrules`](https://github.com/anthony-chaudhary/fak/blob/main/.windsurfrules)
(auto-loaded), which points back here.

## How it fits

```
Windsurf (Cascade)  ──▶  fak serve (gateway)  ──▶  your model (cloud or local)
        ▲                  adjudicates every            │
        └────── results ───  tool call ◀────────────────┘
```

The gateway speaks an OpenAI-compatible API, so any Windsurf "custom / BYOK
OpenAI-compatible" model setting can point at it.

## Prerequisites

```bash
# Build fak (the Go module is the repo root)
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak
./fak version
```

## Quick start: put the kernel in front of Cascade

```bash
# Start the gateway with a read-only capability floor (no key needed to try it)
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b \
  --policy examples/customer-support-readonly-policy.json
```

Then in Windsurf, set the model provider to an **OpenAI-compatible** endpoint with base
URL `http://127.0.0.1:8080/v1` and any non-empty API key (e.g. `fak-local`). Cascade's
edits and commands now flow through the capability floor.

Verify the gateway is healthy:

```bash
curl http://127.0.0.1:8080/healthz
```

## Author a capability floor

Reproduce a denial offline before trusting it in the IDE:

```bash
./fak preflight \
  --tool run_command \
  --args '{"command":"rm -rf /tmp"}' \
  --policy examples/customer-support-readonly-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

Start a custom policy from the built-in default and validate it:

```bash
./fak policy --dump > windsurf-policy.json
./fak policy --check windsurf-policy.json
```

## Cross-references

- **Integration index**: [README.md](README.md) — the universal recipe + which-agent routing
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md)
- **Policy schema**: [../../POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — authoring capability floors
- **Kernel internals**: [../../ARCHITECTURE.md](https://github.com/anthony-chaudhary/fak/blob/main/ARCHITECTURE.md)
- **Windsurf docs**: [https://docs.windsurf.com](https://docs.windsurf.com)

## License

Apache-2.0

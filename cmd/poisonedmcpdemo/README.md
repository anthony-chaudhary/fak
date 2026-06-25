# Poisoned MCP Demo

`poisonedmcpdemo` is the runnable A/B for hardening an MCP server against tool poisoning. It drives the real context-MMU result quarantine and a structural allow-list floor with no model in the loop.

## Prerequisites

Requires Go only. It does not need an MCP server process, model, API key, GPU, or network.

## Quick Start

From the repo root:

```bash
go run ./cmd/poisonedmcpdemo
go run ./cmd/poisonedmcpdemo -json
```

Each run completes in a few seconds and returns exit code 0 on success. Output is deterministic because the poisoned result payloads, fake secret, and allow-list are fixed.

## What You See

```text
poisoned-MCP A/B - the same tool-poisoning MCP server, unmediated vs behind fak.
result: summarize_doc (injection payload)    IN CONTEXT           QUARANTINED
result: search_kb (benign policy text)       IN CONTEXT           ALLOWED
description: exfiltrate_creds                may coerce the model DENY
```

## What This Does Not Claim

This demo does not claim prompt-injection detection is complete. The point is containment plus the lock: caught poisoned bytes are held out of context, and unwired tools are denied by structure.

## Related Docs

- [Harden any MCP server against tool poisoning](../../docs/integrations/harden-any-mcp.md)
- [MCP example](../../examples/mcp/README.md)

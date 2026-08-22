---
title: "Gemini CLI + fak: first-class guarded launch"
description: "Run Gemini CLI through fak's native Gemini gate, including provider-clear session boundaries for /clear and /new."
---

# Gemini CLI + fak Integration Guide

The supported default is one command:

```bash
fak manage -- gemini
```

`manage` routes to the same guarded launch as `fak guard -- gemini`. Fak
auto-detects Gemini's native `generateContent` wire, starts the loopback gate,
and gives only the child process a `GOOGLE_GEMINI_BASE_URL` pointing at that
gate. Gemini CLI's existing authentication is inherited; no credential is
written into settings.

## Clear and new stay aligned with fak

The guarded launch also creates a temporary Gemini system-settings file and
sets `GEMINI_CLI_SYSTEM_SETTINGS_PATH` only for that launch. The file adds a
`SessionStart` command hook with matcher `clear`; it does not edit
`~/.gemini/settings.json` or another persistent user setting.

Gemini implements `/new` as an alias of `/clear`. Both commands end the old
provider session, mint a new provider session id, and emit
`SessionStart(source=clear)`. The hook sends that id over the authenticated
process-local lifecycle socket that fak already passes to the child. Fak then:

- stops the old trace with `PROVIDER_SESSION_CLEAR`;
- creates exactly one deterministic child trace;
- switches requests without an explicit trace to that child; and
- restores only the context-token allowance while carrying cumulative turn,
  token, spend, query, tool-call, wall-clock, and throughput limits.

A repeated delivery for the same provider session id is idempotent. If the
local hook transaction fails, Gemini still clears and fak keeps the stricter
old cumulative state instead of manufacturing a fresh allowance.

## Choose the upstream explicitly when needed

The native public default is the Gemini Generative Language API. To name a
different Gemini-compatible upstream or key source:

```bash
fak manage \
  --provider gemini \
  --base-url https://generativelanguage.googleapis.com/v1beta \
  --api-key-env GEMINI_API_KEY \
  -- gemini
```

The URL above is the upstream API root. The child still receives the bare
loopback gate URL because Gemini's client appends `/v1beta/models/...` itself.

## MCP-only alternative

If only fak's MCP tools need governance, Gemini can launch fak as a stdio MCP
server from `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "fak": {
      "command": "fak",
      "args": ["serve", "--stdio", "--policy", "examples/customer-support-readonly-policy.json"]
    }
  }
}
```

This governs the tools exposed by that MCP server. It is not the full guarded
provider route and does not install the provider-clear session adapter.

## Offline and live witness

```bash
go test ./internal/harnessprofile ./internal/dropin -count=1
go test ./cmd/fak -run 'TestGuardGemini|TestInstallManagedNativeHooksUsesSupportedSeams' -count=1
```

The captured Gemini CLI run and its old/new trace transitions are recorded in
[`issue-8219-gemini-clear-session.json`](../_witnesses/issue-8219-gemini-clear-session.json).

## Cross-references

- [Provider clear/new session semantics](provider-session-reset.md)
- [Compatibility matrix](compatibility-matrix.md)
- [Policy schema](../../POLICY.md)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)

## License

Apache-2.0

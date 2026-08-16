---
title: "Managed-harness parity packet — 2026-08-13"
description: "Documentation for Managed-harness parity packet — 2026-08-13, including the captured behavior, operating context, and reproducible fak evidence."
---

# Managed-harness parity packet — 2026-08-13

## Command

```powershell
fak manage parity --json
```

## Captured receipt

```json
{"schema":"fak-manage-parity/1","verdict":"PASS","cases":[{"name":"claude-windows-separator","manage":{"invocation":"manage","harness":"claude","platform":"windows","separator":true,"provider":"anthropic","base_url":"http://127.0.0.1:<ephemeral>","policy":"<temp>/windows/guard-default-policy.json","child_argv":["C:\\Program Files\\Claude\\claude.exe","-p","review this repo"],"hooks":{"stop":true,"pre_compact":true,"tool":true,"settings":true}},"legacy":{"invocation":"guard","harness":"claude","platform":"windows","separator":true,"provider":"anthropic","base_url":"http://127.0.0.1:<ephemeral>","policy":"<temp>/windows/guard-default-policy.json","child_argv":["C:\\Program Files\\Claude\\claude.exe","-p","review this repo"],"hooks":{"stop":true,"pre_compact":true,"tool":true,"settings":true}},"verdict":"PASS"},{"name":"codex-posix-no-separator","manage":{"invocation":"m","harness":"codex","platform":"posix","separator":false,"provider":"openai","base_url":"http://127.0.0.1:<ephemeral>","policy":"<temp>/windows/guard-default-policy.json","child_argv":["/usr/local/bin/codex","exec","review this repo"],"hooks":{"stop":false,"pre_compact":false,"tool":false,"settings":false}},"legacy":{"invocation":"guard","harness":"codex","platform":"posix","separator":false,"provider":"openai","base_url":"http://127.0.0.1:<ephemeral>","policy":"<temp>/windows/guard-default-policy.json","child_argv":["/usr/local/bin/codex","exec","review this repo"],"hooks":{"stop":false,"pre_compact":false,"tool":false,"settings":false}},"verdict":"PASS"},{"name":"gemini-posix-separator","manage":{"invocation":"manage","harness":"gemini","platform":"posix","separator":true,"provider":"openai","base_url":"http://127.0.0.1:<ephemeral>","policy":"<temp>/windows/guard-default-policy.json","child_argv":["/opt/bin/gemini","-p","review this repo"],"hooks":{"stop":false,"pre_compact":false,"tool":false,"settings":false}},"legacy":{"invocation":"guard","harness":"gemini","platform":"posix","separator":true,"provider":"openai","base_url":"http://127.0.0.1:<ephemeral>","policy":"<temp>/windows/guard-default-policy.json","child_argv":["/opt/bin/gemini","-p","review this repo"],"hooks":{"stop":false,"pre_compact":false,"tool":false,"settings":false}},"verdict":"PASS"}],"operator_probe":{"invocation":"manage","subcommand":"policy","routed":true,"listener_made":false,"verdict":"PASS"},"external_model":false}
```

## Scope

The packet resolves the same provider, normalized ephemeral gateway URL, policy path, child
argv, and installed-hook posture for `manage`/`m` and legacy `guard`. It covers Claude Code,
Codex, Gemini, Windows/POSIX path forms, and with/without the optional separator. The operator
`policy` probe is routed before launch and records `listener_made:false`.

This is a deterministic launch-contract witness. It spends no model traffic and makes no
quality, cost, or latency claim.

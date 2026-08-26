---
title: "Diagnose Codex MCP startup warnings"
description: "Codex CLI 0.147.0 can print MCP startup interrupted even after every named server initialized (upstream openai/codex37418)."
---
# Diagnose Codex MCP startup warnings

Codex CLI 0.147.0 can print `MCP startup interrupted` even after every named
server initialized (upstream [openai/codex#37418](https://github.com/openai/codex/issues/37418)).
Correlate the warning's names against Codex's local structured log:

```powershell
fak doctor codex-mcp-warning --servers codex_apps,dos,fak,openaiDeveloperDocs
fak doctor codex-mcp-warning --servers codex_apps,dos,fak,openaiDeveloperDocs --json
```

The read-only diagnostic opens `~/.codex/logs_2.sqlite` in SQLite read-only and
query-only mode. Output contains only the supplied server names and typed status;
it never emits log bodies, paths, arguments, environment values, tokens, or
server payloads.

Verdicts:

- `CLIENT_STATUS_FALSE_NEGATIVE`: all named servers have subsequent ready evidence.
- `SERVER_FAILURE`: at least one named server has a startup error or timeout.
- `RUNTIME_REFRESH_CANCELLATION`: cancellation or runtime-refresh evidence exists.
- `INSUFFICIENT_EVIDENCE`: the log is missing/unreadable or lacks named evidence.

The false-warning case is distinct from the real plugin-refresh cancellation and
process leak tracked in #37025.

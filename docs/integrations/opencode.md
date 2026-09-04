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

## Integration paths

- **Dedicated launcher (`fak opencode`):** one-command launcher wrapping OpenCode with kernel adjudication, prompt-cache preservation, and live split view.
- **`fak manage` wrapper:** launch `fak manage opencode` or `fak manage -- opencode`.
- **Dogfood scripts:** `scripts/dogfood-opencode.ps1` (Windows) and `scripts/dogfood-opencode.sh` (Linux/macOS).
- **MCP:** add `fak` as a local MCP server in OpenCode's config (`opencode.json`).
- **OpenAI-compatible gateway:** point an OpenCode provider at `fak serve`.

## Prerequisites

```bash
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak
./fak version
```

## Path 1 — Dedicated launcher: `fak opencode`

The fastest way to run guarded OpenCode:

```bash
fak opencode                            # interactive OpenCode guarded by fak
fak opencode --dry-run                  # preview the guarded command
fak opencode --probe "Use bash to echo ok" # one-shot probe turn
fak opencode --local                    # auto-detect local model server (Ollama, LM Studio, Qwen3.6)
fak opencode --base-url http://127.0.0.1:8001/v1  # point at a custom /v1 upstream
```

`fak opencode` injects `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` into the child process only, enforces the capability floor over OpenCode's tool dialect (`bash`, `read`, `write`, `edit`, `grep`, `glob`, `todowrite`, `task`, `question`, `skill`), and records an audit journal.

## Path 2 — Dogfood scripts

Windows (PowerShell):
```powershell
.\scripts\dogfood-opencode.ps1 --dry-run
.\scripts\dogfood-opencode.ps1 --probe "Reply with pong"
.\scripts\dogfood-opencode.ps1 --smoke
.\scripts\dogfood-opencode.ps1 --print-env
```

macOS / Linux:
```bash
./scripts/dogfood-opencode.sh --dry-run
./scripts/dogfood-opencode.sh --probe "Reply with pong"
./scripts/dogfood-opencode.sh --smoke
./scripts/dogfood-opencode.sh --print-env
```

## Path 3 — MCP server

Add a local MCP server to OpenCode's config (`opencode.json`):

```json
{
  "mcp": {
    "fak": {
      "type": "local",
      "command": ["fak", "serve", "--stdio", "--defer-tools=false", "--policy", "examples/opencode-policy.json"]
    }
  }
}
```

> **Note on Tool Advertisement (`--defer-tools=false`):** By default, `fak serve` uses progressive disclosure on MCP `tools/list` (returning only a minimal bootstrap tool set and deferring cold tools). Because OpenCode queries `tools/list` once at session start and does not support dynamic client-side tool discovery via `fak_tools_search`, specify `--defer-tools=false` (or set `FAK_ABLATE_MCP_TOOL_FILTER=1`) so the full tool registry is advertised immediately at startup.

## Path 4 — OpenAI-compatible gateway

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

## Skill portability: importing skills into OpenCode (#10689, #10690, #10691)

OpenCode natively supports the portable [Agent Skills](https://agentskills.io) standard, enabling direct skill reuse across Claude Code, OpenAI Codex, and OpenCode:

### 1. Skill locations and formats

| Harness | Discovery location | Format |
|---|---|---|
| **Claude Code** | `.claude/skills/<name>/SKILL.md` | Canonical semantic body + Claude frontmatter |
| **OpenAI Codex** | `.agents/skills/<name>/SKILL.md` | Agent Skills standard (generated adapter / native) |
| **OpenCode** | `opencode.json` -> `skills.paths` (`.agents/skills`) | Agent Skills standard / portable bundle |

### 2. Bundling and synchronization with `fak-project-assets`

`cmd/fak-project-assets` serves as the portable skill bundler:
- `go run ./cmd/fak-project-assets sync --json`: syncs canonical skills into portable regular-file adapters under `.agents/skills/` (compatible with Windows, Linux, macOS).
- `go run ./cmd/fak-project-assets parity --json`: verifies that 100% of skills are canonical or mapped with `zero_unexplained_gaps: true`.

In `opencode.json`, point `skills.paths` at the synced portable bundle:
```json
{
  "skills": {
    "paths": [".agents/skills"]
  }
}
```

### 3. Access control and frontmatter mediation

Claude Code frontmatter fields (`disable-model-invocation`, `user-invocable`, `allowed-tools`) are dropped by OpenCode's parser. In fak, this gap is bridged cleanly:
1. **Metadata Acknowledgment:** Skills with load-bearing Claude frontmatter specify `metadata.opencode: agent-permission` or `metadata.opencode: claude-only`. Verified via `python tools/skill_frontmatter_lint.py --check`.
2. **Re-expression in `opencode.json`:** Per-skill invocation gates are re-expressed in `opencode.json` under `permission.skill`:
```json
{
  "permission": {
    "skill": {
      "phased-plan": "deny"
    }
  }
}
```

### 4. Round-trip demonstration

A skill authored for Codex or Claude is synchronized via `fak-project-assets sync`, loaded into OpenCode via `skills.paths`, and governed by fak's MCP capability floor (`fak_adjudicate`).

## Cross-references

- **Integration index**: [README.md](README.md)
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md)
- **Policy schema**: [../../POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md)
- **Harden any MCP server**: [harden-any-mcp.md](harden-any-mcp.md)
- **OpenCode docs**: [https://opencode.ai/docs](https://opencode.ai/docs)

## License

Apache-2.0

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

## Disabling workspace snapshotting in large repositories (#11144)

By default, OpenCode captures workspace snapshots across turns and tool executions by running background git diffs and staging operations to record undo checkpoints. In large repositories or repos with large working trees, this default causes severe performance degradation and resource waste:
- **Disk hammering & I/O latency:** Repetitive tree scans and thousands of full git diff executions stall execution turns, spike disk I/O, and introduce latency into tool call responses.
- **Multi-GB SQLite database bloat:** Snapshot metadata and diff text accumulate in OpenCode's SQLite database (`~/.local/share/opencode/opencode.db` or platform equivalent), growing the database by multiple gigabytes over long sessions.
- **Process thrashing:** Rapid-fire child git processes compete with agent tool calls and compiler/test invocations.

To eliminate this overhead, disable workspace snapshotting explicitly in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "snapshot": false
}
```

Setting `"snapshot": false`:
1. Prevents OpenCode from taking redundant background git diff snapshots between turns.
2. Stops SQLite database bloat, keeping session state lightweight.
3. Removes disk hammering and file-tree scanning overhead, ensuring fast, deterministic tool execution.

In fak, this invariant is guarded across the codebase:
- `opencode.json` at repository root sets `"snapshot": false`.
- `tb4bench.GenerateOpenCodeJSON` includes `"snapshot": false` in synthesized configurations.
- `tools/extend_preflight.py` enforces the `opencode-snapshot-disabled` check.
- `projectassets.VerifyOpenCodeSnapshot` verifies that `opencode.json` has snapshotting disabled.

## Reasoning Effort Profiles & Subagent Delegation (#11146)

Modern reasoning models (such as Gemini 3.8 Flash or similar frontier thinking models) support configurable thinking effort variants. However, configuring reasoning effort globally across an entire agent harness creates severe operational trade-offs:

### The problem: global high-effort latency penalty
Setting `"variant": "high"` globally across all agents introduces a massive latency penalty:
- **10s–25s thinking token latency on routine tool turns:** Everyday deterministic actions—such as inspecting files (`read`), listing directories (`glob`), grepping patterns, or running tests—do not require multi-step deep reasoning.
- **Thinking token bloat & quota exhaustion:** High effort forces the model to emit thousands of intermediate reasoning tokens even for simple single-line edits or trivial checks, inflating token spend and draining rate limits prematurely.
- **Degraded interactive responsiveness:** Developers and automated coordinators experience sluggish turn-by-turn execution during mechanical tool loops.

### The solution: tiered reasoning profiles
To balance snappy turn latency with deep analytical power, OpenCode configurations establish tiered reasoning profiles across built-in agents and specialized subagents:
- **Baseline default for routine work (`general` and `explore`):** Configured with `"variant": "default"`, providing instant, low-latency responses for file searches, routine edits, and straightforward tool execution.
- **On-demand high-effort delegation (`@deep-reason`):** Reserving `"variant": "high"` for a dedicated subagent (`deep-reason`) invoked selectively when tackling complex architectural reasoning, concurrency invariants, frozen ABI modifications (`internal/abi`), or elusive debugging puzzles.

In `opencode.json` (or `~/.config/opencode/opencode.jsonc`):
```json
{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "general": {
      "variant": "default"
    },
    "explore": {
      "variant": "default"
    },
    "deep-reason": {
      "variant": "high",
      "description": "High-reasoning subagent for deep architectural analysis, frozen ABI changes, or complex debugging"
    }
  }
}
```

### Dynamic overrides and CLI control
When a full session demands elevated reasoning effort without subagent switching:
1. **CLI variant override:** Pass `--variant` at launch to override the baseline for the entire session:
   ```bash
   opencode run --variant high "Refactor concurrency lock ordering in internal/kernel"
   ```
2. **Interactive model picker & variant selection:** In the OpenCode interactive TUI, press `/model` or use the model picker dialog to select reasoning effort variants (`default`, `high`, `low`) dynamically for the active session.

## Fleet workers use account-bound launch

For unattended super-loop/fleet work, do not run `XDG_CONFIG_HOME=... opencode run`
directly. Use [`fak fleet-accounts launch|exec`](../fleet-accounts-launch.md), supply the
task tier, and use `--allow-tier3-narrow` only for explicitly narrow tier-3 work. Hard
engineering work cannot be overridden onto a restricted tier-3 OpenCode seat.

## Skill portability: importing skills into OpenCode (#10689, #10690, #10691)

OpenCode natively supports the portable [Agent Skills](https://agentskills.io) standard, enabling seamless skill reuse across Claude Code, OpenAI Codex, and OpenCode:

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

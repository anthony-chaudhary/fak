---
title: "Cold-Agent Navigation, MCP Auto-Configuration, and Public-Only Context Audit"
description: "Audit of how fak behaves for an agent with zero prior context across standalone user projects, public-only fak clones, and companion-aware internal workspaces."
---

# Cold-Agent Navigation, MCP Auto-Configuration, and Public-Only Context Audit

This document records the audited behavior of `fak` when driven by an autonomous agent with **no prior repository context**, across all three deployment contexts.

## 1. Context Matrix

| Context | Detection (`workspace_mode`) | What works out of the box | What is absent / degraded |
| --- | --- | --- | --- |
| **Standalone binary** (external project, no `dos.toml`) | `standalone` | `fak --json`, `fak help --json`, `fak mcp config/install/status/doctor/uninstall`, `fak serve --stdio`, `fak preflight`, `fak manage`, `fak capabilities --json`, `fak doctor launch-posture --json` | Repo-scoped `fak-dev` / `fak-private` tools (`dos-mcp`, `fak-flow`, `fak-sync`); `examples/dev-agent-policy.json` is optional |
| **Public-only `fak` clone** (`dos.toml` present, no `../fak-private`) | `public_repo` | Everything in `standalone`, plus `fak-dev` public checks (`buildcheck`, `index`), `go test ./...`, `.mcp.json` `fak` server, `examples/dev-agent-policy.json` | Private companion tools (`fak-boundary`, `fak-sync`, `fak-flow`, `dos-mcp`) degrade with `COMPANION_ABSENT` (`fak dev availability --json`) |
| **Companion-aware internal** (`fak` + `fak-private`) | `companion_repo` | Full public + private governance (5-gate boundary checks, leak scrub, tickets, leases) | None |

## 2. Zero-Context Cold-Agent Flow

An agent landing with zero prior context can orient and wire `fak` in four deterministic commands:

1. `fak --json` (or `fak help --json`) — returns `schema: "fak.help/1"` with `workspace_mode`, `quick_start`, `mcp` commands, and grouped verb summaries.
2. `fak help --verbs --json` / `fak help <verb> --json` — machine-readable verb index and per-verb deep help (use `fak doctor launch-posture --json` or `fak mcp doctor --json` rather than bare `fak doctor`, which expects answer-shape text on stdin).
3. `fak mcp install --client auto` (supports `project`, `claude`, `cursor`, `opencode`, `codex`) — safely merges only the `fak` entry into `.mcp.json`, `.cursor/mcp.json`, `opencode.json` (`mcp.fak`), or `~/.codex/config.toml`, omits repo-only `--policy` (or synthesizes embedded `fak-policy/v1` `default_open` for Codex) when outside `fak`, and verifies the live MCP `initialize` handshake before writing. Use `--dry-run` or `fak mcp config --json` to preview without writing.
4. `fak mcp status --json` / `fak mcp doctor --json` — inspects configuration and verifies live stdio JSON-RPC `initialize` purity.

## 3. Audited Public-Only Gaps & Dispositions

### Fixed in this pass
- **Machine-readable help (`fak --json`, `fak help --json`):** Previously `fak help` only emitted human text, forcing cold agents to scrape prose. `fak --json`, `fak help --json`, `fak help --verbs --json`, and `fak help <verb> --json` now emit structured JSON with `workspace_mode` and `mcp` discovery.
- **Unified MCP auto-configuration (`fak mcp`):** Previously only `fak codex mcp` existed, and it failed with `POLICY_UNREADABLE` outside the `fak` repo when `examples/dev-agent-policy.json` was missing. `fak mcp config|install|status|doctor|uninstall` now supports `auto`, `project`, `claude`, `cursor`, `opencode`, and `codex`, preserves sibling entries, verifies the `initialize` handshake before modifying config, and falls back cleanly (embedded `fak-policy/v1` `default_open` for Codex; omitted `--policy` for stdio JSON harnesses) when `examples/dev-agent-policy.json` is absent.
- **Explicit workspace-mode classification:** `detectWorkspaceMode()` classifies `standalone`, `public_repo`, and `companion_repo` without subprocesses so cold agents know immediately whether repo/companion surfaces apply.

### Remaining Honest Boundaries
- **Maintainer `.mcp.json` / `opencode.json` in `fak` root:** The repository's own `.mcp.json` and `opencode.json` include `"dos"` (`dos-mcp`) alongside `"fak"` for internal maintainer workflows. External/public-only users running in their own project should use `fak mcp install --client <project|opencode|cursor|codex>`, which writes **only** `"fak"`.
- **Maintainer `AGENTS.md` / `CLAUDE.md` vs external contributors:** `AGENTS.md` and `CLAUDE.md` carry shared-trunk rules (`fak commit`, `dos`) for automated contributors in the maintainers' checkout; external contributors should start from `README.md` and `CONTRIBUTING.md`.
- **Advisory stderr in `standalone` `fak serve --stdio`:** When `fak serve --stdio` starts outside a `fak` workspace, it emits `NOT_A_WORKSPACE` (and mock-planner notes when no model backend is specified) on **stderr** while stdout carries pure JSON-RPC; `fak mcp doctor` reports this as a non-fatal `"warn"` on the `stderr_summary` stage (`ok: true`).

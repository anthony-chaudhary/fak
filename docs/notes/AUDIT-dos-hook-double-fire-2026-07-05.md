---
title: "Audit — the dos-hook double-fire (fak .claude/settings.json + enabled"
description: "Audit confirming dos hooks fire twice per Bash call because fak's Python settings duplicate the plugin's native path — a latency tax, not journal corruption."
---

# Audit — the dos-hook double-fire (fak `.claude/settings.json` + enabled `dos-kernel` plugin)

**Issue:** #2703 (part of epic #2702). **Host:** win/amd64, this dev box.
**Captured:** 2026-07-05 (journal ts UTC 2026-07-06T04:xxZ). **Verdict:** double-fire
**confirmed** at the config + process layer; the *durable* step journal is idempotent
(one record/call), but fak's committed copy runs the **slow Python path** and duplicates
work the plugin already does natively.

This is the evidence base for the sibling "decide the boundary" issue. It is a
measurement/audit deliverable, not a cleanup — no boundary change is made here.

## TL;DR

- Both sources wire dos `PreToolUse` / `PostToolUse` / `Stop` for overlapping tools.
  For a **Bash** call, dos hooks fire **twice** (plugin path + fak-settings path).
- The plugin path is **already native-fast** — `dos-hook-windows-amd64.exe` exists and
  runs. fak's `.claude/settings.json` always takes the **Python** path
  (`python -m dos.cli hook ...`), which is ~3.6× slower on pretool and ~10× on posttool.
- The **durable step journal** (`.dos/streams/<session>.jsonl`) records **one** STEP per
  tool call — it is idempotent, so the double-fire is a **latency tax**, not journal
  corruption. The per-invocation **observation metric** (`.dos/metrics/observations.jsonl`)
  is *not* deduped (one row per hook process).
- **Matcher divergence:** fak's `PostToolUse` has **no matcher** → the slow Python posttool
  fires on *every* tool (Read/Grep/Glob/Write/Edit/TodoWrite/MCP…), where the plugin
  deliberately scopes to `Read|Bash|Grep|Glob`.

## Check 1 & 4 — one record or two? (durable journal is idempotent)

`.dos/streams/<this-session>.jsonl` — one `STEP` per tool call, clean monotonic
`step_index` (no duplicate rows despite two hooks wired):

```jsonl
{"op":"STEP","step_index":0,"tool_name":"Bash","ts":"2026-07-06T04:06:05Z", ...}
{"op":"STEP","step_index":1,"tool_name":"Bash","ts":"2026-07-06T04:06:06Z", ...}
{"op":"STEP","step_index":2,"tool_name":"Read","ts":"2026-07-06T04:06:07Z", ...}
...
{"op":"STEP","step_index":12,"tool_name":"Bash","ts":"2026-07-06T04:15:47Z", ...}
```

So the **step stream de-dups** — the second hook does not append a second STEP. But the
observation metric does *not*: a probe that fed the **same** pretool payload through the
Python path 5× produced **5** rows in `.dos/metrics/observations.jsonl`
(`{"dialect":"claude-code","rung":"provenance","verb":"pretool", ...}`), one per process.
Since the two wired paths each run their own process, per-process side effects (observation
rows; the `additionalContext` each returns to Claude Code on stdout) are **not** collapsed
the way the STEP stream is. `additionalContext` double-injection is *plausible* (Claude Code
merges hook outputs) but was **not** separately captured here — see the assumption below.

## Check 2 & 3 — latency table (measured live, n=5, median ms, stdin payload fed)

| Path | pretool | posttool | writes |
|---|---|---|---|
| **NATIVE** `dos-hook-windows-amd64.exe` (plugin) | **189** | **37** | STEP stream |
| **PYTHON** `python -m dos.cli hook …` (fak settings) | **677** | **373** | STEP stream + observation row |

(min/med/max ms: native pre 184/189/240, native post 34/37/91; python pre 610/677/759,
python post 343/373/388. `python -c "import dos"` → `dos ok 0.29.0`.)

Consistent with the issue's baselines (native pre ~326 / post ~28; python pre ~877–1307 /
post ~478–1164) — same shape, faster here (warm caches).

**Combined per Bash call** (both fire): pretool `189 + 677 = ~866 ms`, posttool
`37 + 373 = ~410 ms` → **~1.28 s** of dos-hook latency, of which **~1.05 s** is fak's
Python duplicate of work the plugin already did natively in ~226 ms. **Check 3 answer: yes,
the plugin path is already native-fast; fak's committed copy is the slow tax.**

The native binary exists and runs:
`~/.claude/plugins/cache/dos/dos-kernel/0.28.0/bin/dos-hook-windows-amd64.exe`
(4,530,688 bytes, built 2026-06-21; plugin manifest `dos-kernel` **v0.28.0**). All 7
platform binaries are shipped in `bin/`; the `bin/dos-hook` launcher prefers the native
binary and falls back to Python.

## Check 5 — matcher scope divergence (exact strings)

| Event | fak `.claude/settings.json` (Python) | plugin `hooks.json` v0.28.0 (native-first) |
|---|---|---|
| PreToolUse (dos) | `Bash\|Write\|Edit\|MultiEdit\|NotebookEdit\|mcp__.*dos.*` | **no matcher** (all tools) |
| PostToolUse (dos) | **no matcher** (all tools) | `Read\|Bash\|Grep\|Glob` |
| Stop (dos) | no matcher | no matcher (+ `stop-failure --success`) |
| others | — | `StopFailure`, `SubagentStop`, `UserPromptSubmit` (marker reset), `SessionStart` |

(fak's `.claude/settings.json` also has a *separate*, non-dos PreToolUse `repoguard` entry,
matcher `Bash|Read|Write|Edit|MultiEdit|NotebookEdit` — out of scope here.)

Per-tool fire count for the dos hooks:

- **PreToolUse Read/Grep/Glob** → **1** (plugin only; fak's matcher excludes them).
- **PreToolUse Bash/Write/Edit/MultiEdit/NotebookEdit/mcp-dos** → **2** (plugin native 189 ms + fak Python 677 ms).
- **PostToolUse Read/Bash/Grep/Glob** → **2** (plugin native 37 ms + fak Python 373 ms).
- **PostToolUse everything else** (Write/Edit/TodoWrite/MCP/…) → **1** (fak only) — the
  **slow 373 ms Python posttool fires on tools the plugin intentionally skips.** This is the
  most wasteful divergence: fak's un-matched PostToolUse is strictly broader *and* strictly
  slower than the plugin's scoped native one.

## Boundary-decision inputs (for the sibling issue — not decided here)

- The plugin already delivers native speed for every event fak's settings covers, plus more
  (`StopFailure`/`SubagentStop`/`UserPromptSubmit`/`SessionStart`).
- fak's committed `.claude/settings.json` dos entries are a **pure-Python duplicate**: same
  journal, slower, and its no-matcher PostToolUse over-fires. The only fak-unique PreToolUse
  entry is `repoguard` (non-dos), which the plugin does not provide.
- If the boundary decision removes fak's dos entries and keeps the plugin, retain the
  `repoguard` PreToolUse entry (fak-specific, no plugin equivalent).

## Generation classification (issue-evidence horizon)

- **Promotion evidence:** the audit is complete and reproducible — all five checks answered
  with live witnesses (config strings, native-binary run, latency table, step-journal
  excerpt). The sibling boundary decision is now unblocked. This measurement retires on
  landing; it is a discrete deliverable, not an ongoing program.
- **Demotion / retirement evidence:** none needed — nothing here is a standing program to
  down-scope; the note *is* the terminal artifact for the measurement ask.
- **Invalidating assumption:** the combined-cost figure assumes Claude Code spawns **both**
  hook processes for a single tool event (its documented settings-plus-enabled-plugin merge).
  Each path was measured **independently** and the config overlap is confirmed, but a single
  Bash event was **not** traced side-by-side isolating two live processes on the shared,
  multi-session journal — so "double `additionalContext` injection" stays *plausible, not
  captured*. If Claude Code deduped identical hook commands across sources (it does not for
  distinct commands — Python vs native differ), the second fire would collapse and the tax
  would drop to the single-path numbers.

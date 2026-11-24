---
title: "Campaign record: deveval multi-gold set"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak deveval-multigold (2026-11-23): deveval multi-gold set."
---

# deveval multi-gold set (2026-11-23)

Frozen step-level dev-work eval set: 300 items from 204 sessions, where gold
answers come from four model families and three harnesses instead of Fable
alone (`../deveval-fable-gold-20261122/` has 63 Fable-only items). Each item
carries `gold_model`, `source` and, where the harness records one, `outcome`.
Score it with `fak-deveval run` / `score` exactly like the Fable set.

## Composition

| family | source | gold_model | items | session success signal |
|---|---|---|---|---|
| next_action | claude-code | claude-fable-5-2 | 50 | none |
| next_action | claude-code | claude-opus-5-5 | 50 | none |
| next_action | pi | deepseek-ai/DeepSeek-V4.1-Flash | 50 | none |
| next_action | pi | deepseek-v4.1-flash | 6 | none |
| next_action | opencode | glm-5.3-flash | 43 | `idle_outcome=succeeded` (filtered) |
| next_action | opencode | glm-5.3-flash-free | 5 | `idle_outcome=succeeded` (filtered) |
| next_action | opencode | deepseek-v4.1-flash | 36 | `idle_outcome=succeeded` (filtered) |
| bash_from_intent | claude-code | claude-fable-5-2 | 50 | none |
| bash_from_intent | claude-code | claude-opus-5-5 | 10 | none |

Gold tools: Bash 241, Read 30, Edit 16, Write 8, Grep 2, Skill 2, Agent 1.
18 items are the first tool call of their task.

## Build

```
fak-deveval build \
  --transcripts ~/.claude/projects/C--work-fak-private \
  --pi-sessions ~/.pi/agent/sessions \
  --opencode-db ~/.local/share/opencode/opencode.db --opencode-outcome succeeded \
  --repo-dir 'C:\work\fak-private' \
  --gold-model claude-fable-5-2,claude-opus-5-5,deepseek-ai/DeepSeek-V4.1-Flash,deepseek-v4.1-flash,glm-5.3-flash,glm-5.3,glm-5.3-flash-free \
  --since 2000h --min-bytes 50000 --max-next 240 --max-bash 60 --max-per-gold 50 \
  --out set.jsonl
```

Built 2026-11-23 from 3,639 transcript files plus the OpenCode store; the
window is relative to build time, so a rebuild later yields a different set.
sha256 of `set.jsonl`: `b1bfdaea0f628a45bec440b9c3ecf3433837f545d0b0a0ebae67b2207b318536`.

## Caveats

- **Gold quality differs by model.** Agreement with Opus or Fable is not the
  same claim as agreement with DeepSeek or GLM; report scores per
  `gold_model`, never pooled, and treat non-Claude gold as a peer baseline,
  not ground truth.
- **Success signal is uneven.** Only OpenCode records a session outcome; its
  items come from `succeeded` sessions only. Claude Code and Pi items carry no
  outcome, so a gold step may belong to a session that failed.
- **Shell differs by source.** OpenCode's shell tool is PowerShell: those
  items carry `"shell":"powershell"` and their system prompt (both run
  protocols) says PowerShell; Claude Code and Pi gold commands are bash. Pi
  `powershell` calls appear in history but are never gold.
- **Tool names are normalized** onto the scored vocabulary (Pi `find` → Glob,
  OpenCode `shell` → Bash, `subagent` → Agent, `skill` → Skill; `path` →
  `file_path`).
- **bash_from_intent is Claude Code only**: it needs the Bash `description`
  field, which Pi and OpenCode do not record. The family cap (60) filled with
  Fable first, leaving Opus 10.
- **Main chain only.** Claude Code sidechains and OpenCode child sessions are
  excluded; Claude Code coverage is the main project folder only, not lane
  worktree folders.
- **Weak task context is dropped, not repaired.** Steps after a compaction,
  and resume nudges ("continue", stalled-session sweep resumes) that do not
  restate the task, never become gold under an unknown request. Secrets,
  email addresses and the build host's name are redacted; a gold step
  containing one is skipped.
- **Known extraction limits.** Pi sessions are a tree (`parentId`) read in
  file order, so an abandoned branch can appear in a later item's history.
  OpenCode sessions are picked by their session-level model, so a session that
  switched to a gold model midway is skipped.
- Bash dominates (80%), as in the Fable set; per-tool scores are thin outside Bash.

## More gold available

The host holds far more than this set uses (counts are assistant messages with
a tool call): Claude Code Opus main chain ~98.5k over 919 transcripts
(09-25..11-23) plus ~220k in subagents; OpenCode DeepSeek ~14.5k and GLM
~3.5k; Pi DeepSeek ~69k over 461 sessions (09-17..10-07). Raise
`--max-per-gold` / `--max-next` for a larger set.

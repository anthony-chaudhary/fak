---
name: resume-watchdog-audit
description: The watchdog-watchdog — one read-only pass that proves the resume watchdog (the n¹ layer that revives dead autonomous Claude sessions) is itself ALIVE, TICKING, and reviving PRODUCTIVE instances, versus silently STALLED. Distrusts self-report: reads the live Fleet registry ledgers (%LOCALAPPDATA%\Fleet\registry), the scheduled-task exit codes AND each task's Principal.LogonType, never the watchdog's own "I'm fine". Verifies the n² layer (is the watchdog ticking, is its backlog draining, are resumes witnessed as real transcript turns, or merely launched_unproven) and the n³ layer (is THIS audit itself scheduled/looped and orthogonal to the failure it detects — who watches the watchman's watchman). Turnkey entry point: tools\watchdog_watchdog_audit.ps1 (read-only, emits GREEN/AMBER/RED + exit 0/2/3). Catches the exact failure this repo hit on 2026-07-09: after a boot the watchdog drained the backlog, then every scheduled task with Principal.LogonType=Interactive began returning 0x800710E0 ("operator or...
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/resume-watchdog-audit/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/resume-watchdog-audit/SKILL.md`](../../../.claude/skills/resume-watchdog-audit/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

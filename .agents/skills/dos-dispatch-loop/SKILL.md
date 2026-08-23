---
name: dos-dispatch-loop
description: Run recurring `dos-dispatch` cycles, switching to `dos-replan` when the backlog drains and stopping on the kernel's loop verdict. Use when you need unattended dispatch->replan->dispatch work across disjoint lanes.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/dos-dispatch-loop/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/dos-dispatch-loop/SKILL.md`](../../../.claude/skills/dos-dispatch-loop/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

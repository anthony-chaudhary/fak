---
name: clean-skill
description: Audit a Claude Code skill's per-invocation context use, then propose a context-bundling helper + SKILL.md edits to cut waste. Reads a representative session JSONL, ranks the largest tool results, traces them back to SKILL.md instructions, and proposes a fix. Stops after the proposal — the operator approves before code is written. Use when a skill feels slow, expensive, or "burns context" — typical signal is a SKILL.md over ~300 lines that reads multiple files >5 KB on every run.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/clean-skill/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/clean-skill/SKILL.md`](../../../.claude/skills/clean-skill/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

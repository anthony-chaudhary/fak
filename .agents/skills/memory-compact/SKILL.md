---
name: memory-compact
description: Compact and structure a Claude Code auto-memory store so MEMORY.md stays under the harness load cap (first 200 lines / 25KB load each session — content past that SILENTLY never loads) while every memory stays...
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/memory-compact/SKILL.md
  canonical-description-hash: bb62f1e683355df2091c57d145bb8c6f092edb7be9a1920320fd45711ddd4577
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/memory-compact/SKILL.md`](../../../.claude/skills/memory-compact/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

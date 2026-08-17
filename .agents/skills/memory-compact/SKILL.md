---
name: memory-compact
description: Compact and structure a Claude Code auto-memory store so MEMORY.md stays under the harness load cap (first 200 lines / 25KB load each session — content past that SILENTLY never loads) while every memory stays reachable. Splits the index into two tiers — hot MEMORY.md (laws/preferences/live-keys/in-flight/open-research) + cold MEMORY_archive.md (shipped/fixed/forensic/dated, recalled on demand) — and proves "done" with an integrity witness (check_memory.py) that re-derives both caps + the both-files bijection from disk, never from narration. Use when the operator says "compact memory", "MEMORY.md is too big / keeps truncating / shows as large", "tier the memory", "fix what loads for this project", or after any heavy pass that grew the index. Read-only except the memory files it edits; deletions need explicit sign-off.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/memory-compact/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/memory-compact/SKILL.md`](../../../.claude/skills/memory-compact/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

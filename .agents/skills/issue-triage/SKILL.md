---
name: issue-triage
description: One repeatable pass over the open GitHub issue backlog — classify every open issue (needs-priority / needs-kind / needs-area, orphaned P0-P1, stale, dormant question), rank them into a deterministic "do next" order, propose the mechanical gardening moves (mark stale, close dormant questions), and apply them only on operator approval. The helper is read-only; writing labels, comments, or closes is gated. Use when the operator says "triage the issues", "what should I work on next", "garden the backlog", "the issue labels are a mess", "close stale issues", or on a /loop cadence to keep the backlog honest.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/issue-triage/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/issue-triage/SKILL.md`](../../../.claude/skills/issue-triage/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

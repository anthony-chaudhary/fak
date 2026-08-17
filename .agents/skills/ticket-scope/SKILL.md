---
name: ticket-scope
description: One repeatable pass that decides whether a GitHub ticket is a single dispatchable unit of agent work — or names exactly which of the six scope axes it fails and how to fix it. Wraps the native scope toolkit (`fak issue contract` for structure/size/routing, `fak dispatch issue-smallness-lint` for atomicity, `fak issue cohort` for batch/wave placement) and reads back one verdict per issue: DISPATCHABLE, or TRIAGE (add the missing section), DECOMPOSE (S2+ epic → leaves), or SPLIT (two deliverables / not one witness). Read-only — it fetches, lints, and proposes the fix; editing the issue, splitting it into children, or relabeling is a separate operator-approved step. Use when the operator says "is this ticket ready to dispatch", "scope issue #N", "why won't this issue dispatch", "check the backlog is well-scoped", "split this epic", or on a /loop cadence before a dispatch wave.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/ticket-scope/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/ticket-scope/SKILL.md`](../../../.claude/skills/ticket-scope/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

---
name: wave-harvest
description: The honest closing half of a super loop — after a detached bulk wave (`/super-loop`) has run, HARVEST it: witness what each headless worker actually shipped (not what its log claims), re-queue the leaves that were claimed-done-but-not-shipped, stop workers that are spinning without net gain, and surface any lane a worker stranded dirty. A launch is not a ship, so a bulk loop is only durable if something reconciles its output against git ground truth. Read-mostly — it audits and re-queues; it may stop a spinning PID but never closes an issue by narration. Use when the operator says "harvest the wave", "what did the fleet actually ship", "reconcile the workers", "clean up after the super loop", or on a cadence between waves.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/wave-harvest/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/wave-harvest/SKILL.md`](../../../.claude/skills/wave-harvest/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

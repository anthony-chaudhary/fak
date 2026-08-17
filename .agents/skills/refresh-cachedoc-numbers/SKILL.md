---
name: refresh-cachedoc-numbers
description: Refresh the recent-operational cachevalue numbers in a guarded doc (e.g. docs/integrations/fable5-more-usage-for-free.md) when this-week's telemetry has moved on. Re-derives the frozen snapshots from live `fak cachevalue report`, reconciles the doc's rendered numbers + snapshot_date to the fresh capture, and re-runs the hermetic audit until it is clean. The audit (tools/cachedoc_numbers_audit.py, gated in `make cachedoc-numbers-lint`) binds every rendered number to a committed snapshot field and checks the arithmetic invariants the doc asserts — this skill is the maintenance loop that keeps that binding true as the fleet/dev windows advance. Distrusts the stale doc: the new numbers come from a fresh `--json` capture, never from editing the visible prose in place. Use when the audit WARNs on staleness, when a cachevalue doc looks out of date, or on a cadence to keep the operational docs honest.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/refresh-cachedoc-numbers/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/refresh-cachedoc-numbers/SKILL.md`](../../../.claude/skills/refresh-cachedoc-numbers/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

---
name: stability-score
description: One repeatable pass that keeps fak trustworthy while it iterates fast — the question no other scorecard asks: as we add items quickly, how do we KNOW a regression / tail-wag / confusion landed, and how do we REVERT to a stable version? Runs the stability scorecard (tools/stability_scorecard.py) over the git-tracked tree across four groups — sentinel (a regression turns a gate RED), invariant (the core assumptions are encoded as tests), revert (we can roll back: keep/revert ladder, version pin, CI-gated tags, a documented rollback runbook), and drift (a small thing wagging a big thing gets caught) — turns each HARD defect into a required affordance to ADD (wire the missing CI regression gate, encode the missing invariant test, commit the missing ratchet baseline, write/link the rollback runbook), retires stability-debt worst-first, re-measures + regenerates the snapshot to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The stability counterpart of code-quality (defects) and...
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/stability-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/stability-score/SKILL.md`](../../../.claude/skills/stability-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

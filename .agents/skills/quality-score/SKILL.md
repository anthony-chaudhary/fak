---
name: quality-score
description: One repeatable RSI pass over CODE quality — the code-side counterpart of refresh-readme. Runs the code-quality scorecard (tools/code_quality_scorecard.py), reads the code-debt work-list, retires debt worst-first using ONLY the safe, genuine classes (gofmt, real tests for untested packages, safe god-function extraction), re-measures to PROVE the number dropped, grounds the ship in DOS (dos commit-audit on the new commit, dos review for the ship_integrity KPI), and commits by explicit path. Use to baseline code quality, drive the code-2x program (halve code-debt, then halve again), or on a /loop cadence to keep the kernel from rotting. The code's checking layer, the way refresh-readme is the README's.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/quality-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/quality-score/SKILL.md`](../../../.claude/skills/quality-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

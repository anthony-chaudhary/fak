---
name: milestone-score
description: One repeatable pass that makes the milestone report's CLIMB and ROADMAP retirable by the RSI loop — the milestone counterpart of quality-score (code) and stability-score (trust under iteration). Runs the milestone scorecard (`fak milestone-scorecard --json`) over the report's OWN two dimensions — the distance-from-MATURED climb shortfall across the M0..M7 support-maturity grid PLUS the un-progressed tracked-epic roadmap gaps — folds them into one deterministic milestone_debt integer + a worst-first milestone_worklist, retires debt worst-first (climb the lowest-rung cell to M4, then close the most-open discrete epic), re-pins the climb ratchet on a real climb improvement, re-measures with --compare to PROVE the debt dropped, and commits only the milestone lane by explicit path. COMPOSES — does NOT duplicate — support-maturity (which fences each cell to its regime ceiling); milestone_debt scores raw distance-to-MATURED across the grid as the headline climb, alongside the roadmap. Use after a cell climbs a...
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/milestone-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/milestone-score/SKILL.md`](../../../.claude/skills/milestone-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

---
name: steerability-score
description: One repeatable pass that keeps fak as STEERABLE as it grows — the one scorecard whose every KPI is growth-invariant, so a 2x-larger repo with the same discipline scores the same. Runs the steerability scorecard (tools/steerability_scorecard.py) over the working tree, reads the 0-100 steerability index + the advisory drift signals (coupling hubs, p90 sizes, long-function rate, package drift, churn hot spots), drives the index UP and the worst drift axis DOWN by adding REAL modularity (split a cmd dispatch monolith along its verb seams, break a coupling hub, document a package header) — never by gaming a detector, re-measures to PROVE the index rose, and commits only the scorecard lane by explicit path. The growth-invariant counterpart of code-quality (absolute defects) and repo-hygiene (tree shape). Use after a structural change (a new package, a coupling edge into a hub, a growing dispatch file), when the project "feels" harder to change than its size warrants, or on a /loop cadence to keep steering...
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/steerability-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/steerability-score/SKILL.md`](../../../.claude/skills/steerability-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

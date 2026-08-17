---
name: appeal-score
description: One repeatable pass that makes a doc read like a person wrote it, not a model. Runs the doc-appeal scorecard (tools/doc_appeal_scorecard.py), turns each HARD defect into a required edit (em-dash flood, bold-emphasis flood, run-on / overlong sentences, walls of text, stacked "X, not Y" contrast frames, a dense or unanchored lead, LLM-scaffolding phrases) and each SOFT signal into a judgment call, retires appeal-debt worst-axis-first WITHOUT changing any claim, number, or link, re-measures to PROVE the debt dropped, and commits only the doc lane by explicit path. The prose-voice counterpart to refresh-readme (freshness) and quality-score (code). Use to de-LLM-ify the README or any reader-facing prose doc, or on a /loop cadence to keep the front door human.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/appeal-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/appeal-score/SKILL.md`](../../../.claude/skills/appeal-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

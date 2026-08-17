---
name: negframe-score
description: One repeatable pass that keeps agent-steer prose leading with the AFFORDANCE, not the prohibition. Runs `fak score negframe` over the steer-prose corpus (AGENTS.md, CLAUDE.md, the skills, or explicit paths), reads the negframe_debt (mechanical negatives with an unambiguous positive rewrite) plus the judgement-tier soft signal, retires the mechanical debt worst-first by applying the suggested reframe, checks the `--since <ref>` ratchet before landing a steer-prose change, re-measures to PROVE the debt dropped, and commits only the scorecard lane by explicit path. Use after editing AGENTS.md/CLAUDE.md/a skill, when a new negatively-framed directive is proposed, or on a /loop cadence to keep steer prose reading as "do this" instead of "don't do that."
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/negframe-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/negframe-score/SKILL.md`](../../../.claude/skills/negframe-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

---
name: persona-score
description: One repeatable pass that keeps fak serving the top-10 personas who land on it — from the free-tier dev who downloads a binary and won't read a word, through the infra engineer who has to operate it, to the researcher who wants to reproduce it. Runs the persona-readiness scorecard (tools/persona_readiness_scorecard.py) over the git-tracked tree, turns each unmet HARD affordance into a required thing to ADD (a prebuilt-binary release, a deployment guide, a determinism witness, a refusal vocabulary, a green gate), retires persona-debt worst-served-first, re-measures + regenerates the snapshot to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The go-to-market counterpart of agent-readiness (one persona: the AI agent), product (the concepts), and industry-score (the field). Use after a change to a persona's entry path (a release, a deploy doc, an integration recipe, the policy spec), when adding a persona to the roster, or on a /loop cadence to keep every front door open.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/persona-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/persona-score/SKILL.md`](../../../.claude/skills/persona-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

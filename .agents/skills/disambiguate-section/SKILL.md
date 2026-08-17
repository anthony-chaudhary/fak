---
name: disambiguate-section
description: Apply the concept-disambiguation pass to ANOTHER section of fak - an under-covered watched family (gateway-engine, guard-gate, ...) or a newly-discovered overloaded root (evict, decision, render, plan, pool, layout) - and, when a section is bigger than one sitting, file a GitHub ticket that scopes it with real context. Two modes over the same scorecard (tools/concept_disambiguation_scorecard.py): APPLY picks the worst-covered section from --gaps, surfaces its genuinely-DISTINCT confusable concepts (filtering inflections), adds grounded + glossary-anchored rows, draws the distinctions, re-measures to PROVE coverage rose, and commits the lane by explicit path; TICKET enumerates the sections that still need the pass (under-covered families + missed roots + cross-cluster canonical collisions) and opens one GH issue per section with the confusable tokens, file evidence, and scorecard-tied acceptance criteria. Use after running disambiguation-score when the backlog is too large for one pass, to onboard a new overloaded root, or to turn the coverage backlog into tracked work.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/disambiguate-section/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/disambiguate-section/SKILL.md`](../../../.claude/skills/disambiguate-section/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

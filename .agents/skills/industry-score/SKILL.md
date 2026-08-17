---
name: industry-score
description: One repeatable pass that keeps fak's competitive story honest AND complete — graded industry-first, not from what fak happened to measure. Runs the industry scorecard (tools/industry_scorecard.py) over a modular data directory (tools/industry_scorecard.data/): a researched taxonomy of the dimensions the LLM-serving / agent-infra field competes on (vLLM/SGLang/TensorRT-LLM/llama.cpp), the current SOTA bar on each with a dated source, and fak's honest position — mostly named gaps. It drives two numbers, coverage (of the field) and parity-debt (honesty of the rows), and updates on two cadences: as the industry moves (a new dimension drops coverage; --stale lists SOTA bars due a re-check) and as fak moves (a benchmark turns a no-claim into a measured row). Regenerates the modular doc folder docs/industry-scorecard/ and commits only the scorecard lane by explicit path. The OUTWARD-facing counterpart of repo-hygiene/code-quality/appeal. Use after a benchmark lands, when a competitor ships a number, when the field adds a dimension, or on a /loop cadence. Score both common-case default quality and bounded-superset coverage for legitimate user cohorts; do not confuse a feature pile with supported coverage.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/industry-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/industry-score/SKILL.md`](../../../.claude/skills/industry-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

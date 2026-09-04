---
name: debt-clean
description: One repeatable, evidence-backed pass that retires a bounded batch of maturity debt worst-first across the system's dedicated debt lanes. Inspects `fak debt-lanes`, targets the highest-carrying-cost hotspots (or compounding-interest core lanes), advances maturity with tests, integration, and benchmarks, re-measures with `--compare` to prove the denominator was level-set and total debt dropped, and commits by explicit path with `(fak <leaf>)`. Use when cleaning or retiring maturity debt across units of work.
metadata:
  canonical: ../../../.claude/skills/debt-clean/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/debt-clean/SKILL.md`](../../../.claude/skills/debt-clean/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

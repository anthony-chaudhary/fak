---
name: modularize
description: One focused, repeatable pass that retires the code-quality scorecard's `architecture` debt — the god-files (>1500 lines) and god-functions (>200 lines) that /quality-score flags as RISKY and explicitly defers to "a focused pass". Splits a monolith along REAL concern seams via behavior-preserving code motion (the goimports recipe: hazard-check → plan boundaries with tools/godsplit_plan.py → sed-extract → goimports -w → gofmt → prove-no-decl-dropped with tools/refactor_verify.py → verify → prove → commit by explicit path), and extracts long functions into named helpers (linear flow → helper + struct-unpack) or a named state struct with methods (closure-soup). Re-measures to PROVE architecture-debt dropped, verifies behavior with go build + go vet + go test, and commits ONLY the touched packages by explicit path. The architecture-KPI focused pass /quality-score points to. Use when `architecture` is the heaviest KPI, to drive code-debt toward 0, or on a /loop cadence to keep the kernel from re-accreting...
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/modularize/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/modularize/SKILL.md`](../../../.claude/skills/modularize/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

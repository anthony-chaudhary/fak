---
name: steer-prs
description: The operator loop over the steer-prs overlay (`fak steer prs`) — fak's read-only view that folds the pending dev->release trunk delta into PR-sized units per (fak <leaf>) ship-stamp and renders them WORST-ATTENTION-FIRST (RESIDUAL -> UNVERIFIABLE -> CLEARED). Teaches the four-step loop (run the view and read worst-first; apply the REGIME GATE — a CLEARED unit with a healthy curve is a reason to do nothing; pick the WEAKEST SUFFICIENT RUNG on the observe -> comment -> ack -> redirect -> pause ladder; confirm the effect on the next tick) and names the anti-gaming laws (an ack is not a witness; the residual pile falls when work gets WITNESSED, not when it gets acked; the overlay gates nothing). Positioned against `fak release prplan` (the release-time promotion twin, biggest-first) and /trajectory-control (steering a live declared objective by its curve): steer-prs reads the FORMING units on the trunk and decides whether any owes a human look. Use when the operator says "what is forming on the trunk", "which units owe me a look", "should I intervene in this intent", or on a /loop cadence over a shared trunk that many agents commit to.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/steer-prs/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/steer-prs/SKILL.md`](../../../.claude/skills/steer-prs/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

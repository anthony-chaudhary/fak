---
name: trajectory-control
description: The operator on-ramp to trajectory control (`trajctl`) — fak's live, forward-progress control plane over a DECLARED objective. Teaches the one primitive that carries the family (anything you want to progress gets a named, witnessed score, and every move is either improve-the-score or improve-the-scorer), then the four operator loops: declare an objective + plan + budget, read the CURVE (never a point) for the closed signal vocabulary (HEALTHY / STALL / DRIFT / DETOUR_OVERRUN), apply the when-to-nudge doctrine (the REGIME GATE — a healthy curve means do nothing; intervening in a high-scoring run is harm), budget a detour as a child objective that must RETURN, and run the meta loop (score the scorers against witnessed outcomes). Positioned against trajectory-audit (retrospective token/cost sweep of past transcripts) and trajectory-garden (gardening a recorded corpus): trajectory-control steers a LIVE objective forward, it does not audit or prune the past. Use when the operator says "keep this long session on its goal", "why did the agent drift", "should I interrupt this run", "budget the side-quest", "declare an objective and score progress", or on a /loop cadence to keep a long-horizon session or sub-agent fleet pointed at what it was asked to do.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/trajectory-control/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/trajectory-control/SKILL.md`](../../../.claude/skills/trajectory-control/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

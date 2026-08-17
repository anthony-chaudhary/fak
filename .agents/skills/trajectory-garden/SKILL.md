---
name: trajectory-garden
description: One repeatable gardening pass over a trajectory corpus — the JSONL of per-turn Turn rows a fak trajectory.Recorder exports. Uses the `fak traj` toolkit (the data plane + the simhash reference vector-similarity primitive + the pluggable scorer seam) to find the trajectories worth a human's attention — near-duplicate queries the lexical ranker misses, cost outliers, and traces the kernel kept refusing — then PROPOSES prune candidates (it never deletes). The reference application of fak's trajectory observability primitives: proof that a trivial skill can build memory/trajectory gardening ON TOP of the defaults, and the starting point you fork to add your own scorers. Use after a recording run, when a memory store has bloated with redundant work, or on a /loop cadence to keep an agent's trajectory memory lean and its bad-query clusters visible.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/trajectory-garden/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/trajectory-garden/SKILL.md`](../../../.claude/skills/trajectory-garden/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

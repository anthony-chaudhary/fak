---
name: default-push-and-tracked-issues
description: Default shipping and follow-up tracking posture for this repo.
metadata:
  type: project
  recorded: 2026-06-30
---

In this repo, finished green work should default to committing and pushing when the
guards allow it. Shared trunk moves quickly; leaving completed work local makes the
fleet fall behind and breaks issue/closure witnesses. If push is blocked by a real
guard state, reconcile in place or surface the blocker instead of silently stopping.

When follow-up work is real and actionable, create or update actual GitHub issues
with scope, non-goals, witness, and routing context. A docs note alone is not enough
for work the dispatch loop is expected to drain.


---
name: curate-cluster
description: One curation pass over a research/doc cluster — reconcile the project's index doc (e.g. INDEX.md) against the docs and scripts actually on disk (add missing entries in the house format, fix dangling references, refresh counts/date/git-context), gitignore regenerable build artifacts, then commit ONLY the quiescent curation lane (docs/experiments/tools/index) by explicit path — never an actively-built code tree. Concurrency-safe by construction: when the repo is a live multi-session tree, the pass excludes any file-tree a peer is writing right now. Use after a burst of doc-writing, when the index drifts from disk, or on a /loop cadence to keep the cluster clean and indexed.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/curate-cluster/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/curate-cluster/SKILL.md`](../../../.claude/skills/curate-cluster/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

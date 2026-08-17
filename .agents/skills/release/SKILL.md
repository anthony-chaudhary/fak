---
name: release
description: Perform a full versioned release — bump version, draft release notes, commit, tag, push, and create the GitHub release page. Reads `.claude/project.yaml` for the project's release-context and version-bump helpers; the skill text is universal, the helpers are project-supplied. Use when the user says "cut a release", "ship vX.Y.Z", "release", or after a shippable phase.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/release/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/release/SKILL.md`](../../../.claude/skills/release/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

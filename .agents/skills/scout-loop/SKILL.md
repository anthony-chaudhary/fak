---
name: scout-loop
description: The super-loop that closes the research→backlog loop — it chains the outward CRAWLERS (the daily `idea-scout` arXiv/GitHub feed, the industry scans, the RESEARCH/CONCEPT corpus) into the STUDY pipeline (`/study-repo` → `/field-borrow`) and runs the whole thing on a cadence. The crawler surfaces repo-shaped leads into a needs-triage queue and stops; turning any one into scoped, witnessed, license-clean backlog is still a manual pass someone has to remember to run. This skill is that seam, automated: once per pass it CRAWLS the freshest outward signal, SELECTS the single highest-value repo-shaped lead, STUDIES it with `/study-repo` (clone into scratch, pin the SHA, read the code not the pitch, decompose small), WITNESSES each borrow with `/field-borrow` (`fak_feature_query`/`fak index` → PRESENT/PARTIAL/ABSENT), FILES the surviving PARTIAL/ABSENT borrows as small independently-shippable leaves under the right epic, and REGISTERS a dated `CONCEPT-STUDY-*` note. It re-implements none of those tools — it...
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/scout-loop/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/scout-loop/SKILL.md`](../../../.claude/skills/scout-loop/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.

---
title: "Good first popularization tasks: small, self-contained ways to help fak spread"
description: "A curated on-ramp of small, well-scoped contributions a newcomer can finish in an afternoon — add an objection answer, a glossary term, a comparison row, a translation, an integration recipe — each entry naming the exact file to touch and its difficulty, so an outside reader can become an invested contributor in one sitting."
slug: good-first-popularization-tasks
keywords:
  - good first issue
  - contributor onramp
  - popularization tasks
  - fak community
  - how to contribute to fak
  - starter tasks
  - documentation contribution
  - verify don't trust
date: 2026-07-03
---

# Good first popularization tasks

> **TL;DR:** popularity compounds through contributors, not just readers. This is
> the on-ramp for small wins. Every task below is scoped so one person can finish
> it in an afternoon without touching anyone else's work. Each names the exact file
> to edit and how hard it is. Pick one, edit the file, open a PR.

This is dimension **E — Social proof & community** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
It serves the *verify, don't trust (DOS)* concept: the trust kernel is what lets a
first-time contributor's change be accepted on git evidence rather than on anyone
vouching for them, so the on-ramp can stay open by default.

## Before you start

Read [CONTRIBUTING.md](../../CONTRIBUTING.md) once. The two rules that will bite you
if you skip them: sign off every commit with `git commit -s` (DCO), and commit by
explicit path (`git commit -- <file>`), never `git add -A`. That is the whole
ceremony for a doc-only change.

Every task here is a doc or content edit. None of them require building the Go kernel
or running a GPU. If you can edit a Markdown file, you can finish one of these.

## The tasks

Each row names one existing file and a difficulty. Starter means a single paragraph
or row with no research. Moderate means you write one short original section and keep
it consistent with a sibling doc.

| # | Task | File to edit | Difficulty |
|---|------|--------------|------------|
| 1 | Add one objection fak draws and a crisp, honest one-to-two-line answer | [docs/adoption/objections.md](objections.md) | starter |
| 2 | Add one glossary term with a single-sentence definition | [docs/explainers/glossary.md](../explainers/glossary.md) | starter |
| 3 | Add a searchable keyword to an explainer's SEO front-matter | [docs/explainers/tool-call-is-a-syscall.md](../explainers/tool-call-is-a-syscall.md) | starter |
| 4 | Add a launch-post title option or one-line hook | [docs/adoption/launch-kit.md](launch-kit.md) | starter |
| 5 | Add a real directory or listing where fak belongs | [docs/adoption/directories.md](directories.md) | starter |
| 6 | Add a new locale stub link to the translation index | [docs/i18n/README.md](../i18n/README.md) | starter |
| 7 | Add one row to the capability matrix, with an honest per-cell note | [docs/adoption/compare/matrix.md](compare/matrix.md) | moderate |
| 8 | Translate one explainer's TL;DR into the German front door | [docs/i18n/de/README.md](../i18n/de/README.md) | moderate |
| 9 | Add a persona rung to the pitch ladder | [docs/adoption/pitch-ladder.md](pitch-ladder.md) | moderate |
| 10 | Add a comparison note to the routers page | [docs/adoption/compare/vs-routers.md](compare/vs-routers.md) | moderate |
| 11 | Add an integration recipe for a harness we do not yet cover, following the Aider recipe as a template, then link it from the integrations index | [docs/integrations/README.md](../integrations/README.md) | moderate |
| 12 | Add a one-paragraph note to a runnable example's README so a newcomer knows what it proves | [examples/playground/README.md](../../examples/playground/README.md) | starter |

## The one rule that keeps a task acceptable

Do not overclaim. fak's honest scope is fenced: the detector is evadable by design,
the prior-art audit found 0 of 29 mechanisms novel, and the headline speedup is the
tuned ~4.1x, never the naive 60x. Quote witnessed numbers, label simulated as
simulated, and where an objection lands, concede it. A change that overclaims will be
sent back; a modest, correct one gets merged. This is the *verify, don't trust* idea
applied to your own PR.

## Want a tracked ticket instead?

The tasks above are always-available edits — they never run out, because each one adds a
row or a paragraph to a file that is never finished. If you would rather claim a numbered
issue that nobody else is on, check the newcomer label:

```
gh issue list --repo anthony-chaudhary/fak --label "good first issue" --state open
```

That is the maintained newcomer queue and the one to check first.

This page used to send you to `--label popularization` instead. Do not start there: every
one of that label's 55 issues is now closed, so the command returns an empty list. The
label is kept only as a record of the finished batch.

If the `good first issue` list is empty on the day you look, that is a restocking lapse on
our side, not a shortage of work — take any row from the table above instead. Those need no
ticket and no assignment: edit the file and open the pull request.

## Related

- [CONTRIBUTING.md](../../CONTRIBUTING.md) — the durable contract for every change.
- [The pitch ladder](pitch-ladder.md) — the canonical fak pitch you will echo in most of these.
- [Objections and one-line answers](objections.md) — the honest-rebuttal card task 1 extends.
- [Concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md) — the framing all 50 tickets share.

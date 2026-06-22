---
name: appeal-score
description: One repeatable pass that makes a doc read like a person wrote it, not a model. Runs the doc-appeal scorecard (tools/doc_appeal_scorecard.py), turns each HARD defect into a required edit (em-dash flood, run-on / overlong sentences, walls of text, stacked "X, not Y" contrast frames, LLM-scaffolding phrases) and each SOFT signal into a judgment call, retires appeal-debt worst-axis-first WITHOUT changing any claim, number, or link, re-measures to PROVE the debt dropped, and commits only the doc lane by explicit path. The prose-voice counterpart to refresh-readme (freshness) and quality-score (code). Use to de-LLM-ify the README or any reader-facing prose doc, or on a /loop cadence to keep the front door human.
---

# appeal-score — make the prose read human, and prove it

> **What this does.** Readers (and answer engines) bounce off prose that reads
> like a machine wrote it: an em-dash in every sentence, run-ons that never
> breathe, walls of text, the same "X, not Y" contrast frame over and over,
> formulaic throat-clearing ("here's the thing", "at its core"). The freshness
> auditor checks whether a doc is *correct*; this pass checks whether it *lands*:
> whether a cold reader gets it, trusts it, and can try it without slogging
> through machine prose. It makes "read more human" a **repeatable, provable
> pass** instead of a one-time vibe edit that decays.

The shape: **run the scorecard → fix every HARD defect → weigh every SOFT signal →
re-measure to prove appeal-debt dropped → commit only the doc lane.**

The headline number is **appeal-debt**: the count of concrete, re-derivable prose
defects. Drive it toward zero and "more human" becomes a number you moved, not a
claim you made.

---

## The one rule that overrides everything: never change the claim

This is a **voice** pass, not a content edit. The repo keeps an honesty ledger,
so the prose may change but the meaning may not:

- **Every number, claim, and capability statement stays identical.** Splitting a
  sentence or cutting an em-dash must not drop, weaken, soften, or invent a
  figure. `9.7×` stays `9.7×`; a `[SIMULATED]` caveat stays; an honest fence
  ("self-host only", "design targets, not measurements") stays.
- **Every link target stays byte-identical.** Verify before and after:
  `grep -oE '\]\([^)]+\)' DOC | sort -u` must match. You are rewording prose, not
  re-pointing links.
- **Simplify the words, never the claim** (the Feynman law, shared with
  `refresh-readme`). Lead with the plain idea, name the term in parens on first
  use. Accuracy is non-negotiable.

If a fix would require changing a claim, **stop**. That defect is out of scope
for this pass.

---

## What counts as an LLM tell (the HARD defects this pass retires)

Each is one unit of appeal-debt, deterministic and re-derivable. Retire them
worst-axis-first:

| Tell | Axis | The human fix |
|---|---|---|
| **Em-dash flood** (past ~1 per 200 words) | voice | Replace most with a period, comma, colon, or parentheses. Keep a few where the pause truly earns it. |
| **Overlong sentence** (≥40 words) / **run-on** (≥5 commas) | clarity | Split into two. Pair list items with "and" to cut commas; turn a 6-item enumeration into a real bulleted list. |
| **Wall of text** (a prose paragraph >110 words) | scannability | Break into 2–3 paragraphs, or lift the structure into a list/table. (A genuine multi-bullet list is exempt — it's already scannable.) |
| **Stacked contrast frame** ("X, not Y" / "not X, it's Y" past a budget of 2) | voice | Vary the shape: "rather than", "but not", or just rewrite. One or two are punchy; a habit is a tic. |
| **LLM-scaffolding phrase** ("here's the thing", "at its core", "it's important to note", "when it comes to") | voice | Cut it, or say the thing plainly. A real Feynman move ("think of it as…", a concrete example) is good and is NOT a tell — keep those. |
| **Cliché / marketing AI-tell** ("leverage", "seamless", "robust", "best-in-class") | voice | Use the plain word. |

**SOFT signals** (passive-voice density, low sentence-length variety, a
bold-emphasis flood, hedging, inconsistent name casing) lower the score but are
**never** appeal-debt; they are writing judgment, not mechanical fact. Weigh
them rather than grinding on them.

---

## Step 1 — Run the scorecard (it builds your work-list)

From the repo root:

```bash
python tools/doc_appeal_scorecard.py                       # human scorecard for README.md
python tools/doc_appeal_scorecard.py --target docs/FAQ.md  # any other doc
python tools/doc_appeal_scorecard.py --json                # machine payload (the loop uses this)
```

It scores five axes (clarity · priority · voice · scannability · organization)
into an **appeal score** (0–100, A–F) and an **appeal-debt** integer, and prints
the work-list: every HARD defect with its line and a snippet, then the SOFT
signals. It is read-only and never edits the doc.

## Step 2 — Retire appeal-debt worst-axis-first

Take the weakest axis first (the scorecard names it). For each HARD defect, apply
the human fix from the table above. After a batch, **re-run the scorecard** and
watch the number fall; that loop (fix, re-measure, fix again) is the whole method.
A buried lead or a missing TL;DR (the `priority` axis) is worth a one-line
summary block near the top that a skimmer or an answer engine can lift.

## Step 3 — Weigh the SOFT signals, then stop

Read the SOFT list once. Fix the cheap, real ones (a stray cliché, a monotone run
of same-length sentences). Do **not** chase SOFT signals to zero. Over-correction is its own tell. Prose chopped into robotic staccato to dodge a metric reads worse
than the original; a human writer varies sentence length and keeps the occasional
long sentence. Aim for "a person wrote this", not "the linter is happy".

## Step 4 — Re-measure and confirm the drop

Re-run the scorecard. State the before/after (e.g. "appeal-debt 70 → 2, voice
24 → 97"). Re-confirm links are byte-identical and no number moved. For the README
specifically, also re-run `tools/readme_freshness_audit.py` so a voice edit didn't
trip a freshness FAIL (a dead link, a stale version pin), and re-stamp the
`appeal-verified` marker near the top with today's date and the new score.

## Step 5 — Commit only the doc lane, by explicit path

This is a shared trunk; commit *your* doc, never a peer's work:

```bash
git commit -s -F <msgfile> -- README.md          # options BEFORE --, paths AFTER
```

- **Stage by explicit path, never `git add -A`.** Stage and commit in one shell
  call so a peer's bare commit can't sweep your staged files.
- **Doc-only diff → a `docs(scope): <verb> …` subject** (lead with a recognized
  verb: rewrite/cut/split/trim), end with the `(fak <leaf>)` trailer. A
  code-effect prefix on a docs-only diff overclaims.
- **On Windows, pass the message via a file** (`-F`), not a here-string.
- **If a peer's `MERGE_HEAD` is set**, wait for it to clear; don't abort or work
  around it. Then commit by explicit path.
- **Stay on the trunk (`main`)**; push promptly.

---

## Scope: what this pass touches, and what it must not

- **In scope:** reader-facing **prose** docs, `README.md` first (the front door),
  then the next-most-important by importance × debt (`docs/FAQ.md`,
  `GETTING-STARTED.md`, `docs/fak/tutorial.md`, `docs/concepts-and-story.md`, the
  `docs/explainers/*`).
- **Out of scope: structured ledgers and generated docs.** In a file like
  `CLAIMS.md` the `—` in `- [TAG] claim — detail` lines is a **format separator**,
  not a voice tell; "fixing" it would corrupt the ledger (and trip `claims-lint`).
  The scorecard over-counts em-dashes in such list-structured docs, so judge by
  eye and skip them. For anything machine-generated, fix the generator rather than
  its output.

## When to run this

- After a release, a headline-number change, or any substantial README edit.
- When a doc reads dense or "written by a model" and you want it provably human.
- On a `/loop` cadence over the core prose docs to keep them from drifting back
  toward machine voice.

The scorecard is the README's voice checking-layer, the way `refresh-readme` is
its freshness layer and `quality-score` is the code's. Same discipline: a number
you can move, and prove you moved.

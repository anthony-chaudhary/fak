---
title: "Negframe positivity audit of the steer corpus (#3543)"
description: "A --suggest review + positivity audit of AGENTS.md, CLAUDE.md, and .claude/skills/**: mechanical gate GREEN (debt 0); the 702 judgement-tier negatives are intentional emphatic guard language, so the card advises on them, it does not gate."
---

# Negframe positivity audit of the steer corpus (#3543)

*2026-07-09. Docs audit. Scanned with `internal/negframe` via the throwaway
`cmd/negframescan` harness (identical lexicon to `fak score negframe`).*

## Reproduce

From the repo root:

```
go run ./cmd/negframescan
```

No arguments = the default steer-prose corpus resolved by
`negframe.ResolveTargets` — **56 files**: `AGENTS.md`, `CLAUDE.md`, and the
skills under `.claude/skills/**`. Package sanity: `go test ./internal/negframe/`
(GREEN, `ok github.com/anthony-chaudhary/fak/internal/negframe`).

Once `cmd/fak` builds again, the shipped surface for this same review is
**`fak score negframe --suggest`** (`cmd/fak/negframescore.go`): the
review-diff view. It walks the corpus (or explicit positional paths), keeps
**only the MECHANICAL-tier findings** — the negatives with a confident
positive rewrite — groups them by document, and prints each as a
review-diff pair:

```
<path>:
  L<line> [<category>]
    - <the negatively-framed original>
    + <the positive reframe>
```

ending with `N mechanical reframe suggestion(s)`. This is the
maintainer-facing "apply these by hand" list. Judgement-tier findings are
deliberately excluded from `--suggest`. When there are zero mechanical
findings it prints `no mechanical reframe suggestions -- steer prose already
leads with the affordance` and exits 0. (Siblings on the same card:
`--per-doc` for the full per-document breakdown, `--since <ref>` for the
diff-scoped CI ratchet that exits 1 only on *newly introduced* mechanical
negatives.)

## Result: mechanical gate GREEN

```
# TOTAL mechanical=0 judgement=702 across 56 path(s)
```

**Mechanical debt = 0.** Not one finding in the whole steer corpus carries a
confident positive rewrite — the equivalent `--suggest` run would print the
"no mechanical reframe suggestions" clean line. No reframe edits were made to
`AGENTS.md` or `CLAUDE.md` (the only condition under which this audit would
touch them).

## Judgement-tier tally by category

702 judgement (SOFT, advisory-only) findings across 55 of the 56 files:

| Category | Count | Positive-shape hint |
|---|---:|---|
| prohibition | 630 | lead with the action to take instead of the one to avoid |
| absence | 65 | name what is present or required, not what is missing |
| refusal | 7 | state the permitted path first, then the boundary |
| hedge | 0 | assert the property directly instead of by double negative |

### prohibition (630) — representative examples

- `AGENTS.md` L194: `- **Work directly on the trunk (`main`). Never open a feature branch or new worktree.**`
- `AGENTS.md` L166: `` `MERGE_HEAD`, and **never force-push**. If a guard refuses (`OFF_TRUNK`), a peer merge is ``
- `CLAUDE.md` L15: `- **Commit by explicit path** — `git commit -- <paths>`, never `git add -A` (shared`

### absence (65) — representative examples

- `AGENTS.md` L150: `` `fak issue cohort --from-plan`. The verb refuses to plan without a spine witness — that ``
- `.claude/skills/agent-readiness/SKILL.md` L97: `**SOFT signals** (a missing `llms-full.txt`, a tool without `--json`) lower the`
- `.claude/skills/appeal-score/SKILL.md` L13: `> whether a cold reader gets it, trusts it, and can try it without slogging`

### refusal (7) — representative examples

- `CLAUDE.md` L14: `and any other off-trunk commit stay forbidden. (Details in [`AGENTS.md`](AGENTS.md).)`
- `AGENTS.md` L210: `a feature branch, or any worktree that commits off-trunk — stays forbidden.`
- `.claude/skills/study-repo/SKILL.md` L137: `all-rights-reserved) → you **may not integrate**; fall back to INSPIRE. When in doubt,`

### hedge (0)

No double-negative assertions anywhere in the corpus.

## Conclusion: these negatives are intentional emphasis — advise, do not gate

Reading the findings against their sources, the judgement tier is dominated by
the repo's **load-bearing guard language**: the trunk law ("**never** open a
feature branch"), commit discipline ("never `git add -A`", "never
force-push"), and each skill's safety rails ("read-only; it never edits the
tree"). These are prohibitions *on purpose* — the emphatic negative is the
teeth of the rule, and each sits next to the positive affordance it protects
(work on `main`; commit by explicit path; the sanctioned detached-worktree
flow). The small absence/refusal tails are the same pattern: boundaries stated
as boundaries, with the permitted path already named nearby.

Mass-reframing them would weaken meaning, not improve it. That matches the
card's own two-tier design: only a **mechanical** negative (an unambiguous
positive rewrite, e.g. "don't forget to X" → "remember to X") counts as
`negframe_debt` and gates; judgement findings carry only a category hint. The
correct standing posture, confirmed by this audit:

- **Gate** on the mechanical tier — currently 0, and kept at 0 going forward by
  the `--since <ref>` ratchet on new edits.
- **Advise** on the judgement tier — a per-edit writing prompt via the category
  hints, never a target to grind to zero.

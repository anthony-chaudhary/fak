---
name: negframe-score
description: One repeatable pass that keeps agent-steer prose leading with the AFFORDANCE, not the prohibition. Runs `fak score negframe` over the steer-prose corpus (AGENTS.md, CLAUDE.md, the skills, or explicit paths), reads the negframe_debt (mechanical negatives with an unambiguous positive rewrite) plus the judgement-tier soft signal, retires the mechanical debt worst-first by applying the suggested reframe, checks the `--since <ref>` ratchet before landing a steer-prose change, re-measures to PROVE the debt dropped, and commits only the scorecard lane by explicit path. Use. Use when this named workflow matches the task.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--per-doc] [--suggest] [--since HEAD~N] [--json] [--markdown]  (no args = default scorecard)"
---

# /negframe-score - lead with the affordance, not the prohibition

> An instance of the generic **[scorecard](../scorecard/SKILL.md)** doctrine. Read that
> for the five laws and the RSI loop; this file is the per-surface specialization.

## Why this scorecard exists

Prose that steers an agent lands better when it leads with the ACTION to take instead of
the thing to avoid. A reminder framed as an omission to fix ("remember: the commit stamp
often gets skipped") makes the reader build a negative and invert it; "remember to stamp
the commit" is the instruction already in the form the reader acts on. Across a whole
steer-prose corpus - the guard runtime prose injected into
every session, AGENTS.md, CLAUDE.md, the skills, the refusal vocabulary - that inversion
tax compounds. `internal/negframe` is a TEXT-READING card: the prose on disk IS the data,
so a finding can only be retired by reframing the sentence, never by editing a JSON file.

## The two tiers (read this before touching anything)

Every finding lands in exactly one tier - do not conflate them:

- **MECHANICAL (HARD, gates `negframe_debt`)** - the finding matches a fixed idiom with an
  unambiguous positive rewrite (a "forget-to-remember-to" idiom, or a double negative like
  "not un-<adjective>" that collapses to the plain adjective). The card hands back the
  exact `Suggest` string; apply it verbatim or a close paraphrase that preserves the
  meaning. This is the only tier that counts as debt.
- **JUDGEMENT (SOFT, advisory only)** - negatively framed (a bare "never merge without
  review", "not allowed") but with no confident mechanical rewrite. The card names the
  `Category` (prohibition / absence / refusal / hedge) and a one-line `Hint` - the positive
  SHAPE to aim for - but never a literal reframe, and it never gates. Fix the cheap real
  ones by hand; do not chase this tier to zero, and never invent a mechanical rule just to
  move a soft count (see anti-gaming below).

## The pass (the shared five-step loop)

1. **Run it** - from the repo root:

   ```
   go run ./cmd/fak score negframe                 # human scorecard: debt + per-KPI
   go run ./cmd/fak score negframe --json           # machine payload (Corpus.negframe_debt / grade)
   go run ./cmd/fak score negframe --per-doc         # per-document finding breakdown (mechanical + judgement)
   go run ./cmd/fak score negframe --suggest         # review-diff view: every mechanical finding with its reframe
   ```

   Default scope is the steer-prose corpus (`negframe.ResolveTargets`: AGENTS.md, CLAUDE.md,
   the skills); pass explicit paths as positional args to score anything else. `--per-doc`
   also takes `--json` for a machine-readable finding list; `--suggest` prints text only.

2. **Retire `negframe_debt` worst-first** - for every mechanical finding `--suggest` lists,
   apply the printed reframe to the source prose. This pass changes ONLY the flagged clause's
   wording - never the meaning, a claim, a number, or a link the sentence carries.

3. **Weigh the judgement (SOFT) findings, then stop** - `--per-doc` shows each judgement
   finding's `Hint`. Reframe the cheap, obviously-better ones by hand (state the permitted
   path before the boundary; assert a property directly instead of by double negative).
   Leave a genuinely load-bearing prohibition alone if reframing it would blur the boundary
   it exists to draw - this tier is a nudge, not a work-list to drain.

4. **Check the diff-scoped ratchet before landing a steer-prose change**:

   ```
   go run ./cmd/fak score negframe --since HEAD      # or the branch point / PR base ref
   ```

   This reports only negatives your CHANGE introduced (exit 1 if any new mechanical one
   appears) - a file absent at `<ref>` is treated as wholly new. This is the CI gate shape;
   run it locally before committing so you never introduce a new mechanical negative in the
   same diff that is supposed to be reducing them.

5. **Re-measure + prove, then commit only the scorecard lane, by explicit path**:

   ```
   go run ./cmd/fak score negframe --json > /tmp/after.json
   go run ./cmd/fak score negframe --compare /tmp/before.json     # prints the debt delta
   go run ./cmd/fak score negframe --markdown > docs/NEGFRAME-SCORECARD.md
   ```

   State the delta plainly: `negframe_debt N -> N' (-k)`. Commit the prose files you
   reframed together with the regenerated snapshot (if any); never `git add -A`. End the
   subject with a `(fak negframe)` (or the touched leaf's) trailer so `dos commit-audit`
   binds it.

## Reading the payload

`Corpus` carries `negframe_debt` (the mechanical count - the debt), `documents` (how many
files were scanned), `mechanical_debt`, and `judgement_soft`. `Schema` is
`fak-negframe-scorecard/1`. `DebtKey` is `"negframe_debt"` - the same string
`internal/scorecardpane` folds into the portfolio total under the `negframe` card.

## The anti-gaming rule (specific to this surface)

**Fix a negative by reframing the sentence to genuinely lead with the action, never by
deleting the directive or muting the detector.** A mechanical finding is retired by applying
its `Suggest` (or an equivalent positive rewrite), not by rewording it into a shape the
lexicon happens not to match. Never widen a mechanical `reframeRule`'s pattern just to make a
finding disappear - the existing rules are deliberately high-precision (see the `not un-`
allowlist in `internal/negframe/negframe.go`) precisely so a mechanical rewrite can never be
wrong; loosening one to chase debt risks emitting a garbled "reframe" on prose it should not
have matched. If a genuinely equivalent phrasing is already positive but still flags, the fix
is to teach the lexicon that shape - recognizing real positive framing the rules were blind
to is not weakening the check; suppressing a real negative is.

## When to use this

- After editing AGENTS.md, CLAUDE.md, or a skill's SKILL.md - run `--since` before landing to
  make sure the change did not introduce a new mechanical negative.
- After adding a new skill or a new refusal/guard message - scope negframe at the new file
  directly (`go run ./cmd/fak score negframe path/to/new/doc.md`) before it joins the corpus.
- On a `/loop` cadence to keep negframe_debt at zero as the steer-prose corpus grows.

The scorecard is read-only; this skill's only writes are the reframed prose, the regenerated
`docs/NEGFRAME-SCORECARD.md` snapshot, and `internal/negframe` itself when a real lexicon gap
is found (never a loosened one).

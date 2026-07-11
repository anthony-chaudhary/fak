---
title: "Concept: the scorer-of-scorers — 'an unmeasured axis fails closed, never perfect'"
description: "Turns the #3833 fail-open incident (a KPI that scored 100 when it could not measure) into a standing, self-enforcing audit across the whole scorecard family, closing the meta-loop: the tool that grades tools now grades whether a scorer is honest about what it could not measure."
---

# Concept: the scorer-of-scorers — "an unmeasured axis fails closed, never perfect"

*2026-07-11*

## The seed incident (#3833)

`code_quality_scorecard.py`'s `ship_integrity` KPI leans on `dos review` as its
witness. When `dos` was **absent or errored**, the KPI returned a perfect
**score 100** — "no residual commits found." But it had found nothing because it
had *looked* at nothing. A missing kernel read as flawless witness discipline:
the classic **fail-OPEN** — *no detection reads as perfect*. #3833 patched that
one branch to fail **closed** (score 0, `errored`), so an unmeasured witness can
never present as a clean pass and the composite ceiling drops when the axis could
not be witnessed.

That fixed **one** branch of a **systemic** pattern. The scorecard family is ~a
dozen `*_scorecard.py` tools, each with several KPIs, several of which shell out
to a toolchain (`ruff`, `dos`, `git`, `gh`) that can be absent. Every one of
those is a candidate fail-open. A one-off patch does not bind the next one.

## The law

> **An unmeasured axis fails closed, never perfect.**
>
> A KPI that cannot run its witness must score **0 + errored** (fail-closed), or
> be an **explicit operator opt-out** (`--no-toolchain`, `--no-dos`) that scores
> 100 *without speaking failure language*. What it may never do is score high
> from a branch that admits it could not measure.

Two look-alikes are **not** fail-open and must not be flagged:

- **Explicit opt-out** — `skipped (--no-toolchain)`, `ruff not run`. A deliberate
  choice not to measure; legitimately 100.
- **Absent subject** — `no CLAIMS.md (skipped)`, `no modules`. Nothing to grade;
  vacuously 100.

The tell that separates fail-open from these is the **witness-failure vocabulary**
the branch writes about *itself*: "unavailable", "UNMEASURED", "not installed",
"could not". A high score sitting next to those words is the bug.

## The mechanism — `audit_failopen` (in `tooling_quality_scorecard.py`)

A pure AST pass over the `*_scorecard.py` sources. For every `return {…}` of a
score-dict, it flags the branch when **all** hold:

1. the literal `score` is `>= 90` (a "high" score a fail-open branch would emit);
2. the branch's own string constants (detail + soft) contain a term from the
   narrow **measurement-failure lexicon** (`unavailable`, `unmeasured`,
   `not installed`, `could not`, `errored`, `no binary`, …) — deliberately
   **excluding** `not run` / `skipped` (opt-out) and `not found` / `no <file>`
   (absent subject), so the two legitimate look-alikes never trip it;
3. no `# failopen-ok: <reason>` waiver sits on or just above the return.

Findings surface as an **advisory** `failopen_debt` meter on the corpus payload
(+ a render worklist) — a **stage-1 ratchet**: it counts and names, but does not
gate `ok`, so introducing the meter cannot red a live trunk. Ratchet-to-HARD once
the family has held `failopen_debt == 0` for a cycle.

This is the **meta-loop**: `tooling_quality_scorecard` already grades the `tools/`
tree; now it also grades whether a **scorer** is honest about what it could not
measure. The scorer-of-scorers. The fix outlives the incident because the pattern,
not the instance, is now checked on every run.

## First harvest

Run against the tree the day it landed, the audit immediately surfaced genuine,
**independent** fail-opens the hand-fix of #3833 had not touched:

- **`observability_scorecard.py:ship_integrity`** — the exact #3833 pattern,
  unfixed, its soft note literally self-labelled `"(fail-open, not a
  witnessed-clean review)"` while scoring **100**. **Fixed** in this change (score
  → 0 + `errored`, mirroring #3833); its test, which had asserted the *bug*
  (`test_ship_integrity_dos_error_fails_open`), was corrected to assert
  fail-closed.
- **`code_quality_scorecard.py` / `observability` `--no-dos` branches** — legit
  opt-outs whose soft text still *said* "dos unavailable" (stale wording from
  before genuine unavailability was routed to the fail-closed branch). Text
  corrected — a real precision fix, and it clears the flag.
- **`steerability_scorecard.py:churn_concentration`** — `available=False` scores
  100 and conflates *no-git* (measurement failure → should fail closed) with
  *empty-range* (nothing to concentrate → arguably fine). Genuinely gray; left on
  the **advisory worklist** for the owning author's semantic call, which is
  exactly what stage-1 advisory is for. `failopen_debt` shipped at **1**, honestly.

And `tooling_quality_scorecard.py`'s **own** `lint`/`format` KPIs were the same
bug: they returned 100 when `ruff` was absent. Fixed to distinguish opt-out (100)
from probed-and-absent (0 + `errored`), and the tool now passes its own audit
(`test_tooling_quality_is_clean_under_its_own_audit`).

## Why this shape

- **Text-keyed, not config-keyed.** The detector reads the confession the scorer
  *already writes*. No per-scorer registration to drift; a new scorecard is
  covered the moment it ships.
- **Floor, not ceiling.** It cannot prove a scorer is honest — only catch the
  branch that says, in its own words, that it measured nothing yet scored high.
  The residual (a fail-open that stays silent, or hides behind opt-out phrasing)
  is what a human review still owns — and the detector deliberately flags the
  *conflated* phrasing too, to shrink that residual.
- **Advisory-first.** The meter that measures honesty is introduced the same way
  every other ratchet in the family was: surface, hold at zero, then gate. Forcing
  it to zero by hastily "fixing" a gray case — or by weakening the detector to
  hide a real finding — would be the very dishonesty the tool exists to catch.

## Related

- `#3833` — the seed fix (`code_quality_scorecard` `ship_integrity` fail-closed).
- `docs/notes/TRAJECTORY-CONTROL-SCORE-ANYTHING-2026-07-03.md` — the "score
  anything" family this generalizes.
- The five prior family laws (HARD-vs-SOFT split, no-gaming SOFT axes, py-debt as
  a drivable integer, read-only by construction, advisory-then-ratchet CI). This
  is the **sixth**: *an unmeasured axis fails closed, never perfect.*

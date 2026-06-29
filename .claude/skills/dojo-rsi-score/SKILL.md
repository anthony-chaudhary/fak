---
name: dojo-rsi-score
description: One repeatable pass that keeps the dojo's self-improving RSI loop actually CLOSING on our own billed usage - the real calibration history the dojo measures - instead of a plan-mode scaffold that never acts. Drives the native dojo-RSI verbs (`fak dojo-rsi fold|propose|rewrite|run|loop|trend`) over a scored dojo report, picks the worst-calibrated MEASURED, NON-FLOOR cell, previews the one-literal recalibration, self-scores it on a strict FoldCalibrable drop + a sample floor + an external witness, routes REPROJECT/HARVEST/floor cells to the human/agent arm, and commits only the dojo lane by explicit path. The calibration-loop counterpart of guard-rsi-score (which scores the guard's verdict loop). Use after a change to a dojo lever or the claim registry, after a fresh corpus scan, or on a /loop cadence to keep the gym's predictors getting better calibrated over time.
---

# dojo-rsi-score - the dojo's self-improving loop, scored against reality

> An instance of the generic **[scorecard](../scorecard/SKILL.md)** doctrine. Read that
> for the five laws and the RSI loop; this file is the per-surface specialization.

## Why this skill exists

The [dojo](../../../docs/fak/dojo.md) closes four RSI rungs - predict, run, measure, score,
trend - against the provider's billed reality. The fifth rung, *act on the calibration gap
and prove the act helped*, is the [dojo-RSI loop](../../../docs/fak/dojo-rsi-loop.md). Its
twin, `guard-rsi-score`, keeps the guard's VERDICT loop honest; this skill keeps the dojo's
CALIBRATION loop honest.

The law, in one line: **the dojo's RSI loop must close on a re-measured `FoldCalibrable` drop
from OUR OWN billed corpus + an external witness - never on a fabricated number, and never by
recalibrating a floor it exists to defend.**

## The load-bearing idea (read before touching a number)

Optimise **`FoldCalibrable`, not raw `mean_calib_err`**. Raw `calib_err` is minimised at
`claimed := realized`, so a constant rewrite is a free, content-free "gain". For a **floor**
metric (`vcache-warmth/false_warm_rate` claim 0.0, the lethal false-warm class; the bimodal
`resume-posture/cross_session_warm_hit_rate` default) closing the gap would *erase the guard
the dojo exists to defend*. `FoldCalibrable` folds a genuine estimate by its `calib_err` but a
floor by its **breach**, so a floor recalibration RAISES the fold and reverts. That is the
structural reason the loop cannot optimise itself into dishonesty - do not weaken it.

## The pass (the shared loop)

1. **Score the corpus** - produce the report the loop reads from real billed usage:
   `fak dojo run --corpus ~/.claude/projects --json > report.json`. An empty/UNMEASURED
   corpus is not a failure; it is honestly uncandidatable (point the scenario at a billed
   corpus, never fabricate rows).
2. **Fold the calibrable metric** - `fak dojo-rsi fold --report report.json` shows the
   estimate calib-err and the floor-breach term separately, so a floor breach can never hide
   inside an averaged-down estimate gain.
3. **Propose worst-first** - `fak dojo-rsi propose --report report.json` ranks the cells by
   novelty x value x staleness. Only a RECALIBRATE is self-keepable; REPROJECT / HARVEST /
   ROUTE_FLOOR / ROUTE_UNMEASURED are routed, never kept by the pure loop.
4. **Preview the recalibration** - `fak dojo-rsi rewrite --report report.json` renders the
   exact one-literal change to `internal/dojo/claims.go` a RECALIBRATE would make. It is a DRY
   RUN (writes nothing, opens no worktree, commits nothing) and refuses a floor/REPROJECT/
   HARVEST cell with the reason it routes. Read the diff before trusting the swap.
5. **Self-score + check the keep-bit** - `fak dojo-rsi run --report report.json --witness
   '{"ok":true}' --check`. KEEP requires measured rows > 0 AND a strict `FoldCalibrable` drop
   AND the per-lever sample >= `--min-sample` (default 3) AND the external witness. `--check`
   re-derives those invariants from the iteration's own fields, so a fabricated KEEP is caught
   at the gate. Never pass a witness the loop authored.
6. **Route what the loop could not keep (close the loop)** - a ROUTE_FLOOR breach is a
   belief-code bug to escalate (a bug in the lever's projection, never a claim swap); a
   REPROJECT routes an agent patch constrained to declared paths; a HARVEST files a goal-gated
   issue with the board row + corpus evidence. The pure loop never auto-lands these.
7. **Trend + stop saturated** - `fak dojo-rsi trend` folds the committed
   `docs/dojo/rsi-journal.jsonl` into the KEEP/REVERT/ESCALATE rollup. A freshly-touched cell
   is marked saturated so a `fak dojo-rsi loop` stops rather than thrashing the same row.
8. **Commit only the dojo lane, by explicit path** - the claim literal (and its pinned-claim
   test, when an honest recalibration updates it), the lever/registry change, the doc, this
   skill. Never `git add -A`. End the subject with `(fak dojo)` or `(fak dojocal)`.

## The anti-gaming rule (specific to this surface)

**Close the loop by re-pointing a genuine estimate at measured reality, never by widening the
ruler or erasing a floor.** The calibration band (`CalibBand` / `calibErr` / `MaxCalibErr`)
lives in `internal/dojo`; the rewrite target is one anchored literal in `claims.go`. A diff
that touches the band changes a second file and fails the one-file keep-bit. Lowering a floor's
claim to its empirical rate RAISES `FoldCalibrable` and reverts. A KEEP requires a strict
`FoldCalibrable` gain on real billed bytes AND a sample floor AND a green external witness;
weakening `CheckIteration` to pass an unwitnessed, too-few-samples, or floor-target keep is the
one move that turns the gym into theater - exactly the auto-erasure this loop exists to forbid.

## ASCII gotcha

Keep the dojo + dojocal SOURCE ASCII-only (no em-dashes / smart quotes): the repo's provenance
pre-commit hook crashes on non-cp1252 UTF-8 before its own escape fires. Use ` - ` for a dash
and straight quotes in `.go` / `.py`. (The `.md` docs may use UTF-8.)

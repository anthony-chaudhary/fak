---
name: guard-rsi-score
description: One repeatable pass that keeps the RSI loop for `fak guard` actually CLOSING on our own usage - the real, hash-chained decision journal `fak guard` writes - instead of a plan-mode scaffold that never closes. Runs the native guard-RSI scorecard (`fak guard-rsi-scorecard`) over the two guard RSI loops (the hardware-gated LATENCY loop tools/guard_hop_rsi.py and the hardware-free VERDICT loop `fak guard-verdict-rsi`), scores maturity (can it honestly close?) + realized value (does it run on our usage?), turns each HARD defect into a required affordance to ADD (a journal-grounded loop, a deterministic verdict-quality metric, a non-forgeable keep-bit, real adjudicated rows, control-pane membership, a paired honesty test), retires guard_rsi_debt worst-first, re-measures to PROVE the debt dropped, and commits only the guard-RSI lane by explicit path. The product-loop counterpart of rsi-maturity (which scores the generic internal/rsiloop engine). Use after a change to either guard RSI loop, after a guarded session adds journal rows, or on a /loop cadence to keep the guard learning from our workflow.
---

# guard-rsi-score - the RSI loop for `fak guard`, scored against reality

> An instance of the generic **[scorecard](../scorecard/SKILL.md)** doctrine. Read that
> for the five laws and the RSI loop; this file is the per-surface specialization.

## Why this scorecard exists

`rsi_maturity_scorecard.py` scores the generic RSI ENGINE (`internal/rsiloop`) - the demo
self-improver. Nothing scored the loop attached to the PRODUCT a user actually runs:
`fak guard`. That loop has a real signal - the default-on, hash-chained decision journal
`fak guard` writes - and a real failure mode: the latency loop (`tools/guard_hop_rsi.py`,
#733) is honestly hardware-gated (#734) and so runs in plan-mode forever, never closing on
a normal box.

The fix is not to fake the latency number. It is the SIBLING loop the journal CAN close:
the verdict-pattern loop (`fak guard-verdict-rsi`), which reads the real journal, scores
the verdict distribution's honesty, and keeps a refinement only on a strict gain + a witness
it did not author. See [docs/fak/guard-verdict-rsi-loop.md](../../../docs/fak/guard-verdict-rsi-loop.md).

The law, in one line: **the guard's RSI loop must close on a re-measured number from OUR OWN
usage journal + an external witness - never on a fabricated latency baseline, and never on an
empty journal.**

## The KPIs (what it grades)

Two axes (the industry-scorecard shape: structure vs realized value). Each HARD failure is
one unit of `guard_rsi_debt`.

MATURITY - can the loop honestly close?
- **verdict_loop_present (HARD)** - a journal-grounded loop exists (closes without hardware).
- **deterministic_metric (HARD)** - verdict-quality is a pure function of the journal bytes
  (no clock, no RNG), so a KEEP can't be a fluke.
- **nonforgeable_keepbit (HARD)** - the honesty gate rejects a keep lacking rows / a strict
  delta / a green witness.
- **empty_journal_honesty (HARD)** - refuses a keep on 0 rows; the row count IS the gate.
- **latency_loop_honest (SOFT)** - the latency loop discloses its hardware gate, not silently broken.

REALIZED - does it run on our own usage?
- **loop_reads_real_journal (HARD)** - a loop READS the real journal (not a dangling
  telemetry string no code consumes).
- **registered_in_control_pane (HARD)** - the scorecard is in `SCORECARDS` AND the baseline
  is pinned, so the ratchet folds + gates this debt.
- **kept_iteration_on_real_rows (HARD)** - real adjudicated rows exist, so the loop can bank
  a kept iteration on our usage.
- **paired_honesty_test (HARD)** - a test proves KEEP-on-gain, REVERT-on-no-gain, and the
  empty-journal refusal.
- **documented (SOFT)** - the real-usage loop has a doc + this skill.

## The pass (the shared five-step loop)

1. **Run it** - `fak guard-rsi-scorecard` (work-list), `--json` (payload),
   `--compare baseline.json` (prove the drop and the Nx verdict).
2. **Retire guard_rsi_debt worst-first** - fix the heaviest HARD KPI by ADDING the real
   thing: the journal-grounded loop, the deterministic metric, the control-pane row, the
   paired test. If `kept_iteration_on_real_rows` is red, the journal has no rows yet - run a
   guarded session (or seed via `fak serve --policy ... ` + `/v1/fak/adjudicate`, then
   `fak audit verify`) so REAL verdicts land. Never fabricate rows.
3. **Weigh the SOFT signals, then stop** - add the doc/skill if missing; keep the latency
   loop's hardware-gate disclosure honest. Don't chase soft signals to zero.
4. **Re-measure + prove** - `--compare` prints the debt delta; the scorecard reads A
   (debt 0) on the disciplined tree once registered + pinned.
5. **Commit only the guard-RSI lane, by explicit path** - the scorecard tool/test, the
   verdict loop tool/test, the control-pane row + pinned baseline, the doc + this skill.
   Never `git add -A`. End the subject with `(fak guard)`.

## The anti-gaming rule (specific to this surface)

**Close the loop by making it learn from real verdicts, never by lowering the bar.** A red
`kept_iteration_on_real_rows` is fixed by producing REAL rows (a guarded session), not by a
fixture file checked into the journal path. A red `deterministic_metric` is fixed by removing
the clock/RNG, not by deleting the check. A KEEP requires a strict verdict-quality gain on
real bytes AND a green witness; weakening `check_iteration` to pass an unwitnessed or
empty-journal keep is the one move that turns the loop into theater - exactly the
plan-mode-forever failure this scorecard exists to catch.

## ASCII gotcha

Keep the scorecard + loop source ASCII-only (no em-dashes / smart quotes). The repo's
provenance pre-commit hook crashes on non-cp1252 UTF-8 before its own escape fires. Use
` - ` for a dash and straight quotes. (The `.md` docs may use UTF-8.)

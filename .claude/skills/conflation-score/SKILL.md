---
name: conflation-score
description: One repeatable pass that keeps every number and status fak reports PROVENANCE-HONEST - each value labeled by what fak CONTROLS (witnessed/authored) vs what it only OBSERVES (relayed from an external party), and no bad observed value blamed on a fak action. Runs the conflation scorecard (tools/conflation_scorecard.py) over the fact-reporting surfaces (Prometheus metric help, the fak guard exit summary), turns each HARD defect into a required edit (label an unlabeled external value OBSERVED; correct prose that attributes a provider-side miss to a fak action), retires conflation-debt worst-first WITHOUT changing any number or logic, re-measures to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The truth-maintenance counterpart of appeal-score (prose voice) and observability (what is measured). Use after adding a metric/exit-summary that reports an external value, when a dashboard mislabels whose number it is, or on a /loop cadence to keep the reporting surface honest about its own boundary.
---

# conflation-score - the anti-conflation / provenance-honesty pass

> An instance of the generic **[scorecard](../scorecard/SKILL.md)** doctrine. Read that
> for the five laws and the RSI loop; this file is the per-surface specialization.

## Why this scorecard exists

The lesson it generalizes (from the compaction-metrics fix): a system that reports numbers
must keep a hard line between **what it controls** and **what it only observes**. fak
*guarantees* the prefix it ships is byte-identical (witnessed, in-process); whether the
*provider* reuses the cache is the provider's call (observed, relayed). The original metric
prose said a cratered `cache_read` meant "the cache broke" - blaming fak for a provider-side
miss (TTL expiry, eviction, the client moving its own breakpoint). That is a
truth-maintenance bug: it erodes trust in the one number fak can actually stand behind.

The anti-conflation law, in one line: **label every reported fact by its provenance
(WITNESSED vs OBSERVED), and never attribute a bad observed value to a fak action unless a
witnessed signal proves the fault is fak's.**

## The KPIs (what it grades)

- **provenance_labeled (HARD)** - every help/summary string that reports an EXTERNAL value
  (a provider counter, upstream-reported usage) carries an OBSERVED/provider-relayed
  qualifier. One unlabeled external value = one debt.
- **no_false_attribution (HARD)** - no prose attributes a bad observed value to a fak action
  ("the cache broke", "the splice is producing", "we re-bill") without a co-located
  disambiguating qualifier. One violation = one debt.
- **fault_signal_isolated (SOFT)** - a family mixing witnessed + observed values names
  exactly one fak-fault signal the reader can point at (the "only `prefix_mismatch`>0 is our
  bug" pattern).

## The pass (the shared five-step loop)

1. **Run it** - `python tools/conflation_scorecard.py` (work-list), `--json` (payload),
   `--compare baseline.json` (prove the drop).
2. **Retire conflation_debt worst-first** - fix the heaviest KPI by **editing the prose**:
   add the OBSERVED label to an external value, or rewrite an attribution sentence so it
   names the provider-side cause and the single fak-fault signal. This pass changes only
   comments and help/summary STRINGS - never a number, a counter, or a branch.
3. **Weigh the SOFT signal** - if a mixed family names no fault signal, add the one sentence;
   don't invent signals that don't exist.
4. **Re-measure + prove** - `--compare` prints the debt delta; the scorecard must read A
   (debt 0) on the disciplined tree.
5. **Commit only the scorecard lane, by explicit path** - the scorecard tool/test, the
   reporting-surface file(s) whose prose you corrected, the control-pane baseline. Never
   `git add -A`. End the subject with `(fak <leaf>)`.

## The anti-gaming rule (specific to this surface)

**Fix a conflation by making the prose honest, never by deleting the number or muting the
detector.** A flagged external value is fixed by *labeling* it OBSERVED, not by removing the
metric. A false-attribution sentence is fixed by naming the real provider-side cause, not by
deleting the sentence so the substring no longer matches. If a flagged string is *already*
honest in different words (e.g. "provider-side reuse, distinct from the local caches"), the
fix is to teach the detector that phrasing in `OBSERVED_QUALIFIERS` - recognizing real
honesty the keyword list was blind to is NOT weakening the check; relaxing it to pass a
genuine conflation is.

## Adding a surface

When a new file renders operator-facing numbers (a metric family, a new exit summary), add
its path to `REPORTING_SURFACES` in `tools/conflation_scorecard.py`. The scorecard then holds
it to the same provenance discipline, and the live-tree smoke test pins the new floor.

## ASCII gotcha

Keep the scorecard source ASCII-only (no em-dashes / smart quotes). The repo's provenance
pre-commit hook crashes on non-cp1252 UTF-8 before its own escape fires. Use ` - ` for a
dash and straight quotes.

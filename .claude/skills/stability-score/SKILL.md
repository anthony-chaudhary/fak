---
name: stability-score
description: One repeatable pass that keeps fak trustworthy while it iterates fast — the question no other scorecard asks: as we add items quickly, how do we KNOW a regression / tail-wag / confusion landed, and how do we REVERT to a stable version? Runs the stability scorecard (tools/stability_scorecard.py) over the git-tracked tree across four groups — sentinel (a regression turns a gate RED), invariant (the core assumptions are encoded as tests), revert (we can roll back: keep/revert ladder, version pin, CI-gated tags, a documented rollback runbook), and drift (a small thing wagging a big thing gets caught) — turns each HARD defect into a required affordance to ADD (wire the missing CI regression gate, encode the missing invariant test, commit the missing ratchet baseline, write/link the rollback runbook), retires stability-debt worst-first, re-measures + regenerates the snapshot to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The stability counterpart of code-quality (defects) and rsi-maturity (the self-improver's structure). Use after a change to a CI gate, an invariant test, a baseline, the release/rollback path, or on a /loop cadence to keep the trunk safe to move fast on.
metadata:
  opencode: claude-only
---

# stability-score — keep fak trustworthy while it iterates fast, and prove it

> **What this does.** A trunk that changes fast lives or dies on a question no other
> scorecard asks: when something breaks, do we **find out** — and can we **get back**?
> This pass turns that from a vibe ("we have lots of tests") into a **repeatable,
> deterministic number** — `stability-debt` — and a process that drives it down by
> ADDING the real sentinel / invariant / revert affordance, never by relaxing a check.

It is an instance of the generic [`scorecard`](../scorecard/SKILL.md) doctrine pointed
at one surface: **stability under fast iteration**. The tool
(`tools/stability_scorecard.py`) reads the git-tracked tree, scores thirteen KPIs in
four groups, and folds them into one `stability_debt` integer plus an A–F grade and the
control-pane envelope. It folds into `tools/scorecard_control_pane.py` alongside the
rest of the family.

## The four groups (the four ways a fast-moving trunk stays trustworthy)

- **sentinel** — when a regression lands, SOMETHING fails (`regression_gates_wired`,
  `ratchet_baselines_committed`, `honesty_ledger_clean`). The roster of HARD CI gates —
  build/vet/test, gofmt, `-race`, claims-lint, the portfolio ratchet, the main-KPI track
  gate, dos-review, the no-blackhole tool-test runner, index-sync, leak-scan — must be
  wired, each with a committed baseline to ratchet against.
- **invariant** — the core assumptions are ENCODED as executable tests
  (`invariant_tests_present`, `frozen_pins_present`, `fail_closed_witnessed`,
  `determinism_witness`): the ABI freeze, the tier/import DAG, the interpreter-free +
  exec-free request path, a fail-closed witness (a bad input is REFUSED), and the
  `proofs_witness` determinism convention so a pass can't be a one-box fluke.
- **revert** — we can get back to a known-good state (`keep_revert_ladder`,
  `version_pin`, `release_tagging_gated`, `rollback_runbook`): the shipgate keep/revert
  ladder, the single `VERSION` marker + resolver, CI-gated tagging of stable versions,
  and a DOCUMENTED, linked operator runbook ([`docs/ROLLBACK.md`](../../../docs/ROLLBACK.md)).
- **drift** — a small thing silently distorting a big thing gets caught
  (`drift_detectors_wired`, and the SOFT `confusion_escalation_signal`): the readme
  freshness / index-sync / commit-stamp / claims-salience / portfolio-trend detectors,
  plus the frontier note on the missing INDETERMINATE ("I can't decide → escalate")
  disposition.

## Run it as an RSI pass (the five steps)

1. **Run it** — `python tools/stability_scorecard.py` for the work-list; `--json` for the
   machine payload. Save a baseline first: `python tools/stability_scorecard.py --json >
   /tmp/stability-before.json`.
2. **Retire `stability_debt` worst-first** — fix the heaviest KPI by ADDING the real
   affordance: wire the missing CI gate HARD, encode the missing invariant as a named
   test, commit the missing ratchet baseline, or write + link the rollback runbook. Never
   weaken a gate, an invariant, or a verdict to score.
3. **Weigh the SOFT signals, then stop** — the frontier notes (no tail-wag backing tool, no
   early-warning trend, no INDETERMINATE verdict) are judgment nudges, not work-list items.
   Don't chase them to zero or stub them to silence.
4. **Re-measure + prove** — `python tools/stability_scorecard.py --compare
   /tmp/stability-before.json` prints the debt delta and the ≥2× verdict; regenerate the
   committed snapshot: `python tools/stability_scorecard.py --markdown --stamp DATE >
   docs/STABILITY-SCORECARD.md`.
5. **Commit only the scorecard lane, by explicit path** —
   `git commit -s -- tools/stability_scorecard.py tools/stability_scorecard_test.py
   docs/STABILITY-SCORECARD.md <the affordance you added>`. End the subject with the
   `(fak stability)` trailer. Never `git add -A`.

## The anti-gaming law

Retire a defect by changing reality, never by changing the detector. A missing regression
gate is fixed by wiring the gate HARD in CI, not by deleting the gate from the roster; a
missing invariant is fixed by writing the real test, not by removing the required symbol;
the rollback runbook is satisfied by documenting the real revert mechanisms an operator
runs, not by stuffing the tokens the KPI greps for into an empty file. If "fixing" a
defect would mean gaming the check instead of making the trunk safer to move fast on,
**stop — that's not a real gap.**

## Re-pinning the portfolio baseline (handle with care)

Adding stability raised the portfolio total, but it ships at zero debt and the rest of the
family fell, so `scorecard_control_pane.py --check` stays GREEN without a re-pin. Do **not**
blindly `--pin`: a portfolio that reads "improved" overall can hide a per-metric regression
(e.g. a sibling scorecard rose while another fell more), and a blind re-pin blesses it.
Re-pin only as a deliberate portfolio pass, after confirming no individual metric regressed.
The stability scorecard's own live-smoke test (`stability_debt == 0`, HARD in CI) is the
real regression sentinel for this surface, independent of the portfolio baseline.

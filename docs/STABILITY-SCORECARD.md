---
title: "fak stability scorecard — the stability-debt measuring stick"
description: "fak's deterministic stability scorecard: KPIs across the four ways a fast-moving trunk stays trustworthy — sentinel (we find out a regression landed), invariant (the assumptions are encoded as tests), revert (we can roll back to a stable version), and drift (a small thing wagging a big thing gets caught) — folded into a composite score and the headline stability-debt metric, re-derived from the git-tracked tree."
---

# Stability scorecard — can we tell when we broke something, and roll back

<!-- stability-scorecard: 2026-06-24 · process: tools/stability_scorecard.py -->

This is the measuring stick for fak's **stability under fast iteration** — the question a team living on a rapidly-changing trunk loses sleep over: as we add items fast, how do we **know** a regression, tail-wag, or confusion landed, and how do we **revert** to a stable version? Every number below is re-derived from the git-tracked tree by `tools/stability_scorecard.py` — no hand-entry. The headline metric is **stability-debt**: the count of concrete, mechanical defects that leave fak unable to catch a regression or roll one back — a missing CI gate, an unencoded invariant, an uncommitted baseline, a missing rollback runbook. Driving stability-debt to zero is what lets fak change fast *and* stay trustworthy.

> Regenerate: `python tools/stability_scorecard.py --markdown --stamp DATE > docs/STABILITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Stability-debt (total HARD defects)** | **0** |
| Composite score | 98.0/100 (grade A) |
| Trustworthiness | sentinel 100 · invariant 100 · revert 100 · drift 83 |
| Advisory (soft) signals | 3 |
| Debt by group | sentinel:0 · invariant:0 · revert:0 · drift:0 |

> The composite tops out below 100 even at zero debt: `confusion_escalation_signal` is a SOFT frontier signal (no INDETERMINATE verdict is wired in the adjudication core yet), scored 50 at weight 0.04, so it subtracts ~2 from the composite without adding HARD debt. Zero stability-debt is the headline; the ~2 is the honest frontier the soft signals track.

## The four ways a fast-moving trunk stays trustworthy

13 KPIs, each 0–100, grouped by the property they defend. `debt` = units of HARD stability-debt. `confusion_escalation_signal` is advisory (it scores but emits no hard debt — wiring a new verdict is a frontier feature, not a checklist item).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| drift | `confusion_escalation_signal` | 50 | 0 | no explicit INDETERMINATE / escalate disposition (confusion is fail-open/closed only) |
| sentinel | `regression_gates_wired` | 100 | 0 | 10/10 HARD regression gates wired in CI |
| sentinel | `ratchet_baselines_committed` | 100 | 0 | 3/3 ratchet baselines committed |
| sentinel | `honesty_ledger_clean` | 100 | 0 | CLAIMS.md present, 0 untagged claim(s) |
| invariant | `invariant_tests_present` | 100 | 0 | 7/7 core invariants have a named test |
| invariant | `frozen_pins_present` | 100 | 0 | 3/3 frozen pins on disk |
| invariant | `fail_closed_witnessed` | 100 | 0 | 3/3 fail-closed behaviours witnessed |
| invariant | `determinism_witness` | 100 | 0 | 18 proofs_witness determinism witness file(s) |
| revert | `keep_revert_ladder` | 100 | 0 | committed keep/revert ladder exists + tested |
| revert | `version_pin` | 100 | 0 | VERSION marker + resolver present |
| revert | `release_tagging_gated` | 100 | 0 | release tagging is CI-gated |
| revert | `rollback_runbook` | 100 | 0 | rollback runbook present (docs/ROLLBACK.md), covers every mechanism |
| drift | `drift_detectors_wired` | 100 | 0 | 5/5 silent-drift detectors present |

## Stability-debt work-list

No stability-debt: a regression fails loudly and there is a written path back to a stable version. 🎉

## Advisory (soft) signals — the frontier, not debt

- the /tail-wag inverted-priority finder is a manual skill with no deterministic backing tool (tools/*tail_wag*.py) — it can't run in a gate
- the portfolio trend only flags a regression ABOVE the pinned baseline; no early-warning on a first downward move WITHIN a healthy envelope
- no 'I could not decide -> escalate' (INDETERMINATE) verdict in internal/abi, internal/adjudicator -- tracked as the frontier in docs/notes/verification-ladder-doctrine.md


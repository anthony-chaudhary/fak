---
title: "fak observability scorecard — the observability-debt measuring stick"
description: "fak's deterministic observability scorecard: eight KPIs across correlation, instrumentation, and verifiability, folded into a composite score and the headline observability-debt metric, re-derived from the git-tracked tree with the Go binary as the metric source of truth."
---

# Observability scorecard

<!-- observability-scorecard: 2026-06-29 · process: tools/observability_scorecard.py -->

This is the measuring stick for the observability-10x program — the counterpart of the code, docs, and repo-hygiene scorecards aimed at the **observability plane**: the metrics the gateway emits, the dashboards and alerts that read them, the docs that tell an operator which metric to query, the trace-id that ties a request across logs, and the proofs / ship-audit that let a claim be verified rather than asserted. Every number below is re-derived from the git-tracked tree by `tools/observability_scorecard.py` — no hand-entry. The **Go binary is the metric source of truth**: a dashboard, alert, or doc reference is a defect iff the binary emits no such `fak_*` family. The headline metric is **observability-debt**: the count of concrete, mechanical defects you fix by *making the live system more visible and more verifiable* — a phantom panel, an alert on a metric that does not exist, a doc that misdirects an operator, a broken trace surface, a log line that leaks a payload, a PROVEN proof with no witness, an unwitnessed ship.

> Regenerate: `python tools/observability_scorecard.py --markdown --stamp DATE > docs/OBSERVABILITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Observability-debt (total HARD defects)** | **1** |
| Composite score | 95.9/100 (grade A) |
| Emitted `fak_*` metric families (source of truth) | 214 |
| Advisory (soft) signals | 31 |
| Debt by group | correlation:0 · instrumentation:0 · verifiability:1 |

## Per-KPI

Eight KPIs, each 0–100, in three groups. `debt` = units of HARD observability-debt. `metric_doc_coverage` is advisory (it scores but emits no hard debt — documenting every internal counter is noise, not visibility); `ship_integrity` is HEAD-relative and fails open when `dos` is absent.

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| verifiability | `proof_witness` | 92 | 1 | 1 theorem(s) unadjudicated or PROVEN-without-witness across 79 theorem(s) |
| instrumentation | `metric_doc_coverage` | 47 | 0 | 52/214 emitted families surfaced in a doc/dashboard/alert (24%) |
| correlation | `dashboard_integrity` | 100 | 0 | every fak_* reference in 4 dashboard(s) is emitted (56 ref(s)) |
| correlation | `alert_integrity` | 100 | 0 | every fak_* reference in 1 alert(s) is emitted (7 ref(s)) |
| correlation | `doc_metric_drift` | 100 | 0 | every fak_* reference in 17 doc(s) is emitted (73 ref(s)) |
| instrumentation | `trace_correlation` | 100 | 0 | trace surface intact: X-Trace-Id honored/minted, response header set, all 5 log event(s) carry trace_id |
| instrumentation | `log_privacy` | 100 | 0 | 5 structured-log event(s) carry no payload field |
| verifiability | `ship_integrity` | 100 | 0 | 12 checkable commit(s) in HEAD~20..HEAD, 0 residual, cleared_rate 1.0 |

## Observability-debt work-list

### `proof_witness` (verifiability) — 1 defect(s), score 92
- unadjudicated proof theorem in docs/proofs/async-addressing.md (#1 “Theorem A.1 — the seam is identity-and-routing only; no behavior rides”): no VERDICT line — adjudicate it (PROVEN/OPEN/REFUTED)


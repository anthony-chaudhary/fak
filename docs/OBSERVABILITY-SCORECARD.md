---
title: "fak observability scorecard — the observability-debt measuring stick"
description: "fak's deterministic observability scorecard: eight KPIs across correlation, instrumentation, and verifiability, folded into a composite score and the headline observability-debt metric, re-derived from the git-tracked tree with the Go binary as the metric source of truth."
---

# Observability scorecard

<!-- observability-scorecard: 2026-07-01 · process: tools/observability_scorecard.py -->

This is the measuring stick for the observability-10x program — the counterpart of the code, docs, and repo-hygiene scorecards aimed at the **observability plane**: the metrics the gateway emits, the dashboards and alerts that read them, the docs that tell an operator which metric to query, the trace-id that ties a request across logs, and the proofs / ship-audit that let a claim be verified rather than asserted. Every number below is re-derived from the git-tracked tree by `tools/observability_scorecard.py` — no hand-entry. The **Go binary is the metric source of truth**: a dashboard, alert, or doc reference is a defect iff the binary emits no such `fak_*` family. The headline metric is **observability-debt**: the count of concrete, mechanical defects you fix by *making the live system more visible and more verifiable* — a phantom panel, an alert on a metric that does not exist, a doc that misdirects an operator, a broken trace surface, a log line that leaks a payload, a PROVEN proof with no witness, an unwitnessed ship.

> Regenerate: `python tools/observability_scorecard.py --markdown --stamp DATE > docs/OBSERVABILITY-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Observability-debt (total HARD defects)** | **0** |
| Composite score | 96.8/100 (grade A) |
| Emitted `fak_*` metric families (source of truth) | 241 |
| Advisory (soft) signals | 31 |
| Debt by group | correlation:0 · instrumentation:0 · verifiability:0 |

## Per-KPI

Eight KPIs, each 0–100, in three groups. `debt` = units of HARD observability-debt. `metric_doc_coverage` is advisory (it scores but emits no hard debt — documenting every internal counter is noise, not visibility); `ship_integrity` is HEAD-relative and fails open when `dos` is absent.

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| instrumentation | `metric_doc_coverage` | 46 | 0 | 54/241 emitted families surfaced in a doc/dashboard/alert (22%) |
| correlation | `dashboard_integrity` | 100 | 0 | every fak_* reference in 4 dashboard(s) is emitted (56 ref(s)) |
| correlation | `alert_integrity` | 100 | 0 | every fak_* reference in 1 alert(s) is emitted (7 ref(s)) |
| correlation | `doc_metric_drift` | 100 | 0 | every fak_* reference in 19 doc(s) is emitted (74 ref(s)) |
| instrumentation | `trace_correlation` | 100 | 0 | trace surface intact: X-Trace-Id honored/minted, response header set, all 5 log event(s) carry trace_id |
| instrumentation | `log_privacy` | 100 | 0 | 5 structured-log event(s) carry no payload field |
| verifiability | `proof_witness` | 100 | 0 | 79 theorem(s) across 34 proof(s); all adjudicated and every PROVEN one witnessed |
| verifiability | `ship_integrity` | 100 | 0 | 15 checkable commit(s) in HEAD~20..HEAD, 0 residual, cleared_rate 1.0 |

## Observability-debt work-list

No observability-debt: every dashboard, alert, and doc points at a metric the binary emits; the trace surface is intact; the log leaks no payload; every PROVEN proof is witnessed; no ship is unwitnessed. 🎉


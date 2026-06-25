---
title: "fak Demo-Quality Scorecard: Demos a Skeptic Can Run"
description: "fak's demo-quality scorecard grades 32 demos on five deterministic axes into a demo-score (0-100, A-F) and a re-derivable demo-debt count."
---

# Demo-quality scorecard

<!-- demo-quality-scorecard: 2026-06-25 · process: tools/demo_quality_scorecard.py -->

> Regenerate: `python tools/demo_quality_scorecard.py --markdown --stamp DATE > docs/DEMO-QUALITY-SCORECARD.md`
> Verify snapshot freshness: `python tools/demo_quality_scorecard.py --check-doc`

> The measuring stick for **demos a skeptic can run**: can I run it in one command, reproduce what the README promises, trust what it claims, and understand what I saw — without a model, a key, or a babysitter? Five deterministic axes (runnable · reproducible · honest_scope · self_contained · documented), folded into a **demo-score** (0–100, A–F) and a **demo-debt** integer (the count of concrete, re-derivable demo defects). Every number below is re-derived from disk by `tools/demo_quality_scorecard.py` — no hand-entry. Drive demo-debt to zero to make "better demos" provable.

## Corpus

| Metric | Value |
|---|---|
| Demos scored | 32 |
| **Demo-debt (total defects)** | **0** |
| Mean score | 99.8/100 |
| Median / min / max | 100.0 / 94.2 / 100.0 |
| Grade distribution | A:32 B:0 C:0 D:0 F:0 |

## Per-demo scores

Five axes, each 0–100 (runnable · reproducible · honest_scope · self_contained · documented), weighted into a score and an A–F grade. `def` = units of demo-debt.

| Score | Grade | Debt | run | repro | scope | self | docs | Demo |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| 94.2 | A | 0 | 100 | 100 | 90 | 88 | 90 | `cmd/dropindemo` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/openai-agents-guardrail` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/adjudication-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/agentdojo-redteam` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/auth-hardening` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/escalation-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/extdriver` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/mcp` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/mcp-client` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/observability` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/policy-hot-reload` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/presets` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/shared-task-record` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/shared-task-record-verdicts` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/trace-reset` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/wire-proof` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/wire-quarantine-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/a2ademo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/ctxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/ctxplandemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/cxlpooldemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/guarddemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/hwcachedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/memqdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/poisonedmcpdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/simpledemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/tokendemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/turntaxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/unseedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/causalbench` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/deletioncert` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/demorace` |

## Demo-debt work-list

No demo-debt: every demo runs, reproduces, scopes itself, and cleans up. 🎉

## Soft signals (score only, not debt)

### `cmd/dropindemo`
- honest_scope: no link to a deeper doc (CLAIMS / STATUS / an explainer) to back the claim
- self_contained: no stated prerequisites — a cold runner can't tell what to install first
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/openai-agents-guardrail`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone


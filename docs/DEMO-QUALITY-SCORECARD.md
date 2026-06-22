# Demo-quality scorecard

<!-- demo-quality-scorecard: 2026-06-22 · process: tools/demo_quality_scorecard.py -->

> Regenerate: `python tools/demo_quality_scorecard.py --markdown --stamp DATE > docs/DEMO-QUALITY-SCORECARD.md`

> The measuring stick for **demos a skeptic can run**: can I run it in one command, reproduce what the README promises, trust what it claims, and understand what I saw — without a model, a key, or a babysitter? Five deterministic axes (runnable · reproducible · honest_scope · self_contained · documented), folded into a **demo-score** (0–100, A–F) and a **demo-debt** integer (the count of concrete, re-derivable demo defects). Every number below is re-derived from disk by `tools/demo_quality_scorecard.py` — no hand-entry. Drive demo-debt to zero to make "better demos" provable.

## Corpus

| Metric | Value |
|---|---|
| Demos scored | 5 |
| **Demo-debt (total defects)** | **0** |
| Mean score | 99.7/100 |
| Median / min / max | 100.0 / 98.4 / 100.0 |
| Grade distribution | A:5 B:0 C:0 D:0 F:0 |

## Per-demo scores

Five axes, each 0–100 (runnable · reproducible · honest_scope · self_contained · documented), weighted into a score and an A–F grade. `def` = units of demo-debt.

| Score | Grade | Debt | run | repro | scope | self | docs | Demo |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/wire-proof` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/adjudication-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/agentdojo-redteam` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/mcp` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/simpledemo` |

## Demo-debt work-list

No demo-debt: every demo runs, reproduces, scopes itself, and cleans up. 🎉

## Soft signals (score only, not debt)

### `examples/wire-proof`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone


---
title: "fak Demo-Robustness Scorecard: Simplicity, Speed, Durability"
description: "The fak demo-robustness scorecard grades 32 demos on simplicity, speed, and durability into a 0-100 robustness-score, A-F grade, and a robustness-debt count."
---

# Demo-robustness scorecard

<!-- demo-robustness-scorecard: 2026-06-25 · process: tools/demo_robustness_scorecard.py -->

> Regenerate: `python tools/demo_robustness_scorecard.py --markdown --stamp DATE > docs/DEMO-ROBUSTNESS-SCORECARD.md`
> Verify snapshot freshness: `python tools/demo_robustness_scorecard.py --check-doc`

> The measuring stick for **demos that stay simple, fast, and durable**: will it still run next month, on a fresh box, in one obvious command, in seconds — without surprises? Three deterministic axes (simplicity · speed · durability), folded into a **robustness-score** (0–100, A–F) and a **robustness-debt** integer (the count of concrete, re-derivable robustness defects). The sibling `demo_quality_scorecard.py` asks *can a skeptic trust the claim*; this one asks *will it keep running*. Both score the same corpus. Every number below is re-derived from disk by `tools/demo_robustness_scorecard.py` — no hand-entry. Drive robustness-debt down to make "more robust demos" provable.

## Corpus

| Metric | Value |
|---|---|
| Demos scored | 32 |
| **Robustness-debt (total defects)** | **0** |
| Axis-debt | simplicity:0 · speed:0 · durability:0 |
| Mean score | 98.9/100 |
| Median / min / max | 100.0 / 92.2 / 100.0 |
| Grade distribution | A:32 B:0 C:0 D:0 F:0 |

## Per-demo scores

Three axes, each 0–100 (simplicity · speed · durability), weighted into a score and an A–F grade. `def` = units of robustness-debt.

| Score | Grade | Debt | simplicity | speed | durability | Demo |
|---:|:--:|:--:|:--:|:--:|:--:|---|
| 92.2 | A | 0 | 88 | 88 | 100 | `examples/adjudication-demo` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/auth-hardening` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/escalation-demo` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/observability` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/policy-hot-reload` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/trace-reset` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/wire-quarantine-demo` |
| 97.6 | A | 0 | 92 | 100 | 100 | `examples/shared-task-record` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/agentdojo-redteam` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/extdriver` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/mcp` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/mcp-client` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/openai-agents-guardrail` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/presets` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/shared-task-record-verdicts` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/wire-proof` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/a2ademo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/ctxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/ctxplandemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/cxlpooldemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/dropindemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/guarddemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/hwcachedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/memqdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/poisonedmcpdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/simpledemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/tokendemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/turntaxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/unseedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/causalbench` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/deletioncert` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/demorace` |

## Robustness-debt work-list

No robustness-debt: every demo is simple, fast, and durable. 🎉

## Soft signals (score only, not debt)

### `examples/adjudication-demo`
- simplicity: 4 prerequisites (curl, go, ollama, python) — more moving parts than a one-tool demo
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/auth-hardening`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/escalation-demo`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/observability`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/policy-hot-reload`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/trace-reset`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/wire-quarantine-demo`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/shared-task-record`
- simplicity: 20 files in the demo dir — a larger surface to skim


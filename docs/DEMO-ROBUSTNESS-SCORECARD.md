---
title: "fak Demo-Robustness Scorecard: Simplicity, Speed, Durability"
description: "The fak demo-robustness scorecard grades 8 demos on simplicity, speed, and durability into a 0-100 robustness-score, A-F grade, and a robustness-debt count."
---

# Demo-robustness scorecard

<!-- demo-robustness-scorecard: 2026-06-22 · process: tools/demo_robustness_scorecard.py -->

> Regenerate: `python tools/demo_robustness_scorecard.py --markdown --stamp DATE > docs/DEMO-ROBUSTNESS-SCORECARD.md`

> The measuring stick for **demos that stay simple, fast, and durable**: will it still run next month, on a fresh box, in one obvious command, in seconds — without surprises? Three deterministic axes (simplicity · speed · durability), folded into a **robustness-score** (0–100, A–F) and a **robustness-debt** integer (the count of concrete, re-derivable robustness defects). The sibling `demo_quality_scorecard.py` asks *can a skeptic trust the claim*; this one asks *will it keep running*. Both score the same corpus. Every number below is re-derived from disk by `tools/demo_robustness_scorecard.py` — no hand-entry. Drive robustness-debt down to make "more robust demos" provable.

## Corpus

| Metric | Value |
|---|---|
| Demos scored | 8 |
| **Robustness-debt (total defects)** | **0** |
| Axis-debt | simplicity:0 · speed:0 · durability:0 |
| Mean score | 95.3/100 |
| Median / min / max | 95.8 / 88.0 / 100.0 |
| Grade distribution | A:7 B:1 C:0 D:0 F:0 |

## Per-demo scores

Three axes, each 0–100 (simplicity · speed · durability), weighted into a score and an A–F grade. `def` = units of robustness-debt.

| Score | Grade | Debt | simplicity | speed | durability | Demo |
|---:|:--:|:--:|:--:|:--:|:--:|---|
| 88.0 | B | 0 | 88 | 88 | 88 | `examples/adjudication-demo` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/auth-hardening` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/escalation-demo` |
| 95.8 | A | 0 | 100 | 100 | 88 | `examples/agentdojo-redteam` |
| 95.8 | A | 0 | 100 | 100 | 88 | `examples/mcp-client` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/mcp` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/wire-proof` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/simpledemo` |

## Robustness-debt work-list

No robustness-debt: every demo is simple, fast, and durable. 🎉

## Soft signals (score only, not debt)

### `examples/adjudication-demo`
- simplicity: 4 prerequisites (curl, go, ollama, python) — more moving parts than a one-tool demo
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/auth-hardening`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/escalation-demo`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/agentdojo-redteam`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/mcp-client`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it


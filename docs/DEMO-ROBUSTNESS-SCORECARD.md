---
title: "fak Demo-Robustness Scorecard: Simplicity, Speed, Durability"
description: "The fak demo-robustness scorecard grades 66 demos on simplicity, speed, and durability into a 0-100 robustness-score, A-F grade, and a robustness-debt count."
---

# Demo-robustness scorecard

<!-- demo-robustness-scorecard: 2026-06-30 · process: tools/demo_robustness_scorecard.py -->

> Regenerate: `python tools/demo_robustness_scorecard.py --markdown --stamp DATE > docs/DEMO-ROBUSTNESS-SCORECARD.md`
> Verify snapshot freshness: `python tools/demo_robustness_scorecard.py --check-doc`

> The measuring stick for **demos that stay simple, fast, and durable**: will it still run next month, on a fresh box, in one obvious command, in seconds — without surprises? Three deterministic axes (simplicity · speed · durability), folded into a **robustness-score** (0–100, A–F) and a **robustness-debt** integer (the count of concrete, re-derivable robustness defects). The sibling `demo_quality_scorecard.py` asks *can a skeptic trust the claim*; this one asks *will it keep running*. Both score the same corpus. Every number below is re-derived from disk by `tools/demo_robustness_scorecard.py` — no hand-entry. Drive robustness-debt down to make "more robust demos" provable.

## Corpus

| Metric | Value |
|---|---|
| Demos scored | 66 |
| **Robustness-debt (total defects)** | **0** |
| Axis-debt | simplicity:0 · speed:0 · durability:0 |
| Mean score | 97.6/100 |
| Median / min / max | 100.0 / 91.6 / 100.0 |
| Grade distribution | A:66 B:0 C:0 D:0 F:0 |

## Per-demo scores

Three axes, each 0–100 (simplicity · speed · durability), weighted into a score and an A–F grade. `def` = units of robustness-debt.

| Score | Grade | Debt | simplicity | speed | durability | Demo |
|---:|:--:|:--:|:--:|:--:|:--:|---|
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/admit-and-log` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/agent-ab` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/context-debugger` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/deny-in-60s` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/preflight-ladder` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/remote-vm-guard` |
| 91.6 | A | 0 | 100 | 88 | 88 | `examples/session-reload` |
| 92.2 | A | 0 | 88 | 88 | 100 | `examples/adjudication-demo` |
| 92.2 | A | 0 | 88 | 88 | 100 | `examples/quarantine-demo` |
| 93.4 | A | 0 | 92 | 100 | 88 | `examples/shared-task-record` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/auth-hardening` |
| 95.8 | A | 0 | 100 | 100 | 88 | `examples/deny-as-value` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/escalation-demo` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/federated-changes` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/fleet-reuse-demo` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/grammar-repair` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/ifc-taint-flow` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/loader-properties` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/observability` |
| 95.8 | A | 0 | 100 | 100 | 88 | `examples/plan-cfi` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/policy-hot-reload` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/self-modify-floor` |
| 95.8 | A | 0 | 100 | 100 | 88 | `examples/shared-task-record-verdicts` |
| 95.8 | A | 0 | 100 | 100 | 88 | `examples/trace-authoring` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/trace-reset` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/vdso-cache-hit` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/wire-quarantine-demo` |
| 95.8 | A | 0 | 100 | 88 | 100 | `examples/witness-gate` |
| 97.6 | A | 0 | 92 | 100 | 100 | `cmd/tokendemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/agentdojo-redteam` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/autogen-groupchat` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/bench-latency` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/crewai-crew` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/dogfood-claude` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/extdriver` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/fanbench` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/frontierswe-harbor-shim` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/gpu-smoke` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/grammar-repair-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/mcp` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/mcp-client` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/normgate-evasion` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/openai-agents-guardrail` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/presets` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/radixattention` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/steward-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/turntax` |
| 100.0 | A | 0 | 100 | 100 | 100 | `examples/wire-proof` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/a2ademo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/agentbenchdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/ctxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/ctxplandemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/cxlpooldemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/dropindemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/guarddemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/hwcachedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/memqdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/poisonedmcpdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/simpledemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/timewolfdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/trychatdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/turntaxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/unseedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/causalbench` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/deletioncert` |
| 100.0 | A | 0 | 100 | 100 | 100 | `cmd/demorace` |

## Robustness-debt work-list

No robustness-debt: every demo is simple, fast, and durable. 🎉

## Soft signals (score only, not debt)

### `examples/admit-and-log`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/agent-ab`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/context-debugger`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/deny-in-60s`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/preflight-ladder`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/remote-vm-guard`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/session-reload`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/adjudication-demo`
- simplicity: 4 prerequisites (curl, go, ollama, python) — more moving parts than a one-tool demo
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/quarantine-demo`
- simplicity: 4 prerequisites (curl, go, ollama, python) — more moving parts than a one-tool demo
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/shared-task-record`
- simplicity: 21 files in the demo dir — a larger surface to skim
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/auth-hardening`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/deny-as-value`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/escalation-demo`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/federated-changes`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/fleet-reuse-demo`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/grammar-repair`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/ifc-taint-flow`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/loader-properties`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/observability`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/plan-cfi`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/policy-hot-reload`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/self-modify-floor`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/shared-task-record-verdicts`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/trace-authoring`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/trace-reset`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/vdso-cache-hit`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/wire-quarantine-demo`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `examples/witness-gate`
- speed: builds the whole binary (`go build`) with no `go run` fast path — slower cold start and a leftover artifact

### `cmd/tokendemo`
- simplicity: 9 files in the demo dir — a larger surface to skim


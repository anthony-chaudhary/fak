---
title: "fak Demo-Quality Scorecard: Demos a Skeptic Can Run"
description: "fak's demo-quality scorecard grades 64 demos on five deterministic axes into a demo-score (0-100, A-F) and a re-derivable demo-debt count."
---

# Demo-quality scorecard

<!-- demo-quality-scorecard: 2026-06-30 · process: tools/demo_quality_scorecard.py -->

> Regenerate: `python tools/demo_quality_scorecard.py --markdown --stamp DATE > docs/DEMO-QUALITY-SCORECARD.md`
> Verify snapshot freshness: `python tools/demo_quality_scorecard.py --check-doc`

> The measuring stick for **demos a skeptic can run**: can I run it in one command, reproduce what the README promises, trust what it claims, and understand what I saw — without a model, a key, or a babysitter? Five deterministic axes (runnable · reproducible · honest_scope · self_contained · documented), folded into a **demo-score** (0–100, A–F) and a **demo-debt** integer (the count of concrete, re-derivable demo defects). Every number below is re-derived from disk by `tools/demo_quality_scorecard.py` — no hand-entry. Drive demo-debt to zero to make "better demos" provable.

## Corpus

| Metric | Value |
|---|---|
| Demos scored | 64 |
| **Demo-debt (total defects)** | **0** |
| Mean score | 98.8/100 |
| Median / min / max | 100.0 / 93.1 / 100.0 |
| Grade distribution | A:64 B:0 C:0 D:0 F:0 |

## Per-demo scores

Five axes, each 0–100 (runnable · reproducible · honest_scope · self_contained · documented), weighted into a score and an A–F grade. `def` = units of demo-debt.

| Score | Grade | Debt | run | repro | scope | self | docs | Demo |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| 93.1 | A | 0 | 86 | 100 | 100 | 100 | 78 | `examples/trace-authoring` |
| 94.2 | A | 0 | 100 | 100 | 90 | 88 | 90 | `examples/grammar-repair-demo` |
| 94.2 | A | 0 | 100 | 100 | 90 | 88 | 90 | `cmd/dropindemo` |
| 95.0 | A | 0 | 86 | 100 | 100 | 100 | 90 | `examples/bench-latency` |
| 95.0 | A | 0 | 86 | 100 | 100 | 100 | 90 | `examples/gpu-smoke` |
| 95.0 | A | 0 | 86 | 100 | 100 | 100 | 90 | `examples/turntax` |
| 96.2 | A | 0 | 100 | 100 | 100 | 88 | 90 | `examples/autogen-groupchat` |
| 96.2 | A | 0 | 100 | 100 | 100 | 88 | 90 | `examples/crewai-crew` |
| 96.2 | A | 0 | 100 | 100 | 100 | 88 | 90 | `examples/fanbench` |
| 96.2 | A | 0 | 100 | 100 | 100 | 88 | 90 | `examples/plan-cfi` |
| 96.4 | A | 0 | 100 | 100 | 90 | 100 | 90 | `examples/normgate-evasion` |
| 96.6 | A | 0 | 86 | 100 | 100 | 100 | 100 | `examples/deny-as-value` |
| 96.6 | A | 0 | 86 | 100 | 100 | 100 | 100 | `examples/grammar-repair` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/admit-and-log` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/agent-ab` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/context-debugger` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/dogfood-claude` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/fleet-reuse-demo` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/loader-properties` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/openai-agents-guardrail` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/radixattention` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/remote-vm-guard` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/self-modify-floor` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/session-reload` |
| 98.4 | A | 0 | 100 | 100 | 100 | 100 | 90 | `examples/vdso-cache-hit` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/adjudication-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/agentdojo-redteam` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/auth-hardening` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/escalation-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/extdriver` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/federated-changes` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/ifc-taint-flow` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/mcp` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/mcp-client` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/observability` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/policy-hot-reload` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/preflight-ladder` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/presets` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/quarantine-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/shared-task-record` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/shared-task-record-verdicts` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/steward-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/trace-reset` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/wire-proof` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/wire-quarantine-demo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `examples/witness-gate` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/a2ademo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/agentbenchdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/ctxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/ctxplandemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/cxlpooldemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/guarddemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/hwcachedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/memqdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/poisonedmcpdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/simpledemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/timewolfdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/tokendemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/trychatdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/turntaxdemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/unseedemo` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/causalbench` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/deletioncert` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | `cmd/demorace` |

## Demo-debt work-list

No demo-debt: every demo runs, reproduces, scopes itself, and cleans up. 🎉

## Soft signals (score only, not debt)

### `examples/trace-authoring`
- runnable: a runnable script exists but the README shows no paste-able command to launch it
- documented: no run/usage section and no visible run command — hard to find how to start
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/grammar-repair-demo`
- honest_scope: no link to a deeper doc (CLAIMS / STATUS / an explainer) to back the claim
- self_contained: no stated prerequisites — a cold runner can't tell what to install first
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `cmd/dropindemo`
- honest_scope: no link to a deeper doc (CLAIMS / STATUS / an explainer) to back the claim
- self_contained: no stated prerequisites — a cold runner can't tell what to install first
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/bench-latency`
- runnable: a runnable script exists but the README shows no paste-able command to launch it
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/gpu-smoke`
- runnable: a runnable script exists but the README shows no paste-able command to launch it
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/turntax`
- runnable: a runnable script exists but the README shows no paste-able command to launch it
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/autogen-groupchat`
- self_contained: no stated prerequisites — a cold runner can't tell what to install first
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/crewai-crew`
- self_contained: no stated prerequisites — a cold runner can't tell what to install first
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/fanbench`
- self_contained: no stated prerequisites — a cold runner can't tell what to install first
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/plan-cfi`
- self_contained: no stated prerequisites — a cold runner can't tell what to install first
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/normgate-evasion`
- honest_scope: no link to a deeper doc (CLAIMS / STATUS / an explainer) to back the claim
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/deny-as-value`
- runnable: a runnable script exists but the README shows no paste-able command to launch it

### `examples/grammar-repair`
- runnable: a runnable script exists but the README shows no paste-able command to launch it

### `examples/admit-and-log`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/agent-ab`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/context-debugger`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/dogfood-claude`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/fleet-reuse-demo`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/loader-properties`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/openai-agents-guardrail`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/radixattention`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/remote-vm-guard`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/self-modify-floor`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/session-reload`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone

### `examples/vdso-cache-hit`
- documented: no 'what you see' / output-explainer section — the reader is left to interpret the run alone


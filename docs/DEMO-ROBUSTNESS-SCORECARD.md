# Demo-robustness scorecard

<!-- demo-robustness-scorecard: 2026-06-22 · process: tools/demo_robustness_scorecard.py -->

> Regenerate: `python tools/demo_robustness_scorecard.py --markdown --stamp DATE > docs/DEMO-ROBUSTNESS-SCORECARD.md`

> The measuring stick for **demos that stay simple, fast, and durable**: will it still run next month, on a fresh box, in one obvious command, in seconds — without surprises? Three deterministic axes (simplicity · speed · durability), folded into a **robustness-score** (0–100, A–F) and a **robustness-debt** integer (the count of concrete, re-derivable robustness defects). The sibling `demo_quality_scorecard.py` asks *can a skeptic trust the claim*; this one asks *will it keep running*. Both score the same corpus. Every number below is re-derived from disk by `tools/demo_robustness_scorecard.py` — no hand-entry. Drive robustness-debt down to make "more robust demos" provable.

## Corpus

| Metric | Value |
|---|---|
| Demos scored | 8 |
| **Robustness-debt (total defects)** | **16** |
| Axis-debt | simplicity:0 · speed:11 · durability:5 |
| Mean score | 71.5/100 |
| Median / min / max | 74.1 / 52.3 / 88.1 |
| Grade distribution | A:0 B:2 C:3 D:1 F:2 |

## Per-demo scores

Three axes, each 0–100 (simplicity · speed · durability), weighted into a score and an A–F grade. `def` = units of robustness-debt.

| Score | Grade | Debt | simplicity | speed | durability | Demo |
|---:|:--:|:--:|:--:|:--:|:--:|---|
| 52.3 | F | 3 | 88 | 20 | 54 | `examples/adjudication-demo` |
| 55.9 | F | 3 | 100 | 20 | 54 | `examples/auth-hardening` |
| 67.8 | D | 2 | 100 | 20 | 88 | `examples/escalation-demo` |
| 72.0 | C | 2 | 100 | 66 | 54 | `examples/mcp-client` |
| 76.2 | C | 2 | 100 | 66 | 66 | `examples/mcp` |
| 76.2 | C | 2 | 100 | 66 | 66 | `cmd/simpledemo` |
| 83.9 | B | 1 | 100 | 66 | 88 | `examples/agentdojo-redteam` |
| 88.1 | B | 1 | 100 | 66 | 100 | `examples/wire-proof` |

## Robustness-debt work-list

### `examples/adjudication-demo` — 3 defect(s), score 52.3 (F)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')
- speed: unbounded wait loop in `run.sh` — a polling loop sleeps with no timeout / max-attempts; it can hang forever (bound it)
- durability: no stability / determinism guarantee — the README doesn't say whether a re-run is repeatable (deterministic / byte-identical / pinned); state it

### `examples/auth-hardening` — 3 defect(s), score 55.9 (F)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')
- speed: unbounded wait loop in `run.sh` — a polling loop sleeps with no timeout / max-attempts; it can hang forever (bound it)
- durability: no stability / determinism guarantee — the README doesn't say whether a re-run is repeatable (deterministic / byte-identical / pinned); state it

### `examples/escalation-demo` — 2 defect(s), score 67.8 (D)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')
- speed: unbounded wait loop in `run.sh` — a polling loop sleeps with no timeout / max-attempts; it can hang forever (bound it)

### `examples/mcp` — 2 defect(s), score 76.2 (C)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')
- durability: no stability / determinism guarantee — the README doesn't say whether a re-run is repeatable (deterministic / byte-identical / pinned); state it

### `examples/mcp-client` — 2 defect(s), score 72.0 (C)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')
- durability: no stability / determinism guarantee — the README doesn't say whether a re-run is repeatable (deterministic / byte-identical / pinned); state it

### `cmd/simpledemo` — 2 defect(s), score 76.2 (C)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')
- durability: no stability / determinism guarantee — the README doesn't say whether a re-run is repeatable (deterministic / byte-identical / pinned); state it

### `examples/agentdojo-redteam` — 1 defect(s), score 83.9 (B)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')

### `examples/wire-proof` — 1 defect(s), score 88.1 (B)
- speed: no stated expected runtime — the README never says how long a run takes; state it (e.g. 'runs in ~Ns', 'completes in seconds')

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

### `examples/mcp-client`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it

### `examples/agentdojo-redteam`
- durability: shell-only entry (`.sh`) with no `.ps1` and no cross-platform note — a Windows user can't tell how to run it


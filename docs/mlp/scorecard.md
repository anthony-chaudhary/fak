---
title: "MLP first-lovable-cut scorecard"
description: "fak mlp-score grades the five acceptance rows for the first lovable cut of epic #3256, and never counts an untracked working-tree file as evidence."
---

# MLP first-lovable-cut scorecard

`fak mlp-score` reports whether epic #3256 has reached its first lovable cut for
milestone #17. It grades the five acceptance rows named by issue #3284 and never
uses an untracked working-tree file as evidence.

```text
fak mlp-score
fak mlp-score --json
fak mlp-score --markdown
fak mlp-score --check
fak milestone report
```

The default and JSON commands measure the current committed snapshot. `--check`
also gates on the result: exit 0 means `lovable`; exit 1 means `not-yet`.
`fak milestone report` embeds the same criterion rows under milestone #17.

## Criteria

| Workstream | Criterion | Owning issues |
|---|---|---|
| B1 | One command starts both runtimes and a POST completes a governed session. | #3420, #3258 |
| B3 | A Python or TypeScript SDK drives the complete path offline. | #3261 |
| C1/C3 | The audit journal and per-session cost cap are enforced. | #3275, #3273 |
| D2 | `fak init agent` emits a running governed agent. | #3283 |
| D5 | Time-to-first-governed-agent is under 10 minutes. | #3286 |

## Witness contract

Each criterion has one structured manifest under `docs/mlp/witnesses/`. A row is
`witnessed` only when all of these are true in committed `HEAD`:

1. The manifest parses as `fak-mlp-witness/1` and names the right criterion.
2. Every required claim appears exactly once.
3. Every claim links a committed `test` or `captured-run` artifact.
4. Every claim carries the command that reproduces its proof.

Example:

```json
{
  "schema": "fak-mlp-witness/1",
  "criterion": "init_agent_emits_governed_agent",
  "claims": [
    {
      "key": "scaffolded_agent_completed",
      "kind": "captured-run",
      "path": "docs/notes/INIT-AGENT-SELFCHECK-2026-07-10.md",
      "command": "fak init agent && ./generated-agent --selfcheck"
    }
  ]
}
```

The manifest is an index, not the proof itself. The linked artifact is the
captured render/run or test that must show the claimed behavior.

## Scorecard spine witness

The implementation is covered by package tests that drive absent, invalid,
partial, and complete committed snapshots. The captured CLI and milestone-report
run below was taken on 2026-07-10 from the issue working tree while its evidence
snapshot was committed `59b4dd561`:

```text
> go run ./cmd/fak mlp-score --json
"schema": "fak-mlp-score/1"
"finding": "mlp_not_yet"
"reason": "MLP first lovable cut: 0 of 5 criteria witnessed; 5 remain"
"witnessed": 0
"total": 5
"mlp_debt": 5
"mlp_verdict": "not-yet"
```

The same run named each absent manifest and emitted `evidence: []`; it did not
promote any untracked implementation file into a witness.

```text
> go run ./cmd/fak milestone report --json
"program_scorecards": [{
  "key": "mlp",
  "milestone": 17,
  "verdict": "not-yet",
  "witnessed": 0,
  "total": 5,
  "criteria": [B1, B3, C1/C3, D2, D5]
}]
```

The bracketed criterion list above is a compact transcription of the five
objects in the captured JSON, not literal JSON syntax. Re-running either command
prints the full stable payload and current committed witness state.

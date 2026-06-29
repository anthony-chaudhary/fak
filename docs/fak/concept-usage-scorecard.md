---
title: "fak concept-usage scorecard"
description: "How much the agentic DEVELOPMENT of fak routes through fak's own concepts — the ship-stamp/DCO/binding-verb commit discipline and lane arbitration (usage breadth), and the verify/improve witness syscalls over passive recall (witness depth), all re-derived from git and the .dos journals."
---

# fak concept-usage scorecard

**conceptusage_debt: 0**; composite **87/100 (B)**; usage 100/100; witness 78/100

> concept-usage: usage 100/100, witness 78/100, composite 87/100 (B); dev routes through the fak concepts; zero hard gaps

The question: when an agent builds fak, how much does that development route through fak's *own* concepts — committing with the witness contract (ship-stamp, DCO, a binding verb), arbitrating disjoint lanes, and **witnessing its own claims via the verify syscall instead of trusting a self-report** — versus generic agentic dev? Every number is re-derived from `git log` and the `.dos` journals fak's tooling wrote; the score moves only when development actually uses the concepts more.

## Usage — does the development OUTPUT carry the fak discipline?

| ok | criterion | detail |
|---|---|---|
| yes | recent commits carry the (fak <leaf>) ship-stamp the dos verify referee binds | 200/200 (100%) carry the (fak <leaf>) trailer |
| yes | recent commits are DCO signed-off (git commit -s) | 200/200 (100%) signed-off |
| yes | recent commits use a Conventional-Commits type | 199/200 (100%) conventional |
| yes | recent commit subjects lead with a verb the witness BINDS (not surface/print) | 145/200 (73%) lead with a binding verb |
| yes | concurrent dev arbitrated disjoint lanes (dos_arbitrate ACQUIRE/RELEASE rows) | 44 lane ACQUIRE(s) across 49 distinct lane(s) |

## Witness — does development TRUST EVIDENCE over self-report?

| ok | criterion | detail |
|---|---|---|
| yes | development proactively witnessed claims via the verify/improve syscall | 14 verify + 9 improve syscall(s) in the journal |
| yes | a healthy share of RECENT dev decisions are evidence-grounded (verify/improve), not recall-only | 44% of the last 16 dev decision(s) used a proactive witness syscall (target >=15%; 34 passive UNVERIFIABLE auto-recalls excluded as non-decisions) |
| no | recalled memory was re-verified against ground truth, not left UNVERIFIABLE | 74/303 (24%) recalls resolved to a checked verdict |
| yes | the verdict journal exists — development actually ran the witnessing syscalls | 326 verdict-journal row(s) |

## Run it

```bash
go run ./cmd/fak concept-usage-score            # score this tree's concept dogfooding
go run ./cmd/fak concept-usage-score --markdown # regenerate this doc
go test ./internal/conceptusage/...             # prove the fold over a thin vs healthy corpus
```

## The 3× program — grow the witness axis honestly

The usage axis is already saturated (commit discipline + lane arbitration are fully dogfooded); the witness axis is the lever. It is thin because **witnessing is manual and rare** — `dos verify` / `dos improve --observe` rows accrue only when someone runs them by hand, while passive `memory_recall` rows dominate the journal. So a 3× is NOT firing verify calls by hand during the measurement window (that is the data-gaming pattern every fak scorecard refuses) — it is making the witness syscall a **byproduct of real work** so the share rises structurally across sessions:

1. **Witness every ship.** Run `dos verify <PLAN> <PHASE>` (or `dos improve --observe`) at ship time, not `dos commit-audit` alone — commit-audit is read-only and writes no row; `verify`/`improve` are the syscalls this axis counts.
2. **Re-verify recalled memory.** When a memory is recalled, re-check it against ground truth (`dos recall <name>`) so it resolves to FRESH/STALE instead of sitting at the 76%-UNVERIFIABLE floor.
3. **Wire it into the dev loop.** The durable fix is a ship-path step (a post-commit / Stop-hook auto-`dos verify`) so the witness share climbs without anyone remembering to — the same way the usage axis is green because the commit hooks make the stamp/DCO automatic.

Re-run after a dev session and `--compare` against a pinned `--json` baseline: the verdict reports the multiple on the witness score (the lever), so a real 3× (witness 6% → 18% share) is provable, not asserted.

**Next:** hold the line; re-run after a dev session — keep witnessing claims via the verify syscall, not self-report

---
title: "fak concept-usage scorecard"
description: "How much the agentic DEVELOPMENT of fak routes through fak's own concepts — the ship-stamp/DCO/binding-verb commit discipline and lane arbitration (usage breadth), and the verify/improve witness syscalls over passive recall (witness depth), all re-derived from git and the .dos journals."
---

# fak concept-usage scorecard

**conceptusage_debt: 1**; composite **66/100 (D)**; usage 100/100; witness 44/100

> concept-usage carries 1 debt (usage 100/100, witness 44/100, composite 66 D): witness_share

The question: when an agent builds fak, how much does that development route through fak's *own* concepts — committing with the witness contract (ship-stamp, DCO, a binding verb), arbitrating disjoint lanes, and **witnessing its own claims via the verify syscall instead of trusting a self-report** — versus generic agentic dev? Every number is re-derived from `git log` and the `.dos` journals fak's tooling wrote; the score moves only when development actually uses the concepts more.

## Usage — does the development OUTPUT carry the fak discipline?

| ok | criterion | detail |
|---|---|---|
| yes | recent commits carry the (fak <leaf>) ship-stamp the dos verify referee binds | 200/200 (100%) carry the (fak <leaf>) trailer |
| yes | recent commits are DCO signed-off (git commit -s) | 200/200 (100%) signed-off |
| yes | recent commits use a Conventional-Commits type | 199/200 (100%) conventional |
| yes | recent commit subjects lead with a verb the witness BINDS (not surface/print) | 146/200 (73%) lead with a binding verb |
| yes | concurrent dev arbitrated disjoint lanes (dos_arbitrate ACQUIRE/RELEASE rows) | 44 lane ACQUIRE(s) across 49 distinct lane(s) |

## Witness — does development TRUST EVIDENCE over self-report?

| ok | criterion | detail |
|---|---|---|
| yes | development proactively witnessed claims via the verify/improve syscall | 14 verify + 4 improve syscall(s) in the journal |
| no | a healthy share of decisions are evidence-grounded (verify/improve), not recall-only | 6% of 321 decision point(s) used a proactive witness syscall (target >=15%) |
| no | recalled memory was re-verified against ground truth, not left UNVERIFIABLE | 74/303 (24%) recalls resolved to a checked verdict |
| yes | the verdict journal exists — development actually ran the witnessing syscalls | 321 verdict-journal row(s) |

## Run it

```bash
go run ./cmd/fak concept-usage-score            # score this tree's concept dogfooding
go run ./cmd/fak concept-usage-score --markdown # regenerate this doc
go test ./internal/conceptusage/...             # prove the fold over a thin vs healthy corpus
```

**Next:** retire worst-first: witness_share — 6% of 321 decision point(s) used a proactive witness syscall (target >=15%)

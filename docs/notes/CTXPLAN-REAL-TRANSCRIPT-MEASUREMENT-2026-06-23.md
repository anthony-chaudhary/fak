# Planned view measured over real session transcripts

_Generated: 2026-06-23 · issue #559 · `cmd/ctxplanbench` · 5 heaviest sessions on this box, 715 replayed turns._

`internal/ctxplan/scaling.go` takes a mean tokens/turn and a forecast-hit rate and **computes**
the resident-token curve. That is a model — useful for the asymptotic claim, but it is not a
measurement. This note records the measurement: the planned view replayed turn-by-turn over the
heaviest REAL Claude Code transcripts on this box, reporting the three quantities the issue names.

> **Companion:** this note measures the planner's **output** (resident tokens). Its sibling
> [`CTXPLAN-PLANNING-COST-FLATTEN`](CTXPLAN-PLANNING-COST-FLATTEN-2026-06-23.md) measures the
> planner's **own per-turn work** (the `Θ(N²)→Θ(c·N)` candidate-index flatten, #558), and the
> honest cost↔fidelity dial that the bounded probe trades on real sessions.

## How (the honest construction)

Each transcript is ingested through the shipped `cdb.IngestSession` → `recall` core image, so a
result the write-time gate quarantines is **sealed here too** (8 of 723 pages on these sessions).
Each `recall.Page` is bridged 1:1 into a `ctxplan.Span`; the session is replayed one benign span
per turn through the **real planner** (`ctxplan.Materialize`) and the **real page-fault handler**
(`ctxplan.DemandPage`). Nothing about the win is modeled:

- **resident tokens** — the real per-span sizes (`ceil(bytes/4)`, the planner's own `TokenCost`),
  summed cumulatively, the empirical analogue of `scaling.go`'s `cumLinear` / `cumCapped`.
- **fault rate** — the ground-truth "reference" signal is real lexical overlap: a turn references
  an older span when the arriving result's content fingerprint appears in that span's descriptor
  (the same extractive signal the planner's relevance uses). A miss is a referenced span the PRIOR
  resident view had elided; each is served through `DemandPage`, proving recovery.
- **quality vs compaction** — `ctxplan.Audit` on the real plans (exact recall every turn) vs
  `ctxplan.CompactionView` (handles stripped → UNFAITHFUL). The served faults ARE the facts the
  planned regime recovers that compaction would lose.

The forecast is a deliberately cheap heuristic a deployment could actually run: the last 6 turn
descriptors as intents (a recency window) + the durable spans pinned. It is **not an oracle**.

## The numbers (budget W=8000 tokens, window K=6)

| session (heaviest first) | turns | linear cum | planned cum | peak | refs | faults | served | faithful |
|---|---|---|---|---|---|---|---|---|
| 0c073826… | 217 | 25.82M | 1.65M | 8.2K | 9558 | 2410 (25.2%) | 2410 | 217/217 |
| af549457… | 359 | 20.58M | 2.73M | 8.0K | 26286 | 8915 (33.9%) | 8915 | 359/359 |
| 4d528918… | 90 | 16.08M | 657.6K | 8.0K | 1662 | 633 (38.1%) | 633 | 90/90 |
| a5d0581a… | 32 | 7.21M | 195.3K | 10.2K† | 467 | 125 (26.8%) | 125 | 32/32 |
| 0227e3e1… | 17 | 124.4K | 11.8K | 953 | 95 | 0 (0.0%) | 0 | 17/17 |
| **aggregate** | **715** | **69.81M** | **5.24M** | — | **38068** | **12083 (31.7%)** | **12083** | **715/715** |

† peak exceeds W on a **pin-overrun turn** (durable pins alone exceed the budget — documented
planner behavior: pins stay resident, `OverBudget` is set, reported not hidden).

### What the numbers say

- **Resident tokens: 13.3× fewer than linear.** Planned cum 5.24M vs 69.81M if the whole
  transcript stayed resident (the Θ(N²) prefill tax). The peak held to the 8000-token working set
  on every turn except the two pin-overrun turns noted.
- **Fault rate: 31.7% miss, 100% served.** 12,083 of the 38,068 real back-references were to spans
  the recency-window forecast had elided — and **every one was served by `DemandPage`** (0 refused,
  0 lost). The fault tax (re-prefill) was 21.11M tokens cumulatively, the measured analogue of
  `scaling.go`'s `FaultTaxCum`. A stronger forecast would lower the 31.7%; the point is that a miss
  is a cheap page fault, never a lost fact.
- **Quality vs compaction: exact recall 715/715 turns; compaction lost facts on 695/715.** The
  12,083 served faults are exactly the facts the planned regime recovers that compaction would have
  permanently dropped. Compaction's oldest-fact survival was ρ^k ≈ 0 on every long session.

## Honest fences

- The forecast is a **cheap recency heuristic**, not a learned/oracle forecast — the 31.7% miss rate
  is an upper bound a real forecaster would tighten; the headline is that misses are *served*.
- The reference signal is **lexical descriptor overlap**, not a semantic model — it is the same
  signal the planner scores with, so "referenced" means the same thing at measure time and plan
  time, but it under-counts paraphrased references.
- Resident units are the **bytes/4 proxy** (`ctxplan.TokenCost`); a real BPE tokenizer shifts the
  absolutes, not the regime.
- Off the live path: a bench cmd, not wired into the agent HTTP loop.

## Reproduce

```bash
go run ./cmd/ctxplanbench -selfcheck                       # pipeline gate (exit 0)
go run ./cmd/ctxplanbench -heaviest 5 -budget 8000 -window 6
go run ./cmd/ctxplanbench -transcripts <path>.jsonl -out report.json
```

Witnesses: `go test ./cmd/ctxplanbench` (`TestReplayInvariants`, `TestReplayLooseBudgetHoldsEverything`).

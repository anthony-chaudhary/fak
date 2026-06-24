# The planner's own work, measured: bounding per-turn compute over real sessions

_Generated: 2026-06-23 · issue #558/#559 · `cmd/ctxplanbench` · 5 heaviest sessions on this box, 851 replayed turns._

[`CTXPLAN-REAL-TRANSCRIPT-MEASUREMENT`](CTXPLAN-REAL-TRANSCRIPT-MEASUREMENT-2026-06-23.md)
measured the planned view's **output** — resident tokens 13.3× below linear, exact recall every
turn. It did not measure the planner's **own work**: the full-scan planner
(`ctxplan.Materialize` → `PlanCells`) scores *every* span each turn, so at turn `i` it scores `i`
candidates and `Σ i = Θ(N²)` cumulatively — the one cost "O(1) resident" never bounded
(`scaling.go`'s `PlannerComputeCum`). `internal/ctxplan/index.go` is the fix (a candidate
**index**, the planner's access path) and `IndexBoundedPlannerCompute` proves the model
(`Θ(c·N)`, linear). This note **measures that flatten end to end** over the heaviest real
transcripts, and reports an honest cost↔fidelity finding the model alone could not surface.

## How (the construction, unchanged from the sibling note + one addition)

Each transcript is ingested through the shipped `cdb.IngestSession` → `recall` core image and
bridged 1:1 into `ctxplan.Span`s, replayed one benign span per turn. The addition: alongside the
existing full-scan plan (`view.Plan`, which scores all `N` spans), the replay maintains **one
persistent `ctxplan.Index`** across turns — the `SessionPlanner` pattern: `Add` each new span
incrementally, then `Index.PlanCells` **probes a bounded candidate set** (≤ cap) instead of
re-scoring all `N`. Both plans go through the **same** `Optimize(ObjGreedy)` under the **same**
forecast and budget, so the only variable measured is the candidate set — full scan (`N`) vs
bounded probe (`c`). `Plan.Candidates` is each turn's scored-candidate count; the resident sets
are compared turn-for-turn (`sameResident`: identical span ids + identical resident cost).

## The numbers (budget W=8000, window K=6, probe cap c=128, recency R=32 — the shipped defaults)

| session (heaviest first) | turns | full-scan cand | probe cand | flatten | peak/cap | plan-agree | bounded-agree |
|---|---|---|---|---|---|---|---|
| 163c21fc… | 119 | 7.4K | 6.5K | 1.1× | 112/128 | 32/119 | 0/87 |
| 928e17d8… | 43 | 968 | 943 | 1.0× | 44/128 | 34/43 | 0/9 |
| 0c073826… | 217 | 23.9K | 18.4K | 1.3× | 128/128 | 45/217 | 9/181 |
| a9fa5236… | 130 | 9.0K | 8.2K | 1.1× | 128/128 | 58/130 | 16/88 |
| e8e026c8… | 342 | 58.9K | 34.1K | **1.7×** | 128/128 | 39/342 | 3/306 |
| **aggregate** | **851** | **100.1K** | **68.0K** | **1.5×** | **128** | **208/851** | **28/671** |

Resident tokens (the token-efficiency win, reproduced): linear cum **76.97M** vs planned **6.36M**
= **12.1× fewer resident than linear**, exact recall **851/851** turns, **15,530** back-reference
faults **all served** (0 refused, 0 lost). The bounded probe's resident cum is **6.29M** — within
**1.2%** of the full-scan's 6.36M (it is in fact marginally *leaner*).

### What the numbers say

- **The per-turn planner compute is bounded — the bound held on every one of 851 turns.** Peak
  probe `≤ cap` (128) for every session; the full re-scan's `Θ(N²)` per-turn growth is replaced by
  a constant `≤ c` ceiling.
- **The cumulative flatten is real and grows with N — visible in the data.** 1.5× less
  candidate-scoring work aggregate; per session it tracks length monotonically (43-turn session
  1.0× → 342-turn 1.7×), exactly the `Θ(N²)→Θ(c·N)` shape. At these horizons (hundreds of turns)
  the flatten is modest; the gap widens *linearly* with the turn count — at 10k turns the model
  puts it near `N/2c ≈ 40×`, at 1M turns it is the enormous gap `index_test.go`'s
  `TestIndexBoundedPlannerComputeFlattensQuadratic` asserts.
- **The token-efficiency win survives the bounded planner.** Bounded resident 6.29M vs full-scan
  6.36M (98.9%): the same ~12× resident-token reduction at bounded planning cost. The "agreement"
  gap below is about *which* spans are resident, **not** how many tokens.

## The honest finding: a cost↔fidelity dial, not a free lunch

The bounded plan reproduces the full-scan plan **exactly when it sees the full candidate set**
(probe = all spans → **342/342 identical** on the largest session — the end-to-end witness of
`index.go`'s incremental==batch / behavior-preservation property, on real data). But at the
**shipped default width** it diverges from the full-scan plan on most long-session turns
(aggregate plan-agree 208/851; among the 671 turns where the cap actually pruned, only 28 agree).
Isolating *why* on the 342-turn session:

| probe config | flatten | plan-agree | what it isolates |
|---|---|---|---|
| cap 128, recency 32 (**default**) | 1.7× | 39/342 | the shipped point |
| cap **∞**, recency 32 | 1.2× | 39/342 | unbounded cap **does not** fix it — not the cap |
| cap 128, recency **4096** | 1.7× | **131/342** | widening coverage **does** — recency tail matters |
| cap ∞, recency ∞ (probe = all) | 1.0× | **342/342** | full coverage ⇒ identical (mechanism correct) |

So the divergence is **access-path coverage**, not the hard cap: a real session spreads
selection-worthy benefit (recall's per-page `utility`, older relevance) across more spans than the
default recency/relevance/durability paths reach. **Every divergence is a bounded efficiency
miss, never a lost fact** — the pruned span stays in the lossless store and pages back on demand
(the 15,530 served faults, 0 lost, are exactly these). The cap and recency window are a measured
**cost↔fidelity dial**: widen them and agreement climbs toward 100% at a higher — still bounded —
planning cost. The clean next lever this measurement names: a **utility access path** in the probe
(index the high-utility old spans recall already flags), which the table predicts would close most
of the 39→342 gap at little added cost.

## Honest fences

- **The flatten is modest at these lengths** (1.5× over 851 turns) — it is `Θ(N)` in the horizon,
  so it pays off on long-lived sessions, not short ones. The asymptote is modeled, not measured
  past ~342 turns.
- **Plan agreement at default width is low on long sessions** (and is reported, not hidden). It is
  a benefit-*optimization* difference between two faithful planners, not a recall loss: bounded
  resident tokens stay within 1.2% of full-scan and exact recall is preserved (faults served).
- **Hardware-independent.** Every number here is an exact integer candidate/turn count or a
  byte/4 token sum — it reproduces bit-for-bit on any box (the M3 Pro `node-macos-a` included);
  only a wall-clock would move, and this measures *work*, not time.
- **Off the live path.** `ctxplanbench` is a bench cmd; the `SessionPlanner` it mirrors
  (`internal/agent/ctxplan_session.go`, #558) is the agent-seam home for the same mechanism,
  unit-witnessed (`TestSessionPlannerBoundedMatchesStatelessFullScan`) but not yet on the gateway
  HTTP loop (the `req.Raw` passthrough, #555, is its own step).

## Reproduce

```bash
go run ./cmd/ctxplanbench -selfcheck                                   # pipeline + planning-cost gate (exit 0)
go run ./cmd/ctxplanbench -heaviest 5 -budget 8000 -window 6           # the table above
go run ./cmd/ctxplanbench -transcripts <s>.jsonl -probe-cap 128 -probe-recency 4096   # the dial
go run ./cmd/ctxplanbench -transcripts <s>.jsonl -probe-cap 100000 -probe-recency 100000  # probe=all → identical
```

Witnesses: `go test ./cmd/ctxplanbench` (`TestReplayInvariants` now asserts the bound + agreement
at N<cap; `TestPlanningCostFlattenWhenBounded` forces the cap to bite and asserts the `Θ(c·N)`
ceiling holds and the flatten is real).

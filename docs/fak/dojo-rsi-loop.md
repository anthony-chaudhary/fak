---
title: "fak dojo RSI loop — the gym's missing 5th rung"
description: "Design for an autonomous, agentic-RSI loop that turns a dojo OVER/UNDER_CLAIM
finding into a worktree-measured, suite-witnessed recalibration kept only on a real gain —
reusing the rsiloop/shipgate keep-bit verbatim, with FoldCalibrable so the loop can never
auto-erase the floors the dojo exists to defend."
---

# fak dojo RSI loop — the gym's missing 5th rung

> **Audience.** Anyone designing the dojo's autonomous fifth rung — by the end you'll understand why the loop optimises `FoldCalibrable` (not raw `MeanCalibErr`) so it can never auto-erase the floors the gym defends.

The [dojo](dojo.md) is fak's prediction-vs-reality gym. It already closes four of the five
RSI rungs:

```
declare a prediction  ->  run the scenario  ->  measure billed reality  ->  score the gap  ->  trend
     (the theory)            (a corpus)            (the provider's bill)      (calib_err)     (ledger)
```

The fifth rung — **act on the gap and prove the act helped** — is today a human reading the
`next_action` string (`harvest the under-claimed saving`, `recalibrate the over-claiming
lever`). This doc designs the loop that closes it autonomously: the dojo's `calib_err` metric
joined to the non-forgeable keep-bit that [`internal/rsiloop`](https://github.com/anthony-chaudhary/fak/tree/main/internal/rsiloop) +
[`internal/shipgate`](https://github.com/anthony-chaudhary/fak/tree/main/internal/shipgate) already provide, exactly the way
[`fak guard-verdict-rsi`](guard-verdict-rsi-loop.md) closed the guard loop on our own journal.

It is the **same machine** as the guard verdict loop, pointed at a different journal: the dojo
history ledger (`docs/dojo/history.jsonl`) is the journal, `calib_err` is the quality signal,
the worst-calibrated measured lever/metric is the candidate, and a recalibration is KEPT only
on a re-measured strict gain plus a green external witness.

## What it optimises — `FoldCalibrable`, not raw `MeanCalibErr`

The naive target is the run's `mean_calib_err`. That is a trap. `calibErr` is minimised at
`claimed := realized`, so a constant rewrite is a *guaranteed, content-free* "gain" — and for a
**floor** metric like `vcache-warmth/false_warm_rate` (claim `0.0`, the lethal false-warm
class) "recalibrating" the claim up to the measured rate would **erase the very guard the dojo
exists to defend**. The loop would optimise itself into dishonesty.

The fix is an additive `IntentionalFloor bool` on `dojo.Prediction` (zero-value-safe, exactly
like the existing `LowerIsBetter`) and a new pure **`FoldCalibrable`** that:

- folds `calib_err` over only the episodes whose claim is a genuine **estimate**
  (`IntentionalFloor == false`), and
- for a **floor** metric contributes a `FloorRespectErr` term: `0` while the floor holds,
  *rising* as the floor is breached.

So a genuine estimate (`resume-posture/cold_write_share 0.85`) recalibrated toward its corpus
mean lowers `FoldCalibrable` and is eligible to KEEP; a floor (`false_warm_rate 0.0`) fitted to
its empirical rate *raises* `FoldCalibrable` and REVERTs. The auto-erasure is mechanically
impossible, not merely discouraged.

## The keep-bit (every Witness field DERIVED, never asserted)

The loop folds each candidate through the **existing** `shipgate.Evaluate(Witness{...})` with
`Class = ClassFull` (all three rungs required). The proposer writes the claim delta but cannot
write any of these:

| Witness field | How it is DERIVED |
|---|---|
| `Before` | `FoldCalibrable` over the **frozen corpus** on the pinned baseline SHA, re-derived from `main` every run (rsiloop's `BaselineMetric` contract pins the SHA once so every candidate forks identically). |
| `After` | `FoldCalibrable` re-run by a child `fak dojo run --json` **inside the candidate worktree**. The number falls out of `dojo.Score` over the corpus's real billed `cache_read`/`cache_creation`, which the claim-edit does not touch. |
| `LowerBetter` | `true` — smaller folded calib metric wins. |
| `SuiteGreen` | exit code of a real `go build ./... && go vet ./...` (Windows-native proxy) or `./test.ps1` / WSL `go test` over the dojo + lever packages. |
| `TruthClean` | `treeChangedOnly(worktree, claimRegistryPath)` — **exactly one file, exactly the recalibrated literal** changed; any stray edit fails closed. |

`shipgate.Evaluate` keeps iff `strictGain AND SuiteGreen AND TruthClean`. The breaker
(`shipgate.Gate`, k=3) upgrades the k-th consecutive non-keep to ESCALATE.

Every tick also emits a structured `dojo_calibration` scorecard beside the scalar
FoldCalibrable value. The scorecard records the before/after folded value, measured
delta, estimate/floor sub-terms, sample floor, selector priority, witness bit, and
agent-arm routing bit. It is telemetry only: DOS observe receipts and JSON output can
explain why a tick was kept, reverted, or escalated, but the keep-bit still reads only
the scalar gain plus the suite/truth witnesses above.

### Anti-gaming defenses (structural, not checks)

1. **Widening the ruler is impossible.** `CalibBand`/`calibErr`/`MaxCalibErr` live in
   `internal/dojo`; the rewrite target is one anchored literal in the claim registry. A diff
   touching the band changes a second file → `treeChangedOnly` → REVERT.
2. **Lowering a claim to fake a gain is defeated twice.** (a) The gain must hold on **two
   disjoint corpus shards** — overfitting the seen shard raises `calib_err` on the held-out one
   and REVERTs. (b) For a floor, `IntentionalFloor` flips the sign of incentive (closing the gap
   *raises* `FloorRespectErr`).
3. **Safety floors are deny-listed independently of report flags.** `false_warm_rate` is
   `NeverRecalibrate`: even if a malformed report drops `IntentionalFloor`, the proposer routes
   it to the floor arm and the keep gate refuses any forced `RECALIBRATE` swap.
4. **Sparse levers owe stricter sample floors.** The global `DefaultMinSample` is only a lower
   bound; cells such as `compaction/token_shed_ratio` can require a higher `MinSample` before a
   mechanical recalibration may KEEP.
5. **`Claimed := Realized` is not free.** The estimate levers carry pinned-claim tests; a
   silent rewrite turns the suite RED → REVERT. An honest recalibration updates the test too —
   a two-file change, so mechanical `treeChangedOnly` forbids it and routes it to the agent arm.
6. **UNMEASURED is uncandidatable by construction.** `compaction` returns `Measured:false`;
   `Fold` skips it, the board never lists it, and the candidate picker can only select a
   measured, non-floor episode. The honesty constraint is structural.
7. **Per-lever non-regression.** A big drop on a noisy lever cannot mask a small regression on a
   critical one — the gate also requires no per-lever `calib_err` increase between Before/After.

## The candidate + the mechanical/agent split

A candidate targets **exactly one** `(lever, metric)` cell of the worst-first board:

```go
type Recal struct {
    Lever, Metric          string
    Kind                   RecalKind // RECALIBRATE | REPROJECT | HARVEST
    OldClaimed, NewClaimed float64
    MeasuredMean           float64
    Sample                 int
    MinSample              int    // effective per-cell floor
    Verdict                string // OVER_CLAIM | UNDER_CLAIM
    IntentionalFloor       bool   // mirrored from the Prediction; gates routing
    NeverRecalibrate       bool   // structural deny-list, independent of the flag
}
```

**Mechanical (no agent) — `RECALIBRATE`:** re-point a genuine *estimate* constant at its corpus
central tendency (one anchored-regex literal swap in the claim registry). The keep-bit is pure
re-measurement. This is the bulk of cycles.

**Agent-required — `REPROJECT` / floor / bimodal / `HARVEST`:**
- `REPROJECT` — a code change to a lever's *projection* (e.g. `resume.Backtest` thresholds,
  `vcacheobserve` belief logic) that moves Realized toward an **unchanged** claim. Gated by the
  same re-measurement (a path-allowlist `treeChangedWithin`, still failing closed).
- **Floor breach** (`false_warm_rate > 0`) — a bug in the belief code, never a recalibration.
- **Bimodal** (`cross_session_warm_hit_rate` 0.00→0.65) — a single scalar claim is the wrong
  model; ESCALATE for a workload-conditioned claim.
- `HARVEST` — a real free saving too big to auto-land: file a GitHub issue with the board row +
  corpus evidence, journaled as a TRACK row, never a KEEP.

Everything that decides KEEP is mechanical; the agent only *proposes* and answers escalations.

## Phased build plan (each phase shippable + witness-gated, safest-autonomous-slice first)

**Phase 0 — extract claims to a registry + add `IntentionalFloor`.** New pure
`internal/dojo/claims.go` (`(lever,metric) → {Claimed, IntentionalFloor, Basis}`, one anchored
literal per cell), `IntentionalFloor` on `dojo.Prediction`, `FoldCalibrable`+`FloorRespectErr`.
Edit `cmd/fak/dojo.go` episode builders to read the registry; mark `false_warm_rate` and
`cross_session_warm_hit_rate` as floors. *Witness:* `go build`/`vet` green, the existing
pinned-claim tests still pass (proving extraction preserved every value), a new test asserting
`FoldCalibrable` excludes floors. Pure refactor; ships alone; unblocks the worktree arm.

**Phase 1 — pure proposer + self-scoring loop (no worktree, mutates nothing, safe unattended).**
New pure `internal/dojocal/dojocal.go` (`ProposeRecals`, the floor/UNMEASURED routing, a
`RunIteration` that replays `FoldCalibrable` with the claim swapped — deterministic, CI-able,
`--witness {"ok":true}` exactly like `guard-verdict-rsi`). New `fak dojo-rsi fold|propose|run
[--check]` that journals to `docs/dojo/rsi-journal.jsonl` and emits the proposed diff to stdout
only. New `dojo-rsi-score` skill (twin of `guard-rsi-score`) + this doc. *Witness:* a paired
test proving KEEP-on-estimate-gain, REVERT-on-no-gain, **REVERT-on-floor-target**, and
refuse-on-unmeasured-corpus.

**Phase 2 — worktree arm (real `go` witness, `RECALIBRATE` only, auto-land gated).** Clone
`internal/rsiloop/worktree.go` into a `dojocal.Harness`; `Measure` rewrites one registry
literal, re-runs the dojo, derives `SuiteGreen`/`TruthClean`. The two-shard gate lands here.
`--apply` lands a kept literal on a clean by-path commit (`git commit -- internal/dojo/claims.go`,
never `git add -A`). *Witness:* a real RECALIBRATE cycle over `~/.claude/projects` banks a KEEP
with `SuiteGreen` from a real `go test`; `dos commit-audit` + `dos review` on the landed commit.

**Phase 3 — self-pacing loop + agent `REPROJECT`/`HARVEST` arm + CI feed.** `internal/dojocal/
select.go` reuses `nightrun`'s `novelty×value×staleness` to pick the worst-AND-least-recently-
touched cell each tick (no thrashing). `fak dojo-rsi loop` self-paces via `ScheduleWakeup`,
wraps each tick in `dos arbitrate` (a `dojocal` lane lease), routes verdicts through
`rsiloop.Observer` to `dos improve --observe`, and gates issue-filing/landing behind
`dos-goal-gate`. New `dojo-rsi-feed.yml` posts the journal's KEEP/REVERT trend (no corpus on CI).

## What stays a human decision

- **Landing a real harvest** — `HARVEST` files an issue; a human decides whether to build the
  saving. `--apply` is operator-invoked and by-path.
- **Escalations** — breaker trips, floor breaches, or high cross-corpus variance: a human/agent
  decides whether the gap is a mis-claim, a belief-code bug, or a genuinely un-improvable
  workload-conditioned metric — the judgment `FoldCalibrable` correctly refuses to make alone.
- **Authoring `REPROJECT` patches** — the agent proposes; a human reviews the kept diff before
  it lands on the shared trunk.

The one move no single proposal owned and this design makes safe: **the optimisation target is
`FoldCalibrable`, not `MeanCalibErr`** — so the gym can never auto-erase the floors it exists to
defend, and a constant-rewrite "gain" is admissible only where the claim is genuinely an
estimate.

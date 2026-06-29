---
title: "fak dojo: the prediction-vs-reality gym"
description: "A closed predict -> run -> measure -> score -> calibrate -> trend loop for fak's token-saving levers, now with the 5th rung wired: each optimization declares a theory, the scenario runs, billed reality is measured, the calibration error is scored and trended, and a miscalibration is fed back as a recalibration kept only on a re-measured gain — demonstrated end to end on a real session corpus."
---

# The dojo: a prediction-vs-reality gym for fak's savers

*Who this is for: anyone adding or trusting a token-saving lever in fak (resume
cache posture, history compaction, vCache warmth, KV-prefix reuse, dispatch
dedup). You'll learn the one loop the dojo runs, how to read its scored verdicts,
and how the loop now closes its own calibration gaps without erasing the floors it
defends.*

fak has many levers that **predict** a saving: "this resume posture saves ~48k
tokens", "compaction preserves the cache prefix", "vCache stays warm". Each is a
*theory*. What fak lacked was a single place that closes the loop on those
theories: a theory is declared, the actual thing runs, we look at billed reality,
we score the gap, and we feed a miscalibration back as a fix and re-measure. Today
`fak resume validate` does the measure step for one lever and emits one number (a
posture accuracy). The dojo generalizes that into a **gym**: any lever registers,
scores against the provider's own billed reality, and accrues a calibration trend.

## The loop

```
  declare a prediction  ->  run the scenario  ->  measure ground truth  ->  score the gap  ->  calibrate  ->  trend
       (the theory)            (a workload)          (the provider's bill)     (verdict)      (feed back)     (ledger)
```

- **Prediction** — a lever's `Claimed` number for one metric, declared *before*
  reality is consulted, with the `Basis` that produced it. Every claim lives in one
  anchored literal in the [claim registry](#the-claim-registry-one-anchored-number-per-cell).
- **Scenario** — the workload the lever runs against. Today that is an *offline*
  replay of a corpus of real Claude Code transcripts (deterministic, hardware-free,
  CI-able); the *live* mode (`--dojo` session markers) is the same shape with a
  different source.
- **Outcome** — the `Realized` number, lifted from the provider's own usage
  records. Every number is tagged `WITNESSED` (fak authored it) or `OBSERVED`
  (relayed from the provider — a bad value is not, by itself, a fak fault).
- **Episode** — one scored prediction-vs-reality: the residual, its normalized
  magnitude (`calib_err`), a letter grade, and a **verdict**:
  - `CALIBRATED` — reality met the claim (within the band).
  - `OVER_CLAIM` — reality fell short of the claim (a theory promising more than
    billed reality delivered).
  - `UNDER_CLAIM` — reality beat the claim (a saving the model is not crediting).
  - `UNMEASURED` — no ground truth existed to score against.
- **Ledger + trend** — `fak dojo run --append-history` appends one row per tick to
  `docs/dojo/history.jsonl` and trends the **mean calibration error** against the
  last row, so the gym answers not just "what did this lever save" but "are our
  predictors getting *better calibrated* over time".

The fold is a **report contract, not a second quality gate**: it is `ACTION` only
when the run could not be measured, and `OK` otherwise — surfacing an over-claim as
an *advisory* line. This mirrors the cadence report's advisory posture, so the gym
measures without double-gating.

### The 5th rung

The four rungs above stop at *trend*. The fifth — **act on the gap and prove the
act helped** — is the self-improving RSI loop described in
[the 5th rung](#the-5th-rung-the-self-improving-rsi-loop) below. It turns an
`OVER_CLAIM` / `UNDER_CLAIM` finding into a re-measured recalibration kept only on a
real gain, and it is built so it can never erase the safety floors the dojo exists
to defend.

## The registered levers

`fak dojo list` prints the live registry. Three levers ship today; each scores one
or more metrics against the provider's billed reality.

| lever | metric | theory (the claim) |
|---|---|---|
| **resume-posture** | `posture_accuracy` | the projection's per-boundary cold/warm call is correct (claim 1.0) |
| | `cold_write_share` | a cold resume re-prefill rewrites ~85% of the resident at the write premium (claim 0.85) |
| | `cross_session_warm_hit_rate` | ~0% by default; workload-dependent and bimodal across corpora, observed 0.00→0.65 (floor 0.0) |
| **compaction** | `cache_prefix_preserved` | a fired compaction keeps the prefix byte-identical so the provider still cache-reads it (claim 1.0) |
| | `token_shed_ratio` | the projected shed matches the billed input-token delta (claim 1.0) |
| **vcache-warmth** | `false_warm_rate` | the warmth belief NEVER predicts warm on a call that bills cache_read=0 (floor 0.0 — the lethal class) |
| | `warm_recall` | the belief recalls every genuinely-warm read it could have predicted (claim 1.0) |

`resume-posture` reuses the exact corpus scan and `resume.Backtest` residual the
`fak resume validate` shell uses. `vcache-warmth` replays the corpus through the
shipped `vcacheobserve.Observe` (the same fold `fak vcache observe` uses).
`compaction` needs a paired ON/OFF corpus that standard transcripts don't carry, so
it **refuses to score** rather than invent a number (see
[honest refusal](#honest-refusal-a-lever-that-wont-fabricate)).

Two of these metrics are **floors**, not estimates: `false_warm_rate` (a believed-
warm call the provider bills as cold is the lethal class) and
`cross_session_warm_hit_rate` (a bimodal default that must not be fitted to one
corpus). A floor is a guard the dojo asserts reality must not breach, and the RSI
loop is built so it can never "recalibrate" a floor away.

## Subcommands

```sh
# score every registered lever against your real session history, and trend it
fak dojo run --corpus ~/.claude/projects --append-history

# machine-readable, or an advisory gate (non-zero only if nothing could be measured)
fak dojo run --corpus ~/.claude/projects --json
fak dojo run --corpus ~/.claude/projects --check

# the cross-lever leaderboard (one row per lever, verdict distribution, worst-first)
fak dojo board --corpus ~/.claude/projects

# the vDSO on-vs-off ablation as a WITNESSED episode (CPU-only, corpus-free)
fak dojo ablate

# what levers are registered, and what each one scores
fak dojo list

# post a calibration rollup to the dojo Slack channel (trend = committed history, no scan)
fak dojo post --rollup trend --dry-run
```

- **run** scores the levers over the corpus, folds the episodes, and (with
  `--append-history`) trends the mean calibration error in `docs/dojo/history.jsonl`.
  With `--live` (auto-selected when no `--corpus` is given and a live corpus exists)
  it discovers the `.dojo/live-episodes` markers that `fak guard --dojo` /
  `fak serve --dojo` write. Those markers are start-only today, so the live path
  *surfaces what it found and reports what is missing to score it* rather than
  inventing a calibration.
- **board** folds the same run into a cross-lever leaderboard, worst-first.
- **ablate** replays a trace with the vDSO fast path ON vs OFF and scores the
  engine-call elision the fast path actually delivered against the claim — a
  WITNESSED episode from a real measured drop, not a projection.
- **list** shows the registered levers and the metrics each one scores.
- **post** posts a calibration rollup to the dojo channel (twin of `fak bench
  post`). `--rollup trend` reads the committed history ledger without a corpus scan;
  `--rollup latest` scores the corpus and posts the latest run. Safe by default
  (`--dry-run` renders without posting).

## The loop in action — from miscalibration to calibrated

This is the whole point, shown end to end on a real corpus of **~2,500 Claude Code
transcripts (~2,160 scorable sessions, ~57k scored boundaries)**.

### 1. First measurement — the naive theory

The first lever, **resume-posture**, wraps the existing `resume.Backtest` residual
and scores three metrics of the cache-posture projection. Run with the naive theory
(a cold re-prefill rewrites the *whole* resident; a resume re-prefills cold):

| metric | theory (claimed) | billed reality | verdict | grade |
|---|---|---|---|---|
| `posture_accuracy` | 1.000 | 0.983 | CALIBRATED | A |
| `cold_write_share` | 1.000 | 0.852 | **OVER_CLAIM** | B |
| `cross_session_warm_hit_rate` | 0.000 | 0.171 | **UNDER_CLAIM** | F |

Overall grade **D**, mean calib-err **0.389**, 1/3 calibrated. The bare "98.3%
accurate" headline hid two real, actionable miscalibrations: the projection
over-states cold cost by ~15% (it never rewrites the *whole* resident), and it
under-claims because ~17% of large first-turn resumes hit a still-warm
cross-session prefix — free savings the within-session model ignores.

### 2. The loop closing — feed the gap back, re-measure (#955)

The dojo's job is not to stop at "we found a gap." The `cold_write_share`
over-claim was fed back as a calibration: the projection's claim was moved from the
magic `1.0` to the observed `0.85`. Re-running on the same kind of corpus:

| metric | theory (claimed) | billed reality | verdict | grade |
|---|---|---|---|---|
| `posture_accuracy` | 1.000 | 0.983 | CALIBRATED | A |
| `cold_write_share` | **0.850** | 0.849 | **CALIBRATED** | A |
| `cross_session_warm_hit_rate` | 0.000 | 0.159 | UNDER_CLAIM | F |

Overall grade **C** (up from D), mean calib-err **0.340** (down from 0.389), 2/3
calibrated (up from 1/3), and the ledger now carries a live `trend: calibration
improved` row. That is the full cycle — predict -> measure -> calibrate ->
re-measure -> *improved* — closing on real billed data.

The `cross_session_warm_hit_rate` under-claim is left open on purpose: it is
bimodal across corpora (observed 0.00 -> 0.65 depending on workload), so its claim
is held at `0.0` as a floor and harvesting it is tracked separately rather than fit
to one corpus.

### Honest refusal: a lever that won't fabricate

The **compaction** lever (#953) scores cache-prefix preservation + token shed. On a
standard transcript corpus it has no paired ON/OFF billing, so it **refuses to
score** rather than invent a number, reporting the episode as `UNMEASURED` with the
missing witness named. That refusal is the gym working: an `UNMEASURED` beats a
fabricated `CALIBRATED`.

## The 5th rung: the self-improving RSI loop

The four rungs stop at *trend*. A human still has to read the `next_action` string
("recalibrate the over-claiming lever") and act. The RSI loop closes that fifth rung
autonomously, reusing the same non-forgeable keep-bit machine
(`internal/rsiloop` + `internal/shipgate`) that `fak guard-verdict-rsi` uses. Full
design: [dojo-rsi-loop.md](dojo-rsi-loop.md).

### The claim registry: one anchored number per cell

Every theory number lives in exactly one place: `internal/dojo/claims.go`. Each
`(lever, metric)` cell carries a single `claim(<float>, ...)` or `floor(<float>,
...)` literal plus its prose basis. The registry is the seam the loop rewrites: a
recalibration re-points one anchored literal and proves the re-measurement gained,
and its keep-bit demands that exactly one file and exactly one literal changed.

### The honesty target: `FoldCalibrable`, not raw `mean_calib_err`

The naive target would be the run's mean calibration error. That is a trap. Raw
`calib_err` is minimised at `claimed := realized`, so a constant rewrite is a free,
content-free "gain" — and for a **floor** like `false_warm_rate` (claim 0.0, the
lethal false-warm class), "recalibrating" the claim up to the measured rate would
*erase the very guard the dojo exists to defend*.

The fix is the `IntentionalFloor` bit on each claim plus a pure `FoldCalibrable`
that folds estimates by their `calib_err` but folds a floor by its **breach** (zero
while the floor holds, rising as it is breached). So re-pointing a genuine estimate
toward its corpus mean *lowers* `FoldCalibrable` and is eligible to keep, while
fitting a floor to its empirical rate *raises* it and reverts. Auto-erasure is
mechanically impossible, not merely discouraged.

### The proposer and the keep-bit

`fak dojo-rsi` folds a dojo report's scored episodes into worst-first recalibration
candidates and decides KEEP/REVERT. Every candidate targets exactly one cell:

- **RECALIBRATE** — re-point a genuine *estimate* at its corpus mean. This is the
  only self-keepable, mechanical move.
- **REPROJECT** — a claim pinned at perfection (1.0) that reality falls short of is
  a code-quality finding, not a recalibration. Routed to an agent arm with a
  declared path allow-list; never self-kept.
- **HARVEST** — a real under-claimed saving too big to auto-land. Routed to a
  goal-gated issue, never auto-landed.
- **ROUTE_FLOOR** — the target is an intentional floor. A breach is a belief-code
  bug to escalate, never a claim swap.
- **ROUTE_UNMEASURED** — the lever had no measured episode. Uncandidatable by
  construction (the honesty floor, surfaced rather than silently dropped).

A RECALIBRATE keeps only when every rung holds, and none of them is asserted by the
proposer:

> KEEP iff measured rows > 0 **and** the folded `FoldCalibrable` value strictly
> drops when the candidate claim is swapped in **and** the cell's per-lever sample
> meets the floor (`--min-sample`, default 3) **and** an external green witness is
> supplied. A floor target raises the fold (breach folded) and an unmeasured corpus
> is uncandidatable — both REVERT by construction.

`fak dojo-rsi run --check` re-derives those invariants from the iteration's own
fields, so a fabricated KEEP is caught at the gate.

### dojo-rsi subcommands

```sh
# 1. score the levers and emit the report the loop reads
fak dojo run --corpus ~/.claude/projects --json > report.json

# 2. fold the report's calibrable metric (estimates vs floor breach)
fak dojo-rsi fold --report report.json

# 3. the worst-first recalibration candidates, ranked novelty x value x staleness
fak dojo-rsi propose --report report.json

# 4. preview the exact one-literal change a RECALIBRATE would make — a DRY RUN
fak dojo-rsi rewrite --report report.json

# 5. self-score one tick (KEEP/REVERT) with an external witness; --check the keep-bit
fak dojo-rsi run --report report.json --witness '{"ok":true}' --check

# 6. self-paced loop over N ticks; stops "saturated" instead of thrashing a fresh cell
fak dojo-rsi loop --report report.json --ticks 3

# 7. fold the committed KEEP/REVERT journal for the CI feed
fak dojo-rsi trend
```

`fak dojo-rsi rewrite` is the safe, no-side-effect preview of the recalibration: it
renders the anchored one-line diff a RECALIBRATE candidate would write to
`claims.go`, without opening a worktree, writing the file, or committing. It refuses
a floor / REPROJECT / HARVEST cell with the reason it routes to the agent arm, so
the preview can never suggest erasing a guard.

The loop appends each tick to `docs/dojo/rsi-journal.jsonl` (separate from the
calibration-trend `history.jsonl`): one durable row per KEEP / REVERT / ESCALATE,
with the measured before/after fold and the routing context the CI feed trends
without re-running a corpus. The selector ranks cells by `novelty x value x
staleness` and marks a freshly-touched cell saturated so a loop stops rather than
thrashing the same row.

## Architecture

The scoring is a pure, stdlib-only package (`internal/dojo`): `Prediction` /
`Outcome` / `Episode`, `Score`, `FoldCalibrable`, `Fold` (-> the same
`schema/ok/verdict/finding/reason/next_action` envelope as the cadence report), the
claim registry (`claims.go`), and the durable JSONL ledger + `TrendVsLast`. The
corpus scan and the concrete levers live in the `cmd/fak/dojo.go` shell. A `Lever`
is the extension point:

```go
type Lever interface {
    Name() string
    Episodes(s Scenario) ([]ScoredInput, error) // one (prediction, outcome) per metric
}
```

The RSI loop is a second pure package (`internal/dojocal`): the proposer
(`ProposeRecals`), the self-scoring iteration (`RunIteration` / `CheckIteration`),
the cell-anchored rewrite (`RewriteClaim`), and the novelty-value-staleness selector
(`RankCandidates`). The worktree harness, which runs the real `go` witness, lives in
the `cmd/fak/dojorsi.go` shell, mirroring how the scoring keeps its I/O at the edge.

## Roadmap

The measure-and-trend gym (epic #951) is complete, and the self-improving 5th rung
(epic #1021) is most of the way there:

- [x] Pure `internal/dojo` loop (Score / Fold / ledger / trend), unit-tested, and
      the first lever (`resume-posture`) scored end to end on a real corpus.
- [x] #955 — feed the resume-posture miscalibrations back (cold-write share
      recalibrated to the observed value; the over-claim is now CALIBRATED).
- [x] #953 — compaction lever (refuses honestly without a paired ON/OFF corpus).
- [x] #954 — vCache-warmth lever (false-warm floor + warm recall, scored against
      billed cache_read).
- [x] `fak dojo board` — the cross-lever leaderboard, worst-first.
- [x] #957 — `fak dojo ablate`, the vDSO on-vs-off ablation as a WITNESSED episode.
- [x] `fak dojo post` — the calibration rollup to the dojo channel.
- [x] #956 (partial) — live mode discovers the `--dojo` session markers; they are
      start-only, so the path surfaces them and names what is missing to score them.
- [x] RSI **Phase 0** (#1022) — the claim registry, `IntentionalFloor`, and
      `FoldCalibrable`.
- [x] RSI **Phase 1** — the pure proposer + self-scoring loop
      (`fak dojo-rsi fold|propose|run`).
- [x] RSI **Phase 3** — the self-pacing selector, the KEEP/REVERT journal, and the
      `dojo-rsi-feed` CI rollup.
- [ ] RSI **Phase 2** (#1024) — the worktree arm: the cell-anchored `RewriteClaim`
      and the `fak dojo-rsi rewrite` dry-run preview have landed; the remaining
      wiring is the throwaway-worktree harness (a real `go` witness + the
      two-disjoint-shard `FoldCalibrable` gate) and the `--apply` by-path auto-land,
      gated behind the dos witness.
- [ ] More levers — KV-prefix reuse, dispatch dedup; the full live-episode writer
      (capturing per-turn billed usage on the `--dojo` markers, #1089/#1093).

## See also

- [dojo-rsi-loop.md](dojo-rsi-loop.md) — the full design of the self-improving 5th
  rung, including the anti-gaming defenses and the phased build plan.
- [`fak resume validate`](server-config.md) — the single-lever prototype the gym
  generalizes.
- [observability.md](observability.md) — the WITNESSED-vs-OBSERVED provenance
  boundary the dojo keeps on every number.
- `internal/dojo` — the pure scoring / fold / ledger / claim-registry package.
- `internal/dojocal` — the pure RSI proposer / self-scoring / rewrite package.

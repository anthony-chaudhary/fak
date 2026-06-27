---
title: "fak dojo: the prediction-vs-reality gym"
description: "A closed predict -> run -> measure -> eval -> calibrate loop for fak's token-saving levers: each optimization declares a theory, the scenario runs, billed reality is measured, the calibration error is scored and trended, and a miscalibration is fed back as a fix — demonstrated end to end on a real session corpus."
---

# The dojo: a prediction-vs-reality gym for fak's savers

*Who this is for: anyone adding or trusting a token-saving lever in fak (resume
cache posture, history compaction, vCache warmth, KV-prefix reuse, dispatch
dedup). You'll learn the one loop the dojo runs and how to read its scored
verdicts.*

fak has many levers that **predict** a saving — "this resume posture saves ~48k
tokens", "compaction preserves the cache prefix", "vCache stays warm". Each is a
*theory*. What fak lacked was a single place that closes the loop on those
theories: a **theory** is declared, **the actual thing runs**, we **look at
billed reality**, we **score the gap**, and — crucially — we **feed a
miscalibration back as a fix** and re-measure. Today `fak resume validate` does
the measure step for one lever and emits one number (a posture accuracy). The
dojo generalizes that into a **gym**: any lever registers, scores against the
provider's own billed reality, and accrues a calibration trend.

## The loop

```
  declare a prediction  ->  run the scenario  ->  measure ground truth  ->  score the gap  ->  calibrate  ->  trend
       (the theory)            (a workload)          (the provider's bill)     (verdict)      (feed back)     (ledger)
```

- **Prediction** — a lever's `Claimed` number for one metric, declared *before*
  reality is consulted, with the `Basis` that produced it.
- **Scenario** — the workload the lever runs against. Today that is an *offline*
  replay of a corpus of real Claude Code transcripts (deterministic,
  hardware-free, CI-able); the *live* mode (a running session feed) is the same
  shape with a different source.
- **Outcome** — the `Realized` number, lifted from the provider's own usage
  records. Every number is tagged `WITNESSED` (fak authored it) or `OBSERVED`
  (relayed from the provider — a bad value is not, by itself, a fak fault).
- **Episode** — one scored prediction-vs-reality: the residual, its normalized
  magnitude (`calib_err`), a letter grade, and a **verdict**:
  - `CALIBRATED` — reality met the claim (within 10%).
  - `OVER_CLAIM` — reality fell short of the claim (a theory promising more than
    billed reality delivered).
  - `UNDER_CLAIM` — reality beat the claim (a saving the model is not crediting).
  - `UNMEASURED` — no ground truth existed to score against.
- **Ledger + trend** — `fak dojo run --append-history` appends one row per tick
  to `docs/dojo/history.jsonl` and trends the **mean calibration error** against
  the last row, so the gym answers not just "what did this lever save" but "are
  our predictors getting *better calibrated* over time".

The fold is a **report contract, not a second quality gate**: it is `ACTION`
only when the run could not be measured, and `OK` otherwise — surfacing an
over-claim as an *advisory* line. This mirrors the cadence report's advisory
posture, so the gym measures without double-gating.

## The loop in action — from miscalibration to calibrated

This is the whole point, shown end to end on a real corpus of **~2,500 Claude
Code transcripts (~2,160 scorable sessions, ~57k scored boundaries)**.

### 1. First measurement — the naive theory

The first lever, **resume-posture**, wraps the existing `resume.Backtest`
residual and scores three metrics of the cache-posture projection. Run with the
naive theory (a cold re-prefill rewrites the *whole* resident; a resume re-prefills
cold):

| metric | theory (claimed) | billed reality | verdict | grade |
|---|---|---|---|---|
| `posture_accuracy` | 1.000 | 0.983 | CALIBRATED | A |
| `cold_write_share` | 1.000 | 0.852 | **OVER_CLAIM** | B |
| `cross_session_warm_hit_rate` | 0.000 | 0.171 | **UNDER_CLAIM** | F |

Overall grade **D**, mean calib-err **0.389**, 1/3 calibrated. The bare "98.3%
accurate" headline hid **two real, actionable miscalibrations**: the projection
over-states cold cost by ~15% (it never rewrites the *whole* resident), and it
under-claims because **~17% of large first-turn resumes hit a still-warm
cross-session prefix** — free savings the within-session model ignores.

### 2. The loop closing — feed the gap back, re-measure (#955)

The dojo's job is not to stop at "we found a gap." The `cold_write_share`
over-claim was fed back as a calibration: the projection's claim was moved from
the magic `1.0` to the observed `0.85`. Re-running on the same kind of corpus:

| metric | theory (claimed) | billed reality | verdict | grade |
|---|---|---|---|---|
| `posture_accuracy` | 1.000 | 0.983 | CALIBRATED | A |
| `cold_write_share` | **0.850** | 0.849 | **CALIBRATED** | A |
| `cross_session_warm_hit_rate` | 0.000 | 0.159 | UNDER_CLAIM | F |

Overall grade **C** (up from D), mean calib-err **0.340** (down from 0.389),
**2/3 calibrated** (up from 1/3), and the ledger now carries a live
`trend: calibration improved` row. That is the full cycle — predict -> measure ->
**calibrate** -> re-measure -> *improved* — closing on real billed data.

The `cross_session_warm_hit_rate` under-claim is left open on purpose: it is
**bimodal across corpora** (observed 0.00 -> 0.65 depending on workload), so its
claim is held at `0.0` and harvesting it is tracked separately rather than fit to
one corpus.

### Honest refusal — a lever that won't fabricate

A second lever, **compaction** (#953), scores cache-prefix preservation + token
shed. On a standard transcript corpus it has no paired ON/OFF billing, so it
**refuses to score** rather than invent a number:

```
fak dojo run: lever "compaction" on "projects": compaction lever requires a paired
ON/OFF compaction corpus with shed_tokens and provider billing metrics; not
available from standard transcripts (see #953)
```

That refusal is the gym working: an `UNMEASURED`/error beats a fabricated
`CALIBRATED`.

## Architecture

The scoring is a pure, stdlib-only package (`internal/dojo`): `Prediction` /
`Outcome` / `Episode`, `Score`, `Fold` (-> the same
`schema/ok/verdict/finding/reason/next_action` envelope as the cadence report),
and the durable JSONL ledger + `TrendVsLast`. The corpus scan and the concrete
levers live in the `cmd/fak/dojo.go` shell. A `Lever` is the extension point:

```go
type Lever interface {
    Name() string
    Episodes(s Scenario) ([]ScoredInput, error) // one (prediction, outcome) per metric
}
```

## Usage

```sh
# score every registered lever against your real session history, and trend it
fak dojo run --corpus ~/.claude/projects --append-history

# machine-readable, or an advisory gate (non-zero only if nothing could be measured)
fak dojo run --corpus ~/.claude/projects --json
fak dojo run --corpus ~/.claude/projects --check

# what levers are registered, and what each one scores
fak dojo list
```

## Roadmap (epic #951)

- [x] Pure `internal/dojo` loop (Score/Fold/ledger/trend), unit-tested, and the
      first lever (`resume-posture`) scored end to end on a real corpus.
- [x] #955 — feed the resume-posture miscalibrations back (cold-write share
      recalibrated to the observed value; the over-claim is now CALIBRATED).
- [x] #953 — compaction lever (refuses honestly without a paired ON/OFF corpus).
- [ ] More levers — vCache warmth (#954), KV-prefix reuse, dispatch dedup.
- [ ] #956 — live mode: every real `fak guard` / `fak serve` session becomes an
      episode, so the dojo records predicted-vs-actual continuously.
- [ ] #957 — lever-on-vs-off ablation over a fixed `WorkloadHash` + a cross-lever
      calibration leaderboard.

## See also

- [`fak resume validate`](server-config.md) — the single-lever prototype the gym generalizes.
- [observability.md](observability.md) — the WITNESSED-vs-OBSERVED provenance boundary the dojo keeps on every number.
- `internal/dojo` — the pure scoring/fold/ledger package.

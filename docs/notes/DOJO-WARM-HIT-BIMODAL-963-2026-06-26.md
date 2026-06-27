# Dojo `cross_session_warm_hit_rate` is bimodal by workload (#963)

Date: 2026-06-26

Resolution note for [#963](https://github.com/anthony-chaudhary/fak/issues/963).
Records the finding, the increment that shipped, and the follow-on that is
deliberately deferred — so the calibration decision is durable and not relearned
the next time someone reads a grade-F dojo row for this metric.

## What the gym found

`fak dojo` ran the `resume-posture` lever over three real Claude Code transcript
corpora on this box. The `cross_session_warm_hit_rate` claim — a fixed `0.17` at
`cmd/fak/dojo.go` — was miscalibrated in both directions, because the realized
rate is bimodal by workload:

| corpus | transcripts | warm-hit realized | sample | verdict vs 0.17 |
|---|---|---|---|---|
| `C--work-fak` | 875 | 0.65 | 118 | UNDER_CLAIM |
| `C--work-job` | 1588 | 0.00 | 132 | OVER_CLAIM |
| `C--work-Benchmark` | 12 | 0.00 | 10 | OVER_CLAIM |

The two sibling claims calibrated fine on the same corpora
(`cold_write_share` 0.85 → 0.79/0.88/0.98; `posture_accuracy` 1.0 → 0.978), so
this is not a measurement artifact — the corpus is sound. The warm-hit rate is a
property of the workload, not a universal constant: a long-running dev tree
resumes a still-warm cross-session prefix about 65% of the time, while short
independent runs (job dispatch, one-off benchmarks) essentially never do.

## What shipped

The claim was recalibrated from `0.17` to `0.0` (the conservative lower bound) on
`main`:

- `f306cfb9`, `265ce039` — claim `0.17` → `0.0`; the theory string was rewritten
  to name the bimodal `0.00→0.65` range instead of a single point estimate.
- `76bc0806` — removed a duplicate `registerDojoLevers` that a double-commit had
  left behind. The duplicate redeclared the function and broke `go build`; this
  restored a green tree.

Rationale for `0.0`: it never over-claims. Over-claiming is the harmful
direction — it credits resume savings that batch corpora never realize — while
under-claiming on a warm dev tree is a safe, honest under-promise. With the claim
at `0.0` the gym reports CALIBRATED on the low-warmth corpora and an honest
UNDER_CLAIM (graded F by magnitude) on the high-warmth dev tree.

## Why a single point is still the wrong shape (deferred follow-on)

Setting the claim to `0.0` trades the over-claim on batch corpora for a large
under-claim on the dev tree; it does not make the metric workload-invariant,
because no single point can be. A grade-F row on `C--work-fak` for this metric is
therefore expected and benign: it is a deliberate conservative under-claim, not a
regression.

The issue's preferred resolution is option 2: stop scoring
`cross_session_warm_hit_rate` as a CLAIMED point and report it as an OBSERVED
spread (min / median / max across corpora) that does not fold into the
calibration grade. That needs a framework change in `internal/dojo` — a
report-only verdict that `Score` and `Fold` exclude from `MeanCalibErr` and the
overall grade — plus the wiring in `cmd/fak/dojo.go` to mark this one prediction
report-only. It is **not yet done**; the `0.0` recalibration is the shipped
increment, and the report-only treatment is the named next step.

## Repro

```
fak dojo run --corpus ~/.claude/projects/C--work-fak       --max-files 200 --json | jq '.episodes[]|select(.metric=="cross_session_warm_hit_rate")'
fak dojo run --corpus ~/.claude/projects/C--work-job       --max-files 150 --json | jq '.episodes[]|select(.metric=="cross_session_warm_hit_rate")'
fak dojo run --corpus ~/.claude/projects/C--work-Benchmark --max-files 150 --json | jq '.episodes[]|select(.metric=="cross_session_warm_hit_rate")'
```

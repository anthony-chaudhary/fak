---
title: "The prediction-calibration contract — back-test any projection against witnessed reality, in a closed verdict vocabulary"
description: "The domain-free contract that lifts fak's prediction-vs-reality calibration discipline (internal/dojo Score, internal/resume Backtest) into a primitive any projection-emitting module can call, and the `dos calibrate` verb implements. A calibration takes a declared Prediction, a witnessed Measurement the predictor did not author, and an eval band, and returns one verdict from a CLOSED set — CALIBRATED | OVER_CLAIM | UNDER_CLAIM | INSUFFICIENT. Three properties make it portable and honest: (1) the verdict vocabulary is a closed, validatable enum — an out-of-set verdict is rejected at the boundary (the UNCLASSIFIED-refuse form), never coerced to a pass; (2) an OVER_CLAIM (the theory promised more than reality delivered) is a first-class verdict surfaced never hidden, and the conservative-bias direction is explicit and direction-aware (lower_is_better flips which side is the over-claim); (3) it is fail-closed — a thin or absent corpus scores INSUFFICIENT, reported and never rounded up to CALIBRATED. The honest fence: this lifts the calibration DECISION, which is SHIPPED and offline-witnessed by internal/dojo and internal/resume; the portable `dos calibrate` verb is not yet."
---

# The prediction-calibration contract

Every module that emits a projection — a router's cache-hit-rate estimate, a
planner's residency forecast, a resume cache's cold/warm posture call, a dojo
lever's claimed saving — eventually has to answer one question: *did reality match
the projection, and if not, which way did it miss?* fak already practices the
discipline that answers it well, in two places:

- [`internal/dojo`](../fak/dojo.md) — the gym's `predict → run → measure → eval →
  calibrate` loop: a `Prediction` (the theory a lever declares for a metric, before
  billed reality is consulted) is scored against a measured `Outcome` lifted from the
  provider's own usage records, yielding a `Verdict` from a closed set.
- [`internal/resume`](https://github.com/anthony-chaudhary/fak/blob/main/internal/resume/backtest.go) `Backtest` — the resume-cache
  projection back-tested against the provider's own per-turn `cache_read` /
  `cache_creation` records: the projection is the model, the usage records are the
  ground truth, and the report is the residual (an `Accuracy`, plus the *directional*
  miss — `ProjColdObsWarm` vs `ProjWarmObsCold`).

The rule both keep is one reusable correctness discipline: **back-test a projection
against telemetry the predictor did not author before defaulting it on, and never
silently over-claim.** The gap this page closes is that the rule is locked inside
those packages — there is no domain-free primitive another agent fleet can call. This
page is that primitive, written as an engine-free contract. It is `G5` of the
[agent-programming-grammar epic](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md)
(#1213), the calibration sibling of [net-true-value](net-true-value.md) (a value
claim), [the observer-effect contract](observer-effect.md) (a cost number), and
[the agent-routing schema](agent-routing-schema.md) (a routing decision). The verb
that walks this contract over a `(prediction, measurement, eval-fn)` tuple is
`dos calibrate`; its home is the installed DOS package, and it is **not yet** (see the
[honest fences](#honest-fences)). fak's `internal/dojo` and `internal/resume` are the
reference implementations — the offline witnesses that the contract below is real, not
a wish.

## The contract

A calibration is three inputs and one verdict. None of the inputs mentions a fak
package — the decision is the *shape*, taken with no model in the loop.

### Prediction — the declared theory, before reality is consulted

```
Prediction {
  metric           : string    // what is being predicted (a hit rate, a residency, a saving)
  claimed          : number    // the value the theory asserts
  unit             : string    // the metric's unit (fraction, tokens, seconds, USD, …)
  basis            : string    // the provenance of the claim (how it was derived) — review context
  lower_is_better  : bool       // the metric's DIRECTION: false (default) = higher is better
  intentional_floor: bool       // the claim is a GUARD reality must not breach, not an estimate
}
```

`lower_is_better` names the metric's polarity so the verdict can tell the *worse*
side of the claim from the better side — the field that makes the conservative-bias
direction explicit (see [the direction rule](#the-conservative-bias-direction)).
`intentional_floor` marks a claim that is a guard the predictor defends (a
`false_warm_rate` that must stay `0.0`), not a best-guess central tendency — a floor
is scored by its *breach*, never recalibrated up to its empirical rate.

### Measurement — the witnessed ground truth the predictor did not author

```
Measurement {
  realized   : number       // the value reality delivered
  provenance : Provenance    // CLOSED: "WITNESSED" | "OBSERVED" — whose number it is
  source     : string        // where the ground truth came from (a usage record, a meter)
  measured   : bool          // false when no ground truth existed (scores INSUFFICIENT)
  sample     : int           // how many boundaries/turns stand behind `realized`
}
```

The measurement is **evidence-bound**: `realized` is ground truth lifted from a
source the predictor did not write — the provider's own billed usage records, a meter
fak controls. `provenance` keeps every number honest about whose it is — a
`WITNESSED` value is one fak authored and controls; an `OBSERVED` value is relayed
from an upstream party (the model provider) and fak does not control it. A calibration
over a measurement the predictor *authored* is not a calibration; it is the
projection grading itself. `measured: false` (or a `sample` below the band's floor) is
the [fail-closed `INSUFFICIENT`](#the-fail-closed-floor) case, never a scored zero.

### EvalBand — the closed tolerance the verdict reads

```
EvalBand {
  calibrated_max : number   // a normalized residual at or under this is CALIBRATED (default 0.10)
  min_sample     : int       // fewer measured boundaries than this scores INSUFFICIENT (fail-closed)
}
```

The band is the *eval-fn* made data: a residual within `calibrated_max` of the claim
is calibrated; a corpus thinner than `min_sample` is too thin to score and is
reported `INSUFFICIENT` rather than rounded up. The default `calibrated_max` of `0.10`
(within 10% of the claim) is the dojo's conservative
[`DefaultCalibBand`](https://github.com/anthony-chaudhary/fak/blob/main/internal/dojo/dojo.go).

### Verdict — the closed, validatable output

Calibrating a `Prediction` against a `Measurement` under an `EvalBand` yields exactly
one verdict from a **closed set**. The decision is a finite switch — an out-of-set
token cannot be produced, and a validator decides membership without a live service:

| verdict | when | meaning |
|---|---|---|
| `CALIBRATED` | the normalized residual is within `calibrated_max` | reality met the claim within tolerance |
| `OVER_CLAIM` | reality landed on the **worse-than-claim** side, beyond the band | the theory promised more than reality delivered — the harmful direction |
| `UNDER_CLAIM` | reality landed on the **better-than-claim** side, beyond the band | the theory under-promised; a margin reality beat (free headroom uncredited) |
| `INSUFFICIENT` | no ground truth, or a corpus thinner than `min_sample` | there is no admissible evidence to score the claim — reported, never rounded up |

The normalized residual is the relative error `|realized − claimed| / |claimed|`,
capped so a claim near zero that reality refutes cannot dominate with an unbounded
ratio; a claim of "nothing" is scored by the absolute residual instead (the exact
[`calibErr`](https://github.com/anthony-chaudhary/fak/blob/main/internal/dojo/dojo.go) the reference implementation uses). The
[machine-checkable schema](prediction-calibration.json) publishes this verdict set as
a closed JSON Schema `enum` (Draft 2020-12), so any orchestrator validates a verdict
with an off-the-shelf validator, **no fak engine present**.

### The conservative-bias direction

`OVER_CLAIM` and `UNDER_CLAIM` are not symmetric — which side is the *over*-claim is
the load-bearing, direction-aware rule, and it is explicit by design. For a
higher-is-better metric (the default — a hit rate, an accuracy, a saving), `realized`
**below** the claim is the worse side: billed reality delivered less than the theory
promised, so it scores `OVER_CLAIM`. For a `lower_is_better` metric (a
`false_warm_rate`, a latency), the polarity flips: `realized` **above** the claim is
the worse side. This is the exact
[`worseThanClaim`](https://github.com/anthony-chaudhary/fak/blob/main/internal/dojo/dojo.go) rule the reference implementation
applies, surfaced here as a named contract so the conservative bias — *over-claiming
is the direction that must never hide* — is part of the vocabulary, not an
implementation detail. An `intentional_floor` claim is scored on its breach (the worse
side only), so "recalibrating a floor up to its empirical rate" can never erase the
guard.

## The three properties (the acceptance, made checkable)

This contract is portable and honest because it holds three properties an external
caller can check without fak's engine — the issue's three acceptance criteria, each
bound to a witness:

1. **The verdict vocabulary is closed and validatable.** The four verdicts above are
   the whole set; an out-of-set token is `UNCLASSIFIED` and refused, never coerced to
   a pass — the same fail-closed posture
   [`dos check-reason`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md) keeps for a refusal token. Membership is
   decided by a finite switch, not a lookup against a live service. The
   [`prediction-calibration.json`](prediction-calibration.json) schema publishes the
   `enum` so the claim is machine-checkable, not asserted; the reference
   implementation's verdict constants
   ([`internal/dojo/dojo.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/dojo/dojo.go)) are the same closed set,
   offline-witnessed by `go test ./internal/dojo/`.
2. **An over-claim is surfaced, never hidden; the conservative-bias direction is
   explicit.** `OVER_CLAIM` is a first-class verdict with its own decision arm, and
   the [direction rule](#the-conservative-bias-direction) names *which* side is the
   over-claim for each metric polarity. The dojo's fold surfaces every over-claim as
   an advisory line and its `FoldCalibrable` folds a floor by its breach, so a loop
   optimising the calibration can never "gain" by recalibrating an over-claim away —
   the [warm-hit bimodal lesson](../notes/DOJO-WARM-HIT-BIMODAL-963-2026-06-26.md)
   is the worked case: a fixed `0.17` claim that real corpora refuted scored
   `OVER_CLAIM`, and the fix was to recalibrate the *claim* down, never to hide the
   verdict.
3. **It is generic enough for a non-fak projection.** Nothing in the three inputs is
   fak-shaped. A router's hit-rate estimate calibrates as
   `Prediction{metric:"cache_hit_rate", claimed:0.94}` against a
   `Measurement{realized:0.91, source:"router_telemetry"}`; a planner's residency
   forecast as `Prediction{metric:"resident_tokens", claimed:48000}` against the
   meter's witnessed count. The [worked round-trip below](#the-round-trip-as-data) is
   on disk as fixtures, validated against the schema with a stock validator.

## The fail-closed floor

The single rule that keeps a calibration honest under thin evidence: a corpus that
cannot conclusively score the claim is **reported `INSUFFICIENT`, never rounded up to
`CALIBRATED`**. Two cases collapse to it — `measured: false` (no ground truth existed,
the dojo's `UNMEASURED`) and a `sample` below `min_sample` (the corpus is too thin to
trust a central tendency, the discipline
[`internal/resume`](https://github.com/anthony-chaudhary/fak/blob/main/internal/resume/backtest.go) keeps by *excluding* an
ambiguous partial re-serve from its accuracy denominator rather than scoring it on a
guess, and [`internal/dojocal`](https://github.com/anthony-chaudhary/fak/blob/main/internal/dojocal/dojocal.go) keeps with a
`DefaultMinSample` floor before a recalibration is trusted). The
[warm-hit bimodal note](../notes/DOJO-WARM-HIT-BIMODAL-963-2026-06-26.md) is the
canonical reason this rule matters: a claim recalibrated `0.17 → 0.0` to the
*conservative lower bound* — because over-claiming is the harmful direction, so the
floor is the value that never over-claims. `INSUFFICIENT` is the calibration analogue
of the verification ladder's `INDETERMINATE`: absence of an affirmative score escalates
to "not yet measured," it does not pass.

## The round-trip, as data

The contract's whole claim is that `(prediction, measurement, eval-fn) → verdict` is
data, not narration. The fixtures under [`fixtures/`](fixtures/) are the on-disk
witness — each validates against [`prediction-calibration.json`](prediction-calibration.json)
with a stock Draft 2020-12 validator, no fak engine present:

- [`calibration-over-claim.json`](fixtures/calibration-over-claim.json) — a hit-rate
  projection (`claimed 0.94`) the telemetry refuted (`realized 0.91`) over a real
  sample: verdict `OVER_CLAIM`, the harmful-direction case surfaced.
- [`calibration-calibrated.json`](fixtures/calibration-calibrated.json) — a residency
  forecast reality met within the band: verdict `CALIBRATED`.
- [`calibration-insufficient.json`](fixtures/calibration-insufficient.json) — a claim
  with no ground truth (`measured: false`): verdict `INSUFFICIENT`, fail-closed and
  not rounded up.

A reviewer reads the three inputs and the verdict; no model and no fak engine are
needed to confirm the calibration. fak's `internal/dojo` scores the same prediction
against the same measurement and produces the same verdict — offline, witnessed by
`go test ./internal/dojo/`, the reference implementation's proof that the contract
holds.

## Reference implementation and witness

| Contract element | Reference stick | Status |
|---|---|---|
| Prediction / Measurement / EvalBand | `dojo.Prediction`, `dojo.Outcome`, `dojo.CalibBand` (`internal/dojo/dojo.go`) | [SHIPPED] |
| Closed Verdict vocabulary | `dojo.VerdictCalibrated` / `VerdictOverClaim` / `VerdictUnderClaim` / `VerdictUnmeasured` constants + the `Score` switch | [SHIPPED] |
| Direction-aware conservative bias | `dojo.worseThanClaim` (polarity rule) + `FloorRespectErr` (floor breach) | [SHIPPED] |
| Fail-closed INSUFFICIENT | `Score` → `UNMEASURED` on `!Measured`; `resume.Backtest` ambiguous-exclusion; `dojocal.DefaultMinSample` | [SHIPPED] |
| Projection back-tested vs witnessed telemetry | `resume.Backtest` (projection vs provider's own `cache_read` records) → `BacktestReport` | [SHIPPED] |
| Offline determinism witness | `go test ./internal/dojo/`, `go test ./internal/resume/` (no model in the loop) | [SHIPPED] |
| Machine-checkable verdict enum | [`prediction-calibration.json`](prediction-calibration.json) (Draft 2020-12) | [SHIPPED] |
| Portable `dos calibrate` verb | the installed DOS package | **not yet** |

## Honest fences

- **This lifts the calibration decision, which is shipped; it does not ship a new
  verb.** The `(prediction, measurement, band) → verdict` decision is [SHIPPED] and
  offline-witnessed in `internal/dojo` (the gym's per-episode score) and
  `internal/resume` (the resume-cache back-test). This page is the *contract* an
  external projection-emitter authors against; the portable `dos calibrate` verb that
  walks it lives in the installed DOS package and is a named follow-on, **not yet**.
- **The reference name for `INSUFFICIENT` is `UNMEASURED`.** The dojo's closed set
  names the fail-closed verdict `UNMEASURED` (no ground truth to score against); this
  contract's canonical name is `INSUFFICIENT` because it also covers the *thin-corpus*
  case (a `sample` below `min_sample`), which the reference sticks keep by exclusion
  (`resume.Backtest` drops an ambiguous pair from the accuracy denominator) and by a
  sample floor (`dojocal.DefaultMinSample`) rather than by a fourth verdict. The two
  names denote the same fail-closed outcome — no admissible evidence, so no score.
- **A calibration grades the projection, not the system.** A verdict says whether the
  *claim* matched reality, never whether the underlying system is good — an
  `UNDER_CLAIM` can be a fine system whose theory was pessimistic. The dojo is a
  measurement mirror, not a second quality gate (`CheckGate` fails only on an
  unmeasured run, never on an over-claim); this contract keeps that posture.
- **Evidence-bound is a precondition, not an output.** The contract assumes the
  `Measurement` is ground truth the predictor did not author; it cannot itself prove
  the provenance of a number handed to it. The `provenance` field records the claim;
  the caller (and a reviewer) is responsible for the measurement's independence, the
  same way `dos verify` trusts git evidence but not a worker's self-report.

## Cross-references

- [`prediction-calibration.json`](prediction-calibration.json) — the machine-checkable JSON Schema (Draft 2020-12) for this contract: validate a calibration with any off-the-shelf validator, no fak engine present.
- [The dojo gym](../fak/dojo.md) · [the dojo-RSI loop](../fak/dojo-rsi-loop.md) — fak's reference implementation of the calibration loop, and the self-improving recalibration the floor protects.
- [Warm-hit bimodal lesson (#963)](../notes/DOJO-WARM-HIT-BIMODAL-963-2026-06-26.md) — the worked case for the fail-closed floor: a `0.17 → 0.0` recalibration to the conservative lower bound, because over-claiming is the harmful direction.
- [The agent-programming grammar](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md) — the epic this contract is `G5` of, and the recipe every lift keeps (closed vocabulary, evidence-bound, fail-closed, data-not-code, pays on both lenses).
- [Net-true-value](net-true-value.md) · [The observer-effect contract](observer-effect.md) · [The agent-routing schema](agent-routing-schema.md) · [The support-maturity honesty fence](support-maturity-honesty-fence.md) — the sibling standards in `docs/standards/`.

package vcachecal

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// estimator.go is the vCache warmth-belief estimator — issue #716 scope 1 and §7. On an
// implicit provider you cannot ask "is X warm?"; you only learn warmth AFTER spending a
// real request (cache_read). The estimator is therefore an OPEN-LOOP estimator WITH
// FEEDBACK, not a known-state machine: it reuses a cachemeta.Lifecycle at TierProvider
// as the clock substrate and reinterprets its transitions as BELIEFS.
//
//   - Advance decays belief on the clock (resident → expiring → expired), since we
//     cannot see eviction.
//   - A real call that read cache revives belief (Touch) and resets the TTL clock — the
//     only ground truth we get.
//   - A believed-warm call that reads cache_read=0 demotes to cold AT ONCE and records
//     the divergence (Rule A1: demote + alarm + byte-diff).
//
// Warmth is a BELIEF, never a boolean (Law A): correctness never depends on it. A wrong
// belief can only ever cost money, never corrupt a result.

// BeliefPolicy binds cachemeta's tiered-TTL lifecycle to the provider plane. The
// provider TTL is the §7 constant the probe harness fits (T); the grace window is the
// expiring→expired slack a belief decays through before it goes cold.
type BeliefPolicy struct {
	ProviderTTLMillis int64
	GraceMillis       int64
}

// lifecyclePolicy lowers a BeliefPolicy onto the cachemeta.LifecyclePolicy the unchanged
// Lifecycle.Advance consumes: TierTTL[TierProvider] = the provider TTL, and the grace
// window. This is the SOLE coupling to cachemeta's clock — the estimator adds no clock
// of its own, exactly as the issue demands ("Each manifest entry carries a Lifecycle at
// TierProvider with TierTTL[TierProvider] = provider TTL").
func (p BeliefPolicy) lifecyclePolicy() cachemeta.LifecyclePolicy {
	return cachemeta.LifecyclePolicy{
		TierTTL:     cachemeta.TierTTL{cachemeta.TierProvider: p.ProviderTTLMillis},
		GraceMillis: p.GraceMillis,
	}
}

// DefaultBeliefPolicy is the §7 starting hypothesis (a 5-minute provider TTL and a 30s
// grace window) BEFORE the probe harness measures the real T. Calibrate-don't-assume:
// the live loop replaces this with a measured Calibration.TTLMillis.
func DefaultBeliefPolicy() BeliefPolicy {
	return BeliefPolicy{ProviderTTLMillis: 5 * 60 * 1000, GraceMillis: 30 * 1000}
}

// FromCalibration builds the belief policy from a fitted Calibration so the estimator
// decays on the MEASURED TTL, not the hypothesis. A calibration still on the hypothesis
// (unmeasured) decays on that hypothesis — honest by construction.
func FromCalibration(c Calibration) BeliefPolicy {
	return BeliefPolicy{ProviderTTLMillis: c.TTLMillis, GraceMillis: DefaultBeliefPolicy().GraceMillis}
}

// Belief is one manifest entry's warmth belief: a cachemeta.Lifecycle at TierProvider,
// reused UNCHANGED, plus the §7 open-loop-with-feedback accounting a known-state
// machine does not carry (the divergence counters the prediction-error report folds).
type Belief struct {
	Lifecycle cachemeta.Lifecycle
	// FalseWarm counts believed-warm calls that read cache_read=0 (the lethal "manifest
	// says HIT, provider says MISS" case — Rule A1). It is the false-warm signal.
	FalseWarm int
	// FalseCold counts believed-cold calls that read cache_read>0 — a warming chance the
	// belief missed (§11.5 "invisible regret", made VISIBLE here).
	FalseCold int
	// Confirmed counts real cache reads (cache_read>0) — the only ground-truth signal.
	Confirmed int
}

// NewBelief starts a prefix WARM at TierProvider at nowMillis. A freshly written
// provider prefix is believed warm until the TTL clock decays it; the only ground truth
// that revives it thereafter is a real cache read.
func NewBelief(nowMillis int64) Belief {
	return Belief{Lifecycle: cachemeta.Lifecycle{
		State:             cachemeta.StateResident,
		Tier:              cachemeta.TierProvider,
		AdmittedAtMillis:  nowMillis,
		EnteredTierMillis: nowMillis,
		StateSinceMillis:  nowMillis,
		LastAccessMillis:  nowMillis,
	}}
}

// Advance decays belief on the clock via the UNCHANGED cachemeta.Lifecycle.Advance:
// Resident → Expiring when the provider TTL elapses (warm → cooling), and Expiring →
// Expired after the grace window (cooling → cold). §7: "Advance decays belief on the
// clock, since we can't see eviction."
func (b Belief) Advance(policy BeliefPolicy, nowMillis int64) Belief {
	lc, _ := b.Lifecycle.Advance(policy.lifecyclePolicy(), nowMillis)
	b.Lifecycle = lc
	return b
}

// IsWarm reports the belief's current hit PREDICTION: Resident or Expiring — a prefix
// inside its TTL or grace window. Expired/Evicted predict a miss. This is the per-call
// prediction the estimator makes BEFORE the provider answers (§7 / scope 4).
func (b Belief) IsWarm() bool {
	switch b.Lifecycle.State {
	case cachemeta.StateResident, cachemeta.StateExpiring:
		return true
	default:
		return false
	}
}

// PredictionClass is the closed four-valued reconcile of a predicted belief against the
// real cache_read the provider returned.
type PredictionClass string

const (
	// TrueWarm: predicted warm, read > 0 — a confirmed hit.
	TrueWarm PredictionClass = "true_warm"
	// FalseWarm: predicted warm, read = 0 — the lethal one (Rule A1 demote + alarm).
	FalseWarm PredictionClass = "false_warm"
	// TrueCold: predicted cold, read = 0 — a correct miss.
	TrueCold PredictionClass = "true_cold"
	// FalseCold: predicted cold, read > 0 — a warming chance the belief missed.
	FalseCold PredictionClass = "false_cold"
)

// PredictionOutcome is one reconciled prediction/observation pair.
type PredictionOutcome struct {
	PredictedWarm bool
	ActualWarm    bool
}

// Class folds the outcome into the closed prediction-error vocabulary.
func (o PredictionOutcome) Class() PredictionClass {
	switch {
	case o.PredictedWarm && o.ActualWarm:
		return TrueWarm
	case o.PredictedWarm && !o.ActualWarm:
		return FalseWarm
	case !o.PredictedWarm && o.ActualWarm:
		return FalseCold
	default:
		return TrueCold
	}
}

// Observe reconciles a predicted belief against the real cache_read counter the
// provider returned for this call, and returns the updated belief plus the outcome the
// PredictionError folds. Per §7 / Rule A1:
//   - cache_read > 0: the read CONFIRMS warmth → revive to Resident and RESET the TTL
//     clock (a confirmed read is the only ground truth; it restarts the warm window).
//     A belief that was COLD but read cache is a false-cold (a missed warming).
//   - cache_read == 0 on a BELIEVED-WARM call: DEMOTE to cold immediately and record a
//     false-warm divergence.
//   - cache_read == 0 on a believed-cold call: a true cold; no state change.
//
// The TTL-clock reset on a confirmed read is what makes the estimator open-loop-with-
// feedback rather than a known-state machine: a continuously-read prefix never decays,
// and decay only bites idle prefixes — exactly §7's "Touch revives and resets the TTL
// clock."
func (b Belief) Observe(policy BeliefPolicy, nowMillis int64, cacheReadTokens int64) (Belief, PredictionOutcome) {
	predicted := b.IsWarm()
	outcome := PredictionOutcome{PredictedWarm: predicted, ActualWarm: cacheReadTokens > 0}
	if cacheReadTokens > 0 {
		// Touch bumps Accesses/LastAccess and revives an Expiring entry; for an Expired
		// belief we force-resurrect (Touch alone does not revive Expired). Either way a
		// confirmed read restarts the TTL clock.
		b.Lifecycle = b.Lifecycle.Touch(nowMillis)
		b.Lifecycle.State = cachemeta.StateResident
		b.Lifecycle.Tier = cachemeta.TierProvider
		b.Lifecycle.StateSinceMillis = nowMillis
		b.Lifecycle.EnteredTierMillis = nowMillis
		b.Confirmed++
		if !predicted {
			b.FalseCold++
		}
		return b, outcome
	}
	if predicted {
		// Believed warm but the provider did not serve from cache: demote to cold AT
		// ONCE and record the divergence (Rule A1).
		b.Lifecycle.State = cachemeta.StateExpired
		b.Lifecycle.StateSinceMillis = nowMillis
		b.FalseWarm++
	}
	return b, outcome
}

// PredictionError accumulates the false-warm / false-cold rates over a workload. Issue
// #716 acceptance: "false-warm and false-cold rates are reported, NOT assumed zero."
type PredictionError struct {
	Total                                    int
	TrueWarm, FalseWarm, TrueCold, FalseCold int
}

// Add folds one observed outcome into the report.
func (e *PredictionError) Add(o PredictionOutcome) {
	e.Total++
	switch o.Class() {
	case TrueWarm:
		e.TrueWarm++
	case FalseWarm:
		e.FalseWarm++
	case TrueCold:
		e.TrueCold++
	case FalseCold:
		e.FalseCold++
	}
}

// FalseWarmRate is false-warm / predicted-warm: of the calls we PREDICTED warm, the
// fraction that actually missed. The lethality signal (Rule A1). Zero when nothing was
// predicted warm.
func (e PredictionError) FalseWarmRate() float64 {
	predicted := e.TrueWarm + e.FalseWarm
	if predicted == 0 {
		return 0
	}
	return float64(e.FalseWarm) / float64(predicted)
}

// FalseColdRate is false-cold / predicted-cold: of the calls we predicted cold, the
// fraction that were actually warm — the §11.5 "invisible regret" made visible. Zero
// when nothing was predicted cold.
func (e PredictionError) FalseColdRate() float64 {
	predicted := e.TrueCold + e.FalseCold
	if predicted == 0 {
		return 0
	}
	return float64(e.FalseCold) / float64(predicted)
}

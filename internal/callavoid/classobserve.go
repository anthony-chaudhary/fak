package callavoid

// classobserve.go — the WITNESSED-PRODUCER half of #819: fold a tool class's RAW observed
// per-class statistics into the (m, v, c) calibration GateClass proves, so the admission
// verdict is read off MEASURED inputs, not a hand-asserted guess.
//
// Why this leaf exists. classgate.go's GateClass already runs the break-even gate over a
// ClassMemoInput, but that input's mutation rate / validate cost / capture cost arrive
// pre-computed — a caller could hand-assert the very (m, v, c) that makes a class admit.
// Issue #819 acceptance (3) forbids that: "Calibration (m, v, c) sourced from OBSERVATION
// (internal/vcachecal-style), not hand-asserted — ProveMemo's PROVEN verdict is only as
// honest as its measured inputs." This file supplies the missing measured ground: a
// ClassObservation of the raw counts the seam already has — reuse attempts, how many were
// invalidated by an intervening write (the world-version churn for that class), and the
// cost samples — and FoldClassObservation derives (m, v, c) from them the way
// internal/vcachecal fits its constants from replay samples instead of assuming them.
//
// The relationship to classgate.go mirrors coverage.go's relationship to witness.go: the
// consumer (GateClass) decides from a calibration; this producer GROUNDS that calibration
// in observation and ABSTAINS — declines without a proof — when there is nothing measured
// to ground it, so an un-observed class can never be admitted on a fabricated input. That
// is the witnessed-vs-guess line the package draws everywhere: a measured (m, v, c) is a
// witness; a number a caller typed is a guess.
//
// Tier-1 discipline holds: this imports nothing internal. The kernel.Reap seam (deferred,
// out of this lane) owns counting per-class {reuse attempts, invalidations, cost samples}
// from the live vDSO and handing them here once per class per session; this leaf supplies
// the pure, conservative fold the seam maps onto.

// ClassObservation is one tool class's RAW measured statistics, drawn straight from the
// vDSO's per-class accounting over a window — the same counts internal/vcachecal fits its
// constants from, never asserted. The mutation rate is DERIVED from Invalidations /
// ReuseAttempts (the share of reuses the world-version churn stranded), not supplied; the
// costs are DERIVED from accumulated samples (mean per sample), not supplied. A class with
// zero reuse attempts has nothing to measure and is ABSTAINED, not guessed.
type ClassObservation struct {
	// Class is the tool class these statistics describe (e.g. "Read", "Grep").
	Class string `json:"class"`
	// ReuseAttempts is how many times a cached entry for this class was RE-PROPOSED in the
	// window — the denominator of the observed mutation rate. The first fill is not a reuse;
	// this counts only the proposals that COULD have been served from the cache. Zero here
	// means the class was never reused, so its economics are unmeasured -> abstain.
	ReuseAttempts int `json:"reuse_attempts"`
	// Invalidations is how many of those reuse attempts found the entry invalidated by an
	// intervening write (the world-version churn for this class). m = Invalidations /
	// ReuseAttempts is the OBSERVED mutation rate — high churn (writes strand reads faster
	// than they hit) drives m -> 1 and refutes the class, exactly the write-heavy-session
	// net-loss the issue describes. It is clamped to [0, ReuseAttempts] by the fold.
	Invalidations int `json:"invalidations"`
	// ValidateCostSamples is the SUM of observed per-reuse re-validation costs (world-version
	// / fingerprint checks), in execution-equivalents, over ValidateCostCount samples. The
	// fold divides to a mean; a class with no validate samples falls back to the cheap
	// default a world-version check costs (defaultObservedValidateCost), never to free.
	ValidateCostSamples float64 `json:"validate_cost_samples"`
	ValidateCostCount   int     `json:"validate_cost_count"`
	// CaptureCostSamples is the SUM of observed one-time capture/store costs over
	// CaptureCostCount samples; the fold divides to a mean, falling back to the cheap
	// default an LRU store costs when unsampled.
	CaptureCostSamples float64 `json:"capture_cost_samples"`
	CaptureCostCount   int     `json:"capture_cost_count"`
}

// Cheap, NON-zero fallbacks for the cost constants when a window observed no cost sample —
// the calibrate-don't-assume default (a world-version check and an LRU store ARE cheap, but
// never free; a zero validate cost would make a stale miss read as break-even, the #817
// floor reasoning). These are used ONLY for the cost terms; the mutation rate is never
// defaulted — a class with no reuse attempts to measure m from abstains outright.
const (
	defaultObservedValidateCost = 0.02 // v: a world-version / fingerprint check, in execution-equivalents.
	defaultObservedCaptureCost  = 0.02 // c: storing a fingerprint+result in the LRU, in execution-equivalents.
)

// ObservedClassGate is GateClass's verdict PLUS the provenance the issue demands: whether
// the calibration was MEASURED from observation or the class was ABSTAINED for want of any.
// An abstained class is declined — never admitted — but its decline is honestly attributed
// to "unmeasured", distinct from a class whose measured economics REFUTE (a proven net
// loss). Only Measured && Decision.Admit is a witnessed admit.
type ObservedClassGate struct {
	// Decision is the underlying break-even verdict over the DERIVED calibration. For an
	// abstained class it is a decline with a zeroed proof — there was no measured input to
	// prove over, so the gate cannot manufacture one.
	Decision ClassGateDecision `json:"decision"`
	// Measured is true iff the mutation rate was derived from at least one observed reuse
	// attempt. False means the class was abstained: nothing was measured, so it is declined
	// regardless of any cost samples. This is the witnessed-vs-guess bit — only a Measured
	// admit is honest.
	Measured bool `json:"measured"`
	// Calibration is the (m, v, c) the fold DERIVED and fed to the gate, surfaced so the
	// admission decision is auditable back to the observed counts it came from.
	Calibration ClassMemoInput `json:"calibration"`
	// Note is a one-line human reason: the gate's note for a measured class, or an explicit
	// abstain reason for an unmeasured one.
	Note string `json:"note"`
}

// FoldClassObservation derives a class's (m, v, c) calibration from its RAW observed
// statistics and runs the break-even gate over the derived input — the measured-input
// admission decision #819 requires. The mutation rate is Invalidations / ReuseAttempts
// (the observed world-version churn for the class), the validate/capture costs are the
// means of their samples (cheap non-zero default when unsampled), and the representative
// reuse count is the observed ReuseAttempts itself (the real window's k, capped to the
// break-even probe of 2 by GateClass when degenerate).
//
// The conservative core: a class with ZERO reuse attempts has nothing to measure m from, so
// it ABSTAINS — Measured=false, Decision declines, no fabricated input. This is the
// calibrate-don't-assume law (vcachecal's "measure s BEFORE trusting vCache, else abstain")
// applied to tier-2 admission: an un-observed class is never admitted on a guess. A class
// WITH observations admits iff the DERIVED economics prove: a stable read-heavy class
// (Invalidations << ReuseAttempts) yields a low m and admits; a write-churned class
// (Invalidations rivaling ReuseAttempts) yields a high m and is declined as a measured net
// loss. Pure and deterministic — same observation, same verdict.
func FoldClassObservation(obs ClassObservation) ObservedClassGate {
	// No reuse attempt -> no measured mutation rate -> abstain. We do NOT default m: an
	// unmeasured class is declined honestly, never admitted on a manufactured rate.
	if obs.ReuseAttempts <= 0 {
		return ObservedClassGate{
			Decision: ClassGateDecision{
				Class: obs.Class,
				Admit: false,
				Note:  "abstain: no observed reuse attempts — mutation rate is unmeasured, declining (calibrate-don't-assume)",
			},
			Measured:    false,
			Calibration: ClassMemoInput{Class: obs.Class},
			Note:        "abstain: no observed reuse attempts — mutation rate is unmeasured, declining (calibrate-don't-assume)",
		}
	}

	// Observed mutation rate: the share of reuses the world-version churn stranded. Clamp the
	// numerator into [0, ReuseAttempts] so a malformed count can never push m outside [0,1]
	// (a negative invalidation count means zero churn; an over-count means total churn).
	inval := obs.Invalidations
	if inval < 0 {
		inval = 0
	}
	if inval > obs.ReuseAttempts {
		inval = obs.ReuseAttempts
	}
	m := float64(inval) / float64(obs.ReuseAttempts)

	cal := ClassMemoInput{
		Class:        obs.Class,
		Accesses:     obs.ReuseAttempts, // the real observed window's reuse count is the representative k.
		MutationRate: m,
		ValidateCost: meanSampleOrDefault(obs.ValidateCostSamples, obs.ValidateCostCount, defaultObservedValidateCost),
		CaptureCost:  meanSampleOrDefault(obs.CaptureCostSamples, obs.CaptureCostCount, defaultObservedCaptureCost),
	}
	decision := GateClass(cal)
	return ObservedClassGate{
		Decision:    decision,
		Measured:    true,
		Calibration: cal,
		Note:        decision.Note,
	}
}

// FoldClassObservations folds a batch of per-class observations in one call, returning a
// verdict per class in input order — the shape the kernel.Reap seam (deferred) consults
// once per session to decide its tier-2 admission set from the window's measured statistics.
// Determinism is preserved: same observations, same verdicts, same order.
func FoldClassObservations(obs []ClassObservation) []ObservedClassGate {
	out := make([]ObservedClassGate, len(obs))
	for i, o := range obs {
		out[i] = FoldClassObservation(o)
	}
	return out
}

// AdmittedFromObservations is the projection the seam builds its tier-2 allow-set from: the
// class names whose MEASURED economics proved. An abstained class (unmeasured) and a class
// whose measured economics refute are both absent — only a witnessed, proven admit survives,
// so the cache never warms a class on anything but observed evidence that it pays.
func AdmittedFromObservations(obs []ClassObservation) []string {
	var admitted []string
	for _, g := range FoldClassObservations(obs) {
		if g.Measured && g.Decision.Admit {
			admitted = append(admitted, g.Decision.Class)
		}
	}
	return admitted
}

// meanSampleOrDefault returns sum/count when at least one non-negative sample was observed,
// else the cheap non-zero fallback — the calibrate-don't-assume cost default. A negative sum
// (a malformed accumulator) falls back too rather than crediting a negative cost.
func meanSampleOrDefault(sum float64, count int, fallback float64) float64 {
	if count <= 0 || sum < 0 {
		return fallback
	}
	return sum / float64(count)
}

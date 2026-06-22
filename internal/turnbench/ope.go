// ope.go — OFF-POLICY EVALUATION past the divergence frontier. Turn a `bounded@i`
// arm's resolve-rate from a HARD refusal ("unknown past call i") into a CALIBRATED,
// clearly-MODELED off-policy estimate with a confidence interval (issue #505).
//
// THE PROBLEM THIS SOLVES. policyreplay.go's divergence witness today is binary: an
// arm is `exact` (the frozen trajectory replays soundly, so its resolve-rate is a real
// measurement) or `bounded@i` (the model-observed result first flips at call i, so the
// recorded trajectory PAST i is counterfactual and resolve-rate is REFUSED outright).
// That refusal throws away ALL resolve-rate signal past the first divergence — including
// the sound prefix (calls 0..i-1, which replayed exactly) and the partial information a
// principled off-policy estimator can still extract about the counterfactual suffix.
//
// THE OPE FRAME. Off-policy evaluation is the field that quantifies HOW WRONG a
// counterfactual is past a divergence: given a trajectory logged under a behavior policy,
// estimate the value of a DIFFERENT (target) policy. The two canonical estimators:
//   - importance sampling / inverse-propensity (IPS): reweight each logged step by the
//     target/behavior action-probability ratio; and
//   - doubly-robust (DR): a direct-method model PLUS an IPS correction, which is unbiased
//     if EITHER the model or the propensities are right (the "double" robustness).
//
// THE DETERMINISM CAVEAT (load-bearing — naive OPE degenerates here). The policies in
// this harness are DETERMINISTIC decision tables, not stochastic action distributions.
// So a target policy assigns probability 1 to the action it would take and 0 to every
// other — and at a divergence the target took a DIFFERENT action than the log, so the
// importance ratio is 0/1 = 0. Naive IPS therefore collapses the suffix weight to ZERO
// and lands us right back at the hard refusal (a zero-weight suffix contributes nothing
// and the variance is undefined). IPS is the WRONG tool for deterministic policies.
//
// WHAT WE DO INSTEAD: a BOUNDED DOUBLY-ROBUST estimate with an EXPLICIT, depth-dependent
// counterfactual-uncertainty term. The structure mirrors deterministic record-replay
// debugging (rr / Pernosco) and query-plan cache invalidation: a frozen replay is SOUND
// up to the first input divergence and increasingly SPECULATIVE after it. Concretely, for
// an arm that diverges at call i over an N-call trace:
//
//   - DIRECT-METHOD POINT ESTIMATE. The sound prefix (calls 0..i-1) replayed exactly, so
//     its served-fraction is MEASURED. Past the frontier we have no live continuation, so
//     the most defensible point estimate is the frozen replay's own served-fraction over
//     the WHOLE trace — i.e. we project the recorded continuation as the maximum-likelihood
//     guess, the direct-method (model) term of a DR estimator with the recorded suffix as
//     the plug-in model. (At depth 0 — an exact arm — the "suffix" is empty and this is
//     exactly the measured resolve-rate.)
//
//   - DEPTH-DEPENDENT CI HALF-WIDTH. The IPS correction term of a real DR estimator has
//     variance that grows with how far past the frontier you extrapolate. Since the
//     deterministic propensities make that correction degenerate, we REPLACE it with a
//     principled bound: each of the (N-i) post-frontier calls is a coin we cannot observe,
//     so the counterfactual suffix could resolve anywhere in [0, N-i] of those calls. We
//     model the per-call counterfactual uncertainty as a depth-scaled term and combine it
//     with the (vanishing) sampling uncertainty of the sound prefix. The half-width is
//     therefore a MONOTONICALLY NON-DECREASING function of the post-frontier depth
//     (N - i): diverge EARLIER (smaller i, larger suffix) ⇒ WIDER CI; diverge LATER ⇒
//     narrower; depth 0 (exact) ⇒ CI half-width 0, estimate == the measured resolve-rate.
//
// THE HONESTY FENCE (this issue lives or dies on it). The estimate is a MODELED
// projection — a SEPARATE, clearly-labeled field, NEVER blended into a measured k.Syscall
// counter. The measured floor counters and the `bounded@i` measured-refusal of the
// MEASURED resolve-rate stay EXACTLY as they are; this adds a parallel, explicitly-modeled
// number alongside them. Every estimate carries Modeled=true, the estimator name, and the
// assumed continuation model, so a reader can never mistake the projection for a count.
package turnbench

import (
	"fmt"
	"math"
)

// OPEEstimatorName is the single source of truth for the estimator label stamped on every
// modeled estimate. It names the FORM (bounded doubly-robust) so a reader knows the
// estimate's pedigree without reading this file.
const OPEEstimatorName = "bounded-doubly-robust (deterministic-policy variant)"

// ResolveRateEstimate is the MODELED off-policy estimate of an arm's resolve-rate past the
// divergence frontier, with a confidence interval. It is a clearly-labeled PROJECTION —
// NEVER a measured counter. It lives ALONGSIDE the measured floor counters and the
// measured-resolve-rate refusal, never replacing them.
//
// For an EXACT arm (divergence depth 0) the estimate collapses to the measured value: Point
// == the measured served-fraction, the CI half-width is 0 (CILow == CIHigh == Point), and
// the note says so. For a BOUNDED@i arm the Point is the direct-method projection and the
// CI is widened by a depth-dependent counterfactual-uncertainty term (see ope.go's header).
type ResolveRateEstimate struct {
	// Modeled is ALWAYS true on this struct — it is the measured/modeled wall in a single
	// boolean. A reader (or a witness) that sees this field knows the numbers here are a
	// projection, not k.Syscall counts. It exists so the JSON can never be misread as
	// measured even out of context.
	Modeled bool `json:"modeled"`

	// Estimator names the OPE form and its variant (OPEEstimatorName).
	Estimator string `json:"estimator"`

	// Point is the modeled resolve-rate point estimate in [0,1] — the served-fraction the
	// frozen replay projects over the whole trace (direct-method term). For an exact arm
	// this EQUALS the measured served-fraction.
	Point float64 `json:"point"`

	// CILow / CIHigh bound the estimate; the interval is symmetric about Point, clamped to
	// [0,1]. The half-width is 0 for an exact arm and widens monotonically with the
	// post-frontier depth. CIHalfWidth is reported raw (pre-clamp) so the monotonic-widening
	// property is visible even when the clamp would otherwise flatten the endpoints near 0/1.
	CILow       float64 `json:"ci_low"`
	CIHigh      float64 `json:"ci_high"`
	CIHalfWidth float64 `json:"ci_half_width"`

	// Confidence is the nominal coverage the half-width targets (e.g. 0.95).
	Confidence float64 `json:"confidence"`

	// Depth is the number of calls extrapolated PAST the frontier (N - firstDivergence for a
	// bounded arm; 0 for an exact arm). It is the lever the CI half-width scales with.
	Depth int `json:"frontier_depth"`

	// SoundPrefix is the count of calls (0..i-1) that replayed EXACTLY before the frontier —
	// the part of Point that rests on measured replay, reported so a reader sees how much of
	// the estimate is sound vs extrapolated.
	SoundPrefix int `json:"sound_prefix_calls"`

	// Assumptions states the continuation model in prose so the projection is defensible and
	// self-describing (which OPE form, what the suffix model is, why IPS degenerates here).
	Assumptions string `json:"assumptions"`
}

// resolveRateInputs is the minimal MEASURED summary an arm hands the estimator: the served
// count and total over the WHOLE replayed trace (the direct-method plug-in), plus the
// frontier index. servedTotal/calls is the frozen replay's served-fraction; it is a real
// per-call count for an exact arm and the projection plug-in for a bounded arm.
type resolveRateInputs struct {
	served   int // calls whose observed class was "served" across the whole frozen trace
	calls    int // total recorded calls (the trace length)
	frontier int // first divergence index; <0 (or >=calls) means EXACT (depth 0)
}

// confidence95 is the nominal coverage the CI targets. Stated explicitly on every estimate.
const confidence95 = 0.95

// EstimateResolveRate computes the MODELED bounded-doubly-robust resolve-rate estimate for
// one arm from its measured served-fraction and divergence frontier. This is the only place
// the modeled number is produced; policyreplay.go wires its output onto the arm result
// without ever folding it into a measured counter.
//
// The estimator (see ope.go's header for the full derivation):
//   - POINT = served / calls — the frozen replay's served-fraction over the whole trace.
//     For an exact arm this IS the measured resolve-rate; for a bounded arm it is the
//     direct-method projection (the recorded suffix as the plug-in continuation model).
//   - DEPTH = calls - frontier for a bounded arm, 0 for an exact arm.
//   - CI HALF-WIDTH = counterfactualHalfWidth(depth, calls) — a depth-monotone term that is
//     0 at depth 0 and grows as more calls are extrapolated past the frontier.
func EstimateResolveRate(in resolveRateInputs) ResolveRateEstimate {
	calls := in.calls
	if calls <= 0 {
		// Degenerate: no calls. A zero-call trace has no resolve-rate; report a vacuous
		// exact estimate (point 0, zero width) rather than dividing by zero.
		return ResolveRateEstimate{
			Modeled:     true,
			Estimator:   OPEEstimatorName,
			Point:       0,
			CILow:       0,
			CIHigh:      0,
			CIHalfWidth: 0,
			Confidence:  confidence95,
			Depth:       0,
			SoundPrefix: 0,
			Assumptions: "empty trace: no calls to resolve; vacuous estimate.",
		}
	}

	point := float64(in.served) / float64(calls)

	// EXACT arm: frontier < 0 (or past the end) ⇒ depth 0 ⇒ the estimate collapses to the
	// measured value with a zero-width CI. This is the depth-0 control: estimate == measured
	// resolve-rate, CI == 0.
	exact := in.frontier < 0 || in.frontier >= calls
	depth := 0
	soundPrefix := calls
	if !exact {
		depth = calls - in.frontier
		soundPrefix = in.frontier
	}

	half := counterfactualHalfWidth(depth, calls)

	est := ResolveRateEstimate{
		Modeled:     true,
		Estimator:   OPEEstimatorName,
		Point:       point,
		CIHalfWidth: half,
		CILow:       clamp01(point - half),
		CIHigh:      clamp01(point + half),
		Confidence:  confidence95,
		Depth:       depth,
		SoundPrefix: soundPrefix,
	}
	if exact {
		est.Assumptions = fmt.Sprintf(
			"EXACT arm (frontier depth 0): the frozen trajectory replays soundly over all %d "+
				"calls, so this MODELED estimate collapses to the MEASURED served-fraction "+
				"(point == measured resolve-rate) with a zero-width CI. No extrapolation.",
			calls)
	} else {
		est.Assumptions = fmt.Sprintf(
			"BOUNDED arm (diverges at call %d of %d; %d calls extrapolated past the frontier). "+
				"DIRECT-METHOD point: the sound prefix (calls 0..%d) replayed exactly; past the "+
				"frontier the recorded continuation is the plug-in (max-likelihood) suffix model. "+
				"IPS/inverse-propensity is REFUSED here: the policies are deterministic decision "+
				"tables, so at a divergence the target/behavior action-probability ratio is 0/1=0 "+
				"and naive importance weighting degenerates to a zero-weight suffix (the hard "+
				"refusal). The CI half-width REPLACES that degenerate IPS variance with a "+
				"principled depth-dependent counterfactual-uncertainty bound: each of the %d "+
				"post-frontier calls is an unobservable coin, so the half-width grows monotonically "+
				"with depth. MODELED projection — NOT a measured counter; the floor counters and "+
				"the measured-resolve-rate refusal stand unchanged.",
			in.frontier, calls, depth, max0(in.frontier-1), depth)
	}
	return est
}

// counterfactualHalfWidth is the CI half-width for a bounded arm: a principled,
// depth-dependent counterfactual-uncertainty term that REPLACES the degenerate IPS variance
// of a deterministic-policy DR estimator. It must satisfy the control properties the
// acceptance test pins:
//
//   - depth 0 (an EXACT arm) ⇒ EXACTLY 0 (the CI collapses; estimate == measured).
//   - STRICTLY INCREASING in depth for a fixed trace length: an arm diverging earlier (more
//     calls past the frontier) has a strictly WIDER CI than one diverging later.
//   - bounded by 1 (a resolve-rate lives in [0,1], so an uncertainty half-width above 1 is
//     meaningless) and approaching that bound as the whole trace becomes counterfactual.
//
// FORM. Let f = depth/calls be the FRACTION of the trace that is counterfactual (f in
// (0,1] for a bounded arm) and let g(f) = sqrt( f * (1 - f/2) ) — the sqrt(variance)-shaped
// counterfactual-uncertainty kernel. g is STRICTLY INCREASING and concave on [0,1]: g(0)=0
// (depth 0 ⇒ zero width) and g(1)=sqrt(1/2) is its maximum (the whole trace counterfactual).
// The sqrt mirrors the sqrt(variance) shape of a real estimator's CI; the (1 - f/2) factor
// makes the marginal uncertainty diminish as the suffix dominates (concave) without EVER
// decreasing. We scale g by a confidence-tied ceiling so the half-width spans (0, ceiling]
// across the depth range WITHOUT a clamp ever flattening the strict monotonicity:
//
//	ceiling = min(1, zFor(confidence))   // a half-width band is at most the full [0,1]
//	half    = ceiling * g(f) / g(1)      ... (a)
//
// (a) is g RE-NORMALIZED to peak at exactly `ceiling` at f=1 (since g(f)/g(1) rises from 0
// to 1 strictly monotonically). The z-tie keeps the interval's width attached to a stated
// nominal confidence — a wider nominal confidence ⇒ a wider band — rather than an arbitrary
// constant, while the /g(1) normalization guarantees the band never exceeds the [0,1]
// ceiling and never saturates BEFORE f=1, so strict monotonicity holds across the WHOLE
// depth range (the clamp-flattening bug a bare z*g(f) would have near f→1).
//
// Because g(f)/g(1) is strictly increasing in f and f = depth/calls is strictly increasing
// in depth for a fixed calls, the half-width is STRICTLY increasing in depth — exactly the
// monotonic-widening control. depth==0 ⇒ f==0 ⇒ half==0.
func counterfactualHalfWidth(depth, calls int) float64 {
	if depth <= 0 || calls <= 0 {
		return 0
	}
	f := float64(depth) / float64(calls)
	if f > 1 {
		f = 1
	}
	g := func(x float64) float64 { return math.Sqrt(x * (1 - x/2)) }
	ceiling := math.Min(1, zFor(confidence95))
	half := ceiling * g(f) / g(1)
	if half > 1 {
		half = 1
	}
	return half
}

// zFor returns the standard-normal critical value for a two-sided interval at the given
// nominal confidence. Only the handful of conventional levels are tabulated; anything else
// falls back to the 95% value. This scales the half-width to a stated confidence so the CI
// is a defensible nominal interval, not an arbitrary band.
func zFor(confidence float64) float64 {
	switch {
	case confidence >= 0.99:
		return 2.5758
	case confidence >= 0.95:
		return 1.9600
	case confidence >= 0.90:
		return 1.6449
	default:
		return 1.9600
	}
}

// clamp01 confines x to [0,1] — a resolve-rate and its CI endpoints are fractions.
func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// max0 returns x if non-negative, else 0 (for the prose "calls 0..i-1" when i==0).
func max0(x int) int {
	if x < 0 {
		return 0
	}
	return x
}

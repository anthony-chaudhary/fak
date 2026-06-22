package turnbench

import (
	"context"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestOPE_ExactArmCollapsesToMeasured is the depth-0 control: an EXACT arm's MODELED
// estimate must collapse to the MEASURED served-fraction with a ZERO-width CI. point ==
// served/calls, CIHalfWidth == 0, CILow == CIHigh == point, depth == 0 — i.e. at depth 0
// the modeled number IS the measurement.
func TestOPE_ExactArmCollapsesToMeasured(t *testing.T) {
	// 4 served of 5 calls, exact (frontier < 0).
	est := EstimateResolveRate(resolveRateInputs{served: 4, calls: 5, frontier: -1})

	if !est.Modeled {
		t.Errorf("estimate must be flagged Modeled=true (the measured/modeled wall)")
	}
	want := 4.0 / 5.0
	if math.Abs(est.Point-want) > 1e-12 {
		t.Errorf("exact point must equal measured served-fraction %v, got %v", want, est.Point)
	}
	if est.CIHalfWidth != 0 {
		t.Errorf("exact arm CI half-width must be 0 (CI collapses), got %v", est.CIHalfWidth)
	}
	if est.CILow != est.Point || est.CIHigh != est.Point {
		t.Errorf("exact arm CI must collapse to the point: low=%v high=%v point=%v",
			est.CILow, est.CIHigh, est.Point)
	}
	if est.Depth != 0 {
		t.Errorf("exact arm frontier depth must be 0, got %d", est.Depth)
	}
	if est.SoundPrefix != 5 {
		t.Errorf("exact arm sound prefix must be the whole trace (5), got %d", est.SoundPrefix)
	}
	if est.Estimator != OPEEstimatorName {
		t.Errorf("estimator label must be %q, got %q", OPEEstimatorName, est.Estimator)
	}
}

// TestOPE_BoundedArmReportsEstimateAndCI is acceptance clause 1 at the estimator level: a
// bounded arm reports a resolve-rate ESTIMATE + a NON-ZERO confidence interval, with the
// point inside [CILow, CIHigh] and the whole interval inside [0,1].
func TestOPE_BoundedArmReportsEstimateAndCI(t *testing.T) {
	// Diverges at call 1 of 5 (depth 4), 3 served.
	est := EstimateResolveRate(resolveRateInputs{served: 3, calls: 5, frontier: 1})

	if !est.Modeled {
		t.Errorf("bounded estimate must be Modeled=true")
	}
	if est.CIHalfWidth <= 0 {
		t.Errorf("a bounded arm must carry a NON-ZERO CI half-width, got %v", est.CIHalfWidth)
	}
	if est.Depth != 4 {
		t.Errorf("frontier depth must be calls-frontier = 4, got %d", est.Depth)
	}
	if est.SoundPrefix != 1 {
		t.Errorf("sound prefix must be the frontier index (1), got %d", est.SoundPrefix)
	}
	if est.CILow < 0 || est.CIHigh > 1 {
		t.Errorf("CI must stay in [0,1], got [%v, %v]", est.CILow, est.CIHigh)
	}
	if !(est.CILow <= est.Point && est.Point <= est.CIHigh) {
		t.Errorf("point %v must lie within [%v, %v]", est.Point, est.CILow, est.CIHigh)
	}
	if est.Assumptions == "" {
		t.Errorf("a bounded estimate must state its assumptions (estimator + suffix model)")
	}
}

// TestOPE_CIWidensMonotonicallyWithDepth is acceptance clause 2 (the control) at the
// estimator level: for a FIXED trace length, an arm that diverges EARLIER (deeper past the
// frontier) must have a STRICTLY WIDER CI than one diverging later — and the width collapses
// to 0 at depth 0. Sweeping the frontier from the end of the trace back to the start must
// yield a strictly increasing half-width sequence.
func TestOPE_CIWidensMonotonicallyWithDepth(t *testing.T) {
	const calls = 8

	// frontier == calls behaves as exact (depth 0); then frontier = calls-1, ..., 0 give
	// strictly increasing depth 1..calls. The half-widths must be strictly increasing.
	var prev float64 = -1
	for frontier := calls; frontier >= 0; frontier-- {
		est := EstimateResolveRate(resolveRateInputs{served: 4, calls: calls, frontier: frontier})
		depth := 0
		if frontier < calls {
			depth = calls - frontier
		}
		if depth == 0 {
			if est.CIHalfWidth != 0 {
				t.Fatalf("depth 0 must give half-width 0, got %v (frontier=%d)", est.CIHalfWidth, frontier)
			}
		} else {
			if !(est.CIHalfWidth > prev) {
				t.Fatalf("CI half-width must STRICTLY widen with depth: depth=%d width=%v not > prev=%v",
					depth, est.CIHalfWidth, prev)
			}
		}
		prev = est.CIHalfWidth
	}
}

// TestOPE_DeeperDivergenceWiderThanShallower is the explicit "diverge@i wider than
// diverge@j>i" pairwise control the acceptance names: an arm diverging at call i has a wider
// CI than one diverging at a LATER call j (j>i, shallower past-frontier extrapolation).
func TestOPE_DeeperDivergenceWiderThanShallower(t *testing.T) {
	const calls = 10
	deep := EstimateResolveRate(resolveRateInputs{served: 5, calls: calls, frontier: 2})    // depth 8
	shallow := EstimateResolveRate(resolveRateInputs{served: 5, calls: calls, frontier: 7}) // depth 3

	if !(deep.Depth > shallow.Depth) {
		t.Fatalf("setup: deep depth %d must exceed shallow depth %d", deep.Depth, shallow.Depth)
	}
	if !(deep.CIHalfWidth > shallow.CIHalfWidth) {
		t.Errorf("earlier/deeper divergence must have a WIDER CI: deep@2 width=%v not > shallow@7 width=%v",
			deep.CIHalfWidth, shallow.CIHalfWidth)
	}
}

// TestOPE_HalfWidthBoundedAndDepthZero pins the two edge invariants the half-width function
// must satisfy: depth 0 ⇒ exactly 0, and the half-width never exceeds 1 (a resolve-rate band
// is at most the full [0,1] either side) even when the WHOLE trace is counterfactual.
func TestOPE_HalfWidthBoundedAndDepthZero(t *testing.T) {
	if got := counterfactualHalfWidth(0, 5); got != 0 {
		t.Errorf("depth 0 half-width must be exactly 0, got %v", got)
	}
	// Whole trace counterfactual (frontier 0): the half-width is the maximum but still <= 1.
	if got := counterfactualHalfWidth(5, 5); got <= 0 || got > 1 {
		t.Errorf("max-depth half-width must be in (0,1], got %v", got)
	}
}

// TestOPE_WiresOntoBoundedAndExactArms is the end-to-end acceptance through RunPolicyReplay:
// every arm result carries a MODELED ResolveRateEstimate; the EXACT arms collapse to the
// measured served-fraction with a zero-width CI, and the BOUNDED arms carry a non-zero CI
// whose width orders by divergence depth (permissive-plus diverges at call 1 — deeper — so
// its CI is wider than strict-no-book's, which diverges at call 3). The measured Counters
// are untouched — the modeled estimate lives strictly alongside them.
func TestOPE_WiresOntoBoundedAndExactArms(t *testing.T) {
	ctx := context.Background()
	tr := spineTrace(t)
	arms := spineArms()

	rep, err := RunPolicyReplay(ctx, tr, arms, "recorded", DefaultCostModel())
	if err != nil {
		t.Fatalf("RunPolicyReplay: %v", err)
	}

	rec := armByName(t, rep, "recorded")
	equiv := armByName(t, rep, "equivalent-on-trace")
	strict := armByName(t, rep, "strict-no-book")
	plus := armByName(t, rep, "permissive-plus")

	// Every arm carries a MODELED estimate (the wall is the Modeled flag).
	for _, a := range rep.Arms {
		if !a.ResolveRateEstimate.Modeled {
			t.Errorf("arm %q estimate must be Modeled=true", a.Name)
		}
		if a.ResolveRateEstimate.Estimator != OPEEstimatorName {
			t.Errorf("arm %q estimator label wrong: %q", a.Name, a.ResolveRateEstimate.Estimator)
		}
	}

	// EXACT arms (recorded, equivalent-on-trace): estimate == measured served-fraction,
	// CI collapsed to 0. Cross-check the point against the served count derived from the
	// arm's own measured class breakdown (served = total calls - denies - quarantines).
	for _, a := range []PolicyArmResult{rec, equiv} {
		if a.FirstDivergence != -1 {
			t.Fatalf("setup: %q expected exact", a.Name)
		}
		if a.ResolveRateEstimate.CIHalfWidth != 0 {
			t.Errorf("exact arm %q CI must collapse to 0, got %v", a.Name, a.ResolveRateEstimate.CIHalfWidth)
		}
		if a.ResolveRateEstimate.CILow != a.ResolveRateEstimate.Point ||
			a.ResolveRateEstimate.CIHigh != a.ResolveRateEstimate.Point {
			t.Errorf("exact arm %q CI must equal the point", a.Name)
		}
		if a.ResolveRateEstimate.Depth != 0 {
			t.Errorf("exact arm %q depth must be 0, got %d", a.Name, a.ResolveRateEstimate.Depth)
		}
	}

	// BOUNDED arms (strict-no-book@3, permissive-plus@1): non-zero CI, and the deeper
	// divergence (permissive-plus, frontier 1) has the WIDER CI.
	if strict.FirstDivergence != 3 || plus.FirstDivergence != 1 {
		t.Fatalf("setup: expected strict@3 plus@1, got strict@%d plus@%d", strict.FirstDivergence, plus.FirstDivergence)
	}
	if strict.ResolveRateEstimate.CIHalfWidth <= 0 {
		t.Errorf("bounded strict-no-book must carry a non-zero CI, got %v", strict.ResolveRateEstimate.CIHalfWidth)
	}
	if plus.ResolveRateEstimate.CIHalfWidth <= 0 {
		t.Errorf("bounded permissive-plus must carry a non-zero CI, got %v", plus.ResolveRateEstimate.CIHalfWidth)
	}
	if !(plus.ResolveRateEstimate.CIHalfWidth > strict.ResolveRateEstimate.CIHalfWidth) {
		t.Errorf("deeper divergence (permissive-plus@1) must have a WIDER CI than shallower (strict-no-book@3): plus=%v strict=%v",
			plus.ResolveRateEstimate.CIHalfWidth, strict.ResolveRateEstimate.CIHalfWidth)
	}

	// HONESTY WALL: the modeled estimate did NOT perturb the MEASURED counters. The bounded
	// arms' measured deny counts stand exactly as the spine test pins them (strict=3, plus=1).
	if strict.Counters.Denies != 3 {
		t.Errorf("modeled estimate must NOT touch measured counters; strict denies=%d want 3", strict.Counters.Denies)
	}
	if plus.Counters.Denies != 1 {
		t.Errorf("modeled estimate must NOT touch measured counters; plus denies=%d want 1", plus.Counters.Denies)
	}
}

// TestOPE_ExactArmPointEqualsMeasuredServedFraction proves through RunPolicyReplay that an
// exact arm's modeled point IS the measured resolve-rate — recomputed independently from the
// arm's measured class breakdown, not from the estimator's own internal served count. This
// is the depth-0 "estimate == measured" half of the control, grounded in measured counters.
func TestOPE_ExactArmPointEqualsMeasuredServedFraction(t *testing.T) {
	ctx := context.Background()
	allow := map[string]bool{"get_user_details": true, "calculate": true}
	tr := &Trace{
		SliceID: "ope-exact-served-fraction",
		Calls: []Call{
			{Tool: "get_user_details", Args: rawArgs(t, map[string]any{"user_id": "u1"})},
			{Tool: "calculate", Args: rawArgs(t, map[string]any{"a": 1, "b": 2})},
		},
	}
	arms := []PolicyArm{
		{Name: "ref", Policy: adjudicator.Policy{Allow: allow}},
		{Name: "ref-copy", Policy: adjudicator.Policy{Allow: allow}},
	}
	rep, err := RunPolicyReplay(ctx, tr, arms, "ref", DefaultCostModel())
	if err != nil {
		t.Fatalf("RunPolicyReplay: %v", err)
	}
	for _, a := range rep.Arms {
		if a.FirstDivergence != -1 {
			t.Fatalf("setup: arm %q expected exact, got divergence %d", a.Name, a.FirstDivergence)
		}
		// Measured served-fraction from the class breakdown: a call is "served" unless it
		// was denied or quarantined.
		denied := int(a.Counters.Denies + a.Counters.Quarantines)
		measuredServed := len(tr.Calls) - denied
		measuredFrac := float64(measuredServed) / float64(len(tr.Calls))
		if math.Abs(a.ResolveRateEstimate.Point-measuredFrac) > 1e-12 {
			t.Errorf("arm %q exact point %v must equal measured served-fraction %v",
				a.Name, a.ResolveRateEstimate.Point, measuredFrac)
		}
		if a.ResolveRateEstimate.CIHalfWidth != 0 {
			t.Errorf("arm %q exact CI must be 0", a.Name)
		}
	}
}

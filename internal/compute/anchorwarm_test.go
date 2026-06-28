package compute

import (
	"math"
	"testing"
)

// index is a small ranked hot-anchor index in descending weight (the shape the artifact persists).
func index() []AnchorWeight {
	return []AnchorWeight{
		{Key: "a", Weight: 50},
		{Key: "b", Weight: 30},
		{Key: "c", Weight: 12},
		{Key: "d", Weight: 8},
	}
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestPlanAnchorWarm_FullHeadromWarmsPlannedTopK(t *testing.T) {
	// Top-2 of a 100-weight index covers (50+30)/100 = 0.80.
	plan := PlanAnchorWarm(AnchorWarmInput{
		Ranked:         index(),
		PlannedAnchors: 2,
		TargetCoverage: 0.8,
		Pressure:       WarmPressureNone,
	})
	if plan.Deferred {
		t.Fatalf("full headroom must not defer: %+v", plan)
	}
	if plan.StarAnchors != 2 {
		t.Fatalf("StarAnchors = %d, want 2", plan.StarAnchors)
	}
	if !approx(plan.ExpectedCoverage, 0.80) {
		t.Fatalf("ExpectedCoverage = %v, want 0.80", plan.ExpectedCoverage)
	}
	if plan.Reason != ReasonWarmedPlanned {
		t.Fatalf("Reason = %q, want %q", plan.Reason, ReasonWarmedPlanned)
	}
	if got, want := plan.SummaryLine(), "warming 2 star anchors, expected coverage 80.0%"; got != want {
		t.Fatalf("SummaryLine = %q, want %q", got, want)
	}
}

func TestPlanAnchorWarm_HighPressureDefersEntirely(t *testing.T) {
	plan := PlanAnchorWarm(AnchorWarmInput{
		Ranked:         index(),
		PlannedAnchors: 3,
		Pressure:       WarmPressureHigh,
	})
	if !plan.Deferred || plan.StarAnchors != 0 || len(plan.Warm) != 0 {
		t.Fatalf("high pressure must warm nothing: %+v", plan)
	}
	if plan.Reason != ReasonPressureDeferred {
		t.Fatalf("Reason = %q, want %q", plan.Reason, ReasonPressureDeferred)
	}
	if !approx(plan.ExpectedCoverage, 0) {
		t.Fatalf("deferred ExpectedCoverage = %v, want 0", plan.ExpectedCoverage)
	}
	if got, want := plan.SummaryLine(), "warming 0 star anchors, expected coverage 0.0%"; got != want {
		t.Fatalf("SummaryLine = %q, want %q", got, want)
	}
}

func TestPlanAnchorWarm_PartialPressureCapsBelowPlanned(t *testing.T) {
	// Planned top-3 but the executor admits only 1 → warm just anchor "a" (0.50 coverage).
	plan := PlanAnchorWarm(AnchorWarmInput{
		Ranked:         index(),
		PlannedAnchors: 3,
		Pressure:       WarmPressurePartial,
		PressureCap:    1,
	})
	if plan.Deferred {
		t.Fatalf("partial pressure with cap>0 must not defer: %+v", plan)
	}
	if plan.StarAnchors != 1 {
		t.Fatalf("StarAnchors = %d, want 1 (capped)", plan.StarAnchors)
	}
	if !approx(plan.ExpectedCoverage, 0.50) {
		t.Fatalf("ExpectedCoverage = %v, want 0.50", plan.ExpectedCoverage)
	}
	if plan.Reason != ReasonPressureCapped {
		t.Fatalf("Reason = %q, want %q", plan.Reason, ReasonPressureCapped)
	}
}

func TestPlanAnchorWarm_PartialPressureCapAtOrAbovePlannedWarmsPlanned(t *testing.T) {
	plan := PlanAnchorWarm(AnchorWarmInput{
		Ranked:         index(),
		PlannedAnchors: 2,
		Pressure:       WarmPressurePartial,
		PressureCap:    5, // cap exceeds planned → planned wins
	})
	if plan.StarAnchors != 2 || plan.Reason != ReasonWarmedPlanned {
		t.Fatalf("cap>=planned should warm the planned top-K: %+v", plan)
	}
}

func TestPlanAnchorWarm_PartialPressureZeroCapDefers(t *testing.T) {
	plan := PlanAnchorWarm(AnchorWarmInput{
		Ranked:         index(),
		PlannedAnchors: 3,
		Pressure:       WarmPressurePartial,
		PressureCap:    0,
	})
	if !plan.Deferred || plan.Reason != ReasonPressureDeferred {
		t.Fatalf("partial pressure with zero cap must defer: %+v", plan)
	}
}

func TestPlanAnchorWarm_EmptyIndexIsNoIndex(t *testing.T) {
	plan := PlanAnchorWarm(AnchorWarmInput{Ranked: nil, PlannedAnchors: 4})
	if plan.StarAnchors != 0 || plan.Reason != ReasonNoIndex || plan.Deferred {
		t.Fatalf("empty index: %+v", plan)
	}
	if got, want := plan.SummaryLine(), "warming 0 star anchors, expected coverage 0.0%"; got != want {
		t.Fatalf("SummaryLine = %q, want %q", got, want)
	}
}

func TestPlanAnchorWarm_NonPositivePlannedWarmsAllRanked(t *testing.T) {
	plan := PlanAnchorWarm(AnchorWarmInput{
		Ranked:         index(),
		PlannedAnchors: 0, // unset → warm the whole ranked set
		Pressure:       WarmPressureNone,
	})
	if plan.StarAnchors != 4 {
		t.Fatalf("StarAnchors = %d, want 4 (all ranked)", plan.StarAnchors)
	}
	if !approx(plan.ExpectedCoverage, 1.0) {
		t.Fatalf("ExpectedCoverage = %v, want 1.0", plan.ExpectedCoverage)
	}
}

func TestPlanAnchorWarm_DropsMalformedRowsAndResorts(t *testing.T) {
	// Out-of-order input with an empty key and a non-positive weight; both dropped, rest re-sorted.
	in := []AnchorWeight{
		{Key: "low", Weight: 10},
		{Key: "", Weight: 99},   // dropped: empty key
		{Key: "neg", Weight: 0}, // dropped: non-positive weight
		{Key: "high", Weight: 90},
	}
	plan := PlanAnchorWarm(AnchorWarmInput{Ranked: in, PlannedAnchors: 1, Pressure: WarmPressureNone})
	if plan.StarAnchors != 1 || len(plan.Warm) != 1 {
		t.Fatalf("plan: %+v", plan)
	}
	if plan.Warm[0].Key != "high" {
		t.Fatalf("top warm anchor = %q, want \"high\" (re-sorted by weight)", plan.Warm[0].Key)
	}
	// total surviving weight = 100; top-1 = 90 → 0.90.
	if !approx(plan.ExpectedCoverage, 0.90) {
		t.Fatalf("ExpectedCoverage = %v, want 0.90", plan.ExpectedCoverage)
	}
}

func TestRealizedCoverage_ClimbsWithObservedAnchors(t *testing.T) {
	plan := PlanAnchorWarm(AnchorWarmInput{
		Ranked:         index(),
		PlannedAnchors: 2, // warm a(50) + b(30); warm weight = 80
		Pressure:       WarmPressureNone,
	})
	if got := plan.RealizedCoverage(nil); got != 0 {
		t.Fatalf("no traffic realized = %v, want 0", got)
	}
	// Observe only "a" → 50/80 = 0.625.
	if got := plan.RealizedCoverage(map[string]bool{"a": true}); !approx(got, 0.625) {
		t.Fatalf("realized(a) = %v, want 0.625", got)
	}
	// Observe both warmed anchors → fully realized.
	if got := plan.RealizedCoverage(map[string]bool{"a": true, "b": true}); !approx(got, 1.0) {
		t.Fatalf("realized(a,b) = %v, want 1.0", got)
	}
	// An anchor NOT in the warm set ("c") never counts toward realized coverage.
	if got := plan.RealizedCoverage(map[string]bool{"c": true}); got != 0 {
		t.Fatalf("realized(c) = %v, want 0 (c was not warmed)", got)
	}
}

func TestRealizedCoverage_DeferredPlanIsZero(t *testing.T) {
	plan := PlanAnchorWarm(AnchorWarmInput{Ranked: index(), PlannedAnchors: 2, Pressure: WarmPressureHigh})
	if got := plan.RealizedCoverage(map[string]bool{"a": true, "b": true}); got != 0 {
		t.Fatalf("deferred plan realized = %v, want 0 (nothing warmed)", got)
	}
}

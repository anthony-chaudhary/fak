package dispatchtick

import (
	"reflect"
	"sort"
	"strings"
	"testing"
)

// admission_edf_test.go is the closure-test half admission_edf.go's header
// promises. The planner is a pure fold with an injected clock and estimator, so
// every property below is witnessable on CPU with no fixture and no I/O.

// TestEDFOrdersByDeadlineNotInputOrder pins the central claim: the plan is
// earliest-deadline-first regardless of how the caller happened to enqueue.
func TestEDFOrdersByDeadlineNotInputOrder(t *testing.T) {
	items := []DeadlineItem{
		{ID: "late", DeadlineTick: 900},
		{ID: "soon", DeadlineTick: 10},
		{ID: "mid", DeadlineTick: 100},
	}
	got := EDFPlanner{}.Plan(items).OrderIDs()
	want := []string{"soon", "mid", "late"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("order = %v, want %v", got, want)
	}
}

// TestEDFTiebreakIsIDThenInputOrder pins the documented deterministic tiebreak.
// Without it, equal deadlines would order by Go's map/sort incidentals and the
// plan would not be reproducible across runs.
func TestEDFTiebreakIsIDThenInputOrder(t *testing.T) {
	items := []DeadlineItem{
		{ID: "c", DeadlineTick: 5},
		{ID: "a", DeadlineTick: 5},
		{ID: "b", DeadlineTick: 5},
	}
	got := EDFPlanner{}.Plan(items).OrderIDs()
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("equal-deadline order = %v, want ID-sorted %v", got, want)
	}

	// Fully identical entries fall back to input order (SliceStable), so a
	// duplicate ID cannot make the plan nondeterministic either.
	dupes := []DeadlineItem{
		{ID: "x", DeadlineTick: 5, CostHintTicks: 1},
		{ID: "x", DeadlineTick: 5, CostHintTicks: 2},
	}
	plan := EDFPlanner{}.Plan(dupes)
	if len(plan.Order) != 2 {
		t.Fatalf("both duplicates must be planned, got %d", len(plan.Order))
	}
	if plan.Order[0].Item.CostHintTicks != 1 || plan.Order[1].Item.CostHintTicks != 2 {
		t.Error("identical keys must retain input order")
	}
}

// TestPlanDoesNotMutateInput: the planner copies before sorting. A caller that
// handed its live queue in and got it reordered underneath would have a bug that
// only shows up under a second, unrelated consumer.
func TestPlanDoesNotMutateInput(t *testing.T) {
	items := []DeadlineItem{
		{ID: "late", DeadlineTick: 900},
		{ID: "soon", DeadlineTick: 10},
	}
	before := append([]DeadlineItem(nil), items...)
	EDFPlanner{}.Plan(items)
	if !reflect.DeepEqual(items, before) {
		t.Errorf("Plan mutated its input: %v, want %v", items, before)
	}
}

// TestShedFreesTheLaneForTheItemsBehind is the reason the shed exists. The lane
// clock must NOT advance past a shed item — if it did, shedding would cost the
// survivors the very time it was supposed to give them, and everyone would miss
// together, which is the outcome this policy exists to prevent.
func TestShedFreesTheLaneForTheItemsBehind(t *testing.T) {
	items := []DeadlineItem{
		{ID: "doomed", DeadlineTick: 3, CostHintTicks: 5, Degradable: true},
		{ID: "rescued", DeadlineTick: 6, CostHintTicks: 5, Degradable: true},
	}
	plan := EDFPlanner{}.Plan(items)

	if got := plan.ShedIDs(); !reflect.DeepEqual(got, []string{"doomed"}) {
		t.Fatalf("shed = %v, want [doomed]", got)
	}
	if got := plan.OrderIDs(); !reflect.DeepEqual(got, []string{"rescued"}) {
		t.Fatalf("order = %v, want [rescued]", got)
	}
	// The survivor starts at the lane's original position, not after the shed
	// item's cost — that is the whole mechanism.
	if got := plan.Order[0].PredictedStartTick; got != 0 {
		t.Errorf("rescued start = %d, want 0 (lane must not advance past a shed item)", got)
	}
	if got := plan.Order[0].PredictedFinishTick; got != 5 {
		t.Errorf("rescued finish = %d, want 5", got)
	}
	if plan.Order[0].PredictedLate {
		t.Error("rescued item finishes at 5 against a deadline of 6; it is not late")
	}
}

// TestNonDegradableIsFlaggedNeverDropped pins the asymmetry that makes this
// policy safe to turn on: an item the caller did not mark degradable is never
// silently disappeared, only flagged, even when it is certain to miss.
func TestNonDegradableIsFlaggedNeverDropped(t *testing.T) {
	items := []DeadlineItem{
		{ID: "mandatory", DeadlineTick: 3, CostHintTicks: 5, Degradable: false},
	}
	plan := EDFPlanner{}.Plan(items)

	if len(plan.Shed) != 0 {
		t.Fatalf("a non-degradable item must never be shed, got %v", plan.ShedIDs())
	}
	if len(plan.Order) != 1 {
		t.Fatalf("order = %v, want the item retained", plan.OrderIDs())
	}
	if !plan.Order[0].PredictedLate {
		t.Error("a retained item predicted to finish past its deadline must be flagged PredictedLate")
	}
}

// TestNonDegradableLatenessCascades pins the documented consequence: a retained
// late item honestly occupies the lane, which can doom a degradable item behind
// it. The header calls this the point rather than a bug, so it is pinned as
// intended behavior — a future change that quietly stops the cascade is a policy
// change, not a fix.
func TestNonDegradableLatenessCascades(t *testing.T) {
	items := []DeadlineItem{
		{ID: "mandatory", DeadlineTick: 3, CostHintTicks: 5, Degradable: false},
		{ID: "victim", DeadlineTick: 6, CostHintTicks: 5, Degradable: true},
	}
	plan := EDFPlanner{}.Plan(items)

	if got := plan.OrderIDs(); !reflect.DeepEqual(got, []string{"mandatory"}) {
		t.Errorf("order = %v, want [mandatory]", got)
	}
	if got := plan.ShedIDs(); !reflect.DeepEqual(got, []string{"victim"}) {
		t.Errorf("shed = %v, want [victim] — the retained late item occupies the lane", got)
	}
	// Evidence: the victim's predicted start is after the mandatory item's cost.
	if got := plan.Shed[0].PredictedStartTick; got != 5 {
		t.Errorf("victim start = %d, want 5 (behind the retained item)", got)
	}
}

// TestDeadlineBoundaryIsInclusive: finishing exactly ON the deadline meets it.
// An off-by-one here sheds work that would have been fine.
func TestDeadlineBoundaryIsInclusive(t *testing.T) {
	onTime := EDFPlanner{}.Plan([]DeadlineItem{
		{ID: "exact", DeadlineTick: 5, CostHintTicks: 5, Degradable: true},
	})
	if len(onTime.Shed) != 0 {
		t.Errorf("finish==deadline must MEET the deadline, got shed %v", onTime.ShedIDs())
	}
	if onTime.Order[0].PredictedLate {
		t.Error("finish==deadline must not be flagged late")
	}

	oneOver := EDFPlanner{}.Plan([]DeadlineItem{
		{ID: "over", DeadlineTick: 4, CostHintTicks: 5, Degradable: true},
	})
	if got := oneOver.ShedIDs(); !reflect.DeepEqual(got, []string{"over"}) {
		t.Errorf("finish one tick past the deadline must miss, got shed %v", got)
	}
}

// TestUnpriceableCostFailsOpen pins the stated fail-safe direction: a negative
// (unpriceable) estimate clamps to zero, so an estimator bug can only FAIL TO
// SHED — keeping today's admit-everything behavior — and can never invent a
// predicted miss that drops real work.
func TestUnpriceableCostFailsOpen(t *testing.T) {
	plan := EDFPlanner{}.Plan([]DeadlineItem{
		{ID: "unpriceable", DeadlineTick: 1, CostHintTicks: -999, Degradable: true},
	})
	if len(plan.Shed) != 0 {
		t.Fatalf("a negative cost must clamp to 0 and admit, got shed %v", plan.ShedIDs())
	}
	if got := plan.Order[0].PredictedFinishTick; got != 0 {
		t.Errorf("clamped finish = %d, want 0", got)
	}

	// Same via a buggy injected estimator.
	buggy := EDFPlanner{EstimateCostTicks: func(DeadlineItem) int64 { return -1 }}
	if got := buggy.Plan([]DeadlineItem{{ID: "a", DeadlineTick: 0, Degradable: true}}); len(got.Shed) != 0 {
		t.Errorf("a negative estimator result must clamp, got shed %v", got.ShedIDs())
	}
}

// TestInjectedClockAndEstimatorAreUsed: the zero value must be tick 0 and cost
// hints (never a wall clock, which would smuggle nondeterminism into a pure
// fold), and both injections must actually take effect.
func TestInjectedClockAndEstimatorAreUsed(t *testing.T) {
	item := DeadlineItem{ID: "a", DeadlineTick: 1000, CostHintTicks: 7}

	zero := EDFPlanner{}.Plan([]DeadlineItem{item})
	if got := zero.Order[0].PredictedStartTick; got != 0 {
		t.Errorf("nil clock start = %d, want tick 0", got)
	}
	if got := zero.Order[0].PredictedFinishTick; got != 7 {
		t.Errorf("nil estimator must use CostHintTicks: finish = %d, want 7", got)
	}

	wired := EDFPlanner{
		NowTick:           func() int64 { return 100 },
		EstimateCostTicks: func(DeadlineItem) int64 { return 50 },
	}.Plan([]DeadlineItem{item})
	if got := wired.Order[0].PredictedStartTick; got != 100 {
		t.Errorf("injected clock start = %d, want 100", got)
	}
	if got := wired.Order[0].PredictedFinishTick; got != 150 {
		t.Errorf("injected estimator finish = %d, want 150 (estimator overrides the hint)", got)
	}
}

// TestEveryItemLandsInExactlyOneList pins the partition invariant the header
// states. A dropped item would be work that vanished with no typed reason —
// precisely the silent disappearance this policy forbids.
func TestEveryItemLandsInExactlyOneList(t *testing.T) {
	items := []DeadlineItem{
		{ID: "a", DeadlineTick: 1, CostHintTicks: 5, Degradable: true},
		{ID: "b", DeadlineTick: 2, CostHintTicks: 5, Degradable: false},
		{ID: "c", DeadlineTick: 50, CostHintTicks: 5, Degradable: true},
		{ID: "d", DeadlineTick: 3, CostHintTicks: 0, Degradable: true},
		{ID: "e", DeadlineTick: 100, CostHintTicks: 1},
	}
	plan := EDFPlanner{}.Plan(items)

	if got, want := len(plan.Order)+len(plan.Shed), len(items); got != want {
		t.Fatalf("planned %d items, input had %d", got, want)
	}
	got := append(plan.OrderIDs(), plan.ShedIDs()...)
	sort.Strings(got)
	want := []string{"a", "b", "c", "d", "e"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("planned IDs = %v, want %v (no item may vanish or be duplicated)", got, want)
	}
}

// TestPlanIsDeterministic: the same inputs must produce byte-for-byte the same
// plan. This is what makes a shed decision auditable after the fact.
func TestPlanIsDeterministic(t *testing.T) {
	items := []DeadlineItem{
		{ID: "a", DeadlineTick: 5, CostHintTicks: 5, Degradable: true},
		{ID: "b", DeadlineTick: 5, CostHintTicks: 3, Degradable: false},
		{ID: "c", DeadlineTick: 1, CostHintTicks: 9, Degradable: true},
	}
	first := EDFPlanner{}.Plan(items)
	for i := 0; i < 16; i++ {
		if !reflect.DeepEqual(EDFPlanner{}.Plan(items), first) {
			t.Fatalf("plan differed on iteration %d", i)
		}
	}
}

// TestShedReasonIsTheWireToken pins the string, not the Go symbol. Renaming the
// constant must never change what goes on the wire — the RefuseHostFootprint
// convention this package already follows.
func TestShedReasonIsTheWireToken(t *testing.T) {
	if ShedPredictedDeadlineMiss != "SHED_PREDICTED_DEADLINE_MISS" {
		t.Errorf("wire token = %q, want SHED_PREDICTED_DEADLINE_MISS", ShedPredictedDeadlineMiss)
	}
	plan := EDFPlanner{}.Plan([]DeadlineItem{
		{ID: "x", DeadlineTick: 1, CostHintTicks: 9, Degradable: true},
	})
	if len(plan.Shed) != 1 {
		t.Fatalf("expected one shed, got %d", len(plan.Shed))
	}
	if plan.Shed[0].Reason != ShedPredictedDeadlineMiss {
		t.Errorf("shed reason = %q, want %q", plan.Shed[0].Reason, ShedPredictedDeadlineMiss)
	}
}

// TestShedStringCarriesRederivableEvidence: a shed line must let a reader check
// the verdict rather than trust it, so the reason AND the four numbers behind it
// have to be present.
func TestShedStringCarriesRederivableEvidence(t *testing.T) {
	s := ShedDecision{
		Item:                DeadlineItem{ID: "job-7", DeadlineTick: 3},
		Reason:              ShedPredictedDeadlineMiss,
		PredictedStartTick:  1,
		PredictedFinishTick: 9,
	}.String()

	for _, want := range []string{
		"SHED_PREDICTED_DEADLINE_MISS",
		"id=job-7",
		"predicted_finish=9",
		"deadline=3",
		"predicted_start=1",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("shed line %q missing %q", s, want)
		}
	}
}

// TestEmptyPlanIsUsable: no items in, no panic, and both lists are safe to range
// over and to marshal.
func TestEmptyPlanIsUsable(t *testing.T) {
	plan := EDFPlanner{}.Plan(nil)
	if len(plan.Order) != 0 || len(plan.Shed) != 0 {
		t.Fatalf("empty input produced %d ordered, %d shed", len(plan.Order), len(plan.Shed))
	}
	if plan.OrderIDs() == nil || plan.ShedIDs() == nil {
		t.Error("ID accessors must return empty, non-nil slices")
	}
}

package deadlineadmit

import (
	"reflect"
	"testing"
)

// equalInts reports whether two ID slices hold the same values in the same
// order, treating nil and empty as equal.
func equalInts(a, b []int) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

// TestAdmit_MiddleDegradableShedNonDegradableMissRetained is the issue's first
// checkable step: three items whose middle (by deadline) is degradable and
// predicted to miss is shed, the EDF survivor order is [earliest, latest], and a
// non-degradable item that is also predicted to miss is retained (never
// silently dropped).
func TestAdmit_MiddleDegradableShedNonDegradableMissRetained(t *testing.T) {
	const now = 0
	items := []Item{
		// earliest deadline, on time, non-degradable -> survives
		{ID: 1, Deadline: 10, PredictedCost: 5, Degradable: false},
		// middle deadline, degradable, predicted miss (finish 100 > 20) -> shed
		{ID: 2, Deadline: 20, PredictedCost: 100, Degradable: true},
		// latest deadline, predicted miss (finish 100 > 30) but NON-degradable
		// -> retained
		{ID: 3, Deadline: 30, PredictedCost: 100, Degradable: false},
	}
	got := Admit(items, now, 1)

	wantOrder := []int{1, 3} // earliest, latest — the shed middle is gone
	if !equalInts(got.Order, wantOrder) {
		t.Errorf("Order = %v, want %v", got.Order, wantOrder)
	}
	wantShed := []int{2}
	if !equalInts(got.Shed, wantShed) {
		t.Errorf("Shed = %v, want %v", got.Shed, wantShed)
	}
}

// TestAdmit_Empty checks that empty input yields empty plan halves.
func TestAdmit_Empty(t *testing.T) {
	got := Admit(nil, 0, 1)
	if len(got.Order) != 0 {
		t.Errorf("Order = %v, want empty", got.Order)
	}
	if len(got.Shed) != 0 {
		t.Errorf("Shed = %v, want empty", got.Shed)
	}
}

// TestAdmit_AllWithinDeadline_ShedNothing checks that when every item is
// predicted to finish on time, nothing is shed and all are ordered by deadline.
func TestAdmit_AllWithinDeadline_ShedNothing(t *testing.T) {
	items := []Item{
		{ID: 1, Deadline: 30, PredictedCost: 5, Degradable: true},
		{ID: 2, Deadline: 10, PredictedCost: 5, Degradable: true},
		{ID: 3, Deadline: 20, PredictedCost: 5, Degradable: true},
	}
	got := Admit(items, 0, 1)
	wantOrder := []int{2, 3, 1} // deadlines 10,20,30
	if !equalInts(got.Order, wantOrder) {
		t.Errorf("Order = %v, want %v", got.Order, wantOrder)
	}
	if len(got.Shed) != 0 {
		t.Errorf("Shed = %v, want empty", got.Shed)
	}
}

// TestOrder_TiesBrokenByID checks that items sharing a deadline are ordered by
// ascending ID for determinism.
func TestOrder_TiesBrokenByID(t *testing.T) {
	items := []Item{
		{ID: 5, Deadline: 10, PredictedCost: 1},
		{ID: 2, Deadline: 10, PredictedCost: 1},
		{ID: 9, Deadline: 10, PredictedCost: 1},
	}
	got := Order(items)
	want := []int{2, 5, 9}
	if !equalInts(got, want) {
		t.Errorf("Order = %v, want %v", got, want)
	}
}

// TestOrder_DeterministicRegardlessOfInputOrder checks that permuting the input
// yields the identical dispatch order, so the policy is reproducible.
func TestOrder_DeterministicRegardlessOfInputOrder(t *testing.T) {
	a := []Item{
		{ID: 1, Deadline: 30, PredictedCost: 1},
		{ID: 2, Deadline: 10, PredictedCost: 1},
		{ID: 3, Deadline: 20, PredictedCost: 1},
		{ID: 4, Deadline: 10, PredictedCost: 1},
	}
	b := []Item{
		{ID: 4, Deadline: 10, PredictedCost: 1},
		{ID: 3, Deadline: 20, PredictedCost: 1},
		{ID: 1, Deadline: 30, PredictedCost: 1},
		{ID: 2, Deadline: 10, PredictedCost: 1},
	}
	gotA := Order(a)
	gotB := Order(b)
	want := []int{2, 4, 3, 1} // deadline 10 (ids 2,4), then 20, then 30
	if !equalInts(gotA, want) {
		t.Errorf("Order(a) = %v, want %v", gotA, want)
	}
	if !equalInts(gotB, want) {
		t.Errorf("Order(b) = %v, want %v", gotB, want)
	}
}

// TestOrder_DoesNotMutateInput checks that Order leaves the caller's slice
// untouched.
func TestOrder_DoesNotMutateInput(t *testing.T) {
	items := []Item{
		{ID: 1, Deadline: 30, PredictedCost: 1},
		{ID: 2, Deadline: 10, PredictedCost: 1},
	}
	before := make([]Item, len(items))
	copy(before, items)
	_ = Order(items)
	if !reflect.DeepEqual(items, before) {
		t.Errorf("input mutated: got %v, want %v", items, before)
	}
}

// TestAdmit_DegradableWithinDeadlineKept checks that a degradable item that is
// predicted to finish in time is kept, not shed.
func TestAdmit_DegradableWithinDeadlineKept(t *testing.T) {
	items := []Item{
		{ID: 1, Deadline: 100, PredictedCost: 10, Degradable: true}, // finish 10 <= 100
	}
	got := Admit(items, 0, 1)
	if !equalInts(got.Order, []int{1}) {
		t.Errorf("Order = %v, want [1]", got.Order)
	}
	if len(got.Shed) != 0 {
		t.Errorf("Shed = %v, want empty", got.Shed)
	}
}

// TestShed_DropThresholdBoundary checks the boundary: a degradable item whose
// predicted overshoot equals the threshold is shed, one just under is kept, and
// an on-time item is never shed even at a zero threshold.
func TestShed_DropThresholdBoundary(t *testing.T) {
	const now = 0
	const threshold = 5
	tests := []struct {
		name string
		item Item
		want bool // shed?
	}{
		// finish 25, deadline 20, overshoot 5 == threshold -> shed
		{"at threshold", Item{ID: 1, Deadline: 20, PredictedCost: 25, Degradable: true}, true},
		// finish 24, deadline 20, overshoot 4 < threshold -> kept
		{"just under threshold", Item{ID: 2, Deadline: 20, PredictedCost: 24, Degradable: true}, false},
		// finish 26, deadline 20, overshoot 6 > threshold -> shed
		{"over threshold", Item{ID: 3, Deadline: 20, PredictedCost: 26, Degradable: true}, true},
		// finish 20, deadline 20, exactly on time -> kept
		{"on time", Item{ID: 4, Deadline: 20, PredictedCost: 20, Degradable: true}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			shed := Shed([]Item{tc.item}, now, threshold)
			got := len(shed) == 1
			if got != tc.want {
				t.Errorf("shed=%v, want %v (shed slice %v)", got, tc.want, shed)
			}
		})
	}
}

// TestShed_OnTimeNotShedAtZeroThreshold checks that a zero threshold still does
// not shed an item predicted to finish exactly at its deadline; only a strictly
// positive overshoot qualifies.
func TestShed_OnTimeNotShedAtZeroThreshold(t *testing.T) {
	items := []Item{
		{ID: 1, Deadline: 10, PredictedCost: 10, Degradable: true}, // overshoot 0
		{ID: 2, Deadline: 10, PredictedCost: 11, Degradable: true}, // overshoot 1
	}
	got := Shed(items, 0, 0)
	if !equalInts(got, []int{2}) {
		t.Errorf("Shed = %v, want [2]", got)
	}
}

// TestAdmit_NonDegradableNeverShed checks that no non-degradable item is ever
// shed regardless of how badly it is predicted to miss, and that they remain in
// EDF order.
func TestAdmit_NonDegradableNeverShed(t *testing.T) {
	items := []Item{
		{ID: 1, Deadline: 5, PredictedCost: 1000, Degradable: false},
		{ID: 2, Deadline: 1, PredictedCost: 1000, Degradable: false},
	}
	got := Admit(items, 0, 1)
	if len(got.Shed) != 0 {
		t.Errorf("Shed = %v, want empty (non-degradable never shed)", got.Shed)
	}
	if !equalInts(got.Order, []int{2, 1}) {
		t.Errorf("Order = %v, want [2 1]", got.Order)
	}
}

// TestShed_OrderedByDeadline checks that multiple shed items are reported in
// earliest-deadline-first order.
func TestShed_OrderedByDeadline(t *testing.T) {
	items := []Item{
		{ID: 1, Deadline: 30, PredictedCost: 1000, Degradable: true},
		{ID: 2, Deadline: 10, PredictedCost: 1000, Degradable: true},
		{ID: 3, Deadline: 20, PredictedCost: 1000, Degradable: true},
	}
	got := Shed(items, 0, 1)
	if !equalInts(got, []int{2, 3, 1}) {
		t.Errorf("Shed = %v, want [2 3 1]", got)
	}
}

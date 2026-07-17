package dispatchtick

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
)

// agingTestNow is a fixed clock so the fold is deterministic (the leaf reads no real clock in tests).
const agingTestNow = int64(1_000_000)

func TestAgingFlagTruthyIsDefaultOff(t *testing.T) {
	on := []string{"1", "true", "TRUE", "yes", "on", " on ", "  1  ", "True"}
	off := []string{"", "0", "off", "no", "false", "disabled", "2", "enable-later"}
	for _, v := range on {
		if !agingFlagTruthy(v) {
			t.Errorf("agingFlagTruthy(%q) = false, want true", v)
		}
	}
	for _, v := range off {
		if agingFlagTruthy(v) {
			t.Errorf("agingFlagTruthy(%q) = true, want false (must be default-off)", v)
		}
	}
}

// TestAgingOrderNoWaitDataMatchesDefaultOrder is the second no-regression guarantee (see
// aging_order.go): with the flag ON but no ReadySince stamps, the aged order reproduces the default
// base-weight-then-recency order byte-for-byte, in both recency directions. So flipping the flag on a
// caller that has not started stamping ready times is a safe no-op, never a silent reshuffle.
func TestAgingOrderNoWaitDataMatchesDefaultOrder(t *testing.T) {
	cands := []LaneCandidate{
		{Number: 10, Weight: PriorityWeightDefault},
		{Number: 900, Weight: PriorityWeightP2},
		{Number: 50, Weight: PriorityWeightP0},
		{Number: 800, Weight: PriorityWeightP1},
		{Number: 40, Weight: PriorityWeightP1},
	}
	for _, preferNewest := range []bool{false, true} {
		want := defaultLaneScorers.Order(cands, preferNewest) // the flag-off path
		// Default params (aging fully tuned on) but zero ReadySince everywhere == all fresh, eff==base.
		got := orderLaneCandidatesAged(cands, preferNewest, dispatchaging.DefaultParams(agingTestNow))
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("preferNewest=%v: aged no-data order = %v, want %v (must match default order)", preferNewest, got, want)
		}
		// The zero-value Params (aging disabled) must also reproduce the default order.
		if gotZero := orderLaneCandidatesAged(cands, preferNewest, dispatchaging.Params{}); !reflect.DeepEqual(gotZero, want) {
			t.Fatalf("preferNewest=%v: aged zero-params order = %v, want %v", preferNewest, gotZero, want)
		}
	}
}

// TestAgingOrderSoftBoostLiftsWaitingUnit: a long-waiting default-tier issue earns enough soft aging
// boost to overtake a fresh P2 it would otherwise sit behind — the anti-starvation reorder the flag
// buys, without bypassing a still-heavier fresh P0.
func TestAgingOrderSoftBoostLiftsWaitingUnit(t *testing.T) {
	cands := []LaneCandidate{
		{Number: 50, Weight: PriorityWeightP0},                                       // fresh P0, eff 1000
		{Number: 900, Weight: PriorityWeightP2},                                      // fresh P2, eff 150
		{Number: 10, Weight: PriorityWeightDefault, ReadySince: agingTestNow - 1200}, // 2 intervals -> boost 120 -> eff 180
	}
	// Off path would be strict priority: [50, 900, 10]. Aging lifts #10 (eff 180) above the P2.
	got := orderLaneCandidatesAged(cands, false, dispatchaging.DefaultParams(agingTestNow))
	if want := []int{50, 10, 900}; !reflect.DeepEqual(got, want) {
		t.Fatalf("aged order = %v, want %v (waiting default lifts above fresh P2, still below fresh P0)", got, want)
	}
}

// TestAgingOrderStarvationForceServes: a unit waiting past the hard starvation deadline is served
// ahead of every non-starved unit this tick, even a fresh P0.
func TestAgingOrderStarvationForceServes(t *testing.T) {
	cands := []LaneCandidate{
		{Number: 50, Weight: PriorityWeightP0},                                        // fresh P0
		{Number: 10, Weight: PriorityWeightDefault, ReadySince: agingTestNow - 30000}, // > 6h -> starved
	}
	got := orderLaneCandidatesAged(cands, false, dispatchaging.DefaultParams(agingTestNow))
	if want := []int{10, 50}; !reflect.DeepEqual(got, want) {
		t.Fatalf("aged order = %v, want %v (starved unit force-served ahead of fresh P0)", got, want)
	}
}

// TestAgingOrderPreservesNumericRecencyTiebreak: units that tie on effective weight and wait break by
// the leaf's NUMERIC recency rule (not Fold's string-ID order), and preferNewest flips it.
func TestAgingOrderPreservesNumericRecencyTiebreak(t *testing.T) {
	cands := []LaneCandidate{
		{Number: 20, Weight: PriorityWeightDefault, ReadySince: agingTestNow - 1200},
		{Number: 5, Weight: PriorityWeightDefault, ReadySince: agingTestNow - 1200},
	}
	if got := orderLaneCandidatesAged(cands, false, dispatchaging.DefaultParams(agingTestNow)); !reflect.DeepEqual(got, []int{5, 20}) {
		t.Fatalf("aged tiebreak order = %v, want oldest-first [5 20]", got)
	}
	if got := orderLaneCandidatesAged(cands, true, dispatchaging.DefaultParams(agingTestNow)); !reflect.DeepEqual(got, []int{20, 5}) {
		t.Fatalf("aged prefer-newest tiebreak = %v, want newest-first [20 5]", got)
	}
}

package model

import (
	"errors"
	"math"
	"testing"
)

// placement_delta_test.go — phase-3 witness for the CONTINUAL-ENGINE PlacementDelta +
// rebalance cadence (ticket continual-engine-placement-delta-cadence-20261104): the
// closed-kind delta grammar, the quarter-first-fire / steady-interval / in-flight
// cadence gate, and the dry-run → apply → receipt loop over ActiveRankMask membership.
// Pure Go, no GPU, no wall clock — every tick is an explicit step argument. [SW-VERIFIED]

// TestRebalanceCadenceFirstFireAtQuarter is the REQUIRED ticket witness: with
// FirstFireFraction=0.25 the first fire is denied TooEarly at step < horizon/4 and
// allowed at >= horizon/4; after the first fire the default steady cadence (the same
// quarter period) denies TooSoon until the interval elapses.
func TestRebalanceCadenceFirstFireAtQuarter(t *testing.T) {
	c, err := NewRebalanceCadence(0.25, 0)
	if err != nil {
		t.Fatalf("NewRebalanceCadence: %v", err)
	}
	const horizon = 100 // threshold = exactly 25

	for step := 0; step < 25; step++ {
		ok, reason := c.Allow(step, horizon)
		if ok || reason != CadenceTooEarly {
			t.Fatalf("step %d: Allow = (%v, %v), want (false, TooEarly)", step, ok, reason)
		}
	}
	if ok, reason := c.Allow(25, horizon); !ok || reason != CadenceOK {
		t.Fatalf("step 25: Allow = (%v, %v), want (true, OK)", ok, reason)
	}

	// Fire the first rebalance; the steady quarter-period cadence then gates the rest.
	if err := c.Begin(); err != nil {
		t.Fatalf("Begin: %v", err)
	}
	if err := c.Complete(25); err != nil {
		t.Fatalf("Complete(25): %v", err)
	}
	for step := 26; step < 50; step++ {
		ok, reason := c.Allow(step, horizon)
		if ok || reason != CadenceTooSoon {
			t.Fatalf("step %d: Allow = (%v, %v), want (false, TooSoon)", step, ok, reason)
		}
	}
	if ok, reason := c.Allow(50, horizon); !ok || reason != CadenceOK {
		t.Fatalf("step 50: Allow = (%v, %v), want (true, OK)", ok, reason)
	}
}

// TestRebalanceCadenceSteadyIntervalGate pins an EXPLICIT MinIntervalSteps: fires are
// spaced by exactly that many steps, independent of the first-fire fraction.
func TestRebalanceCadenceSteadyIntervalGate(t *testing.T) {
	c, err := NewRebalanceCadence(0.25, 30)
	if err != nil {
		t.Fatalf("NewRebalanceCadence: %v", err)
	}
	const horizon = 200 // first-fire threshold = 50

	if ok, reason := c.Allow(49, horizon); ok || reason != CadenceTooEarly {
		t.Fatalf("step 49: Allow = (%v, %v), want (false, TooEarly)", ok, reason)
	}
	if ok, reason := c.Allow(50, horizon); !ok || reason != CadenceOK {
		t.Fatalf("step 50: Allow = (%v, %v), want (true, OK)", ok, reason)
	}
	if err := c.Begin(); err != nil {
		t.Fatalf("Begin: %v", err)
	}
	if err := c.Complete(50); err != nil {
		t.Fatalf("Complete(50): %v", err)
	}

	for step := 51; step < 80; step++ {
		if ok, reason := c.Allow(step, horizon); ok || reason != CadenceTooSoon {
			t.Fatalf("step %d: Allow = (%v, %v), want (false, TooSoon)", step, ok, reason)
		}
	}
	if ok, reason := c.Allow(80, horizon); !ok || reason != CadenceOK {
		t.Fatalf("step 80: Allow = (%v, %v), want (true, OK)", ok, reason)
	}
	if err := c.Begin(); err != nil {
		t.Fatalf("Begin: %v", err)
	}
	if err := c.Complete(80); err != nil {
		t.Fatalf("Complete(80): %v", err)
	}
	if ok, reason := c.Allow(109, horizon); ok || reason != CadenceTooSoon {
		t.Fatalf("step 109: Allow = (%v, %v), want (false, TooSoon)", ok, reason)
	}
	if ok, reason := c.Allow(110, horizon); !ok || reason != CadenceOK {
		t.Fatalf("step 110: Allow = (%v, %v), want (true, OK)", ok, reason)
	}
}

// TestRebalanceCadenceInFlightGate pins the in-flight guard: once a rebalance is
// proposed (Begin), every tick denies InFlight — even past the threshold and interval —
// until the proposal is applied or discarded. An abort consumes no cadence.
func TestRebalanceCadenceInFlightGate(t *testing.T) {
	c := DefaultRebalanceCadence()
	const horizon = 100

	if err := c.Begin(); err != nil {
		t.Fatalf("Begin: %v", err)
	}
	for _, step := range []int{25, 50, 99} {
		if ok, reason := c.Allow(step, horizon); ok || reason != CadenceInFlight {
			t.Fatalf("in flight, step %d: Allow = (%v, %v), want (false, InFlight)", step, ok, reason)
		}
	}
	if err := c.Abort(); err != nil {
		t.Fatalf("Abort: %v", err)
	}
	// The aborted proposal consumed nothing: the first fire is still gated only on the
	// quarter threshold.
	if ok, reason := c.Allow(25, horizon); !ok || reason != CadenceOK {
		t.Fatalf("after abort, step 25: Allow = (%v, %v), want (true, OK)", ok, reason)
	}
}

// TestRebalanceCadenceThresholdCeilingOnOddHorizon pins the fail-LATER direction on a
// non-divisible horizon: 0.25 × 101 = 25.25 → threshold 26, so step 25 stays denied.
func TestRebalanceCadenceThresholdCeilingOnOddHorizon(t *testing.T) {
	c, err := NewRebalanceCadence(0.25, 0)
	if err != nil {
		t.Fatalf("NewRebalanceCadence: %v", err)
	}
	if got := c.FirstFireThreshold(101); got != 26 {
		t.Fatalf("FirstFireThreshold(101) = %d, want 26", got)
	}
	if ok, reason := c.Allow(25, 101); ok || reason != CadenceTooEarly {
		t.Fatalf("step 25 of 101: Allow = (%v, %v), want (false, TooEarly)", ok, reason)
	}
	if ok, reason := c.Allow(26, 101); !ok || reason != CadenceOK {
		t.Fatalf("step 26 of 101: Allow = (%v, %v), want (true, OK)", ok, reason)
	}
}

// TestRebalanceCadenceMalformedTicksFailClosed pins that garbage ticks never allow:
// negative steps and non-positive horizons deny TooEarly instead of firing.
func TestRebalanceCadenceMalformedTicksFailClosed(t *testing.T) {
	c := DefaultRebalanceCadence()
	cases := []struct{ step, horizon int }{
		{-1, 100},
		{0, 0},
		{0, -5},
		{-3, -1},
	}
	for _, tc := range cases {
		if ok, reason := c.Allow(tc.step, tc.horizon); ok || reason != CadenceTooEarly {
			t.Fatalf("Allow(%d, %d) = (%v, %v), want (false, TooEarly)", tc.step, tc.horizon, ok, reason)
		}
	}
}

// TestRebalanceCadenceConstructorValidation pins the knob bounds: fraction in (0,1],
// interval >= 0, and the shipped default.
func TestRebalanceCadenceConstructorValidation(t *testing.T) {
	badFractions := []float64{0, -0.5, 1.5, math.NaN(), math.Inf(1)}
	for _, f := range badFractions {
		if _, err := NewRebalanceCadence(f, 0); err == nil {
			t.Fatalf("NewRebalanceCadence(%g, 0) = nil error, want fail-closed", f)
		}
	}
	if _, err := NewRebalanceCadence(0.25, -1); err == nil {
		t.Fatalf("NewRebalanceCadence(0.25, -1) = nil error, want fail-closed")
	}
	for _, f := range []float64{0.25, 1.0} {
		if _, err := NewRebalanceCadence(f, 0); err != nil {
			t.Fatalf("NewRebalanceCadence(%g, 0) = %v, want accepted", f, err)
		}
	}
	d := DefaultRebalanceCadence()
	if d.FirstFireFraction != 0.25 || d.MinIntervalSteps != 0 {
		t.Fatalf("DefaultRebalanceCadence = (%g, %d), want (0.25, 0)", d.FirstFireFraction, d.MinIntervalSteps)
	}
	if got := d.FirstFireThreshold(100); got != 25 {
		t.Fatalf("default FirstFireThreshold(100) = %d, want 25", got)
	}
	if got := d.SteadyInterval(100); got != 25 {
		t.Fatalf("default SteadyInterval(100) = %d, want 25 (quarter period)", got)
	}
}

// TestRebalanceCadenceStateTransitionsFailClosed pins the paired-transition contract:
// Complete/Abort with nothing in flight, a double Begin, and a negative fire step all
// refuse with ErrRebalanceState.
func TestRebalanceCadenceStateTransitionsFailClosed(t *testing.T) {
	c := DefaultRebalanceCadence()
	if err := c.Complete(1); !errors.Is(err, ErrRebalanceState) {
		t.Fatalf("Complete with nothing in flight: err = %v, want ErrRebalanceState", err)
	}
	if err := c.Abort(); !errors.Is(err, ErrRebalanceState) {
		t.Fatalf("Abort with nothing in flight: err = %v, want ErrRebalanceState", err)
	}
	if err := c.Begin(); err != nil {
		t.Fatalf("Begin: %v", err)
	}
	if err := c.Begin(); !errors.Is(err, ErrRebalanceState) {
		t.Fatalf("double Begin: err = %v, want ErrRebalanceState", err)
	}
	if err := c.Complete(-1); !errors.Is(err, ErrRebalanceState) {
		t.Fatalf("Complete(-1): err = %v, want ErrRebalanceState", err)
	}
	if err := c.Complete(1); err != nil {
		t.Fatalf("Complete(1): %v", err)
	}
}

// TestPlacementDeltaKindsClosedSet pins the closed kind set: all three kinds construct
// with consistent payloads, and any other kind value fails closed with the sentinel
// ErrPlacementDeltaKind — both through the constructor and through Validate on a
// hand-built literal.
func TestPlacementDeltaKindsClosedSet(t *testing.T) {
	all, err := AllActive(4)
	if err != nil {
		t.Fatalf("AllActive: %v", err)
	}
	partial, err := NewActiveRankMask(4, []int{0, 1})
	if err != nil {
		t.Fatalf("NewActiveRankMask: %v", err)
	}
	plan, err := NewTPPlan(8, 4)
	if err != nil {
		t.Fatalf("NewTPPlan: %v", err)
	}

	rerank, err := NewPlacementDelta(PlacementReRank, all, all, plan, "test re-rank")
	if err != nil {
		t.Fatalf("NewPlacementDelta(re-rank): %v", err)
	}
	if rerank.Kind != PlacementReRank || !plansEqual(rerank.Plan, plan) {
		t.Fatalf("re-rank delta = %+v, want kind re-rank carrying the plan", rerank)
	}
	promote, err := NewPlacementDelta(PlacementPromote, partial, all, TPPlan{}, "recover ranks")
	if err != nil {
		t.Fatalf("NewPlacementDelta(promote): %v", err)
	}
	if promote.Kind != PlacementPromote || !plansEqual(promote.Plan, TPPlan{}) {
		t.Fatalf("promote delta = %+v, want kind promote with zero plan", promote)
	}
	demote, err := NewPlacementDelta(PlacementDemote, all, partial, TPPlan{}, "fail ranks")
	if err != nil {
		t.Fatalf("NewPlacementDelta(demote): %v", err)
	}
	if demote.Kind != PlacementDemote {
		t.Fatalf("demote delta kind = %v, want demote", demote.Kind)
	}

	const bogusKind = PlacementDeltaKind(99)
	if _, err := NewPlacementDelta(bogusKind, all, all, plan, ""); !errors.Is(err, ErrPlacementDeltaKind) {
		t.Fatalf("NewPlacementDelta(kind 99): err = %v, want ErrPlacementDeltaKind", err)
	}
	literal := PlacementDelta{Kind: 42, From: all, To: all, Plan: plan}
	if err := literal.Validate(); !errors.Is(err, ErrPlacementDeltaKind) {
		t.Fatalf("Validate(kind 42 literal): err = %v, want ErrPlacementDeltaKind", err)
	}
}

// TestPlacementDeltaKindPayloadConsistency pins each kind's payload contract: re-rank
// keeps membership fixed and carries a valid world-sized plan; promote is a strict
// superset move; demote a strict subset; anything else refuses with ErrPlacementDelta.
func TestPlacementDeltaKindPayloadConsistency(t *testing.T) {
	all, _ := AllActive(4)
	two, _ := NewActiveRankMask(4, []int{0, 1})
	world3, _ := NewActiveRankMask(3, []int{0, 1}) // same live set, different world
	plan4, _ := NewTPPlan(8, 4)
	plan2, _ := NewTPPlan(8, 2) // valid plan, wrong shard count for a world of 4

	cases := []struct {
		name string
		d    PlacementDelta
	}{
		{"re-rank with changed membership", PlacementDelta{Kind: PlacementReRank, From: two, To: all, Plan: plan4}},
		{"re-rank with zero plan", PlacementDelta{Kind: PlacementReRank, From: all, To: all, Plan: TPPlan{}}},
		{"re-rank plan shard count != world", PlacementDelta{Kind: PlacementReRank, From: all, To: all, Plan: plan2}},
		{"promote that is actually a demote", PlacementDelta{Kind: PlacementPromote, From: all, To: two}},
		{"promote with no change", PlacementDelta{Kind: PlacementPromote, From: two, To: two}},
		{"demote that is actually a promote", PlacementDelta{Kind: PlacementDemote, From: two, To: all}},
		{"world size mismatch", PlacementDelta{Kind: PlacementPromote, From: world3, To: all}},
		{"zero-value From", PlacementDelta{Kind: PlacementPromote, From: ActiveRankMask{}, To: all}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.d.Validate(); !errors.Is(err, ErrPlacementDelta) {
				t.Fatalf("Validate(%s): err = %v, want ErrPlacementDelta", tc.name, err)
			}
		})
	}
}

// TestPlacementDeltaRanksDerived pins that Ranks() is derived from From/To/Plan and
// never stored: promote lists the recovered ranks, demote the failed ones, re-rank
// every rank the new partition re-declares.
func TestPlacementDeltaRanksDerived(t *testing.T) {
	all, _ := AllActive(4)
	two, _ := NewActiveRankMask(4, []int{0, 1})
	odd, _ := NewActiveRankMask(4, []int{0, 2})
	plan, _ := NewTPPlan(8, 4)

	promote, err := NewPlacementDelta(PlacementPromote, two, all, TPPlan{}, "")
	if err != nil {
		t.Fatalf("NewPlacementDelta(promote): %v", err)
	}
	if got := promote.Ranks(); len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Fatalf("promote Ranks = %v, want [2 3]", got)
	}
	demote, err := NewPlacementDelta(PlacementDemote, all, odd, TPPlan{}, "")
	if err != nil {
		t.Fatalf("NewPlacementDelta(demote): %v", err)
	}
	if got := demote.Ranks(); len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Fatalf("demote Ranks = %v, want [1 3]", got)
	}
	rerank, err := NewPlacementDelta(PlacementReRank, all, all, plan, "")
	if err != nil {
		t.Fatalf("NewPlacementDelta(re-rank): %v", err)
	}
	if got := rerank.Ranks(); len(got) != 4 || got[0] != 0 || got[3] != 3 {
		t.Fatalf("re-rank Ranks = %v, want [0 1 2 3]", got)
	}
}

// skewedLoad is the shared test load vector: four hot experts then four cold, the same
// skew expert_balance_test.go uses to prove the balanced plan beats count-even bands.
func skewedLoad() []int { return []int{10, 10, 10, 10, 1, 1, 1, 1} }

// TestPlacementDeltaSourcePlanDryRunChangesNothing pins the dry-run contract: Plan
// proposes (the store goes in flight) but changes nothing else — generation, mask, and
// live plan are untouched until Apply.
func TestPlacementDeltaSourcePlanDryRunChangesNothing(t *testing.T) {
	mask, _ := AllActive(2)
	initial, _ := ExpertParallelPlan(8, 2) // count-even bands: all hot experts on rank 0
	store, err := NewPlacementStore(mask, initial)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())
	gen0 := store.Generation()

	d, changed, err := src.Plan(25, 100, skewedLoad())
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if !changed {
		t.Fatalf("Plan: changed = false, want a real re-rank delta")
	}
	if d.Kind != PlacementReRank {
		t.Fatalf("delta kind = %v, want re-rank", d.Kind)
	}
	if store.Generation() != gen0 {
		t.Fatalf("dry run bumped generation %d -> %d", gen0, store.Generation())
	}
	if !sameMembership(store.Mask(), mask) {
		t.Fatalf("dry run changed the mask: %v", store.Mask().ActiveRanks())
	}
	if !plansEqual(store.Plan(), initial) {
		t.Fatalf("dry run swapped the live plan: %v", store.Plan().Shards)
	}
	if store.Pending() == nil {
		t.Fatalf("dry run did not mark the delta in flight")
	}
	// The cadence is in flight too: the gate now denies InFlight.
	if ok, reason := src.cadence.Allow(25, 100); ok || reason != CadenceInFlight {
		t.Fatalf("after Plan, Allow = (%v, %v), want (false, InFlight)", ok, reason)
	}
}

// TestPlacementDeltaSourceApplyBumpsGenerationOnce pins the apply contract: exactly one
// generation bump per applied delta, the receipt stamps Generation and AppliedAt, the
// live placement swaps to the recommended banding, and the pending slot clears.
func TestPlacementDeltaSourceApplyBumpsGenerationOnce(t *testing.T) {
	mask, _ := AllActive(2)
	initial, _ := ExpertParallelPlan(8, 2)
	store, err := NewPlacementStore(mask, initial)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())
	load := skewedLoad()

	wantPlan, err := LoadBalancedExpertBands(load, 2)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	if _, changed, err := src.Plan(25, 100, load); err != nil || !changed {
		t.Fatalf("Plan: changed=%v err=%v, want a real delta", changed, err)
	}

	receipt, err := src.Apply(25, load)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if store.Generation() != 1 {
		t.Fatalf("generation after one apply = %d, want exactly 1", store.Generation())
	}
	if receipt.Delta.Generation != 1 || receipt.Step != 25 || receipt.Delta.AppliedAt != 25 {
		t.Fatalf("receipt stamps = (gen %d, step %d, appliedAt %d), want (1, 25, 25)",
			receipt.Delta.Generation, receipt.Step, receipt.Delta.AppliedAt)
	}
	if !plansEqual(store.Plan(), wantPlan) {
		t.Fatalf("live plan after apply = %v, want the load-balanced banding %v", store.Plan().Shards, wantPlan.Shards)
	}
	if store.Pending() != nil {
		t.Fatalf("apply left a delta pending")
	}
	if store.LastReceipt() == nil || store.LastReceipt().Step != 25 {
		t.Fatalf("LastReceipt = %+v, want the applied receipt at step 25", store.LastReceipt())
	}
	// Nothing left to apply: a second apply must refuse, not double-bump.
	if _, err := src.Apply(26, load); !errors.Is(err, ErrPlacementDelta) {
		t.Fatalf("second Apply: err = %v, want ErrPlacementDelta", err)
	}
	if store.Generation() != 1 {
		t.Fatalf("refused apply bumped generation to %d", store.Generation())
	}
}

// TestPlacementDeltaSourceCASBaseMismatch pins the fail-closed CAS guard: a delta whose
// From is not the live mask is refused at Propose — a stale snapshot is refused, not
// merged.
func TestPlacementDeltaSourceCASBaseMismatch(t *testing.T) {
	live, _ := AllActive(4)
	staleFrom, _ := NewActiveRankMask(4, []int{0, 1, 2})
	plan, _ := NewTPPlan(8, 4)
	store, err := NewPlacementStore(live, plan)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	d, err := NewPlacementDelta(PlacementPromote, staleFrom, live, TPPlan{}, "built against a stale snapshot")
	if err != nil {
		t.Fatalf("NewPlacementDelta: %v", err)
	}
	if err := store.Propose(d); !errors.Is(err, ErrPlacementDelta) {
		t.Fatalf("Propose(stale From): err = %v, want ErrPlacementDelta", err)
	}
	if store.Pending() != nil {
		t.Fatalf("refused proposal left a pending delta")
	}
}

// TestPlacementDeltaSourceAbortClearsInFlight pins the abort path: discarding the
// pending delta releases both the store's pending slot and the cadence's in-flight
// mark without consuming cadence or bumping generation; a double proposal while in
// flight is denied InFlight.
func TestPlacementDeltaSourceAbortClearsInFlight(t *testing.T) {
	mask, _ := AllActive(2)
	initial, _ := ExpertParallelPlan(8, 2)
	store, err := NewPlacementStore(mask, initial)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())
	load := skewedLoad()

	if _, changed, err := src.Plan(25, 100, load); err != nil || !changed {
		t.Fatalf("Plan: changed=%v err=%v", changed, err)
	}
	// A second proposal while in flight is denied by the gate, not stacked.
	if _, _, err := src.Plan(25, 100, load); !errors.Is(err, ErrRebalanceDenied) {
		t.Fatalf("double Plan: err = %v, want ErrRebalanceDenied", err)
	} else if reason, ok := DenialReason(err); !ok || reason != CadenceInFlight {
		t.Fatalf("double Plan reason = (%v, %v), want InFlight", reason, ok)
	}

	if err := src.Abort(); err != nil {
		t.Fatalf("Abort: %v", err)
	}
	if store.Pending() != nil {
		t.Fatalf("abort left a pending delta")
	}
	if store.Generation() != 0 {
		t.Fatalf("abort bumped generation to %d", store.Generation())
	}
	// No cadence consumed: the first fire is still gated only on the quarter threshold.
	if ok, reason := src.cadence.Allow(25, 100); !ok || reason != CadenceOK {
		t.Fatalf("after abort, Allow = (%v, %v), want (true, OK)", ok, reason)
	}
	if err := src.Abort(); !errors.Is(err, ErrRebalanceState) {
		t.Fatalf("Abort with nothing pending: err = %v, want ErrRebalanceState", err)
	}
}

// TestPlacementDeltaSourceNoOpConsumesNoCadence pins the anti-churn property: when the
// live plan is already load-optimal, Plan proposes nothing, marks nothing, and consumes
// no cadence — the engine does not rebalance a balanced workload.
func TestPlacementDeltaSourceNoOpConsumesNoCadence(t *testing.T) {
	flat := []int{4, 4, 4, 4, 4, 4}
	optimal, err := LoadBalancedExpertBands(flat, 3)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	mask, _ := AllActive(3)
	store, err := NewPlacementStore(mask, optimal)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())

	if _, changed, err := src.Plan(25, 100, flat); err != nil || changed {
		t.Fatalf("Plan on already-optimal plan: changed=%v err=%v, want (false, nil)", changed, err)
	}
	if store.Pending() != nil {
		t.Fatalf("no-op Plan marked a delta in flight")
	}
	// Same tick still allowed: nothing fired, so the gate state is unchanged.
	if _, changed, err := src.Plan(25, 100, flat); err != nil || changed {
		t.Fatalf("repeat no-op Plan: changed=%v err=%v", changed, err)
	}
	// A real drift still fires at the same tick once the load skews.
	skewed := []int{9, 9, 9, 1, 1, 1}
	if _, changed, err := src.Plan(25, 100, skewed); err != nil || !changed {
		t.Fatalf("Plan on skewed load: changed=%v err=%v, want a real delta", changed, err)
	}
}

// TestPlacementDeltaSourceGatedByCadence pins the gate in front of the recommendation:
// before the quarter the source refuses with the closed-vocabulary denial and proposes
// nothing; at the quarter it proceeds.
func TestPlacementDeltaSourceGatedByCadence(t *testing.T) {
	mask, _ := AllActive(2)
	initial, _ := ExpertParallelPlan(8, 2)
	store, err := NewPlacementStore(mask, initial)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())
	load := skewedLoad()

	if _, _, err := src.Plan(10, 100, load); !errors.Is(err, ErrRebalanceDenied) {
		t.Fatalf("Plan at step 10: err = %v, want ErrRebalanceDenied", err)
	} else if reason, ok := DenialReason(err); !ok || reason != CadenceTooEarly {
		t.Fatalf("denial reason = (%v, %v), want TooEarly", reason, ok)
	}
	if store.Pending() != nil {
		t.Fatalf("denied Plan proposed a delta anyway")
	}

	if _, changed, err := src.Plan(25, 100, load); err != nil || !changed {
		t.Fatalf("Plan at step 25: changed=%v err=%v, want a real delta", changed, err)
	}
}

// TestPlacementDeltaSourcePromoteDemoteRoundTrip pins the membership-move path: a rank
// recovery promotes (mask widens), a later failure demotes (mask narrows), each through
// one gated, generation-stamped delta; the expert→rank plan is untouched by either.
func TestPlacementDeltaSourcePromoteDemoteRoundTrip(t *testing.T) {
	three, _ := NewActiveRankMask(4, []int{0, 1, 2}) // rank 3 dead
	plan, _ := NewTPPlan(8, 4)
	store, err := NewPlacementStore(three, plan)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())

	d, changed, err := src.PlanMembership(25, 100, []int{0, 1, 2, 3})
	if err != nil || !changed {
		t.Fatalf("PlanMembership(recover): changed=%v err=%v", changed, err)
	}
	if d.Kind != PlacementPromote {
		t.Fatalf("kind = %v, want promote", d.Kind)
	}
	if got := d.Ranks(); len(got) != 1 || got[0] != 3 {
		t.Fatalf("promote Ranks = %v, want [3]", got)
	}
	r, err := src.Apply(25, nil) // no load: membership moves are not scored on expert load
	if err != nil {
		t.Fatalf("Apply(promote): %v", err)
	}
	if got := store.Mask().ActiveRanks(); len(got) != 4 || got[3] != 3 {
		t.Fatalf("mask after promote = %v, want all four ranks live", got)
	}
	if store.Generation() != 1 || r.Delta.Generation != 1 {
		t.Fatalf("generation = store %d / receipt %d, want 1", store.Generation(), r.Delta.Generation)
	}
	if r.BalancedBefore != nil || r.BalancedAfter != nil {
		t.Fatalf("receipt balancedness = (%v, %v), want both undefined without a load", r.BalancedBefore, r.BalancedAfter)
	}
	if !plansEqual(store.Plan(), plan) {
		t.Fatalf("membership move touched the expert plan: %v", store.Plan().Shards)
	}

	// Rank 1 fails next: a strict subset is a demote, gated by the steady cadence
	// (step 50 - 25 = 25 = the quarter-period interval).
	d, changed, err = src.PlanMembership(50, 100, []int{0, 2, 3})
	if err != nil || !changed {
		t.Fatalf("PlanMembership(fail): changed=%v err=%v", changed, err)
	}
	if d.Kind != PlacementDemote {
		t.Fatalf("kind = %v, want demote", d.Kind)
	}
	if got := d.Ranks(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("demote Ranks = %v, want [1]", got)
	}
	if _, err := src.Apply(50, nil); err != nil {
		t.Fatalf("Apply(demote): %v", err)
	}
	if got := store.Mask().ActiveRanks(); len(got) != 3 || got[0] != 0 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("mask after demote = %v, want [0 2 3]", got)
	}
	if store.Generation() != 2 {
		t.Fatalf("generation after two applies = %d, want 2", store.Generation())
	}
}

// TestPlacementDeltaReceiptBalancedness pins the phase-3 witness: an applied re-rank's
// receipt carries the balancedness before and after, scored against the measured load —
// and the load-balanced banding strictly improves it over the count-even incumbent.
func TestPlacementDeltaReceiptBalancedness(t *testing.T) {
	mask, _ := AllActive(2)
	initial, _ := ExpertParallelPlan(8, 2)
	store, err := NewPlacementStore(mask, initial)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())
	load := skewedLoad()

	if _, changed, err := src.Plan(25, 100, load); err != nil || !changed {
		t.Fatalf("Plan: changed=%v err=%v", changed, err)
	}
	receipt, err := src.Apply(25, load)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if receipt.BalancedBefore == nil || receipt.BalancedAfter == nil {
		t.Fatalf("receipt balancedness = (%v, %v), want both defined for routed traffic",
			receipt.BalancedBefore, receipt.BalancedAfter)
	}
	if !(*receipt.BalancedAfter > *receipt.BalancedBefore) {
		t.Fatalf("balancedness after %.4f not above before %.4f", *receipt.BalancedAfter, *receipt.BalancedBefore)
	}
	t.Logf("balancedness before=%.4f after=%.4f (max-per-rank load flattened)",
		*receipt.BalancedBefore, *receipt.BalancedAfter)
}

// TestPlacementDeltaReceiptUndefinedOnZeroTraffic pins the honest-metric rule: with no
// routed traffic the apply still succeeds but the receipt leaves both metrics undefined
// (nil) rather than fabricating a 1.0.
func TestPlacementDeltaReceiptUndefinedOnZeroTraffic(t *testing.T) {
	mask, _ := AllActive(2)
	initial, _ := ExpertParallelPlan(8, 2)
	store, err := NewPlacementStore(mask, initial)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())
	zero := []int{0, 0, 0, 0, 0, 0, 0, 0}

	// The zero-load optimum differs from the count-even bands, so Plan proposes.
	if _, changed, err := src.Plan(25, 100, zero); err != nil || !changed {
		t.Fatalf("Plan on zero load: changed=%v err=%v, want a proposed delta", changed, err)
	}
	receipt, err := src.Apply(25, zero)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if receipt.BalancedBefore != nil || receipt.BalancedAfter != nil {
		t.Fatalf("zero-traffic receipt = (%v, %v), want both undefined", receipt.BalancedBefore, receipt.BalancedAfter)
	}
}

// TestPlacementDeltaMixedMembershipRefused pins the one-delta-one-change rule: a target
// membership that both recovers and loses ranks is neither a pure promote nor a pure
// demote, so it is refused fail-closed with nothing proposed.
func TestPlacementDeltaMixedMembershipRefused(t *testing.T) {
	three, _ := NewActiveRankMask(4, []int{0, 1, 2})
	plan, _ := NewTPPlan(8, 4)
	store, err := NewPlacementStore(three, plan)
	if err != nil {
		t.Fatalf("NewPlacementStore: %v", err)
	}
	src := NewPlacementDeltaSource(store, DefaultRebalanceCadence())

	_, changed, err := src.PlanMembership(25, 100, []int{1, 2, 3}) // rank 0 out, rank 3 in
	if err == nil {
		t.Fatalf("mixed membership change: err = nil, want fail-closed")
	}
	if !errors.Is(err, ErrPlacementDelta) {
		t.Fatalf("mixed membership change: err = %v, want ErrPlacementDelta", err)
	}
	if changed || store.Pending() != nil {
		t.Fatalf("refused proposal left state behind (changed=%v pending=%v)", changed, store.Pending())
	}
}

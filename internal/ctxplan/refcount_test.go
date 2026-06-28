package ctxplan

import (
	"reflect"
	"testing"
)

// TestRefcount_FalseFree_FaultWhileGoalLive is the acceptance test for the false-free arm:
// a span that Faults (is paged out) while the live goal still needs it (it is in the goal's
// reachable set) is flagged false-free — the regression sentinel the goal-as-pin-root child
// (#845) is meant to drive to zero.
func TestRefcount_FalseFree_FaultWhileGoalLive(t *testing.T) {
	rw := NewRefcountWitness(falseRetainK)
	// The plan kept "other" resident but ELIDED "goalspan" — and the turn then faulted it.
	p := Plan{
		Selected: []Selection{sel("other", 100, false)},
		Elided:   []Elision{{ID: "goalspan", Cost: 200}},
	}
	o := Outcome{Faults: []string{"goalspan"}}
	goal := LiveGoal{Active: true, Reachable: []string{"goalspan", "another"}}

	rc := rw.Observe(p, o, goal)

	if got := rc.FalseFree; !reflect.DeepEqual(got, []string{"goalspan"}) {
		t.Fatalf("false-free = %v, want [goalspan]", got)
	}
	if len(rc.FalseRetain) != 0 {
		t.Errorf("false-retain = %v, want empty (this is the false-free axis)", rc.FalseRetain)
	}
	if !rc.Any() {
		t.Error("Any() = false, want true (a false-free fired)")
	}
}

// TestRefcount_NoFalseFree_WhenGoalReachable proves the target state: when the live goal's
// reachable set stays RESIDENT (the keystone #845 pinned it), a fault on some OTHER span is
// NOT a false-free. False-free is structurally tied to the goal's reachable set, not to
// faults in general.
func TestRefcount_NoFalseFree_WhenGoalReachable(t *testing.T) {
	rw := NewRefcountWitness(falseRetainK)
	p := Plan{
		Selected: []Selection{sel("goalspan", 200, true)}, // goal span pinned resident
		Elided:   []Elision{{ID: "scratch", Cost: 50}},
	}
	o := Outcome{Faults: []string{"scratch"}} // a non-goal span faulted — not the goal's loss
	goal := LiveGoal{Active: true, Reachable: []string{"goalspan"}}

	rc := rw.Observe(p, o, goal)

	if len(rc.FalseFree) != 0 {
		t.Errorf("false-free = %v, want empty (goal's reachable set stayed resident)", rc.FalseFree)
	}
}

// TestRefcount_NoFalseFree_WhenNoGoal proves the false-free arm is inert without a live goal
// — an offline / goal-less session is a clean no-op on this axis even when spans fault.
func TestRefcount_NoFalseFree_WhenNoGoal(t *testing.T) {
	rw := NewRefcountWitness(falseRetainK)
	p := Plan{Elided: []Elision{{ID: "x", Cost: 10}}}
	o := Outcome{Faults: []string{"x"}}

	if rc := rw.Observe(p, o, LiveGoal{}); rc.Any() {
		t.Errorf("verdict = %+v, want empty (no live goal => no false-free)", rc)
	}
	// Active goal but empty reachable set is also inert.
	if rc := rw.Observe(p, o, LiveGoal{Active: true}); len(rc.FalseFree) != 0 {
		t.Errorf("false-free = %v, want empty (no reachable ids)", rc.FalseFree)
	}
}

// TestRefcount_FalseRetain_PinnedWastedKTurns is the acceptance test for the false-retain
// arm: a pinned span that is Wasted (resident-untouched) for K consecutive turns emits the
// advisory down-weight — and the pin is NOT freed (the plan's Pinned flag is untouched; the
// witness only reports). Below K it does not fire; at K it does.
func TestRefcount_FalseRetain_PinnedWastedKTurns(t *testing.T) {
	const k = 3
	rw := NewRefcountWitness(k)
	// A pinned span that the turn never touches, every turn.
	p := Plan{Selected: []Selection{sel("pin", 500, true)}}
	o := Outcome{Wasted: []string{"pin"}}

	// Turns 1..K-1: idle but NOT yet flagged (streak building).
	for turn := 1; turn < k; turn++ {
		rc := rw.Observe(p, o, LiveGoal{})
		if len(rc.FalseRetain) != 0 {
			t.Fatalf("turn %d: false-retain = %v, want empty before K turns", turn, rc.FalseRetain)
		}
		if got := rw.Streak("pin"); got != turn {
			t.Fatalf("turn %d: streak = %d, want %d", turn, got, turn)
		}
	}

	// Turn K: the streak reaches K -> flagged false-retain.
	rc := rw.Observe(p, o, LiveGoal{})
	if !reflect.DeepEqual(rc.FalseRetain, []string{"pin"}) {
		t.Fatalf("turn %d: false-retain = %v, want [pin]", k, rc.FalseRetain)
	}

	// CRITICAL: the pin was NOT freed — the plan's span is still Pinned. The witness reports;
	// it never mutates the plan or frees the root (that is the discharge child's job, #847).
	if !p.Selected[0].Pinned {
		t.Error("the pin was freed — false-retain must be advisory only, never auto-free")
	}
}

// TestRefcount_FalseRetain_ResetOnHit proves the streak counts CONSECUTIVE idle turns: a pin
// that idles, gets USED (Hit), then idles again starts its streak over — one use clears the
// rot, so a periodically-useful pin never trips the flag.
func TestRefcount_FalseRetain_ResetOnHit(t *testing.T) {
	const k = 3
	rw := NewRefcountWitness(k)
	p := Plan{Selected: []Selection{sel("pin", 500, true)}}

	wasted := Outcome{Wasted: []string{"pin"}}
	used := Outcome{Hits: []string{"pin"}}

	// Idle K-1 turns (one short of firing).
	for turn := 1; turn < k; turn++ {
		rw.Observe(p, wasted, LiveGoal{})
	}
	if rw.Streak("pin") != k-1 {
		t.Fatalf("streak = %d, want %d before the hit", rw.Streak("pin"), k-1)
	}
	// A turn USES the pin -> streak resets to 0.
	if rc := rw.Observe(p, used, LiveGoal{}); rc.Any() {
		t.Fatalf("verdict = %+v, want empty on a hit turn", rc)
	}
	if rw.Streak("pin") != 0 {
		t.Fatalf("streak = %d, want 0 after a hit (consecutive resets)", rw.Streak("pin"))
	}
	// One more idle turn must NOT fire (the streak is rebuilding from 0, not continuing).
	if rc := rw.Observe(p, wasted, LiveGoal{}); len(rc.FalseRetain) != 0 {
		t.Errorf("false-retain = %v, want empty (streak reset by the hit)", rc.FalseRetain)
	}
}

// TestRefcount_FalseRetain_UnpinnedNotCandidate proves an UNPINNED wasted span is never a
// false-retain candidate — false-retain is specifically about a pin being the only (idle)
// referent. A plain wasted span is ordinary noise (the SignalNoise axis), not refcount rot.
func TestRefcount_FalseRetain_UnpinnedNotCandidate(t *testing.T) {
	const k = 2
	rw := NewRefcountWitness(k)
	p := Plan{Selected: []Selection{sel("plain", 100, false)}} // NOT pinned
	o := Outcome{Wasted: []string{"plain"}}

	for turn := 1; turn <= k+1; turn++ {
		if rc := rw.Observe(p, o, LiveGoal{}); rc.Any() {
			t.Fatalf("turn %d: verdict = %+v, want empty (unpinned span is not a false-retain candidate)", turn, rc)
		}
	}
	if rw.Streak("plain") != 0 {
		t.Errorf("streak = %d, want 0 (unpinned never accumulates)", rw.Streak("plain"))
	}
}

// TestRefcount_TwoClassesDistinct proves false-retain and false-free are SEPARATE classes,
// not one "compaction miss": one turn can carry BOTH, on their own fields, with the right
// span in each — over-resident rot and under-resident loss told apart.
func TestRefcount_TwoClassesDistinct(t *testing.T) {
	const k = 1 // fire false-retain on the first idle turn for a compact test
	rw := NewRefcountWitness(k)
	p := Plan{
		Selected: []Selection{sel("idlepin", 500, true)}, // pinned, about to be wasted
		Elided:   []Elision{{ID: "goalspan", Cost: 200}}, // goal span paged out
	}
	o := Outcome{
		Wasted: []string{"idlepin"},  // -> false-retain (pinned + wasted, K=1)
		Faults: []string{"goalspan"}, // -> false-free (faulted + goal-reachable)
	}
	goal := LiveGoal{Active: true, Reachable: []string{"goalspan"}}

	rc := rw.Observe(p, o, goal)

	if !reflect.DeepEqual(rc.FalseRetain, []string{"idlepin"}) {
		t.Errorf("false-retain = %v, want [idlepin]", rc.FalseRetain)
	}
	if !reflect.DeepEqual(rc.FalseFree, []string{"goalspan"}) {
		t.Errorf("false-free = %v, want [goalspan]", rc.FalseFree)
	}
	// The two are on DISTINCT fields — neither leaks into the other.
	if len(rc.FalseRetain) == len(rc.FalseFree) && reflect.DeepEqual(rc.FalseRetain, rc.FalseFree) {
		t.Error("the two classes are identical — they must be distinct (over- vs under-resident)")
	}
}

// TestRefcountAdvice_CleanTurnZero proves a clean Refcount (nothing flagged) implies the
// zero advisory down-weight — a healthy turn carries no penalty.
func TestRefcountAdvice_CleanTurnZero(t *testing.T) {
	rw := NewRefcountWitness(falseRetainK)
	// A pinned span the turn USED, and no live goal — nothing rots, nothing is lost.
	p := Plan{Selected: []Selection{sel("pin", 100, true)}}
	o := Outcome{Hits: []string{"pin"}}

	rc := rw.Observe(p, o, LiveGoal{})
	if rc.Any() {
		t.Fatalf("verdict = %+v, want clean", rc)
	}
	if adv := rc.DownWeight(); adv != (RefcountAdvice{}) {
		t.Errorf("clean turn down-weight = %+v, want zero", adv)
	}
}

// TestRefcountAdvice_OverRetainAsymmetry is the core property of the advisory down-weight:
// the verdict folds into a magnitude that weighs a false-free (under-resident loss — the
// bug the frame exists to kill) STRICTLY heavier than a false-retain (over-resident rot —
// idle tokens). One false-free must imply a larger down-weight than one false-retain, and
// the Total must be the sum of the two per-class weights.
func TestRefcountAdvice_OverRetainAsymmetry(t *testing.T) {
	oneFree := Refcount{FalseFree: []string{"goalspan"}}
	oneRetain := Refcount{FalseRetain: []string{"idlepin"}}

	advFree := oneFree.DownWeight()
	advRetain := oneRetain.DownWeight()

	// A false-free outweighs a false-retain — the conservative over-retain asymmetry.
	if !(advFree.Total > advRetain.Total) {
		t.Errorf("false-free down-weight %v not > false-retain %v — over-retain asymmetry lost",
			advFree.Total, advRetain.Total)
	}
	if advFree.FalseRetain != 0 || advRetain.FalseFree != 0 {
		t.Errorf("per-class weights leaked across axes: free=%+v retain=%+v", advFree, advRetain)
	}

	// Total is the sum of the two per-class weights, and counts scale it linearly.
	both := Refcount{FalseFree: []string{"a", "b"}, FalseRetain: []string{"c"}}
	adv := both.DownWeight()
	wantFree := 2 * falseFreePenalty
	wantRetain := 1 * falseRetainPenalty
	if !approx(adv.FalseFree, wantFree) || !approx(adv.FalseRetain, wantRetain) {
		t.Errorf("down-weight = %+v, want free=%v retain=%v", adv, wantFree, wantRetain)
	}
	if !approx(adv.Total, wantFree+wantRetain) {
		t.Errorf("total = %v, want %v (sum of the two classes)", adv.Total, wantFree+wantRetain)
	}
}

// TestRefcountAdvice_IsAdvisory_DoesNotChangePlan is the load-bearing acceptance check: the
// refcount down-weight is ADVISORY — observing a turn and reading its implied down-weight
// changes NO plan decision. The same (spans, forecast, budget) plans byte-identically
// whether or not the refcount witness ran, proving the signal never gated correctness
// (epic #844's law: a pin/signal that gated the answer would be a bug).
func TestRefcountAdvice_IsAdvisory_DoesNotChangePlan(t *testing.T) {
	spans := []Span{
		{ID: "sys", Role: "system", Descriptor: "system prompt", Bytes: 160, Durability: DurabilityDurable},
		{ID: "u1", Role: "user", Descriptor: "the goal", Bytes: 240, Step: 1},
		{ID: "scratch", Role: "tool", Descriptor: "noisy tool output", Bytes: 800, Step: 2},
	}
	f := Forecast{Intents: []string{"goal"}, Pins: []string{"sys"}}
	budget := Budget{Tokens: 150} // forces an elision: not every span fits

	// Plan once with NO refcount involvement — the control decision (nil = default cost model).
	want := PlanCells(spans, f, budget, nil)

	// Now run a full refcount observation (false-free + false-retain both fire) against a turn,
	// read its down-weight... and re-plan with the SAME inputs. The plan must be unchanged: the
	// witness is a pure observer, wired into no planner input.
	rw := NewRefcountWitness(1)
	o := Outcome{Wasted: []string{"sys"}, Faults: []string{"scratch"}}
	rc := rw.Observe(want, o, LiveGoal{Active: true, Reachable: []string{"scratch"}})
	if !rc.Any() {
		t.Fatal("expected the observation to flag something so the advisory is non-vacuous")
	}
	if adv := rc.DownWeight(); adv.Total <= 0 {
		t.Fatalf("expected a positive advisory down-weight, got %+v", adv)
	}

	got := PlanCells(spans, f, budget, nil)
	if !reflect.DeepEqual(got.Selected, want.Selected) || !reflect.DeepEqual(got.Elided, want.Elided) {
		t.Errorf("plan changed after a refcount observation — the down-weight is NOT advisory\n got=%+v\nwant=%+v", got, want)
	}
	if got.CostUsed != want.CostUsed {
		t.Errorf("CostUsed changed (%d != %d) — refcount must not affect the plan", got.CostUsed, want.CostUsed)
	}
}

// TestRefcount_ZeroValueWitnessUsable proves the zero-value witness works (K defaults to
// falseRetainK) so a caller can use a plain &RefcountWitness{} without the constructor.
func TestRefcount_ZeroValueWitnessUsable(t *testing.T) {
	var rw RefcountWitness
	p := Plan{Selected: []Selection{sel("pin", 10, true)}}
	o := Outcome{Wasted: []string{"pin"}}
	for turn := 1; turn < falseRetainK; turn++ {
		if rc := rw.Observe(p, o, LiveGoal{}); len(rc.FalseRetain) != 0 {
			t.Fatalf("turn %d fired early on zero-value witness", turn)
		}
	}
	if rc := rw.Observe(p, o, LiveGoal{}); !reflect.DeepEqual(rc.FalseRetain, []string{"pin"}) {
		t.Errorf("zero-value witness: false-retain = %v, want [pin] at default K", rc.FalseRetain)
	}
}

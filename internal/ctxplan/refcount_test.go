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

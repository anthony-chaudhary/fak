package superloop

import (
	"strings"
	"testing"
)

// TestImproveTrajectoryIsRegisteredSuperLoop pins the trajectory intent (issue #2563):
// improve-trajectory is a registered super loop whose single member is a KindTrajectory
// objective source, and it classifies as a super loop — the five-property classifier row
// stays green for the new member kind.
func TestImproveTrajectoryIsRegisteredSuperLoop(t *testing.T) {
	s, ok := Lookup("improve-trajectory")
	if !ok {
		t.Fatal("improve-trajectory not registered")
	}
	if len(s.Members) != 1 {
		t.Fatalf("improve-trajectory must hold exactly one trajectory source member, got %d", len(s.Members))
	}
	if s.Members[0].Kind != KindTrajectory {
		t.Errorf("improve-trajectory member kind = %q, want %q", s.Members[0].Kind, KindTrajectory)
	}
	if v := Classify(FactsFor(s)); !v.IsSuper {
		t.Errorf("improve-trajectory must classify as a super loop: %s", v.Reason)
	}
}

// TestWalkTrajectoryWorstFirst is the trajectory fold witness: an enumerated set of
// trajectory objective statuses (one per open objective, weighed by curve-signal debt)
// folds worst-first — highest debt first, a clean (HEALTHY, debt 0) objective dropped
// from the worklist — and the intent is SATISFIED only when every objective reads clean.
func TestWalkTrajectoryWorstFirst(t *testing.T) {
	s, ok := Lookup("improve-trajectory")
	if !ok {
		t.Fatal("improve-trajectory not registered")
	}
	traj := func(id string, debt int) MemberStatus {
		return MemberStatus{
			Member:   Member{Kind: KindTrajectory, Ref: id},
			Measured: true,
			Debt:     debt,
		}
	}
	// DETOUR_OVERRUN(3) > DRIFT(2) > STALL(1) > HEALTHY(0), in scrambled input order.
	rep := Walk(s, []MemberStatus{
		traj("stall-obj", 1),
		traj("healthy-obj", 0),
		traj("detour-obj", 3),
		traj("drift-obj", 2),
	})
	if rep.Satisfied {
		t.Error("a walk with off-course objectives must not be satisfied")
	}
	if len(rep.Worklist) != 3 {
		t.Fatalf("worklist must drop the one clean objective and keep 3, got %d", len(rep.Worklist))
	}
	wantOrder := []string{"detour-obj", "drift-obj", "stall-obj"}
	for i, want := range wantOrder {
		if got := rep.Worklist[i].Member.Ref; got != want {
			t.Errorf("worklist[%d] = %q, want %q (worst-first by curve debt)", i, got, want)
		}
	}

	// Every objective clean → satisfied, empty worklist.
	clean := Walk(s, []MemberStatus{traj("a", 0), traj("b", 0)})
	if !clean.Satisfied || len(clean.Worklist) != 0 {
		t.Errorf("all-healthy trajectory walk must be satisfied with an empty worklist, got satisfied=%v worklist=%d", clean.Satisfied, len(clean.Worklist))
	}
}

// TestTrajectoryActionHint pins the worklist action for a trajectory member: a measured
// objective surfaces its concrete curve steer command; an unmeasured one surfaces the
// read-the-ledger action (you cannot steer what you have not folded).
func TestTrajectoryActionHint(t *testing.T) {
	measured := actionFor(MemberStatus{
		Member:   Member{Kind: KindTrajectory, Ref: "obj-1", Enter: "fak trajctl curve --objective obj-1"},
		Measured: true,
		Debt:     2,
	})
	if want := "fak trajctl curve --objective obj-1"; !strings.Contains(measured, want) {
		t.Errorf("measured trajectory action = %q, want it to surface %q", measured, want)
	}
	unmeasured := actionFor(MemberStatus{Member: Member{Kind: KindTrajectory, Ref: "obj-2"}})
	if !strings.Contains(unmeasured, "trajctl ledger") {
		t.Errorf("unmeasured trajectory action = %q, want it to point at reading the trajctl ledger", unmeasured)
	}
}

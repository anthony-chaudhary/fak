package superloop

import (
	"strings"
	"testing"
)

// walkOf is a tiny helper: walk an intent over hand-built member statuses so a drive
// test controls exactly which member is worst-first without touching a live surface.
func walkOf(t *testing.T, name string, statuses []MemberStatus) WalkReport {
	t.Helper()
	s, ok := Lookup(name)
	if !ok {
		t.Fatalf("intent %q not registered", name)
	}
	return Walk(s, statuses)
}

// TestDriveSelectsWorstFirstOneMember pins the core SELECT contract: Drive enters the
// single worst-first member (the worklist head Walk already ranked) and no other. Here
// the higher-debt scorecard must be the one entered, at rank 1, carrying its action.
func TestDriveSelectsWorstFirstOneMember(t *testing.T) {
	rep := walkOf(t, "sweep-surfaces", []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "code", Enter: "/quality-score"}, Measured: true, Debt: 3},
		{Member: Member{Kind: KindScorecard, Ref: "slop", Enter: "/slop-score"}, Measured: true, Debt: 9},
		{Member: Member{Kind: KindScorecard, Ref: "appeal", Enter: "/appeal-score"}, Measured: true, Debt: 0},
	})

	dec := Drive(rep)
	if !dec.Enter {
		t.Fatalf("a walk with debt must enter a member; reason=%s", dec.Reason)
	}
	if dec.Member.Ref != "slop" {
		t.Errorf("entered %q, want the worst-first member %q", dec.Member.Ref, "slop")
	}
	if dec.Rank != 1 {
		t.Errorf("the drive enters the head; rank = %d, want 1", dec.Rank)
	}
	if dec.Debt != 9 {
		t.Errorf("entered debt = %d, want the worst-first debt 9", dec.Debt)
	}
	if dec.Action == "" {
		t.Error("an entered member must carry its front-door action")
	}
}

// TestDriveSatisfiedEntersNothing pins that a clean walk (empty worklist) enters
// nothing: Enter is false and no member is named — the caller's re-fold is a clean exit.
func TestDriveSatisfiedEntersNothing(t *testing.T) {
	rep := walkOf(t, "sweep-surfaces", []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "code"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "slop"}, Measured: true, Debt: 0},
	})
	if !rep.Satisfied {
		t.Fatalf("precondition: an all-clean walk must be satisfied, got debt %d unmeasured %d", rep.TotalDebt, rep.Unmeasured)
	}
	dec := Drive(rep)
	if dec.Enter {
		t.Fatalf("a satisfied intent enters nothing, got member %q", dec.Member.Ref)
	}
}

// TestDriveEmptyWorklistUnsatisfiedNotClean pins the drive honesty contract (#3147): an
// empty MEMBER worklist that is UNSATISFIED by an unmet headline (issue shortfall) must
// never read as clean. Every member is measured and at floor, so there is no member to
// enter — but the declared ~200-issue headline is unmet. Drive enters nothing (Enter is
// false), yet its Reason carries the shortfall magnitude, does NOT claim "reads clean",
// and the structural Satisfied/IssueShortfall fields let a consumer gate on the miss.
func TestDriveEmptyWorklistUnsatisfiedNotClean(t *testing.T) {
	// Drive is a pure fold over WalkReport, so build the report directly: an empty
	// worklist (no member is worst-first) coexisting with a 200-issue shortfall.
	rep := WalkReport{
		Name:           "run-the-night",
		IssueTarget:    200,
		IssueShortfall: 200,
		Satisfied:      false,
	}
	dec := Drive(rep)
	if dec.Enter {
		t.Fatalf("no member is worst-first on an empty worklist; the drive must enter nothing, got member %q", dec.Member.Ref)
	}
	if dec.Satisfied {
		t.Error("an unmet headline leaves the drive UNSATISFIED; Satisfied must be false")
	}
	if dec.IssueShortfall != 200 {
		t.Errorf("the drive must carry the structural shortfall; IssueShortfall = %d, want 200", dec.IssueShortfall)
	}
	if strings.Contains(dec.Reason, "reads clean") {
		t.Errorf("an unsatisfied drive must not claim clean; reason = %q", dec.Reason)
	}
	if !strings.Contains(dec.Reason, "200") {
		t.Errorf("the drive reason must carry the shortfall magnitude 200; reason = %q", dec.Reason)
	}
}

// TestDriveBatchSelectsTopKWorstFirst pins the batch SELECT contract: DriveBatch
// offers the top-K worst-first members (the ranked worklist head prefix) in rank
// order, and no more than K. Here the two highest-debt scorecards must be offered,
// worst-first, and the third (lowest debt) left out of a k=2 batch.
func TestDriveBatchSelectsTopKWorstFirst(t *testing.T) {
	rep := walkOf(t, "sweep-surfaces", []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "code", Enter: "/quality-score"}, Measured: true, Debt: 3},
		{Member: Member{Kind: KindScorecard, Ref: "slop", Enter: "/slop-score"}, Measured: true, Debt: 9},
		{Member: Member{Kind: KindScorecard, Ref: "appeal", Enter: "/appeal-score"}, Measured: true, Debt: 5},
	})

	dec := DriveBatch(rep, 2)
	if !dec.Enter {
		t.Fatalf("a walk with debt must offer members; reason=%s", dec.Reason)
	}
	if len(dec.Members) != 2 {
		t.Fatalf("k=2 must offer exactly two members, got %d", len(dec.Members))
	}
	if dec.Members[0].Member.Ref != "slop" || dec.Members[0].Rank != 1 {
		t.Errorf("head must be worst-first slop@rank1, got %q@rank%d", dec.Members[0].Member.Ref, dec.Members[0].Rank)
	}
	if dec.Members[1].Member.Ref != "appeal" || dec.Members[1].Rank != 2 {
		t.Errorf("second must be appeal@rank2, got %q@rank%d", dec.Members[1].Member.Ref, dec.Members[1].Rank)
	}
	for _, m := range dec.Members {
		if !m.Enter || m.Action == "" {
			t.Errorf("every offered member must be an entered decision carrying its action, got %+v", m)
		}
	}
}

// TestDriveBatchClampsAndDefaultsToAll pins the clamp rules: k larger than the
// worklist is clamped to the worklist length, and k <= 0 means "every member".
// The batch head must still equal the single Drive's head — the two rungs share
// the same worst-first selection.
func TestDriveBatchClampsAndDefaultsToAll(t *testing.T) {
	rep := walkOf(t, "sweep-surfaces", []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "code"}, Measured: true, Debt: 3},
		{Member: Member{Kind: KindScorecard, Ref: "slop"}, Measured: true, Debt: 9},
	})
	if got := DriveBatch(rep, 99); len(got.Members) != len(rep.Worklist) {
		t.Errorf("k over the worklist must clamp to %d, got %d", len(rep.Worklist), len(got.Members))
	}
	all := DriveBatch(rep, 0)
	if len(all.Members) != len(rep.Worklist) {
		t.Errorf("k<=0 must offer every worklist member (%d), got %d", len(rep.Worklist), len(all.Members))
	}
	if head := Drive(rep); all.Members[0].Member.Ref != head.Member.Ref {
		t.Errorf("batch head %q must equal single-drive head %q", all.Members[0].Member.Ref, head.Member.Ref)
	}
}

// TestDriveBatchEmptyWorklistMirrorsDriveHonesty pins that a batch over an empty
// worklist carries the SAME clean-vs-unmet-headline honesty as Drive (#3147): a
// satisfied walk reads clean; an unsatisfied empty worklist (issue shortfall)
// enters nothing but is NOT clean and carries the shortfall magnitude.
func TestDriveBatchEmptyWorklistMirrorsDriveHonesty(t *testing.T) {
	clean := walkOf(t, "sweep-surfaces", []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "code"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindScorecard, Ref: "slop"}, Measured: true, Debt: 0},
	})
	if dec := DriveBatch(clean, 3); dec.Enter || !dec.Satisfied || len(dec.Members) != 0 {
		t.Errorf("a satisfied walk offers nothing and reads clean, got %+v", dec)
	}

	shortfall := WalkReport{Name: "run-the-night", IssueTarget: 200, IssueShortfall: 200, Satisfied: false}
	dec := DriveBatch(shortfall, 3)
	if dec.Enter {
		t.Fatalf("an empty worklist offers nothing, got %d members", len(dec.Members))
	}
	if dec.Satisfied {
		t.Error("an unmet headline leaves the batch UNSATISFIED")
	}
	if dec.IssueShortfall != 200 || !strings.Contains(dec.Reason, "200") {
		t.Errorf("the batch must carry the shortfall magnitude 200, got shortfall=%d reason=%q", dec.IssueShortfall, dec.Reason)
	}
	if strings.Contains(dec.Reason, "reads clean") {
		t.Errorf("an unsatisfied batch must not claim clean; reason=%q", dec.Reason)
	}
}

// TestDriveDarkMemberIsWorstFirst pins that a DARK loop member outranks a debt-bearing
// scorecard — a gone-dark member is the most urgent thing to enter (tier 0).
func TestDriveDarkMemberIsWorstFirst(t *testing.T) {
	rep := walkOf(t, "improve-loops", []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "loopindex"}, Measured: true, Debt: 5},
		{Member: Member{Kind: KindLoop, Ref: "cadence"}, Measured: true, Dark: true},
	})
	dec := Drive(rep)
	if !dec.Enter || dec.Member.Ref != "cadence" || !dec.Dark {
		t.Fatalf("a dark loop must be entered first, got enter=%v member=%q dark=%v", dec.Enter, dec.Member.Ref, dec.Dark)
	}
}

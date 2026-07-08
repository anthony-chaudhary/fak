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

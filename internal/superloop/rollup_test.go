package superloop

import (
	"strings"
	"testing"
)

// TestWalkDirectDenominatorTruthfulness pins Problem 1 of issue #10916:
// When template members expand into concrete instances (such as KindLoopFleet:all
// expanding into per-loop statuses), rep.Members must truthfully report the evaluated
// candidate members (Walked + Unmeasured), rather than len(s.Members).
// It must never report paradoxical ratios like "3/1 member(s) could not be read".
func TestWalkDirectDenominatorTruthfulness(t *testing.T) {
	// A super loop with 1 declared template member.
	tmpl := Super{
		Name:    "expand-test",
		Title:   "test template expansion denominator",
		Floor:   0,
		Members: []Member{{Kind: KindLoopFleet, Ref: "all", Why: "the fleet"}},
	}

	// The shell expands the 1 template member into 14 concrete candidate statuses:
	// 11 measured (walked) and 3 unmeasured.
	var statuses []MemberStatus
	for i := 0; i < 11; i++ {
		statuses = append(statuses, MemberStatus{
			Member:   Member{Kind: KindLoopFleet, Ref: string(rune('a' + i))},
			Measured: true,
			Debt:     0,
		})
	}
	for i := 0; i < 3; i++ {
		statuses = append(statuses, MemberStatus{
			Member:   Member{Kind: KindLoopFleet, Ref: string(rune('u' + i))},
			Measured: false,
			Detail:   "no ledger found",
		})
	}

	rep := Walk(tmpl, statuses)

	// Truthful candidate counts: Members must equal Walked + Unmeasured.
	if rep.Walked != 11 {
		t.Errorf("rep.Walked = %d, want 11", rep.Walked)
	}
	if rep.Unmeasured != 3 {
		t.Errorf("rep.Unmeasured = %d, want 3", rep.Unmeasured)
	}
	if rep.Members != 14 {
		t.Errorf("rep.Members = %d, want 14 (Walked + Unmeasured) — got len(s.Members)=%d instead",
			rep.Members, len(tmpl.Members))
	}
	if rep.Members != rep.Walked+rep.Unmeasured {
		t.Errorf("rep.Members (%d) != rep.Walked (%d) + rep.Unmeasured (%d)",
			rep.Members, rep.Walked, rep.Unmeasured)
	}

	// The verdict reason must not report paradoxical "3/1 member(s) could not be read".
	wantRatio := "3/14 member(s) could not be read"
	if !strings.Contains(rep.Reason, wantRatio) {
		t.Errorf("rep.Reason = %q, want it to contain %q", rep.Reason, wantRatio)
	}
	if strings.Contains(rep.Reason, "3/1 member(s)") {
		t.Errorf("rep.Reason still contains paradoxical ratio '3/1 member(s)': %s", rep.Reason)
	}
}

// TestSubwalkStatusPreservesRollupMetrics pins Problem 2 of issue #10916:
// SubwalkStatus must preserve the subwalk's roll-up metrics (denominator/leaf members,
// walked, unmeasured, dark, spinning, orphaned, shortfall, satisfied).
func TestSubwalkStatusPreservesRollupMetrics(t *testing.T) {
	subSuper := Super{
		Name:    "sub-loop",
		Title:   "sub super loop",
		Floor:   0,
		Members: []Member{{Kind: KindScorecard, Ref: "slop"}},
	}
	subStatuses := []MemberStatus{
		{Member: Member{Kind: KindScorecard, Ref: "slop"}, Measured: true, Debt: 2},
		{Member: Member{Kind: KindScorecard, Ref: "code"}, Measured: true, Debt: 0},
		{Member: Member{Kind: KindLoop, Ref: "cron"}, Measured: true, Dark: true},
		{Member: Member{Kind: KindLoop, Ref: "spin"}, Measured: true, Progress: ProgressSpinning},
		{Member: Member{Kind: KindLoop, Ref: "orph"}, Measured: true, FollowOn: FollowonOrphaned},
		{Member: Member{Kind: KindScorecard, Ref: "unm"}, Measured: false},
	}
	subRep := Walk(subSuper, subStatuses)

	m := Member{Kind: KindSuperloop, Ref: "sub-loop"}
	st := SubwalkStatus(m, subRep)

	if st.Rollup == nil {
		t.Fatal("SubwalkStatus must populate Rollup on MemberStatus, got nil")
	}
	r := *st.Rollup
	if r.Members != 6 {
		t.Errorf("rollup Members = %d, want 6", r.Members)
	}
	if r.Walked != 5 {
		t.Errorf("rollup Walked = %d, want 5", r.Walked)
	}
	if r.Unmeasured != 1 {
		t.Errorf("rollup Unmeasured = %d, want 1", r.Unmeasured)
	}
	if r.Dark != 1 {
		t.Errorf("rollup Dark = %d, want 1", r.Dark)
	}
	if r.Spinning != 1 {
		t.Errorf("rollup Spinning = %d, want 1", r.Spinning)
	}
	if r.Orphaned != 1 {
		t.Errorf("rollup Orphaned = %d, want 1", r.Orphaned)
	}
	if r.Satisfied {
		t.Error("rollup Satisfied = true, want false when dark/unmeasured/spinning/orphaned present")
	}
}

// TestRollupLeafDeduplicationSharedSubloops pins the deduplication requirement:
// Leaves from shared sub-walks (fan-in > 1, e.g. drain-issues shared by tend
// and run-the-night) must be deduplicated by member key, upholding the once-only /
// non-double-counting invariant.
func TestRollupLeafDeduplicationSharedSubloops(t *testing.T) {
	// Subwalk B has leaf X and leaf Y.
	leafX := MemberStatus{Member: Member{Kind: KindScorecard, Ref: "leaf-X"}, Measured: true, Debt: 1}
	leafY := MemberStatus{Member: Member{Kind: KindScorecard, Ref: "leaf-Y"}, Measured: true, Debt: 0}
	repB := Walk(Super{Name: "sub-B", Floor: 0}, []MemberStatus{leafX, leafY})

	// Subwalk C has leaf Y (shared!) and leaf Z.
	leafZ := MemberStatus{Member: Member{Kind: KindScorecard, Ref: "leaf-Z"}, Measured: true, Debt: 3}
	repC := Walk(Super{Name: "sub-C", Floor: 0}, []MemberStatus{leafY, leafZ})

	// Parent Superloop A descends both Subwalk B and Subwalk C.
	parentSuper := Super{
		Name:    "parent-A",
		Title:   "parent intent",
		Floor:   0,
		Members: []Member{{Kind: KindSuperloop, Ref: "sub-B"}, {Kind: KindSuperloop, Ref: "sub-C"}},
	}
	stB := SubwalkStatus(Member{Kind: KindSuperloop, Ref: "sub-B"}, repB)
	stC := SubwalkStatus(Member{Kind: KindSuperloop, Ref: "sub-C"}, repC)

	parentRep := Walk(parentSuper, []MemberStatus{stB, stC})

	// Direct candidate counts: 2 descended subwalks.
	if parentRep.Members != 2 {
		t.Errorf("parentRep.Members = %d, want 2 (direct subwalks)", parentRep.Members)
	}
	if parentRep.Walked != 2 {
		t.Errorf("parentRep.Walked = %d, want 2", parentRep.Walked)
	}

	// Rollup summary: leaves X, Y, Z must be deduplicated!
	// Total unique leaves = 3 (X, Y, Z), NOT 4 (2 + 2).
	if parentRep.Rollup.Members != 3 {
		t.Errorf("parentRep.Rollup.Members = %d, want 3 (deduplicated X, Y, Z)", parentRep.Rollup.Members)
	}
	if parentRep.Rollup.Walked != 3 {
		t.Errorf("parentRep.Rollup.Walked = %d, want 3", parentRep.Rollup.Walked)
	}
	if parentRep.Rollup.Unmeasured != 0 {
		t.Errorf("parentRep.Rollup.Unmeasured = %d, want 0", parentRep.Rollup.Unmeasured)
	}
}

// TestRollupSurfacesUnmeasuredAndDarkFromDeepDescents pins that unmeasured and dark
// leaves deep in subwalks are surfaced in the parent's RollupSummary.
func TestRollupSurfacesUnmeasuredAndDarkFromDeepDescents(t *testing.T) {
	// Sub-subwalk with 1 unmeasured leaf and 1 dark leaf.
	deepSuper := Super{Name: "deep", Floor: 0}
	deepLeaf1 := MemberStatus{Member: Member{Kind: KindLoop, Ref: "deep-dark"}, Measured: true, Dark: true}
	deepLeaf2 := MemberStatus{Member: Member{Kind: KindScorecard, Ref: "deep-unm"}, Measured: false}
	deepRep := Walk(deepSuper, []MemberStatus{deepLeaf1, deepLeaf2})

	// Mid-level superloop descending deep.
	midSuper := Super{Name: "mid", Floor: 0}
	midSt := SubwalkStatus(Member{Kind: KindSuperloop, Ref: "deep"}, deepRep)
	midLeaf := MemberStatus{Member: Member{Kind: KindScorecard, Ref: "mid-clean"}, Measured: true, Debt: 0}
	midRep := Walk(midSuper, []MemberStatus{midSt, midLeaf})

	// Root superloop descending mid.
	rootSuper := Super{Name: "root", Floor: 0}
	rootSt := SubwalkStatus(Member{Kind: KindSuperloop, Ref: "mid"}, midRep)
	rootLeaf := MemberStatus{Member: Member{Kind: KindScorecard, Ref: "root-clean"}, Measured: true, Debt: 0}
	rootRep := Walk(rootSuper, []MemberStatus{rootSt, rootLeaf})

	// Direct candidate counts at root:
	// 2 candidates: 1 subwalk (mid) + 1 direct leaf (root-clean).
	if rootRep.Members != 2 {
		t.Errorf("rootRep.Members = %d, want 2", rootRep.Members)
	}
	if rootRep.Walked != 2 {
		t.Errorf("rootRep.Walked = %d, want 2", rootRep.Walked)
	}
	if rootRep.Unmeasured != 0 {
		t.Errorf("rootRep.Unmeasured = %d, want 0 direct unmeasured", rootRep.Unmeasured)
	}

	// Roll-up summary at root across the entire tree:
	// Leaves: deep-dark, deep-unm, mid-clean, root-clean = 4 leaves.
	if rootRep.Rollup.Members != 4 {
		t.Errorf("rootRep.Rollup.Members = %d, want 4 leaves across tree", rootRep.Rollup.Members)
	}
	if rootRep.Rollup.Walked != 3 {
		t.Errorf("rootRep.Rollup.Walked = %d, want 3 walked leaves", rootRep.Rollup.Walked)
	}
	if rootRep.Rollup.Unmeasured != 1 {
		t.Errorf("rootRep.Rollup.Unmeasured = %d, want 1 unmeasured leaf", rootRep.Rollup.Unmeasured)
	}
	if rootRep.Rollup.Dark != 1 {
		t.Errorf("rootRep.Rollup.Dark = %d, want 1 dark leaf", rootRep.Rollup.Dark)
	}
	if rootRep.Rollup.Satisfied {
		t.Error("rootRep.Rollup.Satisfied = true, want false due to unmeasured and dark leaves")
	}
	if rootRep.Satisfied {
		t.Error("rootRep.Satisfied = true, want false when rollup is unsatisfied")
	}
}

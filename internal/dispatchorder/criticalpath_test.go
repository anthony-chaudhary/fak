package dispatchorder

import (
	"fmt"
	"testing"
)

func TestCriticalPathPickThirtySeatsPrioritizesUnlocksAndRefills(t *testing.T) {
	cands := []Candidate{{ID: "spine", Tree: []string{"internal/spine/**"}}, {ID: "wide", Tree: []string{"internal/wide/**"}}}
	for i := 0; i < 28; i++ {
		cands = append(cands, Candidate{ID: fmt.Sprintf("leaf-%02d", i), BlockedBy: []string{"spine"}, Tree: []string{fmt.Sprintf("internal/leaf%02d/**", i)}})
	}
	first := CriticalPathPick(cands, nil, 30)
	if len(first.Admitted) != 2 {
		t.Fatalf("root admission=%v want both ready roots", first.Admitted)
	}
	if first.Admitted[0] != "spine" {
		t.Fatalf("first=%q want dependency-unlocking spine", first.Admitted[0])
	}
	second := CriticalPathPick(cands, []string{"spine"}, 30)
	if !second.Refill || len(second.Admitted) != 29 {
		t.Fatalf("refill=%v admitted=%d want true,29", second.Refill, len(second.Admitted))
	}
	seen := map[string]bool{}
	for _, id := range second.Admitted {
		if seen[id] {
			t.Fatalf("duplicate %s", id)
		}
		seen[id] = true
	}
}

func TestCriticalPathPickFailsClosedAndPreservesTreeDisjointness(t *testing.T) {
	cands := []Candidate{
		{ID: "a", BlockedBy: []string{"b"}, Tree: []string{"internal/a/**"}},
		{ID: "b", BlockedBy: []string{"a"}, Tree: []string{"internal/b/**"}},
		{ID: "missing", BlockedBy: []string{"ghost"}, Tree: []string{"internal/m/**"}},
		{ID: "ready-1", Tree: []string{"internal/shared/**"}},
		{ID: "ready-2", Tree: []string{"internal/shared/child/**"}},
	}
	r := CriticalPathPick(cands, nil, 30)
	if len(r.Admitted) != 1 {
		t.Fatalf("admitted=%v want one tree-disjoint ready issue", r.Admitted)
	}
	disp := map[string]CriticalPathDisposition{}
	for _, s := range r.Scores {
		disp[s.IssueID] = s.Disposition
	}
	if disp["a"] != CriticalPathCycle || disp["b"] != CriticalPathCycle {
		t.Fatalf("cycle dispositions: %#v", disp)
	}
	if disp["missing"] != CriticalPathMissingPrereq {
		t.Fatalf("missing disposition=%s", disp["missing"])
	}
	if (disp["ready-1"] == CriticalPathAdmit) == (disp["ready-2"] == CriticalPathAdmit) {
		t.Fatalf("exactly one overlapping tree must admit: %#v", disp)
	}
}

package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// heldLanePayload is the fixture for the shared soft-hold body: one lane holding a MIX of
// held and unheld issues (so the per-issue maps have something to prune and something to
// keep), one lane held entirely to empty, one lane whose per-issue map covers ONLY a held
// issue, plus a pre-existing human-blocked skip with a LOW number so the re-sort is
// observable rather than incidentally already in order.
func heldLanePayload() dispatchtick.RouterPayload {
	return dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 301, Title: "keep", Lane: "shared", ExpectedSteps: 3},
			{Number: 302, Title: "held-in-shared", Lane: "shared", ExpectedSteps: 4},
			{Number: 303, Title: "keep too", Lane: "shared", ExpectedSteps: 1},
			{Number: 304, Title: "held-alone", Lane: "solo", ExpectedSteps: 2},
			{Number: 305, Title: "held-in-mixed", Lane: "mixed", ExpectedSteps: 5},
			{Number: 306, Title: "keep in mixed", Lane: "mixed", ExpectedSteps: 6},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"shared": {
				Tree: []string{"internal/shared/**"}, Count: 3, StepBudget: 8,
				Issues:     []int{301, 302, 303},
				WorkUnits:  map[int]string{301: "u301", 302: "u302", 303: "u303"},
				IssueSteps: map[int]int{301: 3, 302: 4, 303: 1},
				Priority:   map[int]int{301: 10, 302: 20, 303: 30},
				Generation: map[int]string{301: "g1", 302: "g2", 303: "g3"},
			},
			"solo": {
				Tree: []string{"internal/solo/**"}, Count: 1, StepBudget: 2,
				Issues:   []int{304},
				Priority: map[int]int{304: 40},
			},
			// Every entry in this lane's maps belongs to the held issue, so pruning must
			// leave nil rather than an empty map.
			"mixed": {
				Tree: []string{"internal/mixed/**"}, Count: 2, StepBudget: 11,
				Issues:     []int{305, 306},
				Priority:   map[int]int{305: 50},
				Generation: map[int]string{305: "g5"},
			},
		},
		SkippedHumanBlocked: []dispatchtick.SkippedIssue{
			{Number: 200, Title: "waiting on a person", Reason: dispatchtick.ReasonBlockedByHuman},
		},
		Counts: dispatchtick.RouterCounts{
			Open: 7, Routed: 6, RoutedStepBudget: 21, SkippedHumanBlocked: 1,
			SkippedByReason: map[string]int{dispatchtick.ReasonBlockedByHuman: 1},
		},
	}
}

// holdIssues folds a payload plus a set of issue numbers into the three arguments
// applyDispatchHold takes from every caller, in payload.Issues order.
func holdIssues(p dispatchtick.RouterPayload, nums ...int) (map[int]bool, []dispatchtick.IssueRoute, map[int]int) {
	held := map[int]bool{}
	for _, n := range nums {
		held[n] = true
	}
	var routes []dispatchtick.IssueRoute
	stepByNum := map[int]int{}
	for _, iss := range p.Issues {
		stepByNum[iss.Number] = iss.ExpectedSteps
		if held[iss.Number] {
			routes = append(routes, iss)
		}
	}
	return held, routes, stepByNum
}

// TestApplyDispatchHoldSortsSkippedHighestNumberFirst pins the ordering contract of the
// shared hold body: the newly held rows are APPENDED to whatever was already skipped, then
// the whole set is re-sorted highest-number-first to match the router's own ordering. The
// fixture's pre-existing row (#200) is lower than every held number, so an append with no
// re-sort, or a re-sort the other way, both show up here.
//
// This is a witness the three per-hold copies never had: a mutation that inverted the
// comparator left every existing dispatch-hold test GREEN.
func TestApplyDispatchHoldSortsSkippedHighestNumberFirst(t *testing.T) {
	p := heldLanePayload()
	held, routes, steps := holdIssues(p, 302, 304, 305)
	got := applyDispatchHold(p, held, routes, steps, reasonBlockedByKnownBad,
		func(iss dispatchtick.IssueRoute) string { return "held: " + iss.Title })

	want := []int{305, 304, 302, 200}
	if len(got.SkippedHumanBlocked) != len(want) {
		t.Fatalf("skipped set should hold the 1 pre-existing row plus the 3 held ones, got %+v", got.SkippedHumanBlocked)
	}
	for i, n := range want {
		if got.SkippedHumanBlocked[i].Number != n {
			t.Fatalf("skipped set must be highest-number-first %v, got %v (position %d)",
				want, skippedNumbers(got.SkippedHumanBlocked), i)
		}
	}
	// The pre-existing human-blocked row keeps its own reason: re-sorting must not
	// relabel a row that a different hold booked.
	if got.SkippedHumanBlocked[3].Reason != dispatchtick.ReasonBlockedByHuman {
		t.Errorf("pre-existing #200 must keep its own reason, got %q", got.SkippedHumanBlocked[3].Reason)
	}
	if got.SkippedHumanBlocked[0].NextAction != "held: held-in-mixed" {
		t.Errorf("each held row carries the caller's own hint, got %q", got.SkippedHumanBlocked[0].NextAction)
	}
}

// TestApplyDispatchHoldPrunesStalePerIssueEntries pins the other half of the shared body:
// a lane that SURVIVES a hold must carry no per-issue entry for a held number, or a
// consumer reading Priority/Generation/WorkUnits/IssueSteps could resurrect an issue the
// hold removed from lane.Issues. A map left with nothing must come back nil, not empty.
//
// Also a witness the three copies never had: a mutation that skipped the pruning entirely
// left every existing dispatch-hold test GREEN.
func TestApplyDispatchHoldPrunesStalePerIssueEntries(t *testing.T) {
	p := heldLanePayload()
	held, routes, steps := holdIssues(p, 302, 304, 305)
	got := applyDispatchHold(p, held, routes, steps, reasonBlockedByKnownBad,
		func(iss dispatchtick.IssueRoute) string { return "held" })

	shared, ok := got.Lanes["shared"]
	if !ok {
		t.Fatalf("lane shared keeps two unheld issues and must survive, lanes=%v", laneNames(got))
	}
	if len(shared.Issues) != 2 || shared.Issues[0] != 301 || shared.Issues[1] != 303 {
		t.Fatalf("lane shared should route exactly [301 303] after the hold, got %v", shared.Issues)
	}
	if _, stale := shared.Priority[302]; stale {
		t.Errorf("held #302 must not survive in lane Priority: %v", shared.Priority)
	}
	if _, stale := shared.Generation[302]; stale {
		t.Errorf("held #302 must not survive in lane Generation: %v", shared.Generation)
	}
	if _, stale := shared.WorkUnits[302]; stale {
		t.Errorf("held #302 must not survive in lane WorkUnits: %v", shared.WorkUnits)
	}
	if _, stale := shared.IssueSteps[302]; stale {
		t.Errorf("held #302 must not survive in lane IssueSteps: %v", shared.IssueSteps)
	}
	// Pruning is surgical: the unheld issues keep every one of their entries.
	if shared.Priority[301] != 10 || shared.Generation[303] != "g3" || shared.WorkUnits[301] != "u301" || shared.IssueSteps[303] != 1 {
		t.Errorf("unheld issues must keep their per-issue entries, got %+v", shared)
	}
	// Count and step budget are re-derived from what actually remains.
	if shared.Count != 2 || shared.StepBudget != 4 {
		t.Errorf("lane shared should re-derive to count=2 budget=4 (301's 3 + 303's 1), got count=%d budget=%d",
			shared.Count, shared.StepBudget)
	}

	// A lane held to empty is dropped whole rather than left as a zero-issue lane.
	if _, ok := got.Lanes["solo"]; ok {
		t.Errorf("lane solo held to empty must be dropped, got %+v", got.Lanes["solo"])
	}

	// A surviving lane whose maps covered ONLY held issues comes back nil, not empty --
	// an empty non-nil map would serialize as a present-but-blank field.
	mixed, ok := got.Lanes["mixed"]
	if !ok {
		t.Fatalf("lane mixed keeps #306 and must survive, lanes=%v", laneNames(got))
	}
	if mixed.Priority != nil {
		t.Errorf("lane mixed's Priority covered only the held #305 and must be nil, got %v", mixed.Priority)
	}
	if mixed.Generation != nil {
		t.Errorf("lane mixed's Generation covered only the held #305 and must be nil, got %v", mixed.Generation)
	}

	// The counts reconcile against what survived: 3 of 6 routed, budget 4 (shared) + 6 (mixed).
	if got.Counts.Routed != 3 {
		t.Errorf("routed should drop from 6 to 3, got %d", got.Counts.Routed)
	}
	if got.Counts.RoutedStepBudget != 10 {
		t.Errorf("routed step budget should be 10 after the hold, got %d", got.Counts.RoutedStepBudget)
	}
	if got.Counts.SkippedByReason[reasonBlockedByKnownBad] != 3 {
		t.Errorf("SkippedByReason[%s] should book all 3 holds, got %d",
			reasonBlockedByKnownBad, got.Counts.SkippedByReason[reasonBlockedByKnownBad])
	}
}

// skippedNumbers / laneNames keep the failure messages readable without dumping whole
// payloads into the test log.
func skippedNumbers(rows []dispatchtick.SkippedIssue) []int {
	out := make([]int, 0, len(rows))
	for _, r := range rows {
		out = append(out, r.Number)
	}
	return out
}

func laneNames(p dispatchtick.RouterPayload) []string {
	out := make([]string, 0, len(p.Lanes))
	for lane := range p.Lanes {
		out = append(out, lane)
	}
	return out
}

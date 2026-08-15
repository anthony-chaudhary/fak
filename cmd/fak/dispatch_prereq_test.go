package main

import (
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// prereqPayload is the routed fixture the dependency-hold tests fold: prerequisite #101 in lane
// "foo", dependent #102 in lane "bar" whose BlockedBy the caller supplies. Same shape as the
// known-bad fixture so the two holds are visibly the same class of overlay on different signals.
func prereqPayload(bBlockedBy []string) dispatchtick.RouterPayload {
	return dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 101, Title: "A prereq", Lane: "foo", ExpectedSteps: 3},
			{Number: 102, Title: "B dependent", Lane: "bar", ExpectedSteps: 2, BlockedBy: bBlockedBy},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"foo": {Count: 1, StepBudget: 3, Issues: []int{101}},
			"bar": {Count: 1, StepBudget: 2, Issues: []int{102}},
		},
		Counts: dispatchtick.RouterCounts{
			Open: 2, Routed: 2, RoutedStepBudget: 5,
			SkippedByReason: map[string]int{},
		},
	}
}

// candidateNumbers is the set of numbers the router still offers for dispatch after the hold --
// the list PickTargetIssue walks. A held issue must be absent from it.
func candidateNumbers(p dispatchtick.RouterPayload) []int {
	out := make([]int, 0, len(p.Issues))
	for _, iss := range p.Issues {
		out = append(out, iss.Number)
	}
	return out
}

// TestHoldOpenPrereqBlocksDependent is the Done condition: with dependent #102 naming open
// prerequisite #101, #102 is held out of its lane with reason BLOCKED_BY_OPEN_PREREQ, #101 keeps
// dispatching, PickTargetIssue can never select #102, and the counts reconcile.
func TestHoldOpenPrereqBlocksDependent(t *testing.T) {
	got := holdOpenPrereqForRoute(prereqPayload([]string{"101"}))

	// #102 is HELD: its lane is gone (held empty) and it is no longer a routable candidate.
	if _, ok := got.Lanes["bar"]; ok {
		t.Errorf("lane bar depends on open #101 and must be held out, still present: %+v", got.Lanes["bar"])
	}
	for _, iss := range got.Issues {
		if iss.Number == 102 {
			t.Errorf("held dependent #102 must not remain a routable candidate, got %+v", iss)
		}
	}

	// The prerequisite #101 still dispatches from its lane, and the picker returns it, not #102.
	foo, ok := got.Lanes["foo"]
	if !ok || len(foo.Issues) != 1 || foo.Issues[0] != 101 {
		t.Fatalf("prerequisite #101 must still route in lane foo, got %+v (ok=%v)", foo, ok)
	}
	pick, okPick := dispatchtick.PickTargetIssue(candidateNumbers(got), map[int]bool{})
	if !okPick || pick == 102 {
		t.Fatalf("picker must not select held #102 while #101 is open, picked %d (ok=%v)", pick, okPick)
	}

	// #102 is now a BLOCKED_BY_OPEN_PREREQ skip naming the open prerequisite.
	held := openPrereqBlockedSkipped(got)
	if len(held) != 1 || held[0].Number != 102 {
		t.Fatalf("want exactly one BLOCKED_BY_OPEN_PREREQ row for #102, got %+v", held)
	}
	if !strings.Contains(held[0].NextAction, "#101") {
		t.Errorf("held row's next action should name the open prerequisite #101, got %q", held[0].NextAction)
	}

	// Counts reconcile: one fewer routed, the hold booked under its own reason.
	if got.Counts.Routed != 1 {
		t.Errorf("routed count should drop to 1 after holding #102, got %d", got.Counts.Routed)
	}
	if got.Counts.RoutedStepBudget != 3 {
		t.Errorf("routed step budget should be foo's 3 after the hold, got %d", got.Counts.RoutedStepBudget)
	}
	if got.Counts.SkippedByReason[reasonBlockedByOpenPrereq] != 1 {
		t.Errorf("SkippedByReason[%s] should be 1, got %d", reasonBlockedByOpenPrereq, got.Counts.SkippedByReason[reasonBlockedByOpenPrereq])
	}
}

// TestHoldOpenPrereqFailsOpen pins the fail-open path: a prerequisite ABSENT from the candidate set
// (already closed) never holds its dependent. Once #101 has left the open backlog, #102 dispatches.
func TestHoldOpenPrereqFailsOpen(t *testing.T) {
	// #102 names #999, which is not in the set at all -> closed -> no hold.
	got := holdOpenPrereqForRoute(prereqPayload([]string{"999"}))
	if got.Counts.Routed != 2 {
		t.Errorf("an absent (closed) prerequisite must hold nothing, routed=%d want 2", got.Counts.Routed)
	}
	if len(openPrereqBlockedSkipped(got)) != 0 {
		t.Errorf("fail-open: no BLOCKED_BY_OPEN_PREREQ rows expected, got %d", len(openPrereqBlockedSkipped(got)))
	}

	// Release: with the prerequisite gone (only #102 remains, still naming closed #101), #102 is now
	// pickable -- the self-clearing exit.
	released := holdOpenPrereqForRoute(dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{{Number: 102, Title: "B", Lane: "bar", ExpectedSteps: 2, BlockedBy: []string{"101"}}},
		Lanes:  map[string]dispatchtick.RouterLaneGroup{"bar": {Count: 1, StepBudget: 2, Issues: []int{102}}},
		Counts: dispatchtick.RouterCounts{Open: 1, Routed: 1, RoutedStepBudget: 2, SkippedByReason: map[string]int{}},
	})
	pick, ok := dispatchtick.PickTargetIssue(candidateNumbers(released), map[int]bool{})
	if !ok || pick != 102 {
		t.Fatalf("once prerequisite #101 closes, #102 must dispatch, picked %d (ok=%v)", pick, ok)
	}
}

// TestHoldOpenPrereqCycleSafe pins the 2-cycle break: a mutual A<->B dependency does not deadlock
// both; the engine breaks toward the lowest ID, so #101 keeps the dispatch and #102 is held.
func TestHoldOpenPrereqCycleSafe(t *testing.T) {
	got := holdOpenPrereqForRoute(dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 101, Title: "A", Lane: "foo", ExpectedSteps: 1, BlockedBy: []string{"102"}},
			{Number: 102, Title: "B", Lane: "bar", ExpectedSteps: 1, BlockedBy: []string{"101"}},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"foo": {Count: 1, StepBudget: 1, Issues: []int{101}},
			"bar": {Count: 1, StepBudget: 1, Issues: []int{102}},
		},
		Counts: dispatchtick.RouterCounts{Open: 2, Routed: 2, RoutedStepBudget: 2, SkippedByReason: map[string]int{}},
	})
	held := openPrereqBlockedSkipped(got)
	if len(held) != 1 || held[0].Number != 102 {
		t.Fatalf("a 2-cycle must break toward the lowest ID (hold #102, keep #101), got held %+v", held)
	}
	if _, ok := got.Lanes["foo"]; !ok {
		t.Errorf("lowest-ID #101 must keep dispatching through a 2-cycle")
	}
}

// TestHoldOpenPrereqHoldsOnSkippedOpenPrereq pins the presence universe: a prerequisite that is OPEN
// but currently undispatchable (already in SkippedHumanBlocked) still holds its dependent, because it
// is not CLOSED. Only a closed prerequisite fails open.
func TestHoldOpenPrereqHoldsOnSkippedOpenPrereq(t *testing.T) {
	got := holdOpenPrereqForRoute(dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 102, Title: "B", Lane: "bar", ExpectedSteps: 2, BlockedBy: []string{"101"}},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"bar": {Count: 1, StepBudget: 2, Issues: []int{102}},
		},
		// #101 is open but human-blocked -- present for the prereq check, undispatchable itself.
		SkippedHumanBlocked: []dispatchtick.SkippedIssue{{Number: 101, Title: "A", Reason: dispatchtick.ReasonBlockedByHuman}},
		Counts:              dispatchtick.RouterCounts{Open: 2, Routed: 1, RoutedStepBudget: 2, SkippedHumanBlocked: 1, SkippedByReason: map[string]int{dispatchtick.ReasonBlockedByHuman: 1}},
	})
	if _, ok := got.Lanes["bar"]; ok {
		t.Errorf("#102 must be held while its open-but-blocked prerequisite #101 is unresolved")
	}
	if len(openPrereqBlockedSkipped(got)) != 1 {
		t.Fatalf("want #102 held on the skipped-open prerequisite, got %d holds", len(openPrereqBlockedSkipped(got)))
	}
}

func TestReconcilePrereqReleaseIsOneShotAndPriorityBounded(t *testing.T) {
	root := t.TempDir()
	open := dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 10, Lane: "base", ExpectedSteps: 1},
			{Number: 20, Lane: "next", ExpectedSteps: 1, BlockedBy: []string{"10"}},
			{Number: 30, Lane: "other", ExpectedSteps: 1},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"base":  {Issues: []int{10}, Count: 1, StepBudget: 1},
			"next":  {Issues: []int{20}, Count: 1, StepBudget: 1},
			"other": {Issues: []int{30}, Count: 1, StepBudget: 1},
		},
		Counts: dispatchtick.RouterCounts{Routed: 3, RoutedStepBudget: 3},
	}

	first := reconcilePrereqRelease(root, open)
	if len(first.NewlyUnblocked) != 0 {
		t.Fatalf("first newly_unblocked = %v, want none while prerequisite is open", first.NewlyUnblocked)
	}
	held := holdOpenPrereqForRoute(first)
	heldReason := ""
	for _, row := range held.SkippedHumanBlocked {
		if row.Number == 20 {
			heldReason = row.Reason
		}
	}
	if heldReason != reasonBlockedByOpenPrereq {
		t.Fatalf("held dependent reason = %q, want %s", heldReason, reasonBlockedByOpenPrereq)
	}

	closed := dispatchtick.RouterPayload{Issues: []dispatchtick.IssueRoute{
		{Number: 20, BlockedBy: []string{"10"}},
		{Number: 30},
	}}
	second := reconcilePrereqRelease(root, closed)
	if !reflect.DeepEqual(second.NewlyUnblocked, []int{20}) {
		t.Fatalf("second newly_unblocked = %v, want [20]", second.NewlyUnblocked)
	}
	if got := promoteNewlyUnblocked([]int{30, 20}, map[int]int{}, map[int]bool{20: true}); !reflect.DeepEqual(got, []int{20, 30}) {
		t.Fatalf("same-priority release order = %v, want [20 30]", got)
	}
	if got := promoteNewlyUnblocked([]int{30, 20}, map[int]int{30: dispatchtick.PriorityWeightP1}, map[int]bool{20: true}); !reflect.DeepEqual(got, []int{30, 20}) {
		t.Fatalf("higher explicit priority order = %v, want [30 20]", got)
	}
	base := dispatchWaveOrderStamp(dispatchGoalProfileThroughput, dispatchtick.PriorityWeightDefault, 10, false)
	if dispatchWaveReleaseStamp(base, true) <= dispatchWaveReleaseStamp(base+1, false) {
		t.Fatal("newly unblocked wave stamp must outrank ordinary same-tier work")
	}

	third := reconcilePrereqRelease(root, closed)
	if len(third.NewlyUnblocked) != 0 {
		t.Fatalf("third newly_unblocked = %v, want one-shot signal cleared", third.NewlyUnblocked)
	}
}

func TestPickDispatchLaneNewlyUnblockedLeadsSamePriorityThenClears(t *testing.T) {
	root := t.TempDir()
	oldRoute := dispatchRouteIssues
	defer func() { dispatchRouteIssues = oldRoute }()
	pass := 0
	dispatchRouteIssues = func(string, io.Writer) (dispatchtick.RouterPayload, error) {
		pass++
		newly := []int{20}
		if pass > 1 {
			newly = nil
		}
		return dispatchtick.RouterPayload{
			NewlyUnblocked: newly,
			Issues: []dispatchtick.IssueRoute{
				{Number: 20, Lane: "next"},
				{Number: 30, Lane: "other"},
			},
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"next":  {Issues: []int{20}, Count: 1, StepBudget: 1},
				"other": {Issues: []int{30}, Count: 1, StepBudget: 100},
			},
		}, nil
	}

	first, err := pickDispatchLane(root, io.Discard, "", nil, false, "", dispatchGoalProfileThroughput, 0)
	if err != nil {
		t.Fatal(err)
	}
	if first.Lane != "next" || !reflect.DeepEqual(first.Numbers, []int{20}) {
		t.Fatalf("release pick = lane %q numbers %v, want next/[20]", first.Lane, first.Numbers)
	}
	second, err := pickDispatchLane(root, io.Discard, "", nil, false, "", dispatchGoalProfileThroughput, 0)
	if err != nil {
		t.Fatal(err)
	}
	if second.Lane != "other" {
		t.Fatalf("post-release lane = %q, want ordinary richer lane other", second.Lane)
	}
}

func TestDispatchWaveNewlyUnblockedLeadsOrdinarySameTier(t *testing.T) {
	router := dispatchtick.RouterPayload{
		NewlyUnblocked: []int{20},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"next":  {Issues: []int{20}, Count: 1, StepBudget: 1, Tree: []string{"internal/next"}},
			"other": {Issues: []int{30}, Count: 1, StepBudget: 100, Tree: []string{"internal/other"}},
		},
	}
	price, err := priceDispatchWavePayloadFilteredWithFreshCap(t.TempDir(), router, 1, 1, "", nil, 0, nil, 1, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.RunTargets) != 1 || price.RunTargets[0].Issue != 20 {
		t.Fatalf("run targets = %+v, want newly unblocked issue 20", price.RunTargets)
	}
}

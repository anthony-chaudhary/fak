package main

import (
	"encoding/json"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchReadySincePrecedenceAndUnknownFailClosed(t *testing.T) {
	tests := []struct {
		name                          string
		eligibility, updated, created int64
		want                          int64
	}{
		{"eligibility transition wins", 300, 200, 100, 300},
		{"updated fallback", 0, 200, 100, 200},
		{"created fallback", 0, 0, 100, 100},
		{"unknown", 0, 0, 0, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := dispatchReadySince(tc.eligibility, tc.updated, tc.created); got != tc.want {
				t.Fatalf("dispatchReadySince(%d, %d, %d) = %d, want %d",
					tc.eligibility, tc.updated, tc.created, got, tc.want)
			}
		})
	}

	now := int64(1_000_000)
	res := dispatchaging.Fold([]dispatchaging.Candidate{
		{ID: "unknown", BaseWeight: dispatchtick.PriorityWeightDefault, ReadySince: 0},
	}, dispatchaging.DefaultParams(now))
	if len(res.Order) != 1 || res.Order[0].WaitSeconds != 0 ||
		res.Order[0].Standing == dispatchaging.StandingStarved {
		t.Fatalf("unknown ReadySince must wait zero and never starve: %+v", res.Order)
	}
}

func TestReconcilePrereqReleasePersistsReadySinceProvenance(t *testing.T) {
	root := t.TempDir()
	now := int64(1_000_000)
	rememberDispatchIssueProvenance(root, dispatchIssueSourceRow{
		Issue: dispatchtick.Issue{Number: 10}, CreatedUnix: now - 10_000, UpdatedUnix: now - 300,
	})
	if err := writeDispatchPrereqState(dispatchPrereqStatePath(root), dispatchPrereqState{
		Schema: dispatchPrereqStateSchema,
		Held:   map[string][]string{"20": {"99"}},
	}); err != nil {
		t.Fatal(err)
	}
	payload := dispatchtick.RouterPayload{Issues: []dispatchtick.IssueRoute{
		{Number: 10, Lane: "cmd"},
		{Number: 20, Lane: "cmd"},
	}}
	got := reconcilePrereqReleaseAt(root, payload, now)
	if !reflect.DeepEqual(got.NewlyUnblocked, []int{20}) {
		t.Fatalf("newly unblocked = %v, want [20]", got.NewlyUnblocked)
	}
	state := readDispatchPrereqState(dispatchPrereqStatePath(root))
	if got := state.ReadySince["10"]; got != now-300 {
		t.Fatalf("ordinary ready issue stamp = %d, want updatedAt fallback %d", got, now-300)
	}
	if got := state.ReadySince["20"]; got != now {
		t.Fatalf("released issue stamp = %d, want transition time %d", got, now)
	}
}

func TestDispatchWaveCandidatesFeedDispatchAgingWithEligibilityTime(t *testing.T) {
	root := t.TempDir()
	now := int64(1_000_000)
	created := now - 7*24*3600
	updated := now - 3*3600
	eligible := now - 3600
	rememberDispatchIssueProvenance(root, dispatchIssueSourceRow{
		Issue: dispatchtick.Issue{Number: 20}, CreatedUnix: created, UpdatedUnix: updated,
	})
	if err := writeDispatchPrereqState(dispatchPrereqStatePath(root), dispatchPrereqState{
		Schema: dispatchPrereqStateSchema,
		Held:   map[string][]string{},
		ReadySince: map[string]int64{
			"20": eligible,
		},
	}); err != nil {
		t.Fatal(err)
	}
	router := dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 20, Lane: "cmd", Paths: []string{"cmd/fak/dispatch_ready_since.go"}},
			{Number: 30, Lane: "cmd", Paths: []string{"cmd/fak/dispatch_ready_since_test.go"}},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"cmd": {
				Issues: []int{20, 30}, Count: 2,
				Priority: map[int]int{20: dispatchtick.PriorityWeightP2},
			},
		},
	}
	price, err := priceDispatchWavePayload(root, router, 2, 2, "", nil, 0, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if len(price.Candidates) != 2 {
		t.Fatalf("candidates = %d, want 2", len(price.Candidates))
	}
	byIssue := map[int]dispatchWaveCandidate{}
	for _, c := range price.Candidates {
		byIssue[c.Issue] = c
	}
	if got := byIssue[20].ReadySince; got != eligible {
		t.Fatalf("issue 20 ReadySince = %d, want eligibility transition %d (created=%d updated=%d)",
			got, eligible, created, updated)
	}
	if got := byIssue[20].BaseWeight; got != dispatchtick.PriorityWeightP2 {
		t.Fatalf("issue 20 BaseWeight = %d, want %d", got, dispatchtick.PriorityWeightP2)
	}
	if got := byIssue[30].ReadySince; got != 0 {
		t.Fatalf("issue 30 ReadySince = %d, want unknown 0", got)
	}

	raw, err := json.Marshal(price)
	if err != nil {
		t.Fatal(err)
	}
	cands, err := decodeCandidates(raw)
	if err != nil {
		t.Fatal(err)
	}
	fold := dispatchaging.Fold(cands, dispatchaging.DefaultParams(now))
	waitByID := map[string]int64{}
	standingByID := map[string]dispatchaging.Standing{}
	for _, c := range fold.Order {
		waitByID[c.ID] = c.WaitSeconds
		standingByID[c.ID] = c.Standing
	}
	if got := waitByID[byIssue[20].ID]; got != now-eligible {
		t.Fatalf("issue 20 wait = %d, want now-eligibility = %d (not now-created = %d)",
			got, now-eligible, now-created)
	}
	if got := waitByID[byIssue[30].ID]; got != 0 {
		t.Fatalf("unknown issue wait = %d, want 0", got)
	}
	if standingByID[byIssue[30].ID] == dispatchaging.StandingStarved {
		t.Fatal("unknown issue must never starve")
	}
}

func TestPickDispatchLaneFeedsReadySinceIntoAgingOrder(t *testing.T) {
	root := t.TempDir()
	now := time.Now().Unix()
	if err := writeDispatchPrereqState(dispatchPrereqStatePath(root), dispatchPrereqState{
		Schema: dispatchPrereqStateSchema,
		Held:   map[string][]string{},
		ReadySince: map[string]int64{
			"20": now - 7*3600,
			"30": now - 10,
		},
	}); err != nil {
		t.Fatal(err)
	}

	oldRoute := dispatchRouteIssues
	defer func() { dispatchRouteIssues = oldRoute }()
	dispatchRouteIssues = func(string, io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Issues: []dispatchtick.IssueRoute{
				{Number: 20, Lane: "cmd"},
				{Number: 30, Lane: "cmd"},
			},
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"cmd": {
					Issues: []int{20, 30}, Count: 2,
					Priority: map[int]int{30: dispatchtick.PriorityWeightP0},
				},
			},
		}, nil
	}
	t.Setenv(dispatchtick.FAKDispatchAgingEnv, "1")
	pick, err := pickDispatchLane(root, io.Discard, "cmd", nil, false, "", dispatchGoalProfileThroughput, 0)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{20, 30}; !reflect.DeepEqual(pick.Numbers, want) {
		t.Fatalf("aged lane order = %v, want %v (starved ReadySince force-served)", pick.Numbers, want)
	}
}

package dispatchtick

import (
	"reflect"
	"testing"
)

func TestPriorityWeight(t *testing.T) {
	cases := []struct {
		name   string
		labels []string
		want   int
	}{
		{"no labels", nil, PriorityWeightDefault},
		{"unlabeled by priority", []string{"enhancement", "dispatch"}, PriorityWeightDefault},
		{"p0", []string{"priority/P0"}, PriorityWeightP0},
		{"p1", []string{"priority/P1"}, PriorityWeightP1},
		{"p2", []string{"priority/P2"}, PriorityWeightP2},
		{"heaviest wins", []string{"priority/P2", "priority/P0", "enhancement"}, PriorityWeightP0},
		{"trims whitespace", []string{" priority/P1 "}, PriorityWeightP1},
	}
	for _, tc := range cases {
		if got := PriorityWeight(tc.labels); got != tc.want {
			t.Errorf("%s: PriorityWeight(%v) = %d, want %d", tc.name, tc.labels, got, tc.want)
		}
	}
}

func TestOrderLaneCandidatesPriorityBeatsRecency(t *testing.T) {
	// The whole point of #1395: an old priority/P1 (#300) is picked ahead of a
	// newer unlabeled #1500 instead of pure recency.
	cands := []LaneCandidate{
		{Number: 1500, Weight: PriorityWeightDefault},
		{Number: 300, Weight: PriorityWeightP1},
	}
	if got := OrderLaneCandidates(cands, false); !reflect.DeepEqual(got, []int{300, 1500}) {
		t.Fatalf("order = %v, want [300 1500] (P1 before newer unlabeled)", got)
	}
}

func TestOrderLaneCandidatesNoPriorityIsByNumber(t *testing.T) {
	// Backward-compatible: when every weight is equal the result is exactly the
	// pre-priority by-number order, both directions.
	cands := []LaneCandidate{
		{Number: 20, Weight: PriorityWeightDefault},
		{Number: 5, Weight: PriorityWeightDefault},
		{Number: 13, Weight: PriorityWeightDefault},
	}
	if got := OrderLaneCandidates(cands, false); !reflect.DeepEqual(got, []int{5, 13, 20}) {
		t.Fatalf("default order = %v, want oldest-first [5 13 20]", got)
	}
	if got := OrderLaneCandidates(cands, true); !reflect.DeepEqual(got, []int{20, 13, 5}) {
		t.Fatalf("prefer-newest order = %v, want newest-first [20 13 5]", got)
	}
}

func TestOrderLaneCandidatesTierThenRecency(t *testing.T) {
	cands := []LaneCandidate{
		{Number: 10, Weight: PriorityWeightDefault},
		{Number: 900, Weight: PriorityWeightP2},
		{Number: 50, Weight: PriorityWeightP0},
		{Number: 800, Weight: PriorityWeightP1},
		{Number: 40, Weight: PriorityWeightP1},
	}
	// Tiers descend P0 > P1 > P2 > none; within the P1 tier oldest-first (40 < 800).
	if got := OrderLaneCandidates(cands, false); !reflect.DeepEqual(got, []int{50, 40, 800, 900, 10}) {
		t.Fatalf("tiered order = %v, want [50 40 800 900 10]", got)
	}
	// prefer-newest flips only the within-tier tiebreak (800 before 40), not the tiers.
	if got := OrderLaneCandidates(cands, true); !reflect.DeepEqual(got, []int{50, 800, 40, 900, 10}) {
		t.Fatalf("tiered prefer-newest = %v, want [50 800 40 900 10]", got)
	}
}

func TestBuildRouterPayloadOrdersLaneByPriority(t *testing.T) {
	// RouteIssues end-to-end: an old P1 issue sorts ahead of a newer unlabeled one
	// in the lane group, and the group carries the weight for the labeled issue
	// only (unlabeled issues are omitted and resolve to the default).
	p := RouteIssues(RouterInput{
		Workspace:  "C:/work/fak",
		Taxonomy:   routerTestTaxonomy,
		IssueLimit: 1000,
		Issues: []Issue{
			routerIssue(1500, "fix(gateway): newer noise", nil, ""),
			routerIssue(300, "fix(gateway): old but important", []string{"priority/P1"}, ""),
		},
	})
	grp := p.Lanes["gateway"]
	if !reflect.DeepEqual(grp.Issues, []int{300, 1500}) {
		t.Fatalf("gateway issues = %v, want [300 1500] (P1 first)", grp.Issues)
	}
	if grp.Priority[300] != PriorityWeightP1 {
		t.Fatalf("gateway priority[300] = %d, want %d", grp.Priority[300], PriorityWeightP1)
	}
	if _, ok := grp.Priority[1500]; ok {
		t.Fatalf("unlabeled #1500 must be omitted from the priority map, got %v", grp.Priority)
	}
}

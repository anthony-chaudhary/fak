package dispatchtick

import (
	"reflect"
	"testing"
)

func TestGenerationBucket(t *testing.T) {
	cases := []struct {
		name   string
		labels []string
		want   string
	}{
		{"now", []string{"gen/now"}, GenNow},
		{"next", []string{"gen/next"}, GenNext},
		{"second-next", []string{"gen/second-next"}, GenSecondNext},
		{"future", []string{"gen/future"}, GenFuture},
		{"no generation label", []string{"enhancement", "dispatch"}, GenUnclassified},
		{"no labels at all", nil, GenUnclassified},
		{"nearest horizon wins on double label", []string{"gen/future", "gen/now"}, GenNow},
		{"trims whitespace", []string{" gen/next "}, GenNext},
	}
	for _, tc := range cases {
		if got := GenerationBucket(tc.labels); got != tc.want {
			t.Errorf("%s: GenerationBucket(%v) = %q, want %q", tc.name, tc.labels, got, tc.want)
		}
	}
}

func TestGenerationEligibleDefaultWindow(t *testing.T) {
	// The Default Window rule: now/next launch by default, second-next/future/
	// unclassified are held until explicitly requested.
	cases := []struct {
		bucket string
		want   bool
	}{
		{GenNow, true},
		{GenNext, true},
		{GenSecondNext, false},
		{GenFuture, false},
		{GenUnclassified, false},
	}
	for _, tc := range cases {
		if got := GenerationEligible(tc.bucket, ""); got != tc.want {
			t.Errorf("GenerationEligible(%q, default) = %v, want %v", tc.bucket, got, tc.want)
		}
	}
}

func TestGenerationEligibleExplicitHorizon(t *testing.T) {
	cases := []struct {
		horizon string
		bucket  string
		want    bool
	}{
		{"now", GenNow, true},
		{"now", GenNext, false},
		{"next", GenNext, true},
		{"next", GenNow, false},
		{"second-next", GenSecondNext, true},
		{"second-next", GenFuture, false},
		{"future", GenFuture, true},
		{"future", GenNow, false},
		{"all", GenNow, true},
		{"all", GenSecondNext, true},
		{"all", GenFuture, true},
		{"all", GenUnclassified, false}, // still held for classification repair
	}
	for _, tc := range cases {
		if got := GenerationEligible(tc.bucket, tc.horizon); got != tc.want {
			t.Errorf("GenerationEligible(%q, %q) = %v, want %v", tc.bucket, tc.horizon, got, tc.want)
		}
	}
}

func TestOrderEligibleGenerationCandidatesHoldsLaterHorizonsByDefault(t *testing.T) {
	cands := []GenerationCandidate{
		{Number: 10, Weight: PriorityWeightDefault, Generation: GenNow},
		{Number: 20, Weight: PriorityWeightDefault, Generation: GenNext},
		{Number: 30, Weight: PriorityWeightP0, Generation: GenSecondNext},
		{Number: 40, Weight: PriorityWeightP0, Generation: GenFuture},
		{Number: 50, Weight: PriorityWeightP0, Generation: GenUnclassified},
	}
	got := OrderEligibleGenerationCandidates(cands, "", false)
	want := []int{10, 20}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("default window order = %v, want %v (a P0 second-next/future/unclassified issue must not outrank a default-weight now/next one)", got, want)
	}
}

func TestOrderEligibleGenerationCandidatesPriorityStillWinsInsideHorizon(t *testing.T) {
	cands := []GenerationCandidate{
		{Number: 10, Weight: PriorityWeightDefault, Generation: GenNow},
		{Number: 11, Weight: PriorityWeightP1, Generation: GenNow},
		{Number: 20, Weight: PriorityWeightDefault, Generation: GenNext},
	}
	got := OrderEligibleGenerationCandidates(cands, "", false)
	want := []int{11, 10, 20}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v, want %v (priority/P1 gen/now must still outrank default-weight gen/now and gen/next)", got, want)
	}
}

func TestOrderEligibleGenerationCandidatesExplicitHorizonAdmitsHeldBucket(t *testing.T) {
	cands := []GenerationCandidate{
		{Number: 10, Weight: PriorityWeightDefault, Generation: GenNow},
		{Number: 30, Weight: PriorityWeightDefault, Generation: GenSecondNext},
		{Number: 40, Weight: PriorityWeightDefault, Generation: GenFuture},
	}
	got := OrderEligibleGenerationCandidates(cands, "second-next", false)
	want := []int{30}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("--generation second-next order = %v, want %v (only the requested horizon should be admitted)", got, want)
	}
	got = OrderEligibleGenerationCandidates(cands, "all", false)
	want = []int{10, 30, 40}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("--generation all order = %v, want %v (every classified generation admitted)", got, want)
	}
}

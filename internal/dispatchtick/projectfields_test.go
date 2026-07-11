package dispatchtick

import "testing"

func TestMergeProjectFieldsOverridesPriorityAndFiltersStatus(t *testing.T) {
	p, issues := MergeProjectFields(map[int]int{10: PriorityWeightP2}, []int{10, 20, 30}, map[int]ProjectIssueFields{
		10: {Issue: 10, Priority: "P0", Status: "Ready"},
		20: {Issue: 20, Priority: "P1", Status: "Done"},
	})
	if len(issues) != 2 || issues[0] != 10 || issues[1] != 30 {
		t.Fatalf("issues=%v", issues)
	}
	if p[10] != PriorityWeightP0 || laneIssueWeight(p, 30) != PriorityWeightDefault {
		t.Fatalf("priority=%v", p)
	}
}

func TestProjectPriorityVocabulary(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want int
	}{{"P0", 1000}, {"High", 400}, {"Medium", 150}, {"", 60}} {
		if got := ProjectPriorityWeight(tc.in); got != tc.want {
			t.Fatalf("%q=%d want %d", tc.in, got, tc.want)
		}
	}
}

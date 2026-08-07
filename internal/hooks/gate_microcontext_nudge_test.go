package hooks

import "testing"

func TestParallelFabricNudgeWarnsOnUnboundedFanout(t *testing.T) {
	d := &StagedDiff{StagedPaths: []string{"x.md"}, AddedByFile: map[string][]AddedLine{"x.md": {{Text: "Add massively parallel agent fan-out with workers."}}}}
	r, e := checkParallelFabricNudge(d)
	if e != nil || len(r) != 1 {
		t.Fatalf("findings=%v err=%v", r, e)
	}
}
func TestParallelFabricNudgeSilentOnBoundedRoute(t *testing.T) {
	d := &StagedDiff{StagedPaths: []string{"x.md"}, AddedByFile: map[string][]AddedLine{"x.md": {{Text: "Add parallel agent fan-out with workers through micro-context bounded physical slots."}}}}
	r, e := checkParallelFabricNudge(d)
	if e != nil || len(r) != 0 {
		t.Fatalf("findings=%v err=%v", r, e)
	}
}

package worktype

import (
	"encoding/json"
	"testing"
)

func TestClassifyDispatchPromptClosedRulesAndUnknown(t *testing.T) {
	tests := []struct{ p, w string }{{"Issue title: fix(gateway): stop leak", "wp.issue-to-patch"}, {"Issue title: feat(info): add pane", "wp.spec-to-feature"}, {"Investigate this task", "unknown"}, {"Compare fix(foo) and feat(bar)", "unknown"}}
	for _, tt := range tests {
		a := ClassifyDispatchPrompt("trace-1", tt.p)
		b := ClassifyDispatchPrompt("trace-1", tt.p)
		if a.PatternID != tt.w {
			t.Fatalf("%q => %+v", tt.p, a)
		}
		aj, _ := json.Marshal(a)
		bj, _ := json.Marshal(b)
		if string(aj) != string(bj) {
			t.Fatal("nondeterministic bytes")
		}
	}
}

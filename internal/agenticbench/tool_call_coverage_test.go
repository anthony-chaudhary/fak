package agenticbench

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func readCoverageFixture[T any](t *testing.T, name string) []T {
	t.Helper()
	data, err := os.ReadFile("testdata/tool-call-coverage/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var values []T
	if err := json.Unmarshal(data, &values); err != nil {
		t.Fatal(err)
	}
	return values
}

func TestJoinToolCallCoverageMixedFixture(t *testing.T) {
	calls := readCoverageFixture[importedToolCall](t, "trajectory.json")
	events := readCoverageFixture[fakToolEvent](t, "fak-events.json")
	external := readCoverageFixture[externalToolDisposition](t, "external.json")
	classes := []toolCallClass{{Name: "shell", ToolNames: []string{"Bash"}}, {Name: "desktop", ToolNames: []string{"GUI.Click", "GUI.Hidden"}}, {Name: "file", ToolNames: []string{"File.Write"}}}

	receipt, err := joinToolCallCoverage(calls, classes, events, external, true)
	if err != nil {
		t.Fatal(err)
	}
	wantOverall := (toolCallCoverageCounts{Observed: 4, Mediated: 1, Unmediated: 1, Unknown: 1, Unobservable: 1})
	if receipt.Overall != wantOverall {
		t.Fatalf("overall = %+v, want %+v", receipt.Overall, wantOverall)
	}
	if got := receipt.Overall.MediatedPercent(); got != 25 {
		t.Fatalf("mediated percentage = %v, want 25", got)
	}
	if got := receipt.ByClass["desktop"]; got != (toolCallCoverageCounts{Observed: 2, Unmediated: 1, Unobservable: 1}) {
		t.Fatalf("desktop = %+v", got)
	}
	if receipt.FAKGovernedRun {
		t.Fatal("mixed run must not be declared FAK-governed")
	}
	if !strings.Contains(receipt.GovernanceRefusal, "refused fak_governed_run=true") || !strings.Contains(receipt.GovernanceRefusal, "endpoint/model-call coverage is not effect/tool-call coverage") {
		t.Fatalf("refusal = %q", receipt.GovernanceRefusal)
	}
	if got := receipt.Calls[0]; got.CallID != "call-1" || got.OriginalName != "Bash" || got.Class != "shell" || got.FAKEventID != "policy-101" {
		t.Fatalf("joined identity = %+v", got)
	}
}

func TestJoinToolCallCoverageFailsClosed(t *testing.T) {
	classes := []toolCallClass{{Name: "shell", ToolNames: []string{"Bash"}}}
	tests := []struct {
		name   string
		calls  []importedToolCall
		events []fakToolEvent
		want   string
	}{
		{"missing call id", []importedToolCall{{Name: "Bash", Observable: true}}, nil, "missing call_id"},
		{"duplicate call id", []importedToolCall{{CallID: "x", Name: "Bash", Observable: true}, {CallID: "x", Name: "Bash", Observable: true}}, nil, "duplicate imported call_id"},
		{"unmatched event", []importedToolCall{{CallID: "x", Name: "Bash", Observable: true}}, []fakToolEvent{{EventID: "e", CallID: "y"}}, "unobserved call_id"},
		{"undeclared class", []importedToolCall{{CallID: "x", Name: "Other", Observable: true}}, nil, "no declared normalized class"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := joinToolCallCoverage(tt.calls, classes, tt.events, nil, false)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestJoinToolCallCoverageAllowsFullyMediatedGovernance(t *testing.T) {
	receipt, err := joinToolCallCoverage([]importedToolCall{{CallID: "x", Name: "Bash", Observable: true}}, []toolCallClass{{Name: "shell", ToolNames: []string{"Bash"}}}, []fakToolEvent{{EventID: "e", CallID: "x"}}, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.FAKGovernedRun || receipt.GovernanceRefusal != "" {
		t.Fatalf("receipt = %+v", receipt)
	}
}

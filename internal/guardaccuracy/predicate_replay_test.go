package guardaccuracy

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestReplayPredicatesCountsFalseRejectionsAndEnforcesCeiling(t *testing.T) {
	fixture, err := os.Open("testdata/admitted_calls.json")
	if err != nil {
		t.Fatal(err)
	}
	defer fixture.Close()
	calls, err := LoadAdmittedCalls(fixture)
	if err != nil {
		t.Fatal(err)
	}

	broadShellBlock := ReplayRule{
		Name:    "block-all-shell",
		Ceiling: 1,
		Reject:  func(call AdmittedCall) bool { return call.Tool == "shell" },
	}
	destructiveOnly := ReplayRule{
		Name:    "destructive-shell-only",
		Ceiling: 1,
		Reject: func(call AdmittedCall) bool {
			return call.Tool == "shell" && bytes.Contains(call.Arguments, []byte(`"command":"rm `))
		},
	}

	reports, err := ReplayPredicates(calls, broadShellBlock, destructiveOnly)
	if err == nil || !strings.Contains(err.Error(), `"block-all-shell": false rejections 3 exceed ceiling 1`) {
		t.Fatalf("expected ceiling failure with measured count, got %v", err)
	}
	if len(reports) != 2 {
		t.Fatalf("reports=%d, want 2", len(reports))
	}
	if got := reports[0].FalseRejections; got != 3 {
		t.Errorf("broad false rejections=%d, want 3", got)
	}
	if got := reports[1].FalseRejections; got != 1 {
		t.Errorf("narrow false rejections=%d, want 1", got)
	}
	if got := reports[1].CallsEvaluated; got != 4 {
		t.Errorf("calls evaluated=%d, want 4", got)
	}

	if _, err := ReplayPredicates(calls, destructiveOnly); err != nil {
		t.Fatalf("candidate at its declared ceiling should pass: %v", err)
	}
}

func TestLoadAdmittedCallsRejectsUnknownOutcome(t *testing.T) {
	_, err := LoadAdmittedCalls(strings.NewReader(`[{"tool":"shell","admitted":true,"successful":false}]`))
	if err == nil || !strings.Contains(err.Error(), "requires admitted, successful calls") {
		t.Fatalf("expected corpus contract failure, got %v", err)
	}
}

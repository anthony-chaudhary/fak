package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func TestOrchestrationPlanAcceptsCurrentTaskText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "fan out independent checks in parallel", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Requested.Name != orchestration.ProfileAuto || got.Resolved.Profile != orchestration.ProfileUltracode || got.Resolved.TaskID == "" || got.Resolved.SOLRoute.Mode != orchestration.SOLUltra {
		t.Fatalf("resolved=%+v", got)
	}
	if strings.Contains(stdout.String(), "fan out independent checks") {
		t.Fatal("raw task text leaked into receipt")
	}
}

func TestOrchestrationPlanTaskTextKeepsTinyWorkDirect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "fix typo", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Resolved.Profile != orchestration.ProfileOff || got.Resolved.SOLRoute.Mode != orchestration.SOLStandard {
		t.Fatalf("resolved=%+v, want off/standard", got.Resolved)
	}
}

func TestOrchestrationPlanRequiresExactlyOneTaskSource(t *testing.T) {
	for _, args := range [][]string{{"plan"}, {"plan", "--task", "x", "--task-text", "y"}} {
		var stdout, stderr bytes.Buffer
		if code := runOrchestration(&stdout, &stderr, args); code != 2 {
			t.Fatalf("args=%v code=%d stderr=%q", args, code, stderr.String())
		}
	}
}

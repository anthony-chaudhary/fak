package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/executionroute"
)

func TestExecutionRouteCLIEmitsComposedDecision(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runExecutionRoute(&out, &errOut, []string{"--harnesses", "openai-generic,codex", "--rotatable", "--aspect", "tool_call", "--tool", "write_repository", "--session", "s-1", "--continuity", "--context-utilization", ".9"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got executionroute.Decision
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Harness.Profile.Name != "codex" {
		t.Fatalf("harness=%q", got.Harness.Profile.Name)
	}
	if got.Model.Plan.Primary() == "" {
		t.Fatal("missing model plan")
	}
	if got.Session.Action != executionroute.SessionCompactResume {
		t.Fatalf("session=%q", got.Session.Action)
	}
}

func TestExecutionRouteCLIRejectsImpossibleHarness(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runExecutionRoute(&out, &errOut, []string{"--harnesses", "openai-generic", "--rotatable"})
	if code != 1 {
		t.Fatalf("code=%d want 1", code)
	}
}

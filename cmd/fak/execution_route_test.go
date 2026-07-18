package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
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

func TestExecutionRouteCLIDescriptorPairDrivesCompat(t *testing.T) {
	source := `{"version":1,"id":"s-1","harness":"claude","wire":"anthropic","model_family":"fable","tool_protocol":"native","transcript_format":"cc-jsonl","required_state":["thinking"]}`
	identical := `{"version":1,"harness":"claude","wire":"anthropic","model_family":"fable","tool_protocol":"native","transcript_format":"cc-jsonl"}`
	sourcePath := filepath.Join(t.TempDir(), "source.json")
	if err := os.WriteFile(sourcePath, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}

	// Identical envelopes: the descriptor channel, not the booleans, decides resume.
	var out, errOut bytes.Buffer
	code := runExecutionRoute(&out, &errOut, []string{"--session", "s-1", "--source-descriptor", sourcePath, "--target-descriptor", identical})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got executionroute.Decision
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Session.Action != executionroute.SessionResume {
		t.Fatalf("action=%q want resume", got.Session.Action)
	}
	if got.Session.Compat == nil || got.Session.Compat.Verdict != executionroute.CompatIdentical {
		t.Fatalf("compat=%+v want identical verdict", got.Session.Compat)
	}
	wantAxes := map[executionroute.CompatAxis]bool{
		executionroute.AxisHarness: true, executionroute.AxisWire: true, executionroute.AxisModelFamily: true,
		executionroute.AxisToolProtocol: true, executionroute.AxisTranscriptFormat: true,
	}
	for _, axis := range got.Session.Compat.Axes {
		delete(wantAxes, axis.Axis)
	}
	if len(wantAxes) != 0 {
		t.Fatalf("compatibility result omitted axes: %v", wantAxes)
	}

	// A changed model family strands required thinking: the move is REFUSED.
	moved := `{"version":1,"harness":"claude","wire":"anthropic","model_family":"gpt","tool_protocol":"native","transcript_format":"cc-jsonl"}`
	out.Reset()
	errOut.Reset()
	code = runExecutionRoute(&out, &errOut, []string{"--session", "s-1", "--source-descriptor", sourcePath, "--target-descriptor", moved})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	got = executionroute.Decision{}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Session.Compat == nil || got.Session.Compat.Verdict != executionroute.CompatIncompatible || !got.Session.Compat.Refused {
		t.Fatalf("compat=%+v want refused incompatible", got.Session.Compat)
	}
	if got.Session.Action != executionroute.SessionStart {
		t.Fatalf("action=%q want start after refusal", got.Session.Action)
	}
}

func TestExecutionRouteCLIRefusesLoneDescriptor(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runExecutionRoute(&out, &errOut, []string{"--session", "s-1", "--source-descriptor", `{"version":1,"id":"s-1"}`})
	if code != 2 {
		t.Fatalf("code=%d want 2 (descriptor flags must be paired)", code)
	}
}

func TestExecutionRouteCLIRejectsImpossibleHarness(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runExecutionRoute(&out, &errOut, []string{"--harnesses", "openai-generic", "--rotatable"})
	if code != 1 {
		t.Fatalf("code=%d want 1", code)
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superstream"
)

func TestSuperstreamCLIPlan(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"plan"})
	if code != 0 {
		t.Fatalf("runSuperstream plan exit = %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Super Workstream: stream-sample") {
		t.Fatalf("expected stream-sample header, got: %s", out)
	}
	if !strings.Contains(out, "task-gateway-parity") {
		t.Fatalf("expected queue item in output, got: %s", out)
	}
}

func TestSuperstreamCLIStepJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"step", "--json"})
	if code != 0 {
		t.Fatalf("runSuperstream step exit = %d, stderr: %s", code, stderr.String())
	}
	var dec superstream.StepDecision
	if err := json.Unmarshal(stdout.Bytes(), &dec); err != nil {
		t.Fatalf("failed to parse JSON decision: %v, raw: %s", err, stdout.String())
	}
	if dec.Action != superstream.ActionAcquireLease {
		t.Fatalf("dec.Action = %s, want %s", dec.Action, superstream.ActionAcquireLease)
	}
	if dec.Item == nil || dec.Item.ID != "task-gateway-parity" {
		t.Fatalf("dec.Item = %v, want task-gateway-parity", dec.Item)
	}
}

func TestSuperstreamCLICarryover(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"carryover", "--json"})
	if code != 0 {
		t.Fatalf("runSuperstream carryover exit = %d, stderr: %s", code, stderr.String())
	}
	var seed superstream.StreamCarryoverSeed
	if err := json.Unmarshal(stdout.Bytes(), &seed); err != nil {
		t.Fatalf("failed to parse JSON carryover: %v, raw: %s", err, stdout.String())
	}
	if seed.Schema != superstream.CarryoverSchema {
		t.Fatalf("seed.Schema = %s, want %s", seed.Schema, superstream.CarryoverSchema)
	}
	if seed.StreamID != "stream-sample" {
		t.Fatalf("seed.StreamID = %s, want stream-sample", seed.StreamID)
	}
}

func TestSuperstreamCLIStatus(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"status"})
	if code != 0 {
		t.Fatalf("runSuperstream status exit = %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Super Workstream Status: stream-sample") {
		t.Fatalf("expected status header, got: %s", out)
	}
	if !strings.Contains(out, "Context Safety: SAFE") {
		t.Fatalf("expected Context Safety SAFE, got: %s", out)
	}
}

func writeStreamSpecFile(t *testing.T, path string, spec superstream.StreamSpec) {
	t.Helper()
	b, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		t.Fatalf("marshal spec: %v", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write spec %s: %v", path, err)
	}
}

func writeStepResultFile(t *testing.T, path string, result superstream.StepResult) {
	t.Helper()
	b, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write result %s: %v", path, err)
	}
}

// TestSuperstreamStepApplyPersistsStatus drives the real state machine over two
// items through five separate invocations and then reads the durable result
// back: progress survives process boundaries instead of reading 0 completed.
func TestSuperstreamStepApplyPersistsStatus(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir) // the registry is cwd-relative: keep it out of the package tree
	specPath := filepath.Join(dir, "stream.json")
	writeStreamSpecFile(t, specPath, superstream.StreamSpec{
		ID:     "stream-persist",
		Intent: "persist-progress",
		Queue: []superstream.WorkItem{
			{ID: "task-1", Title: "first", Lane: "docs", Witness: "go test ./docs"},
			{ID: "task-2", Title: "second", Lane: "gateway", Witness: "go test ./internal/gateway"},
		},
	})

	// Each invocation: the witnessed result of the action the previous step
	// asked for. The fifth (ADVANCE_QUEUE) needs no result fields.
	transitions := []struct {
		want   superstream.StepAction
		result *superstream.StepResult
	}{
		{superstream.ActionAcquireLease, &superstream.StepResult{LeaseAcquired: true, FencingToken: "fence-1"}},
		{superstream.ActionExecuteItem, &superstream.StepResult{ExecutedTurns: 3, ExecutedTokens: 1200}},
		{superstream.ActionWitnessAndCommit, &superstream.StepResult{WitnessSuccess: true, CommitSHA: "abc123", WitnessOutput: "ok"}},
		{superstream.ActionReleaseLease, &superstream.StepResult{LeaseReleased: true}},
		{superstream.ActionAdvanceQueue, nil},
	}

	for i, tr := range transitions {
		argv := []string{"step", "--spec", specPath, "--apply", "--json"}
		if tr.result != nil {
			resultPath := filepath.Join(dir, fmt.Sprintf("result-%d.json", i+1))
			writeStepResultFile(t, resultPath, *tr.result)
			argv = append(argv, "--result", resultPath)
		}
		var stdout, stderr bytes.Buffer
		code := runSuperstream(&stdout, &stderr, argv)
		if code != 0 {
			t.Fatalf("step %d exit = %d, stderr: %s", i+1, code, stderr.String())
		}
		var payload struct {
			superstream.StepDecision
			Applied   bool                    `json:"applied"`
			Persisted bool                    `json:"persisted"`
			State     superstream.StreamState `json:"state"`
		}
		if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
			t.Fatalf("step %d: parse JSON: %v, raw: %s", i+1, err, stdout.String())
		}
		if payload.Action != tr.want {
			t.Fatalf("step %d action = %s, want %s (reason: %s)", i+1, payload.Action, tr.want, payload.Reason)
		}
		if !payload.Applied || !payload.Persisted {
			t.Fatalf("step %d applied=%v persisted=%v, want both true", i+1, payload.Applied, payload.Persisted)
		}
	}

	// status --spec reads the persisted state: real progress, item 2 now active.
	var stdout, stderr bytes.Buffer
	if code := runSuperstream(&stdout, &stderr, []string{"status", "--spec", specPath}); code != 0 {
		t.Fatalf("status exit = %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "1 completed") {
		t.Fatalf("status must report persisted progress, got: %s", out)
	}
	if !strings.Contains(out, "Active Item: task-2") {
		t.Fatalf("status must show task-2 as active, got: %s", out)
	}

	// status --all serves the registry row without reading the spec.
	stdout.Reset()
	stderr.Reset()
	if code := runSuperstream(&stdout, &stderr, []string{"status", "--all"}); code != 0 {
		t.Fatalf("status --all exit = %d, stderr: %s", code, stderr.String())
	}
	out = stdout.String()
	if !strings.Contains(out, "stream-persist") {
		t.Fatalf("status --all must list the stream id, got: %s", out)
	}
	if !strings.Contains(out, "1/2") {
		t.Fatalf("status --all must report 1/2 progress, got: %s", out)
	}
	if !strings.Contains(out, "COMPLETED/TOTAL") {
		t.Fatalf("status --all must render the served table header, got: %s", out)
	}
}

func TestSuperstreamStatusAllEmptyRegistry(t *testing.T) {
	t.Chdir(t.TempDir())
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"status", "--all"})
	if code != 0 {
		t.Fatalf("status --all exit = %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "No stream registry") {
		t.Fatalf("expected empty-registry notice, got: %s", stdout.String())
	}
}

func TestSuperstreamStepApplySampleSpecStateless(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"step", "--apply"})
	if code != 0 {
		t.Fatalf("step --apply exit = %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "stateless") {
		t.Fatalf("expected a stateless note on stderr, got: %s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "Persisted: none") {
		t.Fatalf("expected a no-persist line, got: %s", stdout.String())
	}
	if _, err := os.Stat(filepath.Join(dir, ".fak")); !os.IsNotExist(err) {
		t.Fatalf("sample spec must write no registry: .fak stat err = %v", err)
	}
}

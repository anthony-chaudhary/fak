package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskdecision"
)

func TestRunTaskDecisionAppendAndListJSON(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "decisions.jsonl")
	var out, errb bytes.Buffer
	code := runTask(&out, &errb, []string{
		"decision", "append",
		"--task", "issue-2122",
		"--decision", "Use a bounded reset contributor",
		"--rationale", "post-compaction agents need the why",
		"--evidence-ref", "test:task-decision",
		"--open-thread", "wire live reset loader",
		"--log", logPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("append exit = %d; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var appended taskDecisionResult
	if err := json.Unmarshal(out.Bytes(), &appended); err != nil {
		t.Fatalf("append json: %v\n%s", err, out.String())
	}
	if appended.Entry.Schema != taskdecision.Schema || appended.Entry.TaskID != "issue-2122" {
		t.Fatalf("appended entry = %+v", appended.Entry)
	}

	out.Reset()
	errb.Reset()
	code = runTask(&out, &errb, []string{"decision", "list", "--task", "issue-2122", "--log", logPath, "--json"})
	if code != 0 {
		t.Fatalf("list exit = %d; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var listed taskDecisionResult
	if err := json.Unmarshal(out.Bytes(), &listed); err != nil {
		t.Fatalf("list json: %v\n%s", err, out.String())
	}
	if len(listed.Entries) != 1 || listed.Entries[0].Decision != "Use a bounded reset contributor" {
		t.Fatalf("listed entries = %+v", listed.Entries)
	}
}

func TestRunTaskDecisionAppendRequiresEvidence(t *testing.T) {
	var out, errb bytes.Buffer
	code := runTask(&out, &errb, []string{
		"decision", "append",
		"--task", "issue-2122",
		"--decision", "Remember this",
		"--rationale", "because context resets",
		"--log", filepath.Join(t.TempDir(), "decisions.jsonl"),
	})
	if code != 2 {
		t.Fatalf("exit = %d, want usage 2; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "evidence_ref") {
		t.Fatalf("stderr = %q", errb.String())
	}
}

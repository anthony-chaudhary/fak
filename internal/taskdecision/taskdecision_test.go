package taskdecision

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAppendLoadFiltersTaskAndBounds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "decisions.jsonl")
	entries := []Entry{
		{TaskID: "task-1", Decision: "use merge-tree", Rationale: "matches git", EvidenceRef: "issue#1"},
		{TaskID: "other", Decision: "ignore", Rationale: "other task", EvidenceRef: "issue#2"},
		{TaskID: "task-1", Decision: "add fixture", Rationale: "proves conflict", EvidenceRef: "test#1", OpenThreads: []string{"docs later"}},
		{TaskID: "task-1", Decision: "ship", Rationale: "witness green", EvidenceRef: "commit abc"},
	}
	for _, entry := range entries {
		if err := Append(path, entry); err != nil {
			t.Fatal(err)
		}
	}
	got, err := Load(path, "task-1", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2: %+v", len(got), got)
	}
	if got[0].Decision != "add fixture" || got[1].Decision != "ship" {
		t.Fatalf("bounded entries = %+v", got)
	}
	if got[0].Schema != Schema {
		t.Fatalf("schema default = %q", got[0].Schema)
	}
}

func TestRenderCarriesStructuredFields(t *testing.T) {
	text := Render([]Entry{{
		TaskID:      "task-1",
		Decision:    "keep the parser strict",
		Rationale:   "bad JSON should fail closed",
		EvidenceRef: "go test ./internal/taskdecision",
		OpenThreads: []string{"wire CLI", "document reset reload"},
	}})
	for _, want := range []string{"keep the parser strict", "bad JSON should fail closed", "go test ./internal/taskdecision", "wire CLI; document reset reload"} {
		if !strings.Contains(text, want) {
			t.Fatalf("render missing %q:\n%s", want, text)
		}
	}
}

func TestValidateRequiresEvidence(t *testing.T) {
	err := Validate(Normalize(Entry{TaskID: "t", Decision: "d", Rationale: "r"}))
	if err == nil || !strings.Contains(err.Error(), "evidence_ref") {
		t.Fatalf("Validate err = %v, want evidence_ref failure", err)
	}
}

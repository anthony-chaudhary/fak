package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/study"
)

func TestStudyInvocationPersistsAndFreshSearchFindsDecision(t *testing.T) {
	root := t.TempDir()
	store := filepath.Join(root, "store")
	input := filepath.Join(root, "record.json")
	r := study.Record{Schema: study.Schema, Title: "bounded source decision", Observed: "2026-08-26", Sources: []study.Source{{URL: "https://example.test/source", Revision: "v1"}}, Observations: []string{"pinned source"}, Candidates: []study.Candidate{{Effect: "candidate survives fresh invocation", Status: "PARTIAL", Disposition: "WATCH", Evidence: []string{"docs/evidence.md"}, Issue: "https://github.com/anthony-chaudhary/fak/issues/8613"}}}
	b, _ := json.Marshal(r)
	if err := os.WriteFile(input, b, 0600); err != nil {
		t.Fatal(err)
	}
	var out, errout bytes.Buffer
	if code := runStudy(&out, &errout, []string{"add", "--file", input, "--store", store}); code != 0 {
		t.Fatalf("add=%d stderr=%s", code, errout.String())
	}
	var receipt study.Receipt
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.ID == "" {
		t.Fatal("missing durable ID")
	}
	out.Reset()
	errout.Reset()
	if code := runStudy(&out, &errout, []string{"search", "survives fresh", "--store", store}); code != 0 {
		t.Fatalf("search=%d stderr=%s", code, errout.String())
	}
	var matches []study.Match
	if err := json.Unmarshal(out.Bytes(), &matches); err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 || matches[0].ID != receipt.ID || matches[0].Record.Candidates[0].Issue == "" {
		t.Fatalf("fresh invocation did not discover decision: %s", out.String())
	}
}

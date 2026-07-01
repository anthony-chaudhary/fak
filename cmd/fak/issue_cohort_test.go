package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func cohortTestCandidate(key string, paths []string) issuecontract.Candidate {
	return issuecontract.Candidate{
		Schema:         issuecontract.Schema,
		Key:            key,
		Title:          "leaf " + key,
		ParentRef:      "epic #1",
		CurrentState:   "not yet done",
		WhyNow:         "unblocks the next leaf",
		WorkingSpine:   "make the working path more true",
		InScope:        "one file",
		OutOfScope:     "everything else",
		DoneCondition:  "the file changes",
		Witness:        "go test ./... passes",
		AcceptanceGate: "make ci",
		ClosureBinding: "commit cites #1 and (fak leaf)",
		Paths:          paths,
	}
}

func writeCohortPlan(t *testing.T, cands []issuecontract.Candidate) string {
	t.Helper()
	b, err := json.Marshal(cands)
	if err != nil {
		t.Fatalf("marshal candidates: %v", err)
	}
	path := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write plan: %v", err)
	}
	return path
}

func TestRunIssueCohortJSON(t *testing.T) {
	cands := []issuecontract.Candidate{
		cohortTestCandidate("a", []string{"internal/foo/**"}),
		cohortTestCandidate("b", []string{"internal/foo/bar.go"}), // overlaps a
		cohortTestCandidate("c", []string{"internal/baz/**"}),     // disjoint
	}
	big := cohortTestCandidate("big", []string{"internal/big/**"})
	big.ExpectedSteps = 20
	cands = append(cands, big)

	path := writeCohortPlan(t, cands)
	var stdout, stderr bytes.Buffer
	if code := runIssueCohort(&stdout, &stderr, []string{"--from-plan", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}

	var plan issuecohort.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("decode plan: %v\n%s", err, stdout.String())
	}
	if plan.Schema != issuecohort.Schema {
		t.Fatalf("schema = %q", plan.Schema)
	}
	if plan.Dispatchable != 3 {
		t.Fatalf("dispatchable = %d, want 3", plan.Dispatchable)
	}
	if plan.Subdividable != 1 {
		t.Fatalf("subdividable = %d, want 1", plan.Subdividable)
	}
	if plan.CollisionPairs != 1 {
		t.Fatalf("collision pairs = %d, want 1 (a overlaps b)", plan.CollisionPairs)
	}
	// a and b collide, so a wave cannot hold both; c is disjoint from both.
	if plan.NumWaves != 2 {
		t.Fatalf("num waves = %d, want 2", plan.NumWaves)
	}
}

func TestRunIssueCohortText(t *testing.T) {
	path := writeCohortPlan(t, []issuecontract.Candidate{
		cohortTestCandidate("a", []string{"internal/foo/**"}),
	})
	var stdout, stderr bytes.Buffer
	if code := runIssueCohort(&stdout, &stderr, []string{"--from-plan", path}); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte("issue-cohort:")) {
		t.Fatalf("text output missing header:\n%s", stdout.String())
	}
}

func TestRunIssueCohortMissingArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runIssueCohort(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
}

func TestRunIssueCohortBothSourcesRejected(t *testing.T) {
	path := writeCohortPlan(t, []issuecontract.Candidate{
		cohortTestCandidate("a", []string{"internal/foo/**"}),
	})
	var stdout, stderr bytes.Buffer
	if code := runIssueCohort(&stdout, &stderr, []string{"--from-plan", path, "--from-issues", path}); code != 2 {
		t.Fatalf("exit = %d, want 2 (exactly one source)", code)
	}
}

// TestRunIssueCohortFromIssues plans concurrency-safe waves over EXISTING open
// issues by parsing their bodies with the same issuecontract sections the
// contract audit uses.
func TestRunIssueCohortFromIssues(t *testing.T) {
	body := func(paths string) string {
		return "## Current state\n\nnot done\n\n" +
			"## In scope\n\none file\n\n## Out of scope\n\nrest\n\n" +
			"## Done condition\n\nit changes\n\n## Witness\n\ngo test passes\n\n" +
			"## Parent context\n\nepic #1\n\n## Why this is next\n\nunblocks\n\n" +
			"## Working spine\n\nmake it true\n\n## Acceptance gate\n\nmake ci\n\n" +
			"## Closure binding\n\ncites #1 (fak leaf)\n\n## Likely files\n\n" + paths + "\n"
	}
	issues := []issuecontract.IssueDraft{
		{Number: 10, Title: "leaf ten", Body: body("- `internal/foo/**`")},
		{Number: 11, Title: "leaf eleven", Body: body("- `internal/foo/bar.go`")}, // overlaps #10
		{Number: 12, Title: "leaf twelve", Body: body("- `internal/baz/**`")},     // disjoint
	}
	b, err := json.Marshal(issues)
	if err != nil {
		t.Fatalf("marshal issues: %v", err)
	}
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write issues: %v", err)
	}

	var stdout, stderr bytes.Buffer
	if code := runIssueCohort(&stdout, &stderr, []string{"--from-issues", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	var plan issuecohort.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("decode plan: %v\n%s", err, stdout.String())
	}
	if plan.Dispatchable != 3 {
		t.Fatalf("dispatchable = %d, want 3", plan.Dispatchable)
	}
	if plan.CollisionPairs != 1 || plan.NumWaves != 2 {
		t.Fatalf("collisions=%d waves=%d, want 1 collision / 2 waves", plan.CollisionPairs, plan.NumWaves)
	}
}

func TestRunIssueCohortRoutedViaRunIssue(t *testing.T) {
	path := writeCohortPlan(t, []issuecontract.Candidate{
		cohortTestCandidate("a", []string{"internal/foo/**"}),
	})
	var stdout, stderr bytes.Buffer
	if code := runIssue(&stdout, &stderr, []string{"cohort", "--from-plan", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(issuecohort.Schema)) {
		t.Fatalf("routed output missing schema:\n%s", stdout.String())
	}
}

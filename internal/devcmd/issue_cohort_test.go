package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func cohortTestCandidate(key string, paths []string) issuepolicy.Candidate {
	return issuepolicy.Candidate{
		Schema:             issuepolicy.Schema,
		Key:                key,
		Title:              "leaf " + key,
		ParentRef:          "epic #1",
		CurrentState:       "not yet done",
		WhyNow:             "unblocks the next leaf",
		WorkingSpine:       "make the working path more true",
		InScope:            "one file",
		OutOfScope:         "everything else",
		DoneCondition:      "the file changes",
		Witness:            "go test ./... passes",
		AcceptanceGate:     "make ci",
		ClosureBinding:     "commit cites #1 and (fak leaf)",
		Paths:              paths,
		ExpectedSteps:      3,
		WorkEstimate:       "Estimate: 3 points",
		ScopeContribution:  "Contribution: 3/13 points",
		CompletionStandard: "production",
		TargetEnvelope:     "- acceptance pass rate: = 100 percent",
		WitnessedEnvelope:  "- acceptance pass rate: = 100 percent",
	}
}

func writeCohortPlan(t *testing.T, cands []issuepolicy.Candidate) string {
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
	cands := []issuepolicy.Candidate{
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
	path := writeCohortPlan(t, []issuepolicy.Candidate{
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
	path := writeCohortPlan(t, []issuepolicy.Candidate{
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
			"## Closure binding\n\ncites #1 (fak leaf)\n\n" +
			"## Work estimate\n\nEstimate: 3 points\n\n## Overall completion contribution\n\nParent scope baseline: #1, 13 points. Contribution: 3/13 points.\n\n" +
			"## Completion standard\n\nproduction\n\n## Target operating envelope\n\n- acceptance pass rate: = 100 percent\n\n" +
			"## Witnessed operating envelope\n\n- acceptance pass rate: = 100 percent\n\n## Likely files\n\n" + paths + "\n"
	}
	issues := []issuepolicy.IssueDraft{
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

func TestRunIssueCohortFromIssuesCarriesCanonicalProblemFrame(t *testing.T) {
	body := "## Value\n- Centrality: Enabling (managed-context outcome)\n- P1: advanced - shared context is reused\n- P2: advanced - intake rework is removed\n- P3: preserved - classification remains revisable\n- P4: advanced - dispatch sees the frame\n"
	issues := []issuepolicy.IssueDraft{{Number: 44, Title: "enabling leaf", Body: body}}
	b, err := json.Marshal(issues)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runIssueCohort(&stdout, &stderr, []string{"--from-issues", path, "--json", "--strict-project-work=false"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var plan issuecohort.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if len(plan.Portfolio) != 1 || plan.Portfolio[0].Centrality != issuepolicy.CentralityEnabling || plan.Portfolio[0].CentralityTarget != "managed-context outcome" {
		t.Fatalf("canonical frame lost: %+v", plan.Portfolio)
	}
}

func TestRunIssueCohortDispatchWaveCarriesCanonicalProblemFrame(t *testing.T) {
	candidate := cohortTestCandidate("enabling", []string{"internal/enabling/**"})
	candidate.ProblemFrame = issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: issuepolicy.CentralityEnabling, CentralityTarget: "managed-context outcome", Checks: map[string]issuepolicy.ProblemCheck{}}
	path := writeCohortPlan(t, []issuepolicy.Candidate{candidate})
	var stdout, stderr bytes.Buffer
	if code := runIssueCohort(&stdout, &stderr, []string{"--from-plan", path, "--json", "--strict-project-work=false"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var plan issuecohort.Plan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if len(plan.Waves) != 1 || plan.Waves[0].Members[0].ProblemFrame.Centrality != issuepolicy.CentralityEnabling {
		t.Fatalf("dispatch wave lost frame: %+v", plan.Waves)
	}
}

func TestRunIssueCohortRoutedViaRunIssue(t *testing.T) {
	path := writeCohortPlan(t, []issuepolicy.Candidate{
		cohortTestCandidate("a", []string{"internal/foo/**"}),
	})
	var stdout, stderr bytes.Buffer
	if code := RunIssue(&stdout, &stderr, []string{"cohort", "--from-plan", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(issuecohort.Schema)) {
		t.Fatalf("routed output missing schema:\n%s", stdout.String())
	}
}

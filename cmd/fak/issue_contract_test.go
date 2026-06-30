package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func TestIssueContractReviewsDispatchableCandidate(t *testing.T) {
	path := writeIssueContractJSON(t, completeIssueCandidate())
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--file", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		OK      bool `json:"ok"`
		Reviews []struct {
			OK              bool   `json:"ok"`
			Key             string `json:"key"`
			Dispatchability string `json:"dispatchability"`
			Score           struct {
				Total int `json:"total"`
			} `json:"score"`
			SpinePriority struct {
				Total int `json:"total"`
			} `json:"spine_priority"`
		} `json:"reviews"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if !got.OK || len(got.Reviews) != 1 || !got.Reviews[0].OK {
		t.Fatalf("review = %+v, want one OK review", got)
	}
	if got.Reviews[0].Key != "task_push_next/strict-scope" ||
		got.Reviews[0].Dispatchability != issuecontract.Dispatchable ||
		got.Reviews[0].Score.Total != 100 ||
		got.Reviews[0].SpinePriority.Total != 100 {
		t.Fatalf("review identity = %+v", got.Reviews[0])
	}
}

func TestIssueContractRefusesVagueCandidate(t *testing.T) {
	c := completeIssueCandidate()
	c.OutOfScope = ""
	c.DoneCondition = ""
	c.Lane = ""
	c.Paths = nil
	path := writeIssueContractJSON(t, c)
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--file", path})
	if code != 3 {
		t.Fatalf("exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	rendered := out.String()
	for _, want := range []string{"ISSUE_SCOPE_INCOMPLETE", "ISSUE_UNROUTED", "missing: out_of_scope", "missing: done_condition"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered review missing %q:\n%s", want, rendered)
		}
	}
}

func TestIssueContractLiveRequiresDedupeArmor(t *testing.T) {
	path := writeIssueContractJSON(t, completeIssueCandidate())
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--file", path, "--live", "--json"})
	if code != 3 {
		t.Fatalf("unarmored live exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), issuecontract.ReasonLiveUnarmored) {
		t.Fatalf("unarmored live output missing %s:\n%s", issuecontract.ReasonLiveUnarmored, out.String())
	}

	out.Reset()
	errb.Reset()
	code = runIssue(&out, &errb, []string{
		"contract", "--file", path, "--live", "--dedupe-checked", "--dedupe-cap", "300", "--json",
	})
	if code != 0 {
		t.Fatalf("armed live exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
}

func TestIssueContractFromPlanReviewsCandidatesArray(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plan.json")
	body := map[string]any{"candidates": []issuecontract.Candidate{completeIssueCandidate()}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--from-plan", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"mode": "plan"`) {
		t.Fatalf("plan mode missing:\n%s", out.String())
	}
}

func TestIssueContractFromIssuesReviewsGitHubRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuecontract.IssueDraft{{
		Number: 1450,
		Title:  "guardrsi: require block reasons",
		Body: strings.Join([]string{
			"### Parent context",
			"guard-verdict-rsi",
			"### Current state",
			"A guard verdict can reach the journal without a closed reason.",
			"### Why this is next",
			"Reasonless blocks weaken the guard before any tuning work.",
			"### Working spine",
			"Every blocked guard verdict records one closed-vocabulary reason.",
			"### Priority context",
			"Working path: guard preflight to closed reason.",
			"Current blocker: reasonless guard blocks hide the failing gate.",
			"Unblocks: guard tuning depends on reason buckets.",
			"Not polish: fix the smallest guard hole before threshold optimization.",
			"### In scope",
			"Add the missing classification and one regression fixture.",
			"### Out of scope",
			"Do not retune guard thresholds.",
			"### Done condition",
			"The fixture no longer emits a blank reason.",
			"### Witness",
			"go test ./internal/guardrsi",
			"### Acceptance gate",
			"go test ./internal/guardrsi ./internal/guardroute",
			"### Lane",
			"guardrsi",
			"### Path hints",
			"- `internal/guardrsi/**`",
			"### Boundary notes",
			"- Public issue only.",
			"### Closure binding",
			"Resolving commit cites #N and carries `(fak guardrsi)`.",
		}, "\n"),
		Labels: []issuecontract.IssueLabel{{Name: "guardrsi"}},
	}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--from-issues", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"mode": "issues"`) ||
		!strings.Contains(out.String(), `"key": "issue/1450"`) ||
		!strings.Contains(out.String(), `"dispatchability": "dispatchable"`) {
		t.Fatalf("issue review missing expected fields:\n%s", out.String())
	}
}

func TestIssueContractFromIssuesRefusesVagueRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issues.json")
	body := []issuecontract.IssueDraft{{Number: 1451, Title: "make it better", Body: "### Current state\nExists.\n"}}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runIssue(&out, &errb, []string{"contract", "--from-issues", path})
	if code != 3 {
		t.Fatalf("exit = %d, want 3\nstderr:\n%s\nstdout:\n%s", code, errb.String(), out.String())
	}
	for _, want := range []string{"issue/1451", issuecontract.ReasonScopeIncomplete, issuecontract.ReasonUnrouted} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("rendered review missing %q:\n%s", want, out.String())
		}
	}
}

func completeIssueCandidate() issuecontract.Candidate {
	return issuecontract.Candidate{
		Schema:          issuecontract.Schema,
		Key:             "task_push_next/strict-scope",
		Title:           "taskmgr: enforce strict handoff scope",
		ParentRef:       "task_push_next",
		CurrentState:    "Task handoff can already create stable follow-up issues.",
		WhyNow:          "Generated issues are the next weak point before dispatch.",
		WorkingSpine:    "A verified task completion creates one scoped follow-up issue.",
		PriorityContext: "Working path: clean Stop handoff -> scoped issue -> dispatch. Current blocker: vague follow-ups waste dispatch cycles. Unblocks: guard live handoff. Not polish: enforce the smallest leaf before optimization.",
		WorkUnit:        "leaf",
		ExpectedSteps:   3,
		Assumptions:     []string{"The handoff producer can derive the candidate before syncing."},
		ConfusionRisks:  []string{"A broad follow-up can be mistaken for an epic unless scoped."},
		Coordination:    []string{"Do not dispatch concurrently with taskmgr handoff body edits."},
		Trigger:         "A verified completion handoff proposes this next leaf.",
		BatchPolicy:     "At most two follow-up issues per handoff; update by marker key on rerun.",
		InScope:         "Review the next-step candidate and render scoped sections.",
		OutOfScope:      "Do not optimize issue routing or add new scorecards.",
		DoneCondition:   "Legacy handoffs pass by default; strict handoffs refuse vague next steps.",
		Witness:         "go test ./internal/taskmgr",
		AcceptanceGate:  "go test ./cmd/fak -run TestIssueContract",
		Lane:            "taskmgr",
		Paths:           []string{"internal/taskmgr/handoff.go"},
		BoundaryNotes:   []string{"Public issue only; no private lab evidence."},
		ClosureBinding:  "Resolving commit cites #N and carries a matching (fak <leaf>) trailer.",
	}
}

func writeIssueContractJSON(t *testing.T, c issuecontract.Candidate) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "candidate.json")
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

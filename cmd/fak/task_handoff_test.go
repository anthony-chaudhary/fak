package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func TestTaskHandoffDryRunPlansIssueCreate(t *testing.T) {
	dir := t.TempDir()
	handoffPath := writeTaskHandoffFixture(t, dir, true)
	existingPath := filepath.Join(dir, "existing.json")
	if err := os.WriteFile(existingPath, []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write existing: %v", err)
	}

	var out, errb bytes.Buffer
	code := runTask(&out, &errb, []string{"handoff", "--file", handoffPath, "--existing-json", existingPath, "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got taskHandoffResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("parse output: %v\n%s", err, out.String())
	}
	if !got.Review.OK || got.Review.Verdict != "ready" {
		t.Fatalf("review = %+v, want ok ready", got.Review)
	}
	if len(got.Planned) != 1 || got.Planned[0].Action != "create" {
		t.Fatalf("planned = %+v, want one create", got.Planned)
	}
	if got.Planned[0].Key != "task_push_next/live-smoke" {
		t.Fatalf("planned key = %q, want stable next-step key", got.Planned[0].Key)
	}
	labels := map[string]bool{}
	for _, label := range got.Planned[0].Labels {
		labels[label] = true
	}
	for _, want := range []string{"agent-handoff", "generation", "gen/next"} {
		if !labels[want] {
			t.Fatalf("planned labels = %+v, missing %q", got.Planned[0].Labels, want)
		}
	}
}

func TestTaskHandoffEvidenceRefsReachDryRunPlan(t *testing.T) {
	dir := t.TempDir()
	handoff := taskmgr.DraftHandoffFromTask(taskmgr.TaskSnapshot{
		TaskID: "task_origin_handoff",
		Title:  "Origin handoff evidence",
		State:  taskmgr.StateDone,
		Witness: &taskmgr.WitnessRecord{
			VerifiedState: taskmgr.VerifiedDone,
			Source:        "commit-audit",
			SHA:           "abc123",
		},
	}, taskmgr.HandoffDraftOptions{
		CurrentState: "Task is ready for follow-up issue sync.",
		Evidence: taskmgr.HandoffEvidenceInputs{
			ChangedPaths: []string{"internal/taskmgr/handoff.go"},
			TestCommands: []string{"go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence"},
		},
	})
	handoff.NextSteps = []taskmgr.HandoffNextStep{{
		Key:             "task_origin_handoff/evidence-sync",
		Title:           "Keep derived handoff evidence in issue sync",
		Body:            "Ensure task handoff dry-run plans expose derived path and test refs.",
		Reason:          "The handoff producer now derives evidence before the operator edits the JSON.",
		WorkingSpine:    "Task handoff evidence should flow into the issue plan without manual copying.",
		PriorityContext: "Working path: live task snapshot -> derived evidence refs -> handoff issue plan. Current blocker: derived refs used to stop at handoff JSON. Unblocks: issue sync can carry path/test evidence. Not polish: this proves the smallest producer-to-plan path.",
		WorkUnit:        "leaf",
		ExpectedSteps:   3,
		Assumptions:     []string{"The handoff producer can derive changed paths and targeted tests before issue sync."},
		ConfusionRisks:  []string{"Derived refs are suggestions, not a replacement for witnessed completion."},
		Coordination:    []string{"Keep this scoped to task handoff evidence plumbing."},
		Trigger:         "A verified task handoff proposes a follow-up with derived evidence refs.",
		BatchPolicy:     "At most two follow-up issues per handoff; reruns update by marker.",
		InScope:         "Carry derived path/test refs into the dry-run issue plan.",
		OutOfScope:      "Do not run live gh sync in this unit test.",
		DoneCondition:   "The dry-run plan exposes the derived path and test evidence refs.",
		Witness:         "go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence",
		AcceptanceGate:  "go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence",
		Lane:            "taskmgr",
		Paths:           []string{"internal/taskmgr/handoff.go", "cmd/fak/task_handoff_test.go"},
		BoundaryNotes:   []string{"Public task handoff issue only."},
		ClosureBinding:  "Resolving commit cites the issue and carries `(fak taskmgr)`.",
	}}
	b, err := json.Marshal(handoff)
	if err != nil {
		t.Fatalf("marshal handoff: %v", err)
	}
	handoffPath := filepath.Join(dir, "handoff.json")
	if err := os.WriteFile(handoffPath, b, 0o644); err != nil {
		t.Fatalf("write handoff: %v", err)
	}

	var out, errb bytes.Buffer
	code := runTask(&out, &errb, []string{"handoff", "--file", handoffPath, "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got taskHandoffResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("parse output: %v\n%s", err, out.String())
	}
	if len(got.Planned) != 1 {
		t.Fatalf("planned = %+v, want one row", got.Planned)
	}
	for _, want := range []string{
		"path:internal/taskmgr/handoff.go (changed path)",
		"test:go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence (targeted test command)",
	} {
		if !taskHandoffString(got.Planned[0].EvidenceRefs, want) {
			t.Fatalf("planned evidence refs = %+v, missing %q", got.Planned[0].EvidenceRefs, want)
		}
	}
}

func TestTaskHandoffRefusesUnwitnessedCompletion(t *testing.T) {
	dir := t.TempDir()
	handoffPath := writeTaskHandoffFixture(t, dir, false)

	var out, errb bytes.Buffer
	code := runTask(&out, &errb, []string{"handoff", "--file", handoffPath, "--json"})
	if code != 3 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got taskHandoffResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("parse output: %v\n%s", err, out.String())
	}
	if got.Review.OK {
		t.Fatalf("unwitnessed review passed: %+v", got.Review)
	}
	if !taskHandoffReason(got.Review.Reasons, "MISSING_COMPLETION_WITNESS") {
		t.Fatalf("missing witness reason absent: %+v", got.Review.Reasons)
	}
}

func TestTaskHandoffRefusesUnscopedFollowUp(t *testing.T) {
	dir := t.TempDir()
	handoffPath := writeTaskHandoffFixtureWithScope(t, dir, true, false)

	var out, errb bytes.Buffer
	code := runTask(&out, &errb, []string{"handoff", "--file", handoffPath, "--json"})
	if code != 3 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got taskHandoffResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("parse output: %v\n%s", err, out.String())
	}
	if got.Review.OK || len(got.Review.IssueReviews) != 1 {
		t.Fatalf("review = %+v, want one refused issue review", got.Review)
	}
	if !taskHandoffReason(got.Review.Reasons, "NEXT_STEP_1_ISSUE_SCOPE_INCOMPLETE") {
		t.Fatalf("scope reason absent: %+v", got.Review.Reasons)
	}
}

func TestTaskHandoffSyncUsesInjectedRunner(t *testing.T) {
	row := taskmgr.HandoffIssuePlanRow{
		Action: "create",
		Key:    "task_push_next/live-smoke",
		Title:  "Live smoke",
		Body:   "body",
		Labels: []string{"agent-handoff"},
	}
	var calls [][]string
	rows := syncTaskHandoffPlan([]taskmgr.HandoffIssuePlanRow{row}, "owner/repo", []string{"next-step", "agent-handoff"}, func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example.test/issues/9", "", true
	})
	if len(rows) != 1 || !rows[0].OK {
		t.Fatalf("sync rows = %+v", rows)
	}
	joined := strings.Join(calls[0], " ")
	for _, want := range []string{"issue create", "--repo owner/repo", "--label agent-handoff", "--label next-step"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("gh args missing %q: %v", want, calls[0])
		}
	}
}

func writeTaskHandoffFixture(t *testing.T, dir string, witnessed bool) string {
	t.Helper()
	return writeTaskHandoffFixtureWithScope(t, dir, witnessed, true)
}

func writeTaskHandoffFixtureWithScope(t *testing.T, dir string, witnessed, scoped bool) string {
	t.Helper()
	handoff := taskmgr.Handoff{
		Schema:       taskmgr.SchemaHandoff,
		CurrentState: "The implementation is committed; the remaining proof is a live issue sync smoke.",
		Task: taskmgr.HandoffTask{
			TaskID: "task_push_next",
			Title:  "Push next work",
			State:  taskmgr.StateDone,
		},
		NextSteps: []taskmgr.HandoffNextStep{{
			Key:                "task_push_next/live-smoke",
			Title:              "Run live task handoff issue sync smoke",
			Body:               "Exercise `fak task handoff --live` against a disposable follow-up issue.",
			Reason:             "Dry-run planning is covered; live gh behavior still needs an operator-owned witness.",
			Generation:         "gen/next",
			PromotionEvidence:  []string{"The live smoke proves task handoffs can feed dispatch safely."},
			DemotionEvidence:   []string{"Generated follow-ups increase ambiguity or duplicate existing work."},
			GenerationNonGoals: []string{"Do not treat gen/next as a branch or runtime exposure flag."},
			Labels:             []string{"agent-handoff"},
		}},
	}
	if scoped {
		step := &handoff.NextSteps[0]
		step.WorkingSpine = "A verified task handoff creates one scoped follow-up issue."
		step.PriorityContext = "Working path: task completion -> worker-ready follow-up -> dispatch. Current blocker: live sync lacks smoke coverage. Unblocks: task handoffs can feed the issue queue. Not polish: this proves the smallest live path."
		step.WorkUnit = "leaf"
		step.ExpectedSteps = 4
		step.Assumptions = []string{"The disposable issue fixture can be updated by marker key."}
		step.InvalidatingAssumptions = []string{"The generated issue body stops being enough for the next worker to resume."}
		step.ConfusionRisks = []string{"A live smoke is not a broad redesign of task storage."}
		step.Coordination = []string{"Do not run concurrently with other taskmgr issue-body edits."}
		step.Trigger = "Verified task handoff proposes one follow-up after the dry-run path passed."
		step.BatchPolicy = "At most two follow-up issues per handoff; reruns update by marker."
		step.InScope = "Run the live issue sync smoke and keep the generated body parseable by issuecontract."
		step.OutOfScope = "Do not change task storage, scheduling, or unrelated issue producers."
		step.DoneCondition = "The smoke creates or updates the disposable follow-up issue."
		step.Witness = "go test ./cmd/fak -run TestTaskHandoff"
		step.AcceptanceGate = "go test ./cmd/fak -run TestTaskHandoff"
		step.Lane = "taskmgr"
		step.Paths = []string{"cmd/fak/taskmgr.go", "internal/taskmgr/**"}
		step.BoundaryNotes = []string{"Public task handoff issue only."}
		step.ClosureBinding = "Resolving commit cites the issue and carries `(fak taskmgr)`."
	}
	if witnessed {
		handoff.Task.Witness = &taskmgr.WitnessRecord{
			VerifiedState: taskmgr.VerifiedDone,
			Source:        "commit-audit",
			SHA:           "deadbeef",
		}
	}
	b, err := json.Marshal(handoff)
	if err != nil {
		t.Fatalf("marshal handoff: %v", err)
	}
	path := filepath.Join(dir, "handoff.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write handoff: %v", err)
	}
	return path
}

func taskHandoffReason(reasons []string, want string) bool {
	return taskHandoffString(reasons, want)
}

func taskHandoffString(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

package taskmgr

import (
	"strings"
	"testing"
)

func verifiedHandoff() Handoff {
	return Handoff{
		Schema:       SchemaHandoff,
		CurrentState: "Implementation shipped and verified; the remaining work is a narrow follow-up.",
		Task: HandoffTask{
			TaskID: "task_push_next",
			Title:  "Push next work",
			State:  StateDone,
			Witness: &WitnessRecord{
				VerifiedState: VerifiedDone,
				Source:        "commit-audit",
				SHA:           "deadbeef",
			},
		},
		CompletionEvidence: []EvidenceRef{{Kind: "commit", Ref: "deadbeef", Note: "diff-witnessed"}},
		NextSteps: []HandoffNextStep{{
			Key:        "task_push_next/issue-sync",
			Title:      "Add live issue sync smoke",
			Body:       "Wire the dry-run handoff into one live gh smoke on a disposable fixture.",
			Reason:     "The typed handoff exists; the next useful proof is a live end-to-end issue update.",
			Generation: "next",
			PromotionEvidence: []string{
				"Live issue sync can create or update the disposable follow-up issue.",
			},
			DemotionEvidence: []string{
				"The live smoke shows handoff-created issues add more dispatch ambiguity than they remove.",
			},
			InvalidatingAssumptions: []string{
				"Follow-up workers can read the generated issue without reopening the parent task transcript.",
			},
			GenerationNonGoals: []string{
				"Do not create a branch or bypass the shared-trunk handoff gate.",
			},
			WorkingSpine:    "A verified task completion can create one scoped follow-up issue.",
			PriorityContext: "Working path: completed task -> scoped follow-up issue -> dispatch. Current blocker: live sync lacks an operator-owned smoke. Unblocks: task handoffs can feed dispatch safely. Not polish: this proves the minimal live path.",
			WorkUnit:        "leaf",
			ExpectedSteps:   4,
			Assumptions:     []string{"The disposable issue fixture can be updated by marker key."},
			ConfusionRisks:  []string{"A live smoke is not a broad redesign of task storage."},
			Coordination:    []string{"Do not run concurrently with other taskmgr issue-body edits."},
			Trigger:         "Verified task handoff proposes one follow-up after the dry-run path passed.",
			BatchPolicy:     "At most two follow-up issues per handoff; reruns update by marker.",
			InScope:         "Add one live issue sync smoke and keep the handoff body parseable by the issue contract.",
			OutOfScope:      "Do not change task state storage or dispatch routing.",
			DoneCondition:   "The live sync smoke can create or update the disposable follow-up issue.",
			Witness:         "go test ./cmd/fak -run TestTaskHandoff",
			AcceptanceGate:  "go test ./internal/taskmgr ./cmd/fak -run TestTaskHandoff",
			Lane:            "taskmgr",
			Paths:           []string{"internal/taskmgr/**", "cmd/fak/taskmgr.go"},
			Priority:        "p2",
			Labels:          []string{"agent-handoff", "agent-handoff"},
			BoundaryNotes:   []string{"Public handoff issue only; no private operator transcript."},
			ClosureBinding:  "Resolving commit cites the issue and carries `(fak taskmgr)`.",
			EvidenceRefs: []EvidenceRef{{
				Kind: "path", Ref: "internal/taskmgr/handoff.go",
			}},
		}},
	}
}

func TestReviewHandoffRequiresWitnessedCompletionAndNextStep(t *testing.T) {
	h := verifiedHandoff()
	h.Task.Witness = nil
	review := ReviewHandoff(h)
	if review.OK {
		t.Fatalf("unwitnessed handoff passed: %+v", review)
	}
	if !contains(review.Reasons, "MISSING_COMPLETION_WITNESS") {
		t.Fatalf("missing witness reason absent: %+v", review.Reasons)
	}

	h = verifiedHandoff()
	h.NextSteps = nil
	review = ReviewHandoff(h)
	if review.OK {
		t.Fatalf("handoff with no next step or reason passed: %+v", review)
	}
	if !contains(review.Reasons, "MISSING_NEXT_STEP_OR_NOT_APPLICABLE_REASON") {
		t.Fatalf("missing next-step reason absent: %+v", review.Reasons)
	}

	h.NoNextStepReason = "No follow-up is reasonable: the issue was closed and no residual risk remains."
	review = ReviewHandoff(h)
	if !review.OK || review.Verdict != "not_applicable" {
		t.Fatalf("not-applicable handoff = %+v, want ok/not_applicable", review)
	}
}

func TestReviewHandoffCapsFollowUpsAtTwoAndChecksKeys(t *testing.T) {
	h := verifiedHandoff()
	h.NextSteps = append(h.NextSteps,
		HandoffNextStep{Key: "task_push_next/docs", Title: "Document handoff", Body: "Add docs.", Reason: "Users need the input schema."},
		HandoffNextStep{Key: "task_push_next/third", Title: "Third", Body: "Too many.", Reason: "This should be refused."},
	)
	review := ReviewHandoff(h)
	if review.OK {
		t.Fatalf("three follow-ups passed: %+v", review)
	}
	if !contains(review.Reasons, "TOO_MANY_NEXT_STEPS") {
		t.Fatalf("too-many reason absent: %+v", review.Reasons)
	}

	h = verifiedHandoff()
	h.NextSteps[0].Key = "bad key with spaces"
	review = ReviewHandoff(h)
	if review.OK || !contains(review.Reasons, "NEXT_STEP_1_BAD_KEY") {
		t.Fatalf("bad key review = %+v", review)
	}
}

func TestReviewHandoffStrictScopeRejectsVagueNextStep(t *testing.T) {
	h := verifiedHandoff()
	h.NextSteps[0].WorkingSpine = ""
	h.NextSteps[0].OutOfScope = ""
	h.NextSteps[0].DoneCondition = ""
	h.NextSteps[0].Witness = ""
	h.NextSteps[0].Lane = ""
	h.NextSteps[0].Paths = nil

	review := ReviewHandoffWithOptions(h, HandoffReviewOptions{StrictScope: true})
	if review.OK {
		t.Fatalf("strict vague handoff passed: %+v", review)
	}
	if len(review.IssueReviews) != 1 {
		t.Fatalf("issue reviews = %d, want 1", len(review.IssueReviews))
	}
	if !contains(review.Reasons, "NEXT_STEP_1_ISSUE_SCOPE_INCOMPLETE") ||
		!contains(review.Reasons, "NEXT_STEP_1_ISSUE_UNROUTED") {
		t.Fatalf("strict reasons = %+v", review.Reasons)
	}
}

func TestReviewHandoffRejectsBadGeneration(t *testing.T) {
	h := verifiedHandoff()
	h.NextSteps[0].Generation = "later-ish"
	review := ReviewHandoff(h)
	if review.OK || !contains(review.Reasons, "NEXT_STEP_1_BAD_GENERATION") {
		t.Fatalf("bad generation review = %+v, want NEXT_STEP_1_BAD_GENERATION", review)
	}
}

func TestReviewHandoffStrictScopeAcceptsDispatchableNextStep(t *testing.T) {
	review := ReviewHandoffWithOptions(verifiedHandoff(), HandoffReviewOptions{
		StrictScope:   true,
		Live:          true,
		DedupeChecked: true,
		DedupeCap:     300,
	})
	if !review.OK || review.Verdict != "ready" {
		t.Fatalf("strict review = %+v, want ready", review)
	}
	if len(review.IssueReviews) != 1 || review.IssueReviews[0].Score.Total != 100 {
		t.Fatalf("issue review = %+v, want one full-score issue", review.IssueReviews)
	}
	if review.IssueReviews[0].AgentContext.Total != 100 {
		t.Fatalf("agent context = %+v, want full-score issue", review.IssueReviews[0].AgentContext)
	}
}

func TestBuildHandoffIssuePlanDedupesByStableMarker(t *testing.T) {
	h := verifiedHandoff()
	existing := []HandoffIssue{{
		Number: 42,
		State:  "OPEN",
		Body:   "<!-- fak-task-handoff-key: task_push_next/issue-sync -->\nold body",
	}}
	plan := BuildHandoffIssuePlan(h, existing)
	if len(plan) != 1 {
		t.Fatalf("plan rows = %d, want 1", len(plan))
	}
	row := plan[0]
	if row.Action != "update" || row.Number == nil || *row.Number != 42 {
		t.Fatalf("row = %+v, want update #42", row)
	}
	if got := HandoffMarkerKey(row.Body); got != h.NextSteps[0].Key {
		t.Fatalf("marker key = %q, want %q", got, h.NextSteps[0].Key)
	}
	if !strings.Contains(row.Body, "Current state: Implementation shipped and verified") {
		t.Fatalf("body missing current state:\n%s", row.Body)
	}
	if !strings.Contains(row.Body, "Completion witness: `verified_done` via `commit-audit` (`deadbeef`)") {
		t.Fatalf("body missing witness:\n%s", row.Body)
	}
	wantLabels := map[string]bool{"agent-handoff": true, "generation": true, "gen/next": true}
	if len(row.Labels) != len(wantLabels) {
		t.Fatalf("labels = %+v, want %v", row.Labels, wantLabels)
	}
	for _, label := range row.Labels {
		if !wantLabels[label] {
			t.Fatalf("unexpected label %q in %+v", label, row.Labels)
		}
	}
}

func TestHandoffIssueBodyIncludesStrictScopeSections(t *testing.T) {
	h := verifiedHandoff()
	body := HandoffIssueBody(h, h.NextSteps[0])
	for _, want := range []string{
		"## Generation intent",
		"## Promotion evidence",
		"## Demotion or retirement evidence",
		"## Invalidating assumptions",
		"## Generation non-goals",
		"## Working spine",
		"## Priority context",
		"## Work unit",
		"## Expected steps",
		"## Assumptions",
		"## Confusion risks",
		"## Coordination notes",
		"## Trigger",
		"## Batch policy",
		"## In scope",
		"## Out of scope",
		"## Done condition",
		"## Witness",
		"## Acceptance gate",
		"## Lane",
		"## Path hints",
		"## Closure binding",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("body missing %q:\n%s", want, body)
		}
	}
	for _, want := range []string{
		"Generation: `gen/next`",
		"Generation is orthogonal to priority, shared trunk, and runtime feature gates.",
		"Live issue sync can create or update the disposable follow-up issue.",
		"Do not create a branch or bypass the shared-trunk handoff gate.",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("generation body missing %q:\n%s", want, body)
		}
	}
}

func contains(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

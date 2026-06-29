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
			Key:      "task_push_next/issue-sync",
			Title:    "Add live issue sync smoke",
			Body:     "Wire the dry-run handoff into one live gh smoke on a disposable fixture.",
			Reason:   "The typed handoff exists; the next useful proof is a live end-to-end issue update.",
			Priority: "p2",
			Labels:   []string{"agent-handoff", "agent-handoff"},
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
	if len(row.Labels) != 1 || row.Labels[0] != "agent-handoff" {
		t.Fatalf("labels = %+v, want deduped agent-handoff", row.Labels)
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

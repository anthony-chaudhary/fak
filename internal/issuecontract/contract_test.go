package issuecontract

import (
	"strings"
	"testing"
)

func completeCandidate() Candidate {
	return Candidate{
		Schema:          Schema,
		Key:             "task_push_next/strict-scope",
		Title:           "taskmgr: enforce strict handoff scope",
		ParentRef:       "task_push_next",
		CurrentState:    "Task handoff can already create stable follow-up issues.",
		WhyNow:          "Generated issues are the next weak point before dispatch.",
		WorkingSpine:    "A verified task completion creates one scoped follow-up issue.",
		PriorityContext: "Working path: clean Stop handoff -> scoped issue -> dispatch. Current blocker: vague follow-ups waste dispatch cycles. Unblocks: guard live handoff. Not polish: enforce the smallest leaf before optimization.",
		InScope:         "Review the next-step candidate and render scoped sections.",
		OutOfScope:      "Do not optimize issue routing or add new scorecards.",
		DoneCondition:   "Legacy handoffs pass by default; strict handoffs refuse vague next steps.",
		Witness:         "go test ./internal/taskmgr",
		AcceptanceGate:  "go test ./cmd/fak -run TestTaskHandoff",
		Lane:            "taskmgr",
		Paths:           []string{"internal/taskmgr/handoff.go"},
		BoundaryNotes:   []string{"Public issue only; no private lab evidence."},
		ClosureBinding:  "Resolving commit cites #N and carries a matching (fak <leaf>) trailer.",
	}
}

func TestReviewCandidateDispatchableScoresFull(t *testing.T) {
	review := ReviewCandidate(completeCandidate(), Options{})
	if !review.OK || review.Dispatchability != Dispatchable || review.Verdict != "ready" {
		t.Fatalf("review = %+v, want ready dispatchable", review)
	}
	if review.Score.Total != 100 {
		t.Fatalf("score = %+v, want total 100", review.Score)
	}
	if review.SpinePriority.Total != 100 {
		t.Fatalf("spine priority = %+v, want total 100", review.SpinePriority)
	}
	if len(review.Reasons) != 0 || len(review.MissingFields) != 0 {
		t.Fatalf("unexpected reasons/missing: %+v %+v", review.Reasons, review.MissingFields)
	}
}

func TestReviewCandidateScoresGoldPlatingBelowSpineWork(t *testing.T) {
	c := completeCandidate()
	c.PriorityContext = "Nice later: polish helper names after the workflow already works."
	c.CurrentState = "The user-facing workflow already works and has a passing witness."
	c.WhyNow = "This would be cleanup someday."
	c.WorkingSpine = "No working path changes."
	c.OutOfScope = "No adjacent work is addressed."
	review := ReviewCandidate(c, Options{})
	if !review.OK {
		t.Fatalf("gold-plating candidate should still be dispatchable when scoped: %+v", review)
	}
	if review.SpinePriority.Total >= 50 {
		t.Fatalf("spine priority = %+v, want below 50 for scoped polish", review.SpinePriority)
	}
}

func TestReviewCandidateAllowsExistingColonKeyShape(t *testing.T) {
	c := completeCandidate()
	c.Key = "guard-rsi-route/guard-journal:blank_reason_on_deny"
	review := ReviewCandidate(c, Options{})
	if !review.OK {
		t.Fatalf("review = %+v, want colon-bearing stable marker key accepted", review)
	}
}

func TestReviewCandidateNeedsScopeForMissingSpineFields(t *testing.T) {
	c := completeCandidate()
	c.OutOfScope = ""
	c.DoneCondition = ""
	c.WorkingSpine = ""
	review := ReviewCandidate(c, Options{})
	if review.OK || review.Dispatchability != TriageOnly || review.Verdict != "needs_scope" {
		t.Fatalf("review = %+v, want needs_scope triage_only", review)
	}
	if !has(review.Reasons, ReasonScopeIncomplete) {
		t.Fatalf("scope reason missing: %+v", review.Reasons)
	}
	for _, want := range []string{"working_spine", "out_of_scope", "done_condition"} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent: %+v", want, review.MissingFields)
		}
	}
	if review.Score.Total >= 100 {
		t.Fatalf("partial candidate got full score: %+v", review.Score)
	}
}

func TestReviewCandidateNeedsRoute(t *testing.T) {
	c := completeCandidate()
	c.Lane = ""
	c.Paths = nil
	review := ReviewCandidate(c, Options{})
	if review.OK || !has(review.Reasons, ReasonUnrouted) {
		t.Fatalf("review = %+v, want unrouted refusal reason", review)
	}
	if review.Dispatchability != TriageOnly {
		t.Fatalf("dispatchability = %q, want triage_only", review.Dispatchability)
	}
}

func TestReviewCandidateRefusesPrivateBoundary(t *testing.T) {
	c := completeCandidate()
	c.BoundaryNotes = []string{"Requires fak-private Slack control transcript."}
	review := ReviewCandidate(c, Options{})
	if review.OK || review.Dispatchability != Refused || review.Verdict != "refused" {
		t.Fatalf("review = %+v, want refused", review)
	}
	if !has(review.Reasons, ReasonPrivateBoundary) {
		t.Fatalf("private-boundary reason missing: %+v", review.Reasons)
	}
}

func TestReviewCandidateLiveRequiresDedupeArmor(t *testing.T) {
	review := ReviewCandidate(completeCandidate(), Options{Live: true})
	if review.OK || review.Dispatchability != Refused {
		t.Fatalf("unarmored live review = %+v, want refused", review)
	}
	if !has(review.Reasons, ReasonLiveUnarmored) {
		t.Fatalf("live-unarmored reason missing: %+v", review.Reasons)
	}

	armed := ReviewCandidate(completeCandidate(), Options{Live: true, DedupeChecked: true, DedupeCap: 300})
	if !armed.OK {
		t.Fatalf("armed live review refused: %+v", armed)
	}
}

func TestReviewIssueDraftParsesStandardSections(t *testing.T) {
	review := ReviewIssueDraft(IssueDraft{
		Number: 1440,
		Title:  "guardrsi: require a reason on every block",
		Labels: []IssueLabel{{Name: "guardrsi"}},
		Body: strings.Join([]string{
			"### Parent context",
			"guard-verdict-rsi route",
			"### Current state",
			"The guard journal can surface an unexplained block bucket.",
			"### Why this is next",
			"This is a load-bearing honesty hole before threshold tuning.",
			"### Working spine",
			"Every denied guard verdict carries one closed-vocabulary reason.",
			"### Priority context",
			"Working path: guard preflight -> reasoned denial -> worker repair.",
			"Current blocker: blank reasons hide the failing gate.",
			"Unblocks: guard-rsi tuning depends on reason buckets.",
			"Not polish: this fixes the smallest witnessed guard hole before threshold optimization.",
			"### In scope",
			"Add the missing reason mapping and one regression fixture.",
			"### Out of scope",
			"Do not retune model thresholds or rewrite unrelated guard code.",
			"### Done condition",
			"The regression fixture no longer reports a blank reason.",
			"### Witness",
			"go test ./internal/guardrsi ./internal/guardroute",
			"### Acceptance gate",
			"go test ./internal/guardrsi ./internal/guardroute",
			"### Lane",
			"guardrsi",
			"### Path hints",
			"- `internal/guardrsi/**`",
			"### Boundary notes",
			"- Public guard-journal defect only.",
			"### Closure binding",
			"Resolving commit cites #N and carries `(fak guardrsi)`.",
		}, "\n"),
	}, Options{})
	if !review.OK || review.Dispatchability != Dispatchable || review.Score.Total != 100 {
		t.Fatalf("review = %+v, want dispatchable full-score issue draft", review)
	}
	if review.SpinePriority.Total != 100 {
		t.Fatalf("spine priority = %+v, want full-score issue draft", review.SpinePriority)
	}
	if review.Key != "issue/1440" || review.Lane != "guardrsi" {
		t.Fatalf("identity = key %q lane %q", review.Key, review.Lane)
	}
	if len(review.Paths) != 1 || review.Paths[0] != "internal/guardrsi/**" {
		t.Fatalf("paths = %+v", review.Paths)
	}
}

func TestReviewIssueDraftFlagsVagueManualIssue(t *testing.T) {
	review := ReviewIssueDraft(IssueDraft{
		Number: 1441,
		Title:  "make it better",
		Body: strings.Join([]string{
			"### Current state",
			"The feature exists.",
			"### In scope",
			"Improve things.",
		}, "\n"),
	}, Options{})
	if review.OK || review.Dispatchability != TriageOnly {
		t.Fatalf("review = %+v, want triage-only incomplete issue", review)
	}
	for _, want := range []string{"parent_ref", "why_now", "working_spine", "out_of_scope", "done_condition", "witness", "acceptance_gate", "closure_binding"} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent from %+v", want, review.MissingFields)
		}
	}
	if !has(review.Reasons, ReasonUnrouted) {
		t.Fatalf("unrouted reason absent: %+v", review.Reasons)
	}
}

func TestReviewIssueDraftParsesCombinedDoneWitnessSection(t *testing.T) {
	body := strings.Join([]string{
		"### Parent context",
		"task handoff",
		"### Current state",
		"A verified handoff created this follow-up.",
		"### Why this is next",
		"It unblocks the next dispatch cycle before polish.",
		"### Working spine",
		"One handoff issue carries scope and proof.",
		"### Priority context",
		"Working path: handoff -> issue -> worker. Current blocker: old body shape. Unblocks: prompt briefing. Not polish: parser compatibility.",
		"### In scope",
		"Parse the existing combined section.",
		"### Out of scope",
		"Do not redesign handoff sync.",
		"### Done condition / witness",
		"Done condition: The parser extracts a done condition.",
		"Witness: `go test ./internal/issuecontract`",
		"### Acceptance gate",
		"go test ./internal/issuecontract",
		"### Lane",
		"issuecontract",
		"### Closure binding",
		"Resolving commit cites #N.",
	}, "\n")
	review := ReviewIssueDraft(IssueDraft{Number: 1442, Title: "taskmgr: parse combined proof", Body: body}, Options{})
	if !review.OK || review.Score.Witness != 25 {
		t.Fatalf("review = %+v, want combined done/witness parsed as ready", review)
	}
}

func has(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

package issuepolicy

import (
	"strings"
	"testing"
)

func TestAssessProblemFrameAcceptsCentralityClassesAndChecks(t *testing.T) {
	classes := []struct {
		line   string
		class  string
		target string
	}{
		{"Core", CentralityCore, ""},
		{"Enabling (managed-context outcome)", CentralityEnabling, "managed-context outcome"},
		{"Stewardship (release signing obligation)", CentralityStewardship, "release signing obligation"},
		{"Peripheral", CentralityPeripheral, ""},
	}
	for _, tc := range classes {
		t.Run(tc.class, func(t *testing.T) {
			body := "## Value\n- Centrality: " + tc.line + "\n" +
				"- P1: preserved - does not duplicate runtime context\n" +
				"- P2: advanced - removes intake rework\n" +
				"- P3: N/A - no adaptive behavior in this docs-only leaf\n" +
				"- P4: preserved - existing operator path remains intact\n"
			got := AssessProblemFrame(IssueDraft{Body: body})
			if !got.Enforced || !got.Ready {
				t.Fatalf("expected ready frame, got %+v", got)
			}
			if got.Centrality != tc.class || got.CentralityTarget != tc.target {
				t.Fatalf("centrality = %q target %q, want %q target %q", got.Centrality, got.CentralityTarget, tc.class, tc.target)
			}
			for _, id := range []string{"p1", "p2", "p3", "p4"} {
				if !got.Checks[id].Valid {
					t.Fatalf("%s should be valid: %+v", id, got.Checks[id])
				}
			}
		})
	}
}

func TestAssessProblemFrameRejectsMissingMalformedAndCeremonialFields(t *testing.T) {
	body := "## Value\n- Centrality: Enabling\n" +
		"- P1: advanced\n" +
		"- P2: better someday\n" +
		"- P4: N/A\n"
	got := AssessProblemFrame(IssueDraft{Body: body})
	if got.Ready {
		t.Fatalf("expected refusal, got %+v", got)
	}
	for _, reason := range []string{
		"problem_centrality_target_missing",
		"problem_check_p1_ceremonial",
		"problem_check_p2_invalid",
		"problem_check_p3_missing",
		"problem_check_p4_ceremonial",
	} {
		if !containsString(got.Reasons, reason) {
			t.Errorf("missing reason %q in %v", reason, got.Reasons)
		}
	}
}

func TestAssessProblemFrameRejectsStewardshipWithoutObligation(t *testing.T) {
	got := AssessProblemFrame(IssueDraft{Body: "## Value\n- Centrality: Stewardship\n- P1: preserved - context remains reusable\n- P2: preserved - no new operating cost\n- P3: preserved - adaptation remains bounded\n- P4: advanced - obligation is integrated\n"})
	if got.Ready || !containsString(got.Reasons, "problem_centrality_obligation_missing") {
		t.Fatalf("missing stewardship obligation should fail: %+v", got)
	}
}

func TestAssessProblemFrameMigrationBoundary(t *testing.T) {
	legacy := AssessProblemFrame(IssueDraft{Body: "## Scope\nKeep the old contract readable.\n"})
	if legacy.Enforced || !legacy.Ready || legacy.Centrality != CentralityUnclassified {
		t.Fatalf("legacy contract should remain advisory and unclassified: %+v", legacy)
	}

	newBrief := AssessProblemFrame(IssueDraft{Body: "## Scope / tree\ninternal/issuepolicy\n"})
	if !newBrief.Enforced || newBrief.Ready {
		t.Fatalf("new shift-left brief must require the frame: %+v", newBrief)
	}
	if !containsString(newBrief.Reasons, "problem_centrality_invalid") {
		t.Fatalf("missing centrality reason: %v", newBrief.Reasons)
	}
}

func TestReviewCandidateCarriesAndGatesCanonicalProblemFrame(t *testing.T) {
	candidate := Candidate{
		Schema: Schema, Key: "problem-frame/direct", Title: "direct candidate", ParentRef: "#1",
		CurrentState: "gap", WhyNow: "now", WorkingSpine: "path", InScope: "one leaf", OutOfScope: "rest",
		DoneCondition: "done", Witness: "test", AcceptanceGate: "test", ClosureBinding: "commit cites #1",
		Paths: []string{"internal/issuepolicy/**"},
		ProblemFrame: ProblemFrame{Schema: ProblemFrameSchema, Enforced: true, Ready: false, Centrality: CentralityEnabling,
			Reasons: []string{"problem_centrality_target_missing"}, Checks: map[string]ProblemCheck{}},
	}
	review := ReviewCandidate(candidate, Options{})
	if review.OK || review.Verdict != "needs_problem_frame" || !containsString(review.Reasons, ReasonProblemFrameIncomplete) {
		t.Fatalf("direct candidate escaped frame gate: %+v", review)
	}
}

func TestReviewIssueDraftGatesMalformedProblemFrame(t *testing.T) {
	body := "## Value\n- Centrality: Core\n- P1: advanced - context is reused\n- P2: preserved - no efficiency regression\n- P3: preserved - adaptation remains bounded\n- P4: advanced\n"
	review := ReviewIssueDraft(IssueDraft{Number: 7, Title: "issuepolicy: gate problem frame", Body: body}, Options{})
	if review.ProblemFrame.Ready || !containsString(review.Reasons, ReasonProblemFrameIncomplete) {
		t.Fatalf("problem frame did not gate review: %+v", review)
	}
	if !strings.Contains(strings.Join(review.ProblemFrame.RepairActions, "\n"), "P4") {
		t.Fatalf("missing field-specific repair: %v", review.ProblemFrame.RepairActions)
	}
}

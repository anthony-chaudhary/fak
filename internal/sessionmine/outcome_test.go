package sessionmine

import (
	"testing"
)

func TestAttributeOutcomesRequiresExactIndependentWitness(t *testing.T) {
	rows := []OutcomeSession{
		{RegistrationID: "reg-3330", SessionID: "sess-3330", State: "completed"},
		{RegistrationID: "active", SessionID: "sess-active", State: "active"},
	}
	evidence := []OutcomeEvidence{
		{SessionID: "decoy", SHA: "wrong", Claim: "CLAIM_WITNESSED", Verdict: "OK", Witness: "diff-witnessed"},
		{SessionID: "sess-3330", RegistrationID: "other-registration", SHA: "wrong-registration", Claim: "CLAIM_WITNESSED", Verdict: "OK", Witness: "diff-witnessed"},
		{SessionID: "sess-3330", SHA: "abc1234", Claim: "CLAIM_WITNESSED", Verdict: "OK", Witness: "diff-witnessed", Issue: 3330, IssueClosed: true},
	}
	got := AttributeOutcomes(rows, evidence)
	if len(got) != 1 || got[0].Outcome != OutcomeShippedCommit || got[0].SHA != "abc1234" {
		t.Fatalf("got %#v", got)
	}
	if !got[0].IssueClosed || got[0].IssueProvenance != "github_observed" {
		t.Fatalf("closure provenance lost: %#v", got[0])
	}
}

func TestAttributeOutcomesLeavesAmbiguousAndSelfReportedUnknown(t *testing.T) {
	row := OutcomeSession{RegistrationID: "reg", SessionID: "sess", State: "completed"}
	tests := []struct {
		name     string
		evidence []OutcomeEvidence
		reason   string
	}{
		{"self report", []OutcomeEvidence{{SessionID: "sess", SHA: "a", Claim: "CLAIM_UNWITNESSED", Verdict: "OK", Witness: "diff-witnessed"}}, "no_unique_green_diff_witness"},
		{"conflict", []OutcomeEvidence{{SessionID: "sess", SHA: "a", Claim: "CLAIM_WITNESSED", Verdict: "OK", Witness: "diff-witnessed"}, {RegistrationID: "reg", SHA: "b", Claim: "CLAIM_WITNESSED", Verdict: "OK", Witness: "diff-witnessed"}}, "conflicting_green_witnesses"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AttributeOutcomes([]OutcomeSession{row}, tt.evidence)
			if got[0].Outcome != OutcomeUnknown || got[0].Reason != tt.reason {
				t.Fatalf("got %#v", got[0])
			}
		})
	}
}

package issuecontract

import "testing"

func TestRootPointStrictModeRefusesMissingFields(t *testing.T) {
	c := Candidate{
		Key: "qa-root", Title: "fix(qa): catch defect at origin", Generation: "now",
		CurrentState: "the defect escapes", WhyNow: "recurrence", WorkingSpine: "existing checker",
		InScope: "one origin check", OutOfScope: "unrelated cleanup", DoneCondition: "origin check rejects defect",
		Witness: "go test ./internal/issuecontract -run RootPoint", Lane: "issuecontract",
		Paths: []string{"internal/issuecontract/**"}, ExpectedSteps: 2,
	}
	got := ReviewCandidate(c, Options{StrictRootPoint: true})
	if got.Verdict != "needs_scope" {
		t.Fatalf("verdict=%s want %s", got.Verdict, "needs_scope")
	}
	for _, want := range []string{"root_point", "origin_signal", "prevents_recurrence"} {
		if !containsString(got.MissingFields, want) {
			t.Errorf("missing_fields=%v lacks %q", got.MissingFields, want)
		}
	}
	c.RootPoint = "candidate creation"
	c.OriginSignal = "contract review"
	c.PreventsRecurrence = "strict review refuses incomplete candidates"
	got = ReviewCandidate(c, Options{StrictRootPoint: true})
	for _, field := range []string{"root_point", "origin_signal", "prevents_recurrence"} {
		if containsString(got.MissingFields, field) {
			t.Fatalf("complete candidate still missing %q: %v", field, got.MissingFields)
		}
	}
}

func TestRootPointFieldsRoundTripFromIssueBody(t *testing.T) {
	got := CandidateFromIssueDraft(IssueDraft{Number: 1971, Title: "feat(issuecontract): root point", Body: `## Root point
candidate creation

## Origin signal
contract review

## Prevents recurrence
strict review refuses omissions
`})
	if got.RootPoint != "candidate creation" || got.OriginSignal != "contract review" || got.PreventsRecurrence != "strict review refuses omissions" {
		t.Fatalf("root-point fields did not round-trip: %#v", got)
	}
}

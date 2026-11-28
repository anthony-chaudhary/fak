package perfledger

import (
	"slices"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestSubmissionOutcome(t *testing.T) {
	ok := FoldSubmissionOutcome(SubmissionInput{Requests: 10, DurationSeconds: 5, HasEnvelope: true, GatePassed: true})
	if !ok.Valid || len(ok.Reasons) != 0 {
		t.Fatalf("clean submission must be valid: %+v", ok)
	}
	bad := FoldSubmissionOutcome(SubmissionInput{Requests: 10, FailedRequests: 2, DurationSeconds: 5, HasEnvelope: true, GatePassed: false})
	want := []SubmissionReason{ReasonFailedRequests, ReasonGateFailed}
	if bad.Valid || !slices.Equal(bad.Reasons, want) {
		t.Fatalf("got %+v, want reasons %v", bad, want)
	}
	// Empty input folds every applicable reason (failed_requests needs a failure count).
	empty := FoldSubmissionOutcome(SubmissionInput{})
	if empty.Valid || len(empty.Reasons) != 4 || !slices.IsSorted(empty.Reasons) {
		t.Fatalf("empty input must fail with 4 sorted reasons: %+v", empty)
	}
}

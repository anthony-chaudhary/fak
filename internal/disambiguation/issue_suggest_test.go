package disambiguation

import (
	"errors"
	"testing"
)

func TestSuggestIssueContractAndSafety(t *testing.T) {
	r, err := RunIssueSuggestionSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	s := r.Suggestion
	if s.Title == "" || s.Problem == "" || s.DoneCondition == "" || s.AcceptanceGate == "" || len(s.LikelyFiles) == 0 || s.DedupeQuery == "" || !r.UnsafeRejected || !r.NoAutoFile {
		t.Fatalf("report=%#v", r)
	}
	_, err = SuggestIssue(CoverageFinding{Surface: "gpu-01.internal/api.go", Candidate: "x", Reason: "hard"})
	if !errors.Is(err, ErrIssueSuggestionUnsafe) {
		t.Fatalf("err=%v", err)
	}
}

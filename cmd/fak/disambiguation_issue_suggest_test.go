package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"testing"
)

func TestDisambiguationIssueSuggestSelfTestCLI(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runDisambiguation(&out, &errout, []string{"issue-suggest-self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var r disambiguation.IssueSuggestionSelfTestReport
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Suggestion.Title == "" || r.Suggestion.Problem == "" || r.Suggestion.DoneCondition == "" || r.Suggestion.AcceptanceGate == "" || len(r.Suggestion.LikelyFiles) == 0 || r.Suggestion.DedupeQuery == "" || !r.UnsafeRejected || !r.NoAutoFile {
		t.Fatalf("report=%#v", r)
	}
}

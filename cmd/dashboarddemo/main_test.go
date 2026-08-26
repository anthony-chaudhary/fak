package main

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/cmd/internal/democapture"
)

func TestSelfcheck(t *testing.T) {
	got, err := runSelfcheck()
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != schema || got.Verdict != "pass" || got.RichDashboardCount != 9 || !got.RichDashboardsLazy {
		t.Fatalf("unexpected witness: %+v", got)
	}
	out, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	out = append(out, '\n')
	if err := democapture.MatchMarkdown("EXAMPLE-OUTPUT.md", out); err != nil {
		t.Fatal(err)
	}
}

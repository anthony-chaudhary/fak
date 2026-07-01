package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuesmallness"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// A rated backlog with one fail folds to a card whose headline debt is the fail count and
// whose verdict is ACTION, so `fak scoreboard post --from -` publishes a live number.
func TestIssueSmallnessScorecardFoldsFailAsDebt(t *testing.T) {
	report := issuesmallness.ReportOpen([]issuesmallness.Issue{
		{Number: 1, Title: "clean one", Body: dispatchIssueSmallnessSingleBody},
		{Number: 2, Title: "bundled one", Body: dispatchIssueSmallnessThreeBody},
	})
	card := issueSmallnessScorecard(report)

	if card.Schema != issueSmallnessScorecardSchema {
		t.Fatalf("schema = %q, want %q", card.Schema, issueSmallnessScorecardSchema)
	}
	if card.OK || card.Verdict != "ACTION" {
		t.Fatalf("ok=%v verdict=%q, want a failing (ACTION) card", card.OK, card.Verdict)
	}
	if got := card.Corpus[issueSmallnessDebtKey]; got != 1 {
		t.Fatalf("corpus[%s] = %v, want 1 (the single fail)", issueSmallnessDebtKey, got)
	}
	if card.Corpus["pass_count"] != 1 || card.Corpus["fail_count"] != 1 || card.Corpus["scanned"] != 2 {
		t.Fatalf("corpus counts = %#v, want scanned 2 / pass 1 / fail 1", card.Corpus)
	}
}

// An empty backlog is a clean card: score 100, zero debt, OK verdict.
func TestIssueSmallnessScorecardCleanWhenNoIssues(t *testing.T) {
	card := issueSmallnessScorecard(issuesmallness.ReportOpen(nil))
	if !card.OK || card.Verdict != "OK" {
		t.Fatalf("ok=%v verdict=%q, want a clean (OK) card", card.OK, card.Verdict)
	}
	if got := card.Corpus[issueSmallnessDebtKey]; got != 0 {
		t.Fatalf("corpus[%s] = %v, want 0", issueSmallnessDebtKey, got)
	}
	if card.Corpus["score"] != scorecard.Round1(100) {
		t.Fatalf("score = %v, want 100", card.Corpus["score"])
	}
}

// The `--open --scorecard` path emits the folded control-pane payload and returns 0 (the
// producer never gates), so it pipes cleanly into `fak scoreboard post --from -`.
func TestDispatchIssueSmallnessOpenScorecardEmitsPayload(t *testing.T) {
	oldFetch := dispatchIssueSmallnessFetchOpenIssues
	dispatchIssueSmallnessFetchOpenIssues = func(limit int) ([]issuesmallness.Issue, error) {
		return []issuesmallness.Issue{
			{Number: 1, Title: "clean one", Body: dispatchIssueSmallnessSingleBody},
			{Number: 2, Title: "bundled one", Body: dispatchIssueSmallnessThreeBody},
		}, nil
	}
	t.Cleanup(func() { dispatchIssueSmallnessFetchOpenIssues = oldFetch })

	var stdout, stderr strings.Builder
	code := runDispatchIssueSmallnessLint(&stdout, &stderr, strings.NewReader(""), []string{"--open", "--scorecard"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (producer never gates) (stderr: %s)", code, stderr.String())
	}
	var got scorecard.Payload
	if err := json.Unmarshal([]byte(stdout.String()), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, stdout.String())
	}
	if got.Schema != issueSmallnessScorecardSchema || got.Verdict != "ACTION" {
		t.Fatalf("payload = %#v, want the rated-issues card with an ACTION verdict", got)
	}
	if got.Corpus[issueSmallnessDebtKey] != float64(1) {
		t.Fatalf("debt = %v, want 1 through the JSON round-trip", got.Corpus[issueSmallnessDebtKey])
	}
}

// --scorecard without --open is a usage error: there is nothing to rate.
func TestDispatchIssueSmallnessScorecardRequiresOpen(t *testing.T) {
	var stdout, stderr strings.Builder
	code := runDispatchIssueSmallnessLint(&stdout, &stderr, strings.NewReader(""), []string{"--scorecard", "--body-file", "-"})
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "--scorecard only applies to --open") {
		t.Fatalf("stderr = %q, want the --scorecard/--open guard", stderr.String())
	}
}

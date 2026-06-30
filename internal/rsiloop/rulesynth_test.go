package rsiloop

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rulesynth"
)

// guarded is the protected tree fragment the test near-misses reach.
var guarded = []string{"internal/adjudicator/"}

// mineCorpus runs the real rulesynth.Detect predicate over a set of raw calls and
// returns the near-misses — so the test corpus is exactly what the live stream would
// mine, never a hand-faked NearMiss.
func mineCorpus(t *testing.T, cmds []string) []rulesynth.NearMiss {
	t.Helper()
	var corpus []rulesynth.NearMiss
	for _, cmd := range cmds {
		if nm, ok := rulesynth.Detect(rulesynth.Call{Tool: "Bash", Arg: "command", Command: cmd}, guarded); ok {
			corpus = append(corpus, nm)
		}
	}
	if len(corpus) == 0 {
		t.Fatalf("no near-misses mined from %v — fixture would not exercise the loop", cmds)
	}
	return corpus
}

// TestRuleSynthHarness_KeepsCatchingRule drives the synthesis through the REAL engine
// (Run) and asserts the keep-bit kept a rule that newly catches refusal-log near-misses
// without regressing a benign call — the rung-2 loop end to end.
func TestRuleSynthHarness_KeepsCatchingRule(t *testing.T) {
	// An unrecognized interpreter-eval write that reaches the guarded tree: a true
	// near-miss the current floor admits.
	corpus := mineCorpus(t, []string{
		"php -r 'file_put_contents(\"internal/adjudicator/decide.go\", $x);'",
		"php -r 'file_put_contents(\"internal/adjudicator/policy.go\", $y);'",
	})
	// A benign use of the SAME interpreter on an UNguarded path must stay admitted.
	benign := []rulesynth.Call{
		{Tool: "Bash", Arg: "command", Command: "php -r 'echo 1+1;'"},
		{Tool: "Bash", Arg: "command", Command: "php -r 'echo file_get_contents(\"docs/readme.md\");'"},
	}

	h := NewRuleSynthHarness(corpus, benign)

	// Baseline catches nothing — a near-miss is, by construction, admitted by the floor.
	base, ref, err := h.BaselineMetric()
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	if base != 0 {
		t.Fatalf("baseline metric = %v, want 0 (floor catches no near-miss)", base)
	}
	if ref == "" {
		t.Fatal("baseline ref empty")
	}

	res, err := Run(h, nil, 0, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Kept < 1 {
		t.Fatalf("expected >=1 kept rule, got %d (rows: %+v)", res.Kept, res.Rows)
	}
	if res.FinalBaseline < 1 {
		t.Fatalf("kept rule should catch >=1 near-miss, final baseline = %v", res.FinalBaseline)
	}
	// The kept row must be a real, gated KEEP — measured, suite-green, and truth-clean.
	var keptRow *Row
	for i := range res.Rows {
		if res.Rows[i].Kept {
			keptRow = &res.Rows[i]
			break
		}
	}
	if keptRow == nil {
		t.Fatal("no kept row found")
	}
	if !keptRow.SuiteGreen {
		t.Errorf("kept rule regressed a benign call (suite not green): %+v", keptRow)
	}
	if !keptRow.TruthClean {
		t.Errorf("kept rule did not catch its whole cluster (not truth-clean): %+v", keptRow)
	}
	if keptRow.MetricName != RuleSynthMetricName {
		t.Errorf("metric name = %q, want %q", keptRow.MetricName, RuleSynthMetricName)
	}
	if keptRow.Score == nil || keptRow.Score.Name != RuleSynthMetricName || keptRow.Score.Grade != "clean" {
		t.Fatalf("kept rule should carry a clean rulesynth scorecard: %+v", keptRow.Score)
	}
	if got := scoreComponentValue(keptRow.Score, "caught"); got < 1 {
		t.Fatalf("rulesynth score should expose caught near-misses, got %.0f in %+v", got, keptRow.Score)
	}
	if got := scoreComponentValue(keptRow.Score, "regressed"); got != 0 {
		t.Fatalf("kept rulesynth score should expose zero regressions, got %.0f in %+v", got, keptRow.Score)
	}
	if got := scoreComponentValue(keptRow.Score, "catches_cluster"); got != 1 {
		t.Fatalf("kept rulesynth score should expose catches_cluster=true, got %.0f in %+v", got, keptRow.Score)
	}
}

// TestRuleSynthHarness_RevertsRegressingRule proves the keep-bit refuses a candidate
// that would catch a near-miss only by also denying a benign call — the engine, not the
// proposer, holds the no-regression line.
func TestRuleSynthHarness_RevertsRegressingRule(t *testing.T) {
	corpus := mineCorpus(t, []string{
		"php -r 'file_put_contents(\"internal/adjudicator/decide.go\", $x);'",
	})
	// A benign call that the synthesized "php -r ... adjudicator" rule WOULD also deny:
	// it names the same verb and the same guarded tree, but is a read. The synthesized
	// regex cannot tell read from write, so it regresses this — and the keep-bit must
	// therefore REVERT the only candidate.
	benign := []rulesynth.Call{
		{Tool: "Bash", Arg: "command", Command: "php -r 'echo file_get_contents(\"internal/adjudicator/decide.go\");'"},
	}

	h := NewRuleSynthHarness(corpus, benign)
	res, err := Run(h, nil, 0, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Kept != 0 {
		t.Fatalf("expected 0 kept (rule regresses a benign call), got %d (rows: %+v)", res.Kept, res.Rows)
	}
	if len(res.Rows) == 0 {
		t.Fatal("expected at least one candidate row")
	}
	if res.Rows[0].SuiteGreen {
		t.Errorf("regressing candidate should be suite-red, got green: %+v", res.Rows[0])
	}
	if res.Rows[0].Score == nil || res.Rows[0].Score.Grade != "regressing" {
		t.Fatalf("regressing candidate should carry a regressing scorecard: %+v", res.Rows[0].Score)
	}
	if got := scoreComponentValue(res.Rows[0].Score, "regressed"); got == 0 {
		t.Fatalf("regressing score should expose benign regressions: %+v", res.Rows[0].Score)
	}
}

// TestRuleSynthHarness_BadPayload guards the Measure type assertion.
func TestRuleSynthHarness_BadPayload(t *testing.T) {
	h := NewRuleSynthHarness(nil, nil)
	if _, err := h.Measure(Candidate{Label: "x", Payload: 42}); err == nil {
		t.Fatal("expected error for non-rulesynth.Candidate payload")
	}
}

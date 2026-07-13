package quality

import (
	"strings"
	"testing"
)

// mtCarriedContext is the hermetic carried context for the multi-turn tests:
// two facts committed before the dialog under test begins (a prior session /
// conversation summary), one per line.
const mtCarriedContext = "the deploy window is tuesday\nthe rollout owner is priya"

// mtFaithfulDialog re-affirms both carried facts and commits one new one; no
// turn contradicts anything.
const mtFaithfulDialog = `user: when is the deploy window?
assistant: the deploy window is tuesday
user: who owns the rollout?
assistant: the rollout owner is priya; the runbook is up to date`

// mtContradictoryDialog is the defect witness: turn 1 commits the deploy
// window, turn 3 asserts the opposite — the class of drift this oracle exists
// to catch.
const mtContradictoryDialog = `user: when is the deploy window?
assistant: the deploy window is tuesday
user: are we still on for that?
assistant: the deploy window is friday`

// mtContextDialog contradicts a fact carried into the conversation rather than
// one committed inside it.
const mtContextDialog = `user: who owns the rollout?
assistant: the rollout owner is marco`

// mtCase builds a valid case whose Reference.Text carries the dialog's
// committed context.
func mtCase(carried string, minScore float64) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "multiturn-consistency-deploy-dialog",
		Version:   1,
		Prompt:    "Continue the deploy-planning dialog without contradicting committed facts.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: carried},
		Oracles:   []string{"multiturn-consistency"},
		Rubric:    RubricSpec{MinScore: minScore},
	}
}

// TestMultiTurnConsistencyRegistered proves the oracle registered under its
// stable name and kind, so cases can reference it by name.
func TestMultiTurnConsistencyRegistered(t *testing.T) {
	os, err := Lookup([]string{"multiturn-consistency"})
	if err != nil {
		t.Fatalf("Lookup(multiturn-consistency): %v", err)
	}
	if got := os[0].Kind(); got != "rubric" {
		t.Errorf("Kind() = %q, want rubric", got)
	}
}

// TestMultiTurnConsistencyFaithfulDialogPasses is the happy path: every turn
// respects the carried context and earlier turns, so the oracle passes at 1.0.
func TestMultiTurnConsistencyFaithfulDialogPasses(t *testing.T) {
	c := mtCase(mtCarriedContext, 1)
	v := mtConsistency{}.Judge(Trace{Text: mtCarriedContext}, Trace{Text: mtFaithfulDialog}, c)
	if !v.Pass {
		t.Fatalf("faithful dialog must pass; got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0", v.Score)
	}
	if v.FirstDivergence != nil {
		t.Errorf("faithful dialog must carry no divergence; got %+v", v.FirstDivergence)
	}
}

// TestMultiTurnConsistencyContradictionFails is the defect witness: a later
// turn asserting the opposite of an earlier committed fact fails, Detail names
// the contradiction pair, and FirstDivergence localizes the contradicting turn.
func TestMultiTurnConsistencyContradictionFails(t *testing.T) {
	c := mtCase("", 1)
	v := mtConsistency{}.Judge(Trace{}, Trace{Text: mtContradictoryDialog}, c)
	if v.Pass {
		t.Fatalf("contradictory dialog must not pass; got %+v", v)
	}
	if want := 0.5; v.Score != want {
		t.Errorf("score = %v, want %v (1 of 2 assertions contradicts)", v.Score, want)
	}
	for _, clause := range []string{"the deploy window is friday", "the deploy window is tuesday"} {
		if !strings.Contains(v.Detail, clause) {
			t.Errorf("Detail must name the contradiction pair member %q; got %q", clause, v.Detail)
		}
	}
	d := v.FirstDivergence
	if d == nil {
		t.Fatal("failing verdict must localize the contradicting turn via FirstDivergence")
	}
	if d.Index != 3 {
		t.Errorf("FirstDivergence.Index = %d, want 3 (the contradicting turn)", d.Index)
	}
	if d.Reference != "the deploy window is tuesday" || d.Engine != "the deploy window is friday" {
		t.Errorf("FirstDivergence = %+v, want committed vs contradicting clause", d)
	}
}

// TestMultiTurnConsistencyCarriedContextRespected proves the carried-context
// half of the contract: a turn contradicting a fact carried in via the
// reference trace fails, and Detail attributes the committed fact to the
// carried context.
func TestMultiTurnConsistencyCarriedContextRespected(t *testing.T) {
	c := mtCase(mtCarriedContext, 1)
	v := mtConsistency{}.Judge(Trace{Text: mtCarriedContext}, Trace{Text: mtContextDialog}, c)
	if v.Pass {
		t.Fatalf("dialog contradicting carried context must not pass; got %+v", v)
	}
	if !strings.Contains(v.Detail, "carried context") {
		t.Errorf("Detail must attribute the committed fact to the carried context; got %q", v.Detail)
	}
	for _, clause := range []string{"the rollout owner is marco", "the rollout owner is priya"} {
		if !strings.Contains(v.Detail, clause) {
			t.Errorf("Detail must name the contradiction pair member %q; got %q", clause, v.Detail)
		}
	}
}

// TestMultiTurnConsistencyNegationContradicts covers both negation directions:
// denying a committed value, and asserting a value an earlier turn denied.
func TestMultiTurnConsistencyNegationContradicts(t *testing.T) {
	c := mtCase("", 1)
	cases := []struct {
		name, dialog string
	}{
		{"deny-committed", "the feature flag is enabled\nthe feature flag is not enabled"},
		{"assert-denied", "the cache is not warm\nthe cache is warm"},
	}
	for _, tc := range cases {
		v := mtConsistency{}.Judge(Trace{}, Trace{Text: tc.dialog}, c)
		if v.Pass {
			t.Errorf("%s: negation contradiction must not pass; got %+v", tc.name, v)
			continue
		}
		if v.FirstDivergence == nil || v.FirstDivergence.Index != 1 {
			t.Errorf("%s: FirstDivergence must localize turn 1; got %+v", tc.name, v.FirstDivergence)
		}
	}
}

// TestMultiTurnConsistencyMinScoreTolerance proves the threshold gate: the same
// half-contradictory dialog passes when the case tolerates MinScore 0.5, and
// the tolerated contradiction is still named.
func TestMultiTurnConsistencyMinScoreTolerance(t *testing.T) {
	c := mtCase("", 0.5)
	v := mtConsistency{}.Judge(Trace{}, Trace{Text: mtContradictoryDialog}, c)
	if !v.Pass {
		t.Fatalf("1/2 consistent must pass at MinScore 0.5; got %+v", v)
	}
	if !strings.Contains(v.Detail, "the deploy window is friday") {
		t.Errorf("tolerated-contradiction detail should still name the clause; got %q", v.Detail)
	}
}

// TestMultiTurnConsistencyNoFactsDialog defines the nothing-asserted edge: a
// dialog of questions and small talk commits no facts, so nothing can
// contradict — pass at score 1, no panic.
func TestMultiTurnConsistencyNoFactsDialog(t *testing.T) {
	c := mtCase(mtCarriedContext, 1)
	for _, text := range []string{"", "user: when is the deploy window?\nassistant: let me check", "   \n  "} {
		v := mtConsistency{}.Judge(Trace{Text: mtCarriedContext}, Trace{Text: text}, c)
		if !v.Pass || v.Score != 1 {
			t.Errorf("factless dialog %q: got %+v, want pass at score 1", text, v)
		}
	}
}

// TestMultiTurnConsistencySpineIntegration runs the contradictory dialog
// through the full spine: the failure bundle names multiturn-consistency as the
// failing oracle and carries the contradiction pair in its detail.
func TestMultiTurnConsistencySpineIntegration(t *testing.T) {
	c := mtCase(mtCarriedContext, 1)
	eng := ScriptedRunner{Label: "engine-state-drift", Trace: Trace{Text: mtContradictoryDialog}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("contradictory dialog must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "multiturn-consistency" {
		t.Errorf("failing oracle = %q, want multiturn-consistency", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, "the deploy window is friday") {
		t.Errorf("bundle detail must name the contradicting clause; got %q", fb.Detail)
	}
}

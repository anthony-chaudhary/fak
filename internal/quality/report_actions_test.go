package quality

import (
	"strings"
	"testing"
)

// actFaithfulItems is the complete report: a win (no actionability obligation),
// a risk with an owner and a next action, and a decision-needing blocker that
// carries all three fields.
func actFaithfulItems() []actItem {
	return []actItem{
		{Kind: "win", Title: "cache hit rate reached 91%"},
		{Kind: "risk", Title: "vendor API deprecation in Q3",
			Owner: "dana", NextAction: "draft the migration plan by Friday"},
		{Kind: "blocker", Title: "staging migration blocked on DBA review",
			Owner: "lee", NextAction: "escalate the review queue",
			NeedsDecision: true, DecisionAsk: "approve contractor DBA hours"},
	}
}

// actCase builds a hermetic report case judged only by the action-completeness
// oracle, with the default threshold (every risk/blocker must be actionable).
func actCase() QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "report-action-completeness",
		Version:   1,
		Prompt:    "Write the weekly engineering status report as structured items for the executive rollup.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: actItemsText(actFaithfulItems())},
		Oracles:   []string{"action-completeness"},
	}
}

// actVerdict pulls the action-completeness verdict out of a result or fails the
// test.
func actVerdict(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "action-completeness" {
			return v
		}
	}
	t.Fatalf("no action-completeness verdict in %s", Explain(res))
	return Verdict{}
}

// TestActionCompletenessFaithfulReportPasses is the faithful path: every raised
// risk/blocker carries an owner, a next action, and the flagged decision ask,
// and the win carries no obligation — full score, no failure bundle.
func TestActionCompletenessFaithfulReportPasses(t *testing.T) {
	c := actCase()
	eng := ScriptedRunner{Label: "engine-faithful", Trace: Trace{Text: actItemsText(actFaithfulItems())}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("complete actionable report should pass; got %s", Explain(res))
	}
	if v := actVerdict(t, res); v.Score != 1 {
		t.Errorf("faithful report score = %v, want 1", v.Score)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestActionCompletenessIncompleteItemsFail is the defect Witness for #4554: a
// report that raises its blocker with a "TBD" owner and its risk with no next
// action must fail, and the Detail must name each incomplete item with the
// exact missing field while never listing the complete one.
func TestActionCompletenessIncompleteItemsFail(t *testing.T) {
	c := actCase()
	items := []actItem{
		{Kind: "risk", Title: "flaky-test quarantine growth",
			Owner: "mira", NextAction: "burn the quarantine list down to 3"},
		{Kind: "risk", Title: "vendor API deprecation in Q3", Owner: "dana"}, // no next action
		{Kind: "blocker", Title: "staging migration blocked on DBA review",
			Owner: "TBD", NextAction: "escalate the review queue",
			NeedsDecision: true, DecisionAsk: "approve contractor DBA hours"}, // placeholder owner
	}
	eng := ScriptedRunner{Label: "engine-unowned", Trace: Trace{Text: actItemsText(items)}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("report with an ownerless blocker and an actionless risk must not pass; got %s", Explain(res))
	}
	v := actVerdict(t, res)
	if v.Pass {
		t.Fatal("action-completeness verdict should have failed")
	}
	if want := 1.0 / 3.0; v.Score != want {
		t.Errorf("score = %v, want %v (1 of 3 actionable items complete)", v.Score, want)
	}
	for _, want := range []string{
		`risk "vendor API deprecation in Q3" missing next action`,
		`blocker "staging migration blocked on DBA review" missing owner`,
	} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail %q missing incomplete item %s", v.Detail, want)
		}
	}
	if strings.Contains(v.Detail, "flaky-test quarantine growth") {
		t.Errorf("Detail %q lists the complete item as incomplete", v.Detail)
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "action-completeness" {
		t.Errorf("first failing oracle = %q, want action-completeness", fb.FailingOracle)
	}
}

// TestActionCompletenessDecisionAskGate covers the "where applicable" clause:
// the SAME owned, actioned blocker fails when it flags needs_decision without
// stating the ask, and passes once no decision is declared needed.
func TestActionCompletenessDecisionAskGate(t *testing.T) {
	c := actCase()
	flagged := actItem{Kind: "blocker", Title: "ledger cutover frozen",
		Owner: "lee", NextAction: "stage the dual-write shim", NeedsDecision: true}
	eng := ScriptedRunner{Label: "engine-no-ask", Trace: Trace{Text: actItemsText([]actItem{flagged})}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("needs_decision blocker without a decision ask must not pass; got %s", Explain(res))
	}
	if v := actVerdict(t, res); !strings.Contains(v.Detail, `blocker "ledger cutover frozen" missing decision ask`) {
		t.Errorf("Detail %q does not localize the missing decision ask", v.Detail)
	}

	unflagged := flagged
	unflagged.NeedsDecision = false
	eng = ScriptedRunner{Label: "engine-no-decision-needed", Trace: Trace{Text: actItemsText([]actItem{unflagged})}}
	res, err = RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("owned+actioned blocker with no decision needed should pass; got %s", Explain(res))
	}
}

// TestActionCompletenessEdgeBehavior pins the two defined edges: a report with
// no risk/blocker items passes vacuously, and a Text that is not a JSON item
// array fails closed with score 0.
func TestActionCompletenessEdgeBehavior(t *testing.T) {
	c := actCase()
	wins := []actItem{{Kind: "win", Title: "onboarded two engineers"}}
	eng := ScriptedRunner{Label: "engine-wins-only", Trace: Trace{Text: actItemsText(wins)}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("report with no risk/blocker items should pass vacuously; got %s", Explain(res))
	}

	eng = ScriptedRunner{Label: "engine-prose", Trace: Trace{Text: "Overall a smooth week with no notable escalations."}}
	res, err = RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("unparseable report must fail closed; got %s", Explain(res))
	}
	v := actVerdict(t, res)
	if v.Score != 0 {
		t.Errorf("unparseable report score = %v, want 0", v.Score)
	}
	if !strings.Contains(v.Detail, "not parseable") {
		t.Errorf("Detail %q does not explain the parse refusal", v.Detail)
	}
}

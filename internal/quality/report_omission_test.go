package quality

import (
	"strings"
	"testing"
)

// omissionFaithfulReport covers every material item omissionCase declares — the
// "nothing dropped" engine output.
const omissionFaithfulReport = "Win: cache hit rate reached 91% after the shard rebalance. " +
	"Risk: vendor API deprecation in Q3 still needs a migration owner. " +
	"Blocker: staging migration blocked on DBA review since Tuesday. " +
	"Decision: we chose Postgres over DynamoDB for the ledger store."

// omissionCase builds a hermetic report case judged only by the material-omission
// oracle: one material item in each of the four categories, carried on the case
// via the MaterialItems helper, with the default threshold (nothing may be
// omitted).
func omissionCase() QualityCase {
	items := MaterialItems{
		Wins:      []string{"cache hit rate reached 91%"},
		Risks:     []string{"vendor API deprecation in Q3"},
		Blockers:  []string{"staging migration blocked on DBA review"},
		Decisions: []string{"chose Postgres over DynamoDB"},
	}
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "report-material-omission",
		Version:   1,
		Prompt:    "Write the weekly engineering status report for the executive rollup.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: omissionFaithfulReport},
		Oracles:   []string{"material-omission"},
		Rubric:    items.Rubric(0),
	}
}

// omissionVerdict pulls the material-omission verdict out of a result or fails
// the test.
func omissionVerdict(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "material-omission" {
			return v
		}
	}
	t.Fatalf("no material-omission verdict in %s", Explain(res))
	return Verdict{}
}

// TestMaterialOmissionCompleteReportPasses is the faithful path: a report
// covering every declared win, risk, blocker, and decision passes with a full
// score and no failure bundle.
func TestMaterialOmissionCompleteReportPasses(t *testing.T) {
	c := omissionCase()
	eng := ScriptedRunner{Label: "engine-faithful", Trace: Trace{Text: omissionFaithfulReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("complete report should pass; got %s", Explain(res))
	}
	if v := omissionVerdict(t, res); v.Score != 1 {
		t.Errorf("complete report score = %v, want 1", v.Score)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestMaterialOmissionDroppedItemsFail is the defect Witness for #4552: a report
// that reads fine — wins and risks are covered — but silently drops the material
// blocker and decision must fail, and the Detail must name each omitted item with
// its category while never listing the covered ones.
func TestMaterialOmissionDroppedItemsFail(t *testing.T) {
	c := omissionCase()
	text := "Win: cache hit rate reached 91% after the shard rebalance. " +
		"Risk: vendor API deprecation in Q3 still needs a migration owner. " +
		"Overall a smooth week with no notable escalations."
	eng := ScriptedRunner{Label: "engine-omits", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("report omitting a blocker and a decision must not pass; got %s", Explain(res))
	}
	v := omissionVerdict(t, res)
	if v.Pass {
		t.Fatal("material-omission verdict should have failed")
	}
	if v.Score != 0.5 {
		t.Errorf("score = %v, want 0.5 (2 of 4 items covered)", v.Score)
	}
	for _, want := range []string{
		`blocker "staging migration blocked on DBA review"`,
		`decision "chose Postgres over DynamoDB"`,
	} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail %q missing omitted item %s", v.Detail, want)
		}
	}
	for _, covered := range []string{"cache hit rate", "vendor API deprecation"} {
		if strings.Contains(v.Detail, covered) {
			t.Errorf("Detail %q lists covered item %q as omitted", v.Detail, covered)
		}
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "material-omission" {
		t.Errorf("first failing oracle = %q, want material-omission", fb.FailingOracle)
	}
}

// TestMaterialOmissionExtraContentStillPasses distinguishes this oracle from
// grounding: a report that ADDS extra grounded content but omits nothing is not
// penalized — only omission fails here.
func TestMaterialOmissionExtraContentStillPasses(t *testing.T) {
	c := omissionCase()
	text := omissionFaithfulReport +
		" Additionally, we onboarded two engineers, refreshed the on-call dashboards," +
		" and cut the flaky-test quarantine list from 14 to 3."
	eng := ScriptedRunner{Label: "engine-verbose", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("extra grounded content must not fail an omission check; got %s", Explain(res))
	}
	if v := omissionVerdict(t, res); v.Score != 1 {
		t.Errorf("verbose-but-complete report score = %v, want 1", v.Score)
	}
}

// TestMaterialOmissionMinScoreAndPlainItems covers the two authoring variants: an
// explicit MinScore admits bounded omission, and plain (uncategorized)
// Rubric.Required entries work without the helper — the omitted one is quoted
// bare in the Detail.
func TestMaterialOmissionMinScoreAndPlainItems(t *testing.T) {
	c := omissionCase()
	c.Rubric = RubricSpec{
		Required: []string{"cache hit rate reached 91%", "chose Postgres over DynamoDB"},
		MinScore: 0.5,
	}
	eng := ScriptedRunner{Label: "engine-partial", Trace: Trace{
		Text: "Win: cache hit rate reached 91% after the shard rebalance.",
	}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("score 0.50 >= MinScore 0.50 should pass; got %s", Explain(res))
	}

	tight := c
	tight.Rubric.MinScore = 0 // default: nothing material may be omitted
	res, err = RunCase(tight, ReferenceRunner{}, eng, oraclesFor(t, tight))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("default threshold must refuse any omission; got %s", Explain(res))
	}
	v := omissionVerdict(t, res)
	if want := `omitted: "chose Postgres over DynamoDB"`; !strings.Contains(v.Detail, want) {
		t.Errorf("Detail %q missing bare-quoted plain item %q", v.Detail, want)
	}
}

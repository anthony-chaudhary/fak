package quality

import (
	"strings"
	"testing"
)

// drFaithfulReport leads with both material decisions in the top/priority
// section and keeps the low-signal noise below — the correctly ranked report.
const drFaithfulReport = "# Priority decisions\n" +
	"Decision: adopt the usage-based pricing model for Q4 renewals.\n" +
	"Decision: freeze hiring for the platform team until revenue recovers.\n" +
	"\n" +
	"# Operational notes\n" +
	"Office plants were rotated to the south-facing windows.\n" +
	"The weekly newsletter shipped on time again.\n" +
	"\n" +
	"# Appendix\n" +
	"Full meeting transcripts are available on request.\n"

// drBuriedReport covers the same content — an omission check passes it — but
// parks the hiring freeze decision under the appendix noise.
const drBuriedReport = "# Priority decisions\n" +
	"Decision: adopt the usage-based pricing model for Q4 renewals.\n" +
	"\n" +
	"# Operational notes\n" +
	"Office plants were rotated to the south-facing windows.\n" +
	"The weekly newsletter shipped on time again.\n" +
	"\n" +
	"# Appendix\n" +
	"Full meeting transcripts are available on request.\n" +
	"Decision: freeze hiring for the platform team until revenue recovers.\n"

// drRelevanceCase builds a hermetic sectioned-report case judged only by the
// decision-relevance oracle: two material decisions carried on the case in the
// MaterialItems "decision:" encoding, plus a categorized non-decision entry the
// oracle must ignore, with the default threshold (every decision must lead).
func drRelevanceCase() QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "report-decision-relevance",
		Version:   1,
		Prompt:    "Write the weekly executive rollup, priority decisions first.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 96},
		Reference: Trace{Text: drFaithfulReport},
		Oracles:   []string{"decision-relevance"},
		Rubric: RubricSpec{
			Required: []string{
				"decision: adopt the usage-based pricing model",
				"decision: freeze hiring for the platform team",
				"win: the weekly newsletter shipped on time", // another oracle's item; ignored here
			},
		},
	}
}

// drRelevanceVerdict pulls the decision-relevance verdict out of a result or
// fails the test.
func drRelevanceVerdict(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "decision-relevance" {
			return v
		}
	}
	t.Fatalf("no decision-relevance verdict in %s", Explain(res))
	return Verdict{}
}

// TestDecisionRelevanceLeadingDecisionsPass is the faithful path: a report that
// leads with every material decision passes with a full score and no failure
// bundle, and the categorized non-decision entry does not dilute the score.
func TestDecisionRelevanceLeadingDecisionsPass(t *testing.T) {
	c := drRelevanceCase()
	eng := ScriptedRunner{Label: "engine-ranked", Trace: Trace{Text: drFaithfulReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("correctly ranked report should pass; got %s", Explain(res))
	}
	v := drRelevanceVerdict(t, res)
	if v.Score != 1 {
		t.Errorf("ranked report score = %v, want 1", v.Score)
	}
	if !strings.Contains(v.Detail, `"Priority decisions"`) {
		t.Errorf("Detail %q should name the top section", v.Detail)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestDecisionRelevanceBuriedDecisionFails is the defect Witness for #4553: the
// report still CONTAINS both decisions — omission cannot catch it — but buries
// the hiring freeze under appendix noise. The oracle must fail, score the
// burial, and name the buried decision and the section it sank to, while never
// flagging the correctly placed one.
func TestDecisionRelevanceBuriedDecisionFails(t *testing.T) {
	c := drRelevanceCase()
	eng := ScriptedRunner{Label: "engine-buries", Trace: Trace{Text: drBuriedReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("report burying a material decision must not pass; got %s", Explain(res))
	}
	v := drRelevanceVerdict(t, res)
	if v.Pass {
		t.Fatal("decision-relevance verdict should have failed")
	}
	if v.Score != 0.5 {
		t.Errorf("score = %v, want 0.5 (1 of 2 decisions leads)", v.Score)
	}
	for _, want := range []string{
		`decision "freeze hiring for the platform team" buried`,
		`section 3 ("Appendix")`,
	} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail %q missing %q", v.Detail, want)
		}
	}
	if strings.Contains(v.Detail, `"adopt the usage-based pricing model" buried`) {
		t.Errorf("Detail %q flags the correctly placed decision as buried", v.Detail)
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "decision-relevance" {
		t.Errorf("first failing oracle = %q, want decision-relevance", fb.FailingOracle)
	}
}

// TestDecisionRelevanceAbsentDecisionFailsClosed distinguishes absence from
// burial: a decision missing from the report entirely is certainly not in the
// top section, so the oracle fails closed and says "absent", not "buried".
func TestDecisionRelevanceAbsentDecisionFailsClosed(t *testing.T) {
	c := drRelevanceCase()
	text := "# Priority decisions\n" +
		"Decision: adopt the usage-based pricing model for Q4 renewals.\n" +
		"\n" +
		"# Appendix\n" +
		"Full meeting transcripts are available on request.\n"
	eng := ScriptedRunner{Label: "engine-drops", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatal("report missing a material decision must not pass decision-relevance")
	}
	v := drRelevanceVerdict(t, res)
	if want := `decision "freeze hiring for the platform team" absent`; !strings.Contains(v.Detail, want) {
		t.Errorf("Detail %q missing %q", v.Detail, want)
	}
}

// TestDecisionRelevanceParagraphFallback covers the headerless model: a plain
// prose report's first blank-line-separated paragraph is the top section, so a
// decision leading it passes and one pushed below the noise paragraph fails
// with the burial named by position.
func TestDecisionRelevanceParagraphFallback(t *testing.T) {
	c := drRelevanceCase()
	c.Rubric = RubricSpec{Required: []string{"decision: cut the vendor contract"}}

	lead := "We will cut the vendor contract at renewal.\n\nThe team also repainted the lobby.\n"
	eng := ScriptedRunner{Label: "engine-plain-lead", Trace: Trace{Text: lead}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("decision leading a plain report should pass; got %s", Explain(res))
	}

	buried := "The team repainted the lobby and rotated the plants.\n\nWe will cut the vendor contract at renewal.\n"
	eng = ScriptedRunner{Label: "engine-plain-buries", Trace: Trace{Text: buried}}
	res, err = RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatal("decision below the noise paragraph must not pass")
	}
	v := drRelevanceVerdict(t, res)
	if want := `decision "cut the vendor contract" buried in section 2`; !strings.Contains(v.Detail, want) {
		t.Errorf("Detail %q missing %q", v.Detail, want)
	}
}

// TestDecisionRelevanceMinScoreTolerates covers the explicit-threshold variant:
// MinScore 0.5 admits one buried decision — the verdict passes but the Detail
// still names the tolerated burial so it is observed, not hidden.
func TestDecisionRelevanceMinScoreTolerates(t *testing.T) {
	c := drRelevanceCase()
	c.Rubric.MinScore = 0.5
	eng := ScriptedRunner{Label: "engine-buries", Trace: Trace{Text: drBuriedReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("score 0.50 >= MinScore 0.50 should pass; got %s", Explain(res))
	}
	v := drRelevanceVerdict(t, res)
	if !strings.Contains(v.Detail, "tolerated") || !strings.Contains(v.Detail, "freeze hiring for the platform team") {
		t.Errorf("Detail %q should name the tolerated burial", v.Detail)
	}
}

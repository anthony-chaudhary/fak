package quality

import (
	"strings"
	"testing"
)

// oscPrevBlock is the previous report's status block oscCase carries in
// Reference.Text: auth-service arrives with a green -> red history (red is
// what the previous report asserted), billing and search are steady-state.
const oscPrevBlock = "auth-service | green -> red\n" +
	"billing | green\n" +
	"search | yellow"

// oscCase builds a hermetic report case judged only by the status-oscillation
// oracle, with the previous report's statuses serialized into Reference.Text
// and the default threshold (no unexplained flip tolerated).
func oscCase() QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "report-status-oscillation",
		Version:   1,
		Prompt:    "Write this week's engineering status rollup; the previous report's statuses are attached.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: oscPrevBlock},
		Oracles:   []string{"status-oscillation"},
	}
}

// oscVerdictFrom pulls the status-oscillation verdict out of a result or
// fails the test.
func oscVerdictFrom(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "status-oscillation" {
			return v
		}
	}
	t.Fatalf("no status-oscillation verdict in %s", Explain(res))
	return Verdict{}
}

// TestStatusOscillationExplainedFlipPasses is the faithful path: the report
// flips auth-service red -> green but narrates WHY in the flip sentence, and
// keeps the steady workstreams steady — a legitimate status change passes
// with a full score and no failure bundle.
func TestStatusOscillationExplainedFlipPasses(t *testing.T) {
	c := oscCase()
	text := "Auth-service is back to green after the login fix rolled out on Monday. " +
		"Billing remains green. Search is still yellow pending the reindex."
	eng := ScriptedRunner{Label: "engine-explains", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("explained flip should pass; got %s", Explain(res))
	}
	v := oscVerdictFrom(t, res)
	if v.Score != 1 {
		t.Errorf("explained-flip score = %v, want 1", v.Score)
	}
	if !strings.Contains(v.Detail, "1 flip(s) explained") {
		t.Errorf("Detail %q should note the explained flip", v.Detail)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestStatusOscillationUnexplainedFlipFails is the defect witness for #4560:
// auth-service completes a green -> red -> green round trip with NO rationale
// anywhere near the flip — the report reads fine sentence by sentence, and
// only the report-to-report comparison fails it. The Detail must name the
// item and its full oscillation chain.
func TestStatusOscillationUnexplainedFlipFails(t *testing.T) {
	c := oscCase()
	text := "Auth-service is green. Billing remains green. Search is still yellow pending the reindex."
	eng := ScriptedRunner{Label: "engine-oscillates", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("unexplained status oscillation must not pass; got %s", Explain(res))
	}
	v := oscVerdictFrom(t, res)
	if v.Pass {
		t.Fatal("status-oscillation verdict should have failed")
	}
	if want := 2.0 / 3.0; v.Score != want {
		t.Errorf("score = %v, want %v (2 of 3 statuses consistent)", v.Score, want)
	}
	for _, want := range []string{"oscillation", `"auth-service"`, "green -> red -> green"} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail %q missing %q", v.Detail, want)
		}
	}
	for _, steady := range []string{`"billing"`, `"search"`} {
		if strings.Contains(v.Detail, steady) {
			t.Errorf("Detail %q names steady workstream %s as a violation", v.Detail, steady)
		}
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "status-oscillation" {
		t.Errorf("first failing oracle = %q, want status-oscillation", fb.FailingOracle)
	}
}

// TestStatusOscillationRationaleInFollowingSentencePasses pins the rationale
// scope: the flip sentence itself is bare, but the immediately following
// sentence carries the "because" — normal report prose — and the flip counts
// as explained.
func TestStatusOscillationRationaleInFollowingSentencePasses(t *testing.T) {
	c := oscCase()
	c.Reference = Trace{Text: "payments | red"}
	text := "Payments is now green. This is because the checkout fix landed and the soak stayed clean."
	eng := ScriptedRunner{Label: "engine-next-sentence", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("rationale in the following sentence should explain the flip; got %s", Explain(res))
	}
	if v := oscVerdictFrom(t, res); v.Score != 1 {
		t.Errorf("score = %v, want 1", v.Score)
	}
}

// TestStatusOscillationEdgeBehavior pins the defined edges: an unparseable
// previous block constrains nothing; a declared workstream the report never
// gives a checkable status is skipped, not failed; and word-bounding keeps
// "delivered" from reading as a flip to red.
func TestStatusOscillationEdgeBehavior(t *testing.T) {
	// No parseable previous statuses ("purple" is outside the vocabulary).
	c := oscCase()
	c.Reference = Trace{Text: "no structured status block here\nauth | purple"}
	eng := ScriptedRunner{Label: "engine-a", Trace: Trace{Text: "Auth-service is red."}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("unparseable previous block must constrain nothing; got %s", Explain(res))
	}
	if v := oscVerdictFrom(t, res); !strings.Contains(v.Detail, "no previous statuses") {
		t.Errorf("Detail %q should note the empty previous block", v.Detail)
	}

	// Declared workstream never mentioned with a status: skipped, not failed.
	c = oscCase()
	c.Reference = Trace{Text: "ghost-stream | red"}
	eng = ScriptedRunner{Label: "engine-b", Trace: Trace{Text: "Billing remains green."}}
	res, err = RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("workstream without a checkable current status must be skipped; got %s", Explain(res))
	}
	if v := oscVerdictFrom(t, res); !strings.Contains(v.Detail, "no checkable current status") {
		t.Errorf("Detail %q should note nothing was checkable", v.Detail)
	}

	// Word-bounding: "delivered" contains "red" but asserts no status, so the
	// green workstream is skipped rather than read as an unexplained flip.
	c = oscCase()
	c.Reference = Trace{Text: "delivery | green"}
	eng = ScriptedRunner{Label: "engine-c", Trace: Trace{Text: "Delivery delivered the milestone on time."}}
	res, err = RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("'delivered' must not read as a flip to red; got %s", Explain(res))
	}
}

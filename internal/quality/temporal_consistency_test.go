package quality

import (
	"strings"
	"testing"
)

// tmpTemporalFactBlock is the dated ground truth for the temporal cases: the
// payments migration moved from "in progress" (2026-06-30) to "completed"
// (2026-07-10), and the vendor contract carries a single current fact. A
// malformed line proves the parser skips garbage instead of panicking.
const tmpTemporalFactBlock = "2026-06-30 | payments migration | in progress\n" +
	"2026-07-10 | payments migration | completed\n" +
	"2026-07-08 | vendor contract | renewed through 2027\n" +
	"not a fact line\n"

// tmpTemporalFaithfulReport asserts every topic's CURRENT status and carries
// the block's newest date as its currency claim.
const tmpTemporalFaithfulReport = "As of 2026-07-10, the payments migration is completed. " +
	"The vendor contract is renewed through 2027."

// tmpTemporalCase builds a hermetic report case judged only by the
// temporal-consistency oracle, carrying the dated fact block in
// Reference.Text per that oracle's contract.
func tmpTemporalCase() QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "report-temporal-consistency",
		Version:   1,
		Prompt:    "Write the weekly status rollup; state only the latest known facts.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: tmpTemporalFactBlock},
		Oracles:   []string{"temporal-consistency"},
	}
}

// tmpTemporalVerdict pulls the temporal-consistency verdict out of a result
// or fails the test.
func tmpTemporalVerdict(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "temporal-consistency" {
			return v
		}
	}
	t.Fatalf("no temporal-consistency verdict in %s", Explain(res))
	return Verdict{}
}

// TestTemporalConsistencyFaithfulReportPasses is the faithful path: a report
// asserting each topic's newest status, dated with the block's newest date,
// passes with a full score and no failure bundle.
func TestTemporalConsistencyFaithfulReportPasses(t *testing.T) {
	c := tmpTemporalCase()
	eng := ScriptedRunner{Label: "engine-current", Trace: Trace{Text: tmpTemporalFaithfulReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("report of current facts should pass; got %s", Explain(res))
	}
	if v := tmpTemporalVerdict(t, res); v.Score != 1 {
		t.Errorf("faithful report score = %v, want 1", v.Score)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestTemporalConsistencyStaleStatusFails is the defect Witness for #4556: a
// fluent report asserting the payments migration is still "in progress" —
// a status the 2026-07-10 datapoint contradicts — must fail, and the Detail
// must name the stale claim AND the newer fact that supersedes it.
func TestTemporalConsistencyStaleStatusFails(t *testing.T) {
	c := tmpTemporalCase()
	text := "The payments migration is in progress and on track. " +
		"The vendor contract is renewed through 2027."
	eng := ScriptedRunner{Label: "engine-stale-status", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("report asserting a superseded status must not pass; got %s", Explain(res))
	}
	v := tmpTemporalVerdict(t, res)
	if v.Pass {
		t.Fatal("temporal-consistency verdict should have failed")
	}
	if v.Score != 0.5 {
		t.Errorf("score = %v, want 0.5 (1 of 2 temporal claims consistent)", v.Score)
	}
	for _, want := range []string{
		`stale claim "The payments migration is in progress and on track"`,
		`superseded status "in progress" (2026-06-30)`,
		`newer fact (2026-07-10) says "completed"`,
	} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail %q missing %s", v.Detail, want)
		}
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "temporal-consistency" {
		t.Errorf("first failing oracle = %q, want temporal-consistency", fb.FailingOracle)
	}
}

// TestTemporalConsistencyStaleAsOfDateFails is the second defect class: the
// statuses are current, but the report stamps itself "as of 2026-06-30" —
// older than the block's newest datapoint — presenting a stale date as
// current.
func TestTemporalConsistencyStaleAsOfDateFails(t *testing.T) {
	c := tmpTemporalCase()
	text := "As of 2026-06-30, the payments migration is completed. " +
		"The vendor contract is renewed through 2027."
	eng := ScriptedRunner{Label: "engine-stale-date", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("report presenting a stale date as current must not pass; got %s", Explain(res))
	}
	v := tmpTemporalVerdict(t, res)
	if v.Pass {
		t.Fatal("temporal-consistency verdict should have failed")
	}
	for _, want := range []string{
		`report claims currency "as of 2026-06-30"`,
		"newer datapoint dated 2026-07-10",
	} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail %q missing %s", v.Detail, want)
		}
	}
}

// TestTemporalConsistencyNarratedHistoryPasses pins the documented tie-break:
// a sentence that narrates the superseded status but ALSO asserts the current
// one ("previously in progress, now completed") is consistent — the current
// status wins.
func TestTemporalConsistencyNarratedHistoryPasses(t *testing.T) {
	c := tmpTemporalCase()
	text := "The payments migration, previously in progress, is now completed. " +
		"The vendor contract is renewed through 2027."
	eng := ScriptedRunner{Label: "engine-narrates", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("narrated history alongside the current status must pass; got %s", Explain(res))
	}
	if v := tmpTemporalVerdict(t, res); v.Score != 1 {
		t.Errorf("narrated-history score = %v, want 1", v.Score)
	}
}

// TestTemporalConsistencyEdgeBehavior pins the defined empty edges: no
// parseable facts means nothing to check, and a report making no temporal
// claims about the declared topics passes with a note — neither is a silent
// nor a spurious failure.
func TestTemporalConsistencyEdgeBehavior(t *testing.T) {
	o := tmpTemporalConsistency{}
	c := tmpTemporalCase()

	v := o.Judge(Trace{Text: "no | valid\nfacts here"}, Trace{Text: tmpTemporalFaithfulReport}, c)
	if !v.Pass || v.Score != 1 {
		t.Errorf("no parseable facts should pass at score 1; got pass=%v score=%v detail=%q", v.Pass, v.Score, v.Detail)
	}
	if !strings.Contains(v.Detail, "no dated facts") {
		t.Errorf("Detail %q should note the missing fact block", v.Detail)
	}

	v = o.Judge(Trace{Text: tmpTemporalFactBlock}, Trace{Text: "We onboarded two engineers this week."}, c)
	if !v.Pass || v.Score != 1 {
		t.Errorf("report with no temporal claims should pass at score 1; got pass=%v score=%v detail=%q", v.Pass, v.Score, v.Detail)
	}
	if !strings.Contains(v.Detail, "no checkable temporal claims") {
		t.Errorf("Detail %q should note the absence of temporal claims", v.Detail)
	}
}

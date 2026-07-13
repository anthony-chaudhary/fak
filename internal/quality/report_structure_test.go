package quality

import (
	"strings"
	"testing"
)

// structFaithfulReport is the canonical engine output for structCase: every
// required section present, colon-labeled headings on their own lines.
const structFaithfulReport = "Summary:\n" +
	"Throughput increased 12% week over week.\n\n" +
	"Risks:\n" +
	"Vendor API deprecation lands in Q3 without a migration owner.\n\n" +
	"Decisions:\n" +
	"We chose Postgres over DynamoDB for the ledger store.\n\n" +
	"Next actions:\n" +
	"Assign a migration owner and schedule the DBA review.\n"

// structRestyledReport carries the SAME four sections with entirely different
// prose, a different section order, and four different heading styles (markdown
// hashes, bold, inline colon label, bare ALL-CAPS). Structure-without-style must
// treat it as equivalent to the faithful report.
const structRestyledReport = "## Decisions\n" +
	"The ledger store will be Postgres rather than DynamoDB.\n\n" +
	"**Next Actions**\n" +
	"- Find an owner for the vendor migration.\n" +
	"- Book the DBA review slot.\n\n" +
	"Risks: the Q3 vendor API deprecation still has nobody driving the migration.\n\n" +
	"SUMMARY\n" +
	"Week-over-week throughput is up twelve percent.\n"

// structCase builds a hermetic report case judged only by the
// structure-without-style oracle: four required section anchors, default
// threshold (every section must be present).
func structCase() QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "report-structure-without-style",
		Version:   1,
		Prompt:    "Write the weekly engineering status report for the executive rollup.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 128},
		Reference: Trace{Text: structFaithfulReport},
		Oracles:   []string{"structure-without-style"},
		Rubric: RubricSpec{
			Required: []string{"Summary", "Risks", "Decisions", "Next actions"},
		},
	}
}

// structVerdict pulls the structure-without-style verdict out of a result or
// fails the test.
func structVerdict(t *testing.T, res Result) Verdict {
	t.Helper()
	for _, v := range res.Verdicts {
		if v.Oracle == "structure-without-style" {
			return v
		}
	}
	t.Fatalf("no structure-without-style verdict in %s", Explain(res))
	return Verdict{}
}

// TestStructureFaithfulReportPasses is the faithful path: a report carrying
// every required section passes with a full score and no failure bundle.
func TestStructureFaithfulReportPasses(t *testing.T) {
	c := structCase()
	eng := ScriptedRunner{Label: "engine-faithful", Trace: Trace{Text: structFaithfulReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful report should pass; got %s", Explain(res))
	}
	if v := structVerdict(t, res); v.Score != 1 {
		t.Errorf("faithful report score = %v, want 1", v.Score)
	}
	if res.FailureBundle != nil {
		t.Fatalf("passing run must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestStructureRestyledReportStillPasses is the style-independence Witness for
// #4559: the same four sections reworded, reordered, and re-decorated (markdown,
// bold, inline label, ALL-CAPS) must still pass with a full score — section
// presence is required, prose style is not frozen.
func TestStructureRestyledReportStillPasses(t *testing.T) {
	c := structCase()
	eng := ScriptedRunner{Label: "engine-restyled", Trace: Trace{Text: structRestyledReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("restyled same-sections report must pass (style is not frozen); got %s", Explain(res))
	}
	if v := structVerdict(t, res); v.Score != 1 {
		t.Errorf("restyled report score = %v, want 1", v.Score)
	}
}

// TestStructureMissingSectionFails is the defect Witness: a report that dissolves
// its Risks section into prose — the word "risks" still appears mid-sentence —
// must fail, and the Detail must name exactly the missing section.
func TestStructureMissingSectionFails(t *testing.T) {
	c := structCase()
	text := "Summary:\n" +
		"Throughput increased 12% week over week, and the top risks were reviewed in standup.\n\n" +
		"Decisions:\n" +
		"We chose Postgres over DynamoDB for the ledger store.\n\n" +
		"Next actions:\n" +
		"Assign a migration owner and schedule the DBA review.\n"
	eng := ScriptedRunner{Label: "engine-dissolved-section", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("report missing its Risks section must not pass (prose mention is not a section); got %s", Explain(res))
	}
	v := structVerdict(t, res)
	if v.Pass {
		t.Fatal("structure-without-style verdict should have failed")
	}
	if v.Score != 0.75 {
		t.Errorf("score = %v, want 0.75 (3 of 4 sections present)", v.Score)
	}
	if want := `missing required section(s): "Risks"`; !strings.Contains(v.Detail, want) {
		t.Errorf("Detail %q missing %q", v.Detail, want)
	}
	for _, present := range []string{`"Summary"`, `"Decisions"`, `"Next actions"`} {
		if strings.Contains(v.Detail, present) {
			t.Errorf("Detail %q names present section %s as missing", v.Detail, present)
		}
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "structure-without-style" {
		t.Errorf("first failing oracle = %q, want structure-without-style", fb.FailingOracle)
	}
}

// TestStructureMinScoreToleratesBoundedMissing covers the explicit-threshold
// authoring variant: MinScore 0.75 admits one missing section (named as
// tolerated in the Detail), while the default threshold refuses it.
func TestStructureMinScoreToleratesBoundedMissing(t *testing.T) {
	c := structCase()
	c.Rubric.MinScore = 0.75
	text := "Summary:\nStrong week.\n\nDecisions:\nPostgres it is.\n\nNext actions:\nAssign the owner.\n"
	eng := ScriptedRunner{Label: "engine-three-sections", Trace: Trace{Text: text}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("score 0.75 >= MinScore 0.75 should pass; got %s", Explain(res))
	}
	if v := structVerdict(t, res); !strings.Contains(v.Detail, `tolerated missing section(s): "Risks"`) {
		t.Errorf("Detail %q should name the tolerated missing section", v.Detail)
	}
}

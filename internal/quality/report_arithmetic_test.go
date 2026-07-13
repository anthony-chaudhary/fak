package quality

import (
	"strings"
	"testing"
)

// arithCase builds a report-arithmetic case whose ground truth (prior, current,
// denominator) travels in Reference.Text as the canonical "key: value" block.
func arithCase(prior, current float64, denominator int) QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "report-arithmetic-demo",
		Version: 1,
		Prompt:  "Summarize this week's throughput for the executive rollup.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 32},
		Reference: Trace{
			Text: ArithmeticGroundTruth(prior, current, denominator),
		},
		Oracles: []string{"report-arithmetic"},
	}
}

// arithEngine replays a fixed report text as the engine path.
func arithEngine(text string) ScriptedRunner {
	return ScriptedRunner{Label: "engine-report", Trace: Trace{Text: text}}
}

// TestReportArithmeticFaithfulReportPasses is the happy path: a report whose
// percentage matches the prior->current delta, whose trend word agrees with the
// delta's sign, and whose "N of M" claim uses the period's denominator passes
// the oracle through the full spine with no failure bundle.
func TestReportArithmeticFaithfulReportPasses(t *testing.T) {
	c := arithCase(100, 112, 40) // +12% week over week, 40 reporting services
	eng := arithEngine("Throughput increased 12% week over week; 38 of 40 services reported.")
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful report should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
	}
	for _, v := range res.Verdicts {
		if v.Oracle == "report-arithmetic" && v.Score != 1 {
			t.Errorf("all claims consistent, want score 1, got %v (%s)", v.Score, v.Detail)
		}
	}
}

// TestReportArithmeticTrendContradictionFails is the arithmetic-layer mirror of
// the spine's increased-vs-decreased decode defect: the report claims
// "increased 12%" while ground truth shows a 12% DECREASE. The oracle must fail
// with a Detail carrying the bad claim and the expected sign/value vs stated.
func TestReportArithmeticTrendContradictionFails(t *testing.T) {
	c := arithCase(100, 88, 40) // ground truth: a 12% decrease
	eng := arithEngine("Throughput increased 12% week over week.")
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("trend contradiction must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "report-arithmetic" {
		t.Errorf("first failing oracle = %q, want report-arithmetic", fb.FailingOracle)
	}
	for _, want := range []string{`"increased 12%"`, "decreased", "-12"} {
		if !strings.Contains(fb.Detail, want) {
			t.Errorf("Detail %q should pinpoint the bad claim; missing %s", fb.Detail, want)
		}
	}
}

// TestReportArithmeticDefectsFail sweeps the remaining defect classes — wrong
// denominator, numerator over denominator, percent-vs-count unit mismatches,
// an overstated percentage, and a "flat" claim on a moved metric — each judged
// against the same +12% / denominator-40 ground truth. Every row must fail with
// a Detail naming the offending claim.
func TestReportArithmeticDefectsFail(t *testing.T) {
	rows := []struct {
		name   string
		report string
		want   []string // substrings the Detail must carry
	}{
		{
			// The consistent trend claim comes first; Detail must still pinpoint
			// the LATER bad denominator, proving first-BAD-claim localization.
			name:   "wrong denominator",
			report: "Throughput increased 12% week over week, but only 38 of 41 services reported.",
			want:   []string{`"38 of 41"`, "41", "40"},
		},
		{
			name:   "numerator exceeds denominator",
			report: "45 of 40 services reported this week.",
			want:   []string{`"45 of 40"`, "exceeds"},
		},
		{
			name:   "percent where a count belongs",
			report: "38% of 40 checks passed.",
			want:   []string{"unit mismatch", `"38% of 40"`},
		},
		{
			name:   "raw count where a percentage belongs",
			report: "Throughput increased 12 week over week.",
			want:   []string{"unit mismatch", "12"},
		},
		{
			name:   "overstated percentage",
			report: "Throughput increased 20% week over week.",
			want:   []string{`"increased 20%"`, "20", "12"},
		},
		{
			name:   "flat contradicts a moved metric",
			report: "Throughput was flat week over week.",
			want:   []string{`"flat"`, "increased"},
		},
	}
	c := arithCase(100, 112, 40)
	ref, err := ReferenceRunner{}.Run(c)
	if err != nil {
		t.Fatalf("reference runner: %v", err)
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			v := ReportArithmetic{}.Judge(ref, Trace{Text: row.report}, c)
			if v.Pass {
				t.Fatalf("defect report must fail: %q", row.report)
			}
			for _, want := range row.want {
				if !strings.Contains(v.Detail, want) {
					t.Errorf("Detail %q should name the bad claim; missing %s", v.Detail, want)
				}
			}
		})
	}
}

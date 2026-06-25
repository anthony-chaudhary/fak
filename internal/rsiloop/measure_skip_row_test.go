package rsiloop

// measure_skip_row_test.go binds issue #686's journal-honesty obligation: the cheaper
// truth-only rung (shipped in 634eb88) must produce an HONEST journal Row — it records
// suite_green=false (the suite was never run, not "passed") and truth_clean=true, and the
// keep-bit reflects the docs-only keep on truth ALONE. This proves the skipped rung does
// not over-claim a suite a downstream ledger reader would otherwise trust.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// TestCheapRungJournalsHonestRow proves the journal Row emitted via the cheap rung does
// not over-claim: suite_green is false (skipped, not passed), truth_clean is true, and
// the keep-bit is set from truth alone. The expensive Measure rung must not run at all —
// it fails the test if invoked — so the row is genuinely the product of the cheap rung.
func TestCheapRungJournalsHonestRow(t *testing.T) {
	h := Harness{
		MetricName:      "vdso_hit_rate",
		LowerBetter:     false,
		BaselineRefName: "test-ref",
		BaselineMetric:  func() (float64, string, error) { return 0.50, "deadbeef", nil },
		Candidates:      func() []Candidate { return []Candidate{{Label: "doc", Payload: []string{"README.md"}}} },
		Measure: func(Candidate) (Measurement, error) {
			t.Fatalf("the expensive Measure rung (worktree+suite) must not run for a docs-only candidate")
			return Measurement{}, nil
		},
		MeasureTruthOnly: func(Candidate) (Measurement, error) {
			return Measurement{Metric: 0.50, SuiteGreen: false, TruthClean: true}, nil
		},
		Classify: func(c Candidate) shipgate.EvidenceClass {
			ps, _ := c.Payload.([]string)
			return shipgate.ClassifyPaths(ps, func(p string) bool { return strings.HasSuffix(p, ".md") })
		},
	}
	res, err := Run(h, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res.Rows) == 0 {
		t.Fatalf("no rows produced")
	}
	row := res.Rows[0]
	if row.SuiteGreen {
		t.Fatalf("cheap rung must journal suite_green=false (the suite was skipped, not passed): %+v", row)
	}
	if !row.TruthClean {
		t.Fatalf("cheap rung must journal truth_clean=true: %+v", row)
	}
	if !row.Kept {
		t.Fatalf("docs-only via the cheap rung must be kept on truth alone: %+v", row)
	}
}

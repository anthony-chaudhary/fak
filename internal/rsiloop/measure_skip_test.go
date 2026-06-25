package rsiloop

// measure_skip_test.go is issue #686: when the harness has PROVEN an evidence class that
// needs neither a metric gain nor a green suite (Profile.NeedsCostlyEvidence()==false,
// e.g. ClassDocsOnly), the loop takes the cheaper truth-only rung — no worktree fork, no
// suite run — instead of the full Measure. The skip is gated on the harness-proven class
// and is opt-in: a nil MeasureTruthOnly seam always takes the full rung.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// skipHarness builds a one-candidate harness with BOTH rungs instrumented: full records
// that the expensive worktree+suite rung ran, cheap records the truth-only rung ran. The
// Classify seam proves the class from the candidate's touched paths (all .md =>
// ClassDocsOnly; any code path => ClassFull). cheapSeam toggles whether the cheaper rung
// is supplied at all.
func skipHarness(full, cheap *bool, paths []string, cheapSeam bool) Harness {
	h := Harness{
		MetricName:      "vdso_hit_rate",
		LowerBetter:     false,
		BaselineRefName: "test-ref",
		BaselineMetric:  func() (float64, string, error) { return 0.50, "deadbeef", nil },
		Candidates:      func() []Candidate { return []Candidate{{Label: "cand", Payload: paths}} },
		Measure: func(Candidate) (Measurement, error) {
			*full = true // the EXPENSIVE rung: a real impl forks a worktree and runs the suite
			return Measurement{Metric: 0.62, SuiteGreen: true, TruthClean: true}, nil
		},
		Classify: func(c Candidate) shipgate.EvidenceClass {
			ps, _ := c.Payload.([]string)
			return shipgate.ClassifyPaths(ps, func(p string) bool { return strings.HasSuffix(p, ".md") })
		},
	}
	if cheapSeam {
		h.MeasureTruthOnly = func(Candidate) (Measurement, error) {
			*cheap = true // the cheap rung: truth probe only — no worktree, no suite
			return Measurement{Metric: 0.50, SuiteGreen: false, TruthClean: true}, nil
		}
	}
	return h
}

// TestDocsOnlyTakesCheapRungSkippingWorktreeAndSuite: a proven docs-only candidate takes
// the truth-only rung — MeasureTruthOnly runs, the expensive Measure does NOT — and still
// KEEPs on truth-clean alone (the metric/suite the cheap rung never measured are
// irrelevant to ClassDocsOnly's keep rule).
func TestDocsOnlyTakesCheapRungSkippingWorktreeAndSuite(t *testing.T) {
	var full, cheap bool
	h := skipHarness(&full, &cheap, []string{"README.md", "docs/x.md"}, true)
	res, err := Run(h, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !cheap {
		t.Fatalf("docs-only must take the cheap truth-only rung (MeasureTruthOnly not called)")
	}
	if full {
		t.Fatalf("docs-only must SKIP the worktree+suite rung (the expensive Measure was called)")
	}
	if res.Kept != 1 || len(res.Rows) == 0 || !res.Rows[0].Kept {
		t.Fatalf("docs-only via the cheap rung must KEEP on truth-clean alone: %+v", res)
	}
}

// TestCostlyClassTakesFullRung: a candidate the harness CANNOT prove docs-only (a code
// path => ClassFull, which needs gain+suite) takes the full Measure rung even though a
// cheaper seam is supplied — the skip is gated on the proven class, not merely on the
// seam being present.
func TestCostlyClassTakesFullRung(t *testing.T) {
	var full, cheap bool
	h := skipHarness(&full, &cheap, []string{"internal/shipgate/shipgate.go"}, true)
	res, err := Run(h, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !full {
		t.Fatalf("a costly-evidence class (ClassFull) must take the full Measure rung")
	}
	if cheap {
		t.Fatalf("a costly-evidence class must NOT take the cheap rung")
	}
	if res.Kept != 1 || len(res.Rows) == 0 || !res.Rows[0].Kept {
		t.Fatalf("ClassFull with gain+green+clean must KEEP via the full rung: %+v", res)
	}
}

// TestNilCheapSeamAlwaysTakesFullRung: with no MeasureTruthOnly seam the loop always
// takes the full Measure rung, even for a proven docs-only candidate — the skip is opt-in
// and a harness that does not supply the cheaper probe is byte-for-byte unchanged.
func TestNilCheapSeamAlwaysTakesFullRung(t *testing.T) {
	var full, cheap bool
	h := skipHarness(&full, &cheap, []string{"README.md"}, false) // cheapSeam=false => MeasureTruthOnly nil
	if _, err := Run(h, nil, 3, 1); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !full {
		t.Fatalf("a nil cheap seam must take the full Measure rung even for docs-only")
	}
	if cheap {
		t.Fatalf("the cheap rung must not run when MeasureTruthOnly is nil")
	}
}

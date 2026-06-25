package rsiloop

// evidence_class_test.go closes the harness half of issue #680 criterion 7: the loop
// KEEPs a candidate the harness has PROVEN is docs-only on truth-clean ALONE, and
// REFUSES to apply a docs-only profile to a candidate it cannot prove is docs-only
// (falling back to ClassFull), and a nil Classify seam defaults to ClassFull (legacy).
// The class flows end-to-end: ClassifyPaths (the gate) -> Harness.Classify (the seam)
// -> Witness.Class -> shipgate.Evaluate (the graduated keep-bit).

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// isDocPath is the doc-path predicate for the classification tests (a real harness
// would supply its repo's doc convention; e.g. *.md / docs/**).
func isDocPath(p string) bool { return strings.HasSuffix(p, ".md") }

// classifyHarness builds a one-candidate harness whose Classify seam PROVES the
// candidate's class from its touched paths via shipgate.ClassifyPaths - the genuine
// harness gate. meas is the fixed witness the candidate earns; the candidate's
// Payload carries its touched paths.
func classifyHarness(meas Measurement, paths []string) Harness {
	return Harness{
		MetricName:      "vdso_hit_rate",
		LowerBetter:     false,
		BaselineRefName: "test-ref",
		BaselineMetric:  func() (float64, string, error) { return 0.50, "deadbeef", nil },
		Candidates: func() []Candidate {
			return []Candidate{{Label: "cand", Payload: paths}}
		},
		Measure: func(Candidate) (Measurement, error) { return meas, nil },
		Classify: func(c Candidate) shipgate.EvidenceClass {
			ps, _ := c.Payload.([]string)
			return shipgate.ClassifyPaths(ps, isDocPath)
		},
	}
}

// discriminating is the witness that separates the classes: a strict metric gain AND
// a clean truth syscall, but a RED suite. ClassDocsOnly KEEPs on it (truth-clean
// alone); ClassFull REVERTs on it (red suite). Same measurement, opposite outcome -
// so the ONLY variable is the class the harness proved.
func discriminating() Measurement {
	return Measurement{Metric: 0.62, SuiteGreen: false, TruthClean: true} // 0.62 > 0.50 baseline = strict gain
}

// criterion 7 (KEEP): a docs-only candidate the harness has PROVEN (every touched
// path is a doc) is KEPT on truth-clean ALONE - the red suite and the gain are
// irrelevant to its keep property - flowing through the real loop.
func TestHarnessKeepsProvenDocsOnlyOnTruthCleanAlone(t *testing.T) {
	h := classifyHarness(discriminating(), []string{"README.md", "docs/x.md"})
	res, err := Run(h, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Kept != 1 || len(res.Rows) == 0 || !res.Rows[0].Kept {
		t.Fatalf("proven docs-only candidate must KEEP on truth-clean alone: %+v", res)
	}
}

// criterion 7 (the refusal): the SAME witness, but the candidate touched a .go file
// so the harness CANNOT prove it docs-only. ClassifyPaths returns ClassFull; the loop
// requires all three signals; the red suite REVERTs it. The harness refuses to apply
// a docs-only profile to a candidate it cannot prove is docs-only.
func TestHarnessRefusesDocsOnlyForCodeCandidate(t *testing.T) {
	h := classifyHarness(discriminating(), []string{"README.md", "internal/shipgate/shipgate.go"})
	res, err := Run(h, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res.Rows) == 0 {
		t.Fatalf("no rows produced")
	}
	if res.Kept != 0 || res.Rows[0].Kept {
		t.Fatalf("unprovable docs-only (code path) must fall back to ClassFull and REVERT: %+v", res)
	}
	if res.Rows[0].Decision != "REVERT" {
		t.Fatalf("code candidate with red suite must REVERT at ClassFull, got %s", res.Rows[0].Decision)
	}
}

// criterion 7 (the default): a nil Classify seam leaves every candidate at the full
// rung - the loop cannot prove a narrower class, so it pins ClassFull. Legacy
// all-three behavior: all-three-good KEEPs, a red suite REVERTs (no downgrade).
func TestNilClassifyDefaultsToClassFull(t *testing.T) {
	allGood := Measurement{Metric: 0.62, SuiteGreen: true, TruthClean: true}
	h := classifyHarness(allGood, []string{"README.md"})
	h.Classify = nil
	res, err := Run(h, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Kept != 1 || len(res.Rows) == 0 || !res.Rows[0].Kept {
		t.Fatalf("nil Classify + all-three-good must KEEP at ClassFull: %+v", res)
	}

	h2 := classifyHarness(discriminating(), []string{"README.md"})
	h2.Classify = nil
	res2, err := Run(h2, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res2.Kept != 0 || len(res2.Rows) == 0 || res2.Rows[0].Kept {
		t.Fatalf("nil Classify + red suite must REVERT at ClassFull (no downgrade): %+v", res2)
	}
}

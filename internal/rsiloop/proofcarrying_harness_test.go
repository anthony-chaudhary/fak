package rsiloop

// proofcarrying_harness_test.go binds the harness half of issue #685 for the
// ClassProofCarrying rung. The Classify seam (shipped in 3a8c388) sets Witness.Class
// from a harness-PROVEN class BEFORE shipgate.Evaluate, never the candidate's say-so. The
// existing evidence_class_test.go exercises only the ClassDocsOnly and ClassFull rungs;
// this adds the untested ClassProofCarrying rung end-to-end through Run: a candidate the
// harness proves is proof-carrying KEEPs on a strict gain + truth-clean ALONE (the red
// suite is irrelevant), and an unprovable narrowing always falls up to ClassFull.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// proofCarryingHarness builds a one-candidate harness whose Classify seam returns the
// fixed harness-proven class. meas is the witness the candidate earns from Measure.
func proofCarryingHarness(meas Measurement, class shipgate.EvidenceClass) Harness {
	return Harness{
		MetricName:      "vdso_hit_rate",
		LowerBetter:     false,
		BaselineRefName: "test-ref",
		BaselineMetric:  func() (float64, string, error) { return 0.50, "deadbeef", nil },
		Candidates:      func() []Candidate { return []Candidate{{Label: "cand"}} },
		Measure:         func(Candidate) (Measurement, error) { return meas, nil },
		Classify:        func(Candidate) shipgate.EvidenceClass { return class },
	}
}

// TestHarnessKeepsProvenProofCarryingOnGainAndTruth: a candidate the harness PROVES is
// proof-carrying KEEPs on a strict metric gain + a clean truth syscall even with a RED
// suite — the suite is irrelevant to its keep property — flowing through the real loop.
// The SAME measurement at ClassFull REVERTs (the full rung needs a green suite), so the
// only variable that flipped the outcome is the harness-proven class, not the candidate.
func TestHarnessKeepsProvenProofCarryingOnGainAndTruth(t *testing.T) {
	// 0.62 > 0.50 baseline = strict gain; truth clean; suite RED.
	meas := Measurement{Metric: 0.62, SuiteGreen: false, TruthClean: true}

	res, err := Run(proofCarryingHarness(meas, shipgate.ClassProofCarrying), nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Kept != 1 || len(res.Rows) == 0 || !res.Rows[0].Kept {
		t.Fatalf("proven proof-carrying candidate must KEEP on gain+truth with a red suite: %+v", res)
	}

	resFull, err := Run(proofCarryingHarness(meas, shipgate.ClassFull), nil, 3, 1)
	if err != nil {
		t.Fatalf("Run (full): %v", err)
	}
	if resFull.Kept != 0 || len(resFull.Rows) == 0 || resFull.Rows[0].Kept {
		t.Fatalf("same witness at ClassFull must REVERT (red suite): %+v", resFull)
	}
}

// TestHarnessProofCarryingRevertsWithoutGain: the proof-carrying rung still REQUIRES the
// gain — truth-clean alone does not carry it (that is ClassDocsOnly's rung). With no
// strict gain the candidate REVERTs even at ClassProofCarrying, proving the class drops
// only the suite, never the proof (the metric gain) itself.
func TestHarnessProofCarryingRevertsWithoutGain(t *testing.T) {
	// 0.50 == 0.50 baseline = NO strict gain; truth clean; suite green.
	meas := Measurement{Metric: 0.50, SuiteGreen: true, TruthClean: true}

	res, err := Run(proofCarryingHarness(meas, shipgate.ClassProofCarrying), nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Kept != 0 || len(res.Rows) == 0 || res.Rows[0].Kept {
		t.Fatalf("proof-carrying with NO gain must REVERT (the gain is required): %+v", res)
	}
}

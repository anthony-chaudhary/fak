package shipgate

// class_docsonly_test.go is issue #682's named acceptance test. The {needTruth}-only
// profile for ClassDocsOnly shipped with the graduated keep-bit; this proves the
// behavior the issue names: a docs-only candidate KEEPs on the truth-clean signal
// ALONE, with the suite-green and metric-gain signals proven IRRELEVANT to the
// decision (not merely incidentally true in one example).

import "testing"

// TestEvaluateClassDocsOnlyTruthCleanSufficient proves truth-clean is SUFFICIENT for a
// ClassDocsOnly keep: across every combination of the two signals the profile does not
// require (suite-green and metric-gain), the keep-bit tracks TruthClean exactly. When
// truth is clean the candidate KEEPs even with a red suite and no gain; when truth is
// dirty it REVERTs even with a green suite and a strict gain. The suite and gain bits
// never move the decision, so neither is consulted for this class.
func TestEvaluateClassDocsOnlyTruthCleanSufficient(t *testing.T) {
	for _, suite := range []bool{false, true} {
		for _, gain := range []bool{false, true} {
			// LowerBetter=false, so After>Before is a strict gain.
			before, after := 1.0, 1.0
			if gain {
				after = 2.0
			}
			for _, truth := range []bool{false, true} {
				w := Witness{
					Class:      ClassDocsOnly,
					Before:     before,
					After:      after,
					SuiteGreen: suite,
					TruthClean: truth,
				}
				d, out := Evaluate(w)
				// The keep-bit must equal truth-clean ALONE — independent of suite/gain.
				if out.Kept() != truth {
					t.Fatalf("docs-only keep-bit=%v but truth-clean=%v (suite=%v gain=%v); truth must be sufficient and sole",
						out.Kept(), truth, suite, gain)
				}
				wantDecision := REVERT
				if truth {
					wantDecision = KEEP
				}
				if d != wantDecision {
					t.Fatalf("docs-only decision=%v want %v (suite=%v gain=%v truth=%v)", d, wantDecision, suite, gain, truth)
				}
			}
		}
	}
}

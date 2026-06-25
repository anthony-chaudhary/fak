package shipgate

// class_proofcarrying_test.go is issue #683's named acceptance test. The
// {needGain, needTruth} profile for ClassProofCarrying shipped with the graduated
// keep-bit; this proves the behavior the issue names: a proof-carrying candidate keeps
// on a strict metric gain PLUS a clean truth syscall while the suite-green signal is
// proven IRRELEVANT — the candidate carries its own proof (the gain), so the full
// suite run is skipped from the keep rule, but truth-clean is still required and a
// missing gain still reverts.

import "testing"

// TestEvaluateClassProofCarryingSkipsSuite proves the suite-green signal is SKIPPED for
// ClassProofCarrying while gain and truth remain mandatory. Across both suite values
// the keep-bit equals (gain AND truth), so a green and a red suite produce the
// identical decision for the same gain/truth — the suite is never consulted. Dropping
// either required signal (no gain, or dirty truth) reverts regardless of the suite.
func TestEvaluateClassProofCarryingSkipsSuite(t *testing.T) {
	for _, gain := range []bool{false, true} {
		for _, truth := range []bool{false, true} {
			before, after := 1.0, 1.0 // LowerBetter=false => After>Before is the strict gain
			if gain {
				after = 2.0
			}
			wantKeep := gain && truth
			var redDecision, greenDecision Decision
			var redKept, greenKept bool
			for _, suite := range []bool{false, true} {
				w := Witness{
					Class:      ClassProofCarrying,
					Before:     before,
					After:      after,
					SuiteGreen: suite,
					TruthClean: truth,
				}
				d, out := Evaluate(w)
				if out.Kept() != wantKeep {
					t.Fatalf("proof-carrying keep-bit=%v want (gain&&truth)=%v (suite=%v gain=%v truth=%v)",
						out.Kept(), wantKeep, suite, gain, truth)
				}
				if suite {
					greenDecision, greenKept = d, out.Kept()
				} else {
					redDecision, redKept = d, out.Kept()
				}
			}
			// The suite must not move the decision: red and green agree for fixed gain/truth.
			if redDecision != greenDecision || redKept != greenKept {
				t.Fatalf("proof-carrying suite changed the outcome (gain=%v truth=%v): red=(%v,%v) green=(%v,%v); suite must be irrelevant",
					gain, truth, redDecision, redKept, greenDecision, greenKept)
			}
		}
	}
}

package main

import "testing"

// TestSelfcheckReproducesSafetyInvariants pins the browser demo's own data path:
// runSelfcheck replays every shipped scenario through the kernel (the same
// turnbench.RunWithCalls the browser drives down both columns) and must reproduce
// the documented safety-floor invariants (exit 0) — WITHOUT fak breaches the
// expected count, WITH fak breaches zero, on every scenario. This is the browserless
// guard the -selfcheck flag exposes to operators, now also gating CI and
// cross-platform (mac/arm64) runs so a regression fails here, not in a demo nobody reran.
//
// turnTaxDir() resolves the fixtures via its "../../testdata/turntax" candidate when
// the test runs with CWD = cmd/guarddemo, so no working-dir juggling is needed.
func TestSelfcheckReproducesSafetyInvariants(t *testing.T) {
	if code := runSelfcheck(); code != 0 {
		t.Fatalf("runSelfcheck() = %d, want 0 (a shipped guarddemo scenario drifted from its documented safety-floor invariants)", code)
	}
}

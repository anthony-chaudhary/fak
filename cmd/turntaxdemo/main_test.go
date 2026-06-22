package main

import "testing"

// TestSelfcheckReproducesDocumentedInvariants pins the browser demo's own data
// path: runSelfcheck replays every shipped suite through the kernel (the same
// turnbench.RunWithCalls the browser drives) and must reproduce the documented
// turn-tax + safety-floor invariants (exit 0). This is the browserless guard the
// -selfcheck flag exposes to operators, now also gating CI and cross-platform
// (mac/arm64) runs so a regression fails here instead of in a demo nobody reran.
//
// turnTaxDir() resolves the fixtures via its "../../testdata/turntax" candidate
// when the test runs with CWD = cmd/turntaxdemo, so no working-dir juggling is
// needed.
func TestSelfcheckReproducesDocumentedInvariants(t *testing.T) {
	if code := runSelfcheck(); code != 0 {
		t.Fatalf("runSelfcheck() = %d, want 0 (a shipped turntax suite drifted from its documented invariants)", code)
	}
}

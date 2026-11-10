package main

import "testing"

// TestSelfcheck runs the same headless witness the -selfcheck flag runs: the
// shared PII-bearing scenario replayed against BOTH arms over the real gateway
// HTTP seams (two in-process gateway servers on ephemeral loopback listeners),
// asserting the documented #14327 invariants — raw pass-through forwards
// byte-identical with zero admissions and no note; the armed floor masks every
// protected span, renders the in-band [fak] note, and admits only the masked
// bytes. One PASS line per assertion; any drift fails the test.
//
// fak-test:runtime fast est=15s
func TestSelfcheck(t *testing.T) {
	lines, failures := runSelfcheckCore()
	for _, l := range lines {
		t.Log(l)
	}
	if failures > 0 {
		t.Fatalf("%d/%d selfcheck assertion(s) failed", failures, len(lines))
	}
}

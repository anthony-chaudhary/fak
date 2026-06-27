package demoui

import "fmt"

// SelfcheckChecker accumulates the per-suite invariant mismatches a demo's
// browserless -selfcheck collects while replaying each fixture through the
// kernel. Every on-box demo (cmd/guarddemo, cmd/turntaxdemo, cmd/tokendemo)
// builds the same fresh checker per fixture, compares a handful of integer
// counters against the documented expectations, and prints/exits on any
// mismatch — so the collector lives here once instead of copy-pasted into each
// main.
//
// The zero value is ready to use:
//
//	var c demoui.SelfcheckChecker
//	c.Check("denies", got, want)
//	c.Notef("ctx_with>ctx_without")
//	if c.Failed() { ... c.Mismatches() ... }
type SelfcheckChecker struct {
	miss []string
}

// Check records a mismatch when got != want, formatted identically to the
// hand-rolled closure the demos previously each defined.
func (c *SelfcheckChecker) Check(name string, got, want int) {
	if got != want {
		c.miss = append(c.miss, fmt.Sprintf("%s=%d(want %d)", name, got, want))
	}
}

// Note records a pre-formatted mismatch message for the ad-hoc invariants a
// demo asserts outside the integer-counter grammar (a string-valued field, a
// cross-meter bound).
func (c *SelfcheckChecker) Note(msg string) {
	c.miss = append(c.miss, msg)
}

// Notef is Note with fmt.Sprintf formatting.
func (c *SelfcheckChecker) Notef(format string, args ...any) {
	c.miss = append(c.miss, fmt.Sprintf(format, args...))
}

// Failed reports whether any mismatch was recorded.
func (c *SelfcheckChecker) Failed() bool { return len(c.miss) > 0 }

// Mismatches returns the recorded mismatch messages in record order.
func (c *SelfcheckChecker) Mismatches() []string { return c.miss }

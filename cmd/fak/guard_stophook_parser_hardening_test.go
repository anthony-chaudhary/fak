package main

import (
	"strings"
	"testing"
)

// TestParseGuardStopHookSignals_UnrelatedSpacedLabelDoesNotFailOpen is the regression for the
// latent silent-governor-disable: the guard Stop hook scans the gateway's WHOLE /metrics scrape,
// and a label value with a space (promQuote escapes \, ", newline — not spaces) splits under
// strings.Fields so a non-numeric token lands mid-line. The old parser ran ParseFloat on every
// line's second field and hard-failed the entire scan on such a line, which fails the caller OPEN
// and silently disables the deny-all auto-continue / bounded give-up governor. The hardened parser
// only reads a value for the four gauges it consumes, so an unrelated spaced-label series is inert.
func TestParseGuardStopHookSignals_UnrelatedSpacedLabelDoesNotFailOpen(t *testing.T) {
	metrics := strings.Join([]string{
		`# HELP fak_gateway_build_info Static fak gateway build and runtime labels.`,
		`# TYPE fak_gateway_build_info gauge`,
		// A label value carrying a space — the exact shape that split the old parser mid-line.
		`fak_gateway_build_info{version="v0.37.0",engine="llama.cpp metal",model="claude opus 4.8"} 1`,
		`fak_gateway_operations_total{operation="syscall",reason="floor refused all"} 3`,
		`fak_guard_deny_all_consecutive 4`,
		`fak_guard_deny_all_same_consecutive 2`,
		`fak_guard_tool_feedback_consecutive 0`,
	}, "\n")
	signals, err := parseGuardStopHookSignals(metrics)
	if err != nil {
		t.Fatalf("hardened parser must not fail open on an unrelated spaced-label line: %v", err)
	}
	if signals.DenyAllConsecutive != 4 {
		t.Fatalf("DenyAllConsecutive = %d, want 4 (governor must still read its gauge past the spaced line)", signals.DenyAllConsecutive)
	}
	if signals.DenyAllSameConsecutive != 2 || !signals.DenyAllSameConsecutiveSeen {
		t.Fatalf("same-consecutive = %d seen=%v, want 2/true", signals.DenyAllSameConsecutive, signals.DenyAllSameConsecutiveSeen)
	}
}

// TestParseGuardStopHookSignals_LabeledOwnGaugeStillMatches proves the de-labeling: if a future
// gateway emits one of the hook's own gauges WITH a label set, the exact-string match the old
// parser used would miss it (leaving DenyAllConsecutive unset → "metric not found" → fail open).
// The hardened parser strips the label set before matching and reads the trailing value.
func TestParseGuardStopHookSignals_LabeledOwnGaugeStillMatches(t *testing.T) {
	signals, err := parseGuardStopHookSignals(strings.Join([]string{
		`fak_guard_deny_all_consecutive{scope="session"} 7`,
		`fak_guard_deny_all_same_consecutive{scope="session"} 3`,
	}, "\n"))
	if err != nil {
		t.Fatalf("labeled own gauge must still parse: %v", err)
	}
	if signals.DenyAllConsecutive != 7 {
		t.Fatalf("DenyAllConsecutive = %d, want 7 from a labeled emission", signals.DenyAllConsecutive)
	}
	if signals.DenyAllSameConsecutive != 3 || !signals.DenyAllSameConsecutiveSeen {
		t.Fatalf("same-consecutive = %d seen=%v, want 3/true from a labeled emission", signals.DenyAllSameConsecutive, signals.DenyAllSameConsecutiveSeen)
	}
}

// TestParseGuardStopHookSignals_MissingGaugeStillFailsOpen pins the preserved contract: when the
// required deny-all gauge is ABSENT the parser must still error, so the caller fails OPEN (allows
// the stop) rather than silently treating a missing gauge as 0 (which would never auto-continue).
// The hardened parser keeps foundDenyAll and the trailing not-found error, so a scrape with only
// OTHER (even labeled/spaced) series errors exactly as before.
func TestParseGuardStopHookSignals_MissingGaugeStillFailsOpen(t *testing.T) {
	metrics := strings.Join([]string{
		`fak_guard_deny_all_stops_total 5`,
		`fak_gateway_build_info{model="claude opus 4.8"} 1`,
	}, "\n")
	if _, err := parseGuardStopHookSignals(metrics); err == nil {
		t.Fatal("missing deny-all gauge must error so the hook fails open, not silently treats 0")
	}
}

// TestParseGuardStopHookSignals_OwnGaugeBadValueStillErrors confirms the hardening did not swallow
// a genuinely malformed value on one of the hook's OWN gauges: that must still error (fail open),
// unlike an unrelated line, which is now inert.
func TestParseGuardStopHookSignals_OwnGaugeBadValueStillErrors(t *testing.T) {
	if _, err := parseGuardStopHookSignals("fak_guard_deny_all_consecutive not_a_number\n"); err == nil {
		t.Fatal("a malformed value on our own gauge must still error (fail open)")
	}
}

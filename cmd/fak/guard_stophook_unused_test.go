package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestParseFakVerbCalls asserts the guard-stophook metrics parser reads the
// fak_mcp_verb_calls_total counter and distinguishes "reported 0" from "absent".
func TestParseFakVerbCalls(t *testing.T) {
	withCounter := "fak_guard_deny_all_consecutive 0\nfak_mcp_verb_calls_total 7\n"
	sig, err := parseGuardStopHookSignals(withCounter)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !sig.FakVerbCallsSeen || sig.FakVerbCalls != 7 {
		t.Fatalf("want seen=true calls=7, got seen=%v calls=%d", sig.FakVerbCallsSeen, sig.FakVerbCalls)
	}

	// Counter absent (older gateway): seen must be false so the advisory stays silent.
	absent := "fak_guard_deny_all_consecutive 0\n"
	sig2, err := parseGuardStopHookSignals(absent)
	if err != nil {
		t.Fatalf("parse absent: %v", err)
	}
	if sig2.FakVerbCallsSeen {
		t.Fatalf("want seen=false when counter absent, got seen=true")
	}
}

// TestUnusedSubstrateAdvisory_FiresOnZeroVerbClean asserts the #3093 advisory fires at a
// clean stop when the counter is present and 0, and stays silent otherwise.
func TestUnusedSubstrateAdvisory_FiresOnZeroVerbClean(t *testing.T) {
	// The env knob defaults to shadow (advise) when unset — assert that default here.
	t.Setenv(guardStopHookUnusedEnvMode, "")

	cases := []struct {
		name      string
		signals   guardStopHookSignals
		wantFires bool
	}{
		{"zero-verbs-seen", guardStopHookSignals{FakVerbCalls: 0, FakVerbCallsSeen: true}, true},
		{"verbs-used", guardStopHookSignals{FakVerbCalls: 4, FakVerbCallsSeen: true}, false},
		{"counter-absent", guardStopHookSignals{FakVerbCalls: 0, FakVerbCallsSeen: false}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var b bytes.Buffer
			emitUnusedSubstrateAdvisory(&b, tc.signals)
			fired := strings.Contains(b.String(), "ZERO fak verbs")
			if fired != tc.wantFires {
				t.Fatalf("%s: fired=%v want=%v (out=%q)", tc.name, fired, tc.wantFires, b.String())
			}
		})
	}
}

// TestUnusedSubstrateAdvisory_OffSuppresses asserts the off knob silences the advisory even
// on the zero-verb clean path.
func TestUnusedSubstrateAdvisory_OffSuppresses(t *testing.T) {
	t.Setenv(guardStopHookUnusedEnvMode, "off")
	var b bytes.Buffer
	emitUnusedSubstrateAdvisory(&b, guardStopHookSignals{FakVerbCalls: 0, FakVerbCallsSeen: true})
	if b.Len() != 0 {
		t.Fatalf("off mode should suppress advisory, got: %q", b.String())
	}
}

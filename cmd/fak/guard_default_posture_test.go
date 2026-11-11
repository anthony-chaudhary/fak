package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// The shipped embedded guard floor is explicit default_open: parsing it must
// yield the open posture in both the adjudicator policy and the runtime
// context, so a strict operator has to say fail_closed to get the v0.1 floor.
func TestEmbeddedGuardFloorParsesDefaultOpen(t *testing.T) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("ParseRuntime(guardDefaultPolicyJSON): %v", err)
	}
	if rt.Adjudicator.Posture != adjudicator.PostureDefaultOpen {
		t.Fatalf("embedded floor Posture = %v, want PostureDefaultOpen", rt.Adjudicator.Posture)
	}
	if rt.PolicyContext.Posture.String() != "default_open" {
		t.Fatalf("embedded floor PolicyContext.Posture = %v, want default_open", rt.PolicyContext.Posture)
	}
}

// Parity invariant claimed by internal/adjudicator/default_policy.go: every
// lowercase built-in the programmatic DefaultPolicy admits is admitted by the
// shipped on-disk floor too, so the two floors cannot drift on a native-loop
// tool name.
func TestEmbeddedGuardFloorCoversDefaultPolicyLowercaseAllow(t *testing.T) {
	m, err := policy.ParseManifest(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("ParseManifest(guardDefaultPolicyJSON): %v", err)
	}
	onDisk := map[string]bool{}
	for _, name := range m.Allow {
		onDisk[name] = true
	}
	var missing []string
	for name := range adjudicator.DefaultPolicy().Allow {
		if name == strings.ToLower(name) && !onDisk[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("lowercase DefaultPolicy allow entries missing from the embedded guard floor: %v", missing)
	}
}

// Omitted chat mode is the default-open developer floor; fail_closed and its
// strict alias stay the explicit operator lever.
func TestParseChatModeDefaults(t *testing.T) {
	cases := []struct {
		in   string
		want adjudicator.Posture
	}{
		{"", adjudicator.PostureDefaultOpen},
		{"default_open", adjudicator.PostureDefaultOpen},
		{"fail_closed", adjudicator.PostureFailClosed},
		{"strict", adjudicator.PostureFailClosed},
		{"admit_and_log", adjudicator.PostureAdmitAndLog},
	}
	for _, tc := range cases {
		if got := parseChatMode(tc.in); got != tc.want {
			t.Fatalf("parseChatMode(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

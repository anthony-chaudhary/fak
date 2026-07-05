package main

import (
	"strings"
	"testing"
)

// guardInfoManagedCacheText is the live-line rendering of the #2190 managed-cache posture.
// The contract mirrors the gateway block: passive/cold omits the clause, an ACTIVE-but-inert
// session reads as a visible zero (with its dominant refusal reason), and a paying session
// reports the upgrade count.
func TestGuardInfoManagedCacheText(t *testing.T) {
	cases := []struct {
		name    string
		mc      *guardInfoManagedCache
		want    string // "" means: expect empty
		wantSub []string
	}{
		{name: "nil block is silent", mc: nil, want: ""},
		{name: "off posture named", mc: &guardInfoManagedCache{Active: false}, want: "managed cache off"},
		{
			name:    "active paying reports count",
			mc:      &guardInfoManagedCache{Active: true, Upgraded: 4},
			wantSub: []string{"ACTIVE", "4"},
		},
		{
			name:    "active but inert names dominant reason",
			mc:      &guardInfoManagedCache{Active: true, Inert: true, Reasons: map[string]uint64{"no_stable_breakpoint": 9, "volatile_head": 2}},
			wantSub: []string{"ACTIVE but inert", "0 upgrades", "no_stable_breakpoint"},
		},
		{
			name:    "active inert without reasons still visible",
			mc:      &guardInfoManagedCache{Active: true, Inert: true},
			wantSub: []string{"ACTIVE but inert", "0 upgrades"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardInfoManagedCacheText(guardInfoVars{ManagedCache: tc.mc})
			if tc.want != "" && got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
			if tc.want == "" && len(tc.wantSub) == 0 && got != "" {
				t.Fatalf("expected empty clause, got %q", got)
			}
			for _, sub := range tc.wantSub {
				if !strings.Contains(got, sub) {
					t.Fatalf("clause %q missing %q", got, sub)
				}
			}
		})
	}
}

// topTTLUpgradeReason must be deterministic despite Go's randomized map iteration: the
// most-frequent reason wins, ties broken by name.
func TestTopTTLUpgradeReason(t *testing.T) {
	if got := topTTLUpgradeReason(nil); got != "" {
		t.Fatalf("empty map: got %q, want \"\"", got)
	}
	got := topTTLUpgradeReason(map[string]uint64{"volatile_head": 3, "no_stable_breakpoint": 3, "ambiguous": 1})
	if got != "no_stable_breakpoint" {
		t.Fatalf("tie should break by name to no_stable_breakpoint, got %q", got)
	}
	if got := topTTLUpgradeReason(map[string]uint64{"a": 1, "b": 7}); got != "b" {
		t.Fatalf("max should win, got %q", got)
	}
}

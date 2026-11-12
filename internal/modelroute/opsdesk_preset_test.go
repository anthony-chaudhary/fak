package modelroute

import (
	"encoding/json"
	"os"
	"testing"
)

// TestOpsDeskPresetTombstoned is the tombstone witness for the retired
// ops-desk routing preset and its sidecar roster (operator decision
// 2026-11-08: the hive-ai provider is retired everywhere). The preset's whole
// shape was hive-hosted same-provider rotation — one OpenAI-compatible wire
// (hive-ai) carrying BOTH hosted-flash seats (glm-5.3-flash and
// deepseek-v4.1-flash) so each could rotate onto the other — so it is
// tombstoned rather than repointed to other providers, and the sidecar roster
// (examples/model-accounts.hosted-hive.json) that resolved those seats onto
// the hive-ai account is tombstoned with it. Both files must carry the stub
// contract: a tombstoned date, the retirement reason, and a pointer to the
// examples index row. The semantic witnesses those files used to carry
// (TestOpsDeskPresetDecisions, TestOpsDeskPresetAlternationIsTierShaped,
// TestOpsDeskHostedHiveRosterResolvesBothSeats,
// TestOpsDeskPresetWorkerElectedSeatsAreThemselves) retired with them. The
// provider-neutral rotation primitive itself (rotation.go:
// NextSeat/RotationPeers/ValidateRotationGroups) is untouched and stays
// witnessed by TestOpsDeskRotationPoolNextSeat and
// TestOpsDeskRotationGroupsValidateFailClosed.
func TestOpsDeskPresetTombstoned(t *testing.T) {
	for _, path := range []string{
		"../../examples/routing-presets/ops-desk.json",
		"../../examples/model-accounts.hosted-hive.json",
	} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		var doc map[string]any
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatalf("%s must stay valid JSON: %v", path, err)
		}
		if got, want := doc["tombstoned"], "2026-11-08"; got != want {
			t.Errorf("%s tombstoned = %v, want %q", path, got, want)
		}
		if reason, _ := doc["reason"].(string); reason == "" {
			t.Errorf("%s carries no tombstone reason", path)
		}
		if see, _ := doc["see"].(string); see != "examples/README.md row" {
			t.Errorf("%s see = %v, want the examples index row pointer", path, see)
		}
	}
}

// TestOpsDeskRotationPoolNextSeat witnesses the GENERAL same-class rotation
// primitive (rotation.go) the desk was built on: RotationPeers returns the
// same-class alternates declared in a pool and NextSeat is a pure cyclic step
// across the pool — the seam dispatch wiring applies per call to spread load
// or step off an exhausted/degraded seat. A seat NOT declared in any pool
// (e.g. the unavailable gpt-6-astra) must not rotate anywhere. The primitive
// is provider-neutral code and survives the 2026-11-08 hive-ai tombstone; the
// shipped preset that exercised it is retired (TestOpsDeskPresetTombstoned).
func TestOpsDeskRotationPoolNextSeat(t *testing.T) {
	if err := ValidateRotationGroups(DeskRotationGroups()); err != nil {
		t.Fatalf("declared rotation pools invalid: %v", err)
	}

	class, ok := RotationClass("glm-5.3-flash")
	if !ok || class != "hosted-flash" {
		t.Fatalf("RotationClass(glm-5.3-flash) = %q, %v; want hosted-flash", class, ok)
	}
	peers, ok := RotationPeers("glm-5.3-flash")
	if !ok || len(peers) != 1 || peers[0] != "deepseek-v4.1-flash" {
		t.Fatalf("RotationPeers(glm-5.3-flash) = %v, %v; want [deepseek-v4.1-flash]", peers, ok)
	}

	// The pure cyclic step: glm53 -> deepseek -> glm53 (pool wraps). No state,
	// so two calls from the same seat agree - rotation never disturbs the
	// deterministic Route spine.
	for _, cycle := range []struct{ from, want string }{
		{"glm-5.3-flash", "deepseek-v4.1-flash"},
		{"deepseek-v4.1-flash", "glm-5.3-flash"},
	} {
		next, class, ok := NextSeat(cycle.from)
		if !ok || next != cycle.want || class != "hosted-flash" {
			t.Errorf("NextSeat(%q) = %q, %q, %v; want %q in hosted-flash", cycle.from, next, class, ok, cycle.want)
		}
	}

	// Peers is a copy: mutating it must not leak into the declared pools.
	peers[0] = "mutated"
	fresh, _ := RotationPeers("glm-5.3-flash")
	if fresh[0] != "deepseek-v4.1-flash" {
		t.Fatalf("RotationPeers leaked mutable state: %v", fresh)
	}

	// An undeclared seat is not a rotation candidate: the unavailable astra
	// seat must fail closed rather than rotate onto anything.
	if _, _, ok := NextSeat("gpt-6-astra"); ok {
		t.Error("NextSeat(gpt-6-astra) returned ok; an undeclared seat must not rotate")
	}
	if _, ok := RotationPeers("gpt-6-astra"); ok {
		t.Error("RotationPeers(gpt-6-astra) returned ok; an undeclared seat has no peers")
	}
}

// TestOpsDeskRotationGroupsValidateFailClosed proves the rotation validator is
// a fail-loud boundary: duplicate classes, single-seat pools, ambiguous
// membership, and non-token seat ids each refuse rather than silently
// mis-rotate.
func TestOpsDeskRotationGroupsValidateFailClosed(t *testing.T) {
	cases := []struct {
		name   string
		groups []RotationGroup
	}{
		{
			name:   "duplicate class",
			groups: []RotationGroup{{Class: "x", Seats: []string{"a", "b"}}, {Class: "x", Seats: []string{"c", "d"}}},
		},
		{
			name:   "single-seat pool",
			groups: []RotationGroup{{Class: "solo", Seats: []string{"a"}}},
		},
		{
			name:   "seat in two classes",
			groups: []RotationGroup{{Class: "x", Seats: []string{"a", "b"}}, {Class: "y", Seats: []string{"b", "c"}}},
		},
		{
			name:   "seat with route delimiter",
			groups: []RotationGroup{{Class: "x", Seats: []string{"openai:gpt", "b"}}},
		},
		{
			name:   "empty class name",
			groups: []RotationGroup{{Class: "", Seats: []string{"a", "b"}}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateRotationGroups(tc.groups); err == nil {
				t.Fatalf("ValidateRotationGroups(%v) = nil; want a fail-loud refusal", tc.groups)
			}
		})
	}
}

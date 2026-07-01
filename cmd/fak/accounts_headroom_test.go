package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func hrBool(b bool) *bool    { return &b }
func hrStr(s string) *string { return &s }

// TestHeadroomFromRoster pins the coarse offerability tiering that turns the live runtime
// roster into the config-plane rotation signal: offerable -> +1, walled/throttled -> -1,
// unknown -> 0, keyed by the account bucket key; non-claude and identity-less rows are ignored.
func TestHeadroomFromRoster(t *testing.T) {
	rows := []fleetaccounts.Account{
		// Offerable claude worker -> room (+1).
		{Product: "claude", AccountUUID: hrStr("u-day30"), Available: hrBool(true), Blocked: hrBool(false)},
		// Blocked claude worker -> walled (-1).
		{Product: "claude", AccountUUID: hrStr("u-gem7"), Available: hrBool(false), Blocked: hrBool(true)},
		// Usage-throttled claude worker -> walled (-1), even if a stale Available lingered true.
		{Product: "claude", AccountUUID: hrStr("u-cap"), Available: hrBool(true), Throttled: hrBool(true)},
		// No runtime availability signal -> unknown (0).
		{Product: "claude", AccountUUID: hrStr("u-unknown")},
		// Non-claude row -> ignored (config-plane rotation is over Claude seats).
		{Product: "opencode", AccountUUID: hrStr("u-glm"), Available: hrBool(true)},
		// No resolved identity -> ignored (nothing to key on).
		{Product: "claude", Available: hrBool(true)},
	}

	hr := headroomFromRoster(rows)
	want := map[string]float64{
		"uuid:u-day30":   1,
		"uuid:u-gem7":    -1,
		"uuid:u-cap":     -1,
		"uuid:u-unknown": 0,
	}
	if len(hr) != len(want) {
		t.Fatalf("headroom map = %v, want %d entries", hr, len(want))
	}
	for k, v := range want {
		if got := hr[k]; got != v {
			t.Fatalf("headroom[%q] = %v, want %v", k, got, v)
		}
	}
	if _, ok := hr["uuid:u-glm"]; ok {
		t.Fatal("opencode bucket must not appear in the claude rotation headroom")
	}
}

// TestHeadroomFromRosterBucketBestScore checks that when several dirs map to ONE account
// bucket, the best score wins — a bucket has room if ANY of its dirs can be offered.
func TestHeadroomFromRosterBucketBestScore(t *testing.T) {
	rows := []fleetaccounts.Account{
		// Same bucket u-gem7: one blocked dir, one offerable dir -> the bucket has room.
		{Product: "claude", AccountUUID: hrStr("u-gem7"), Available: hrBool(false), Blocked: hrBool(true)},
		{Product: "claude", AccountUUID: hrStr("u-gem7"), Available: hrBool(true), Blocked: hrBool(false)},
	}
	hr := headroomFromRoster(rows)
	if got := hr["uuid:u-gem7"]; got != 1 {
		t.Fatalf("bucket best score = %v, want 1 (any offerable dir gives the bucket room)", got)
	}
}

// TestHeadroomFromRosterEmptyIsNil ensures an empty/irrelevant roster yields a nil signal, so
// the pure planner falls back to stable-by-name rather than a spurious all-zero headroom order.
func TestHeadroomFromRosterEmptyIsNil(t *testing.T) {
	if hr := headroomFromRoster(nil); hr != nil {
		t.Fatalf("nil roster -> %v, want nil signal", hr)
	}
	if hr := headroomFromRoster([]fleetaccounts.Account{{Product: "opencode", AccountUUID: hrStr("u-glm")}}); hr != nil {
		t.Fatalf("no-claude roster -> %v, want nil signal", hr)
	}
}

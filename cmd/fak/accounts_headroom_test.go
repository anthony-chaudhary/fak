package main

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func hrBool(b bool) *bool    { return &b }
func hrStr(s string) *string { return &s }
func hrInt(i int) *int       { return &i }

// hrNow is a fixed anchor for the reset-soonness tie-break, so a "3pm"-style reset resolves
// deterministically relative to it.
var hrNow = time.Date(2026, time.June, 30, 10, 0, 0, 0, time.UTC)

// TestHeadroomFromRoster pins the BANDED offerability tiering that turns the live runtime
// roster into the config-plane rotation signal: offerable -> (1,2], walled/throttled ->
// [-1,0), unknown -> 0, keyed by the account bucket key; non-claude and identity-less rows
// are ignored. The exact within-tier value is a separate test; here we assert the bands so a
// bucket never jumps tier.
func TestHeadroomFromRoster(t *testing.T) {
	rows := []fleetaccounts.Account{
		// Offerable claude worker -> room (band (1,2]).
		{Product: "claude", AccountUUID: hrStr("u-day30"), Available: hrBool(true), Blocked: hrBool(false)},
		// Blocked claude worker -> walled (band [-1,0)).
		{Product: "claude", AccountUUID: hrStr("u-gem7"), Available: hrBool(false), Blocked: hrBool(true)},
		// Usage-throttled claude worker -> walled ([-1,0)), even if a stale Available lingered true.
		{Product: "claude", AccountUUID: hrStr("u-cap"), Available: hrBool(true), Throttled: hrBool(true)},
		// No runtime availability signal -> unknown (0).
		{Product: "claude", AccountUUID: hrStr("u-unknown")},
		// Non-claude row -> ignored (config-plane rotation is over Claude seats).
		{Product: "opencode", AccountUUID: hrStr("u-glm"), Available: hrBool(true)},
		// No resolved identity -> ignored (nothing to key on).
		{Product: "claude", Available: hrBool(true)},
	}

	hr := headroomFromRoster(rows, hrNow)
	if len(hr) != 4 {
		t.Fatalf("headroom map = %v, want 4 entries", hr)
	}
	// Offerable band is (1,2] — strictly above the unknown/walled tiers. Walled band is [-1,0)
	// — its floor -1 is INCLUSIVE (a walled bucket with no parseable reset carries the bare
	// base, zero soonness bonus, which is exactly what u-gem7/u-cap have here).
	assertOfferable(t, "u-day30 offerable", hr["uuid:u-day30"])
	assertWalled(t, "u-gem7 walled", hr["uuid:u-gem7"])
	assertWalled(t, "u-cap throttled", hr["uuid:u-cap"])
	if got := hr["uuid:u-unknown"]; got != 0 {
		t.Fatalf("unknown bucket = %v, want 0", got)
	}
	if _, ok := hr["uuid:u-glm"]; ok {
		t.Fatal("opencode bucket must not appear in the claude rotation headroom")
	}
}

// assertOfferable fails unless v is strictly inside the offerable band (1,2], so an offerable
// bucket always outranks unknown (0) and walled (<0).
func assertOfferable(t *testing.T, name string, v float64) {
	t.Helper()
	if v <= 1 || v > 2 {
		t.Fatalf("%s score = %v, want in (1,2]", name, v)
	}
}

// assertWalled fails unless v is inside the walled band [-1,0): the floor -1 is inclusive (no
// reset bonus), the ceiling 0 is exclusive so a walled bucket never reaches unknown/offerable.
func assertWalled(t *testing.T, name string, v float64) {
	t.Helper()
	if v < -1 || v >= 0 {
		t.Fatalf("%s score = %v, want in [-1,0)", name, v)
	}
}

// TestHeadroomOfferableLeastLoadedFirst proves the within-offerable tie-break: two offerable
// buckets with different live-session counts order least-loaded-first, and both stay above 1.
func TestHeadroomOfferableLeastLoadedFirst(t *testing.T) {
	rows := []fleetaccounts.Account{
		{Product: "claude", AccountUUID: hrStr("u-busy"), Available: hrBool(true), LiveSessions: hrInt(5)},
		{Product: "claude", AccountUUID: hrStr("u-idle"), Available: hrBool(true), LiveSessions: hrInt(0)},
	}
	hr := headroomFromRoster(rows, hrNow)
	busy, idle := hr["uuid:u-busy"], hr["uuid:u-idle"]
	if !(idle > busy) {
		t.Fatalf("idle (%v) must outrank busy (%v) among offerable buckets", idle, busy)
	}
	if busy <= 1 || idle <= 1 {
		t.Fatalf("both offerable scores must stay > 1: busy=%v idle=%v", busy, idle)
	}
}

// TestHeadroomWalledSoonestResetFirst proves the within-walled tie-break: two walled buckets
// with different reset times order soonest-reset-first, and both stay below 0. Anchored at
// 10:00 UTC, "11am" is sooner than "3pm".
func TestHeadroomWalledSoonestResetFirst(t *testing.T) {
	rows := []fleetaccounts.Account{
		{Product: "claude", AccountUUID: hrStr("u-late"), Blocked: hrBool(true), Reset: hrStr("3pm")},
		{Product: "claude", AccountUUID: hrStr("u-soon"), Blocked: hrBool(true), Reset: hrStr("11am")},
	}
	hr := headroomFromRoster(rows, hrNow)
	late, soon := hr["uuid:u-late"], hr["uuid:u-soon"]
	if !(soon > late) {
		t.Fatalf("soonest-reset (%v) must outrank later reset (%v) among walled buckets", soon, late)
	}
	if late >= 0 || soon >= 0 {
		t.Fatalf("both walled scores must stay < 0: late=%v soon=%v", late, soon)
	}
}

// TestHeadroomTierBandsNeverOverlap is the load-bearing invariant: any offerable bucket
// outranks any unknown, which outranks any walled — regardless of within-tier bonuses. A
// heavily-loaded offerable bucket must still beat a soon-to-reset walled one.
func TestHeadroomTierBandsNeverOverlap(t *testing.T) {
	rows := []fleetaccounts.Account{
		{Product: "claude", AccountUUID: hrStr("u-off"), Available: hrBool(true), LiveSessions: hrInt(99)},
		{Product: "claude", AccountUUID: hrStr("u-unk")},
		{Product: "claude", AccountUUID: hrStr("u-wall"), Blocked: hrBool(true), Reset: hrStr("11am")},
	}
	hr := headroomFromRoster(rows, hrNow)
	off, unk, wall := hr["uuid:u-off"], hr["uuid:u-unk"], hr["uuid:u-wall"]
	if !(off > unk && unk > wall) {
		t.Fatalf("tier order broken: offerable=%v unknown=%v walled=%v (want offerable>unknown>walled)", off, unk, wall)
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
	hr := headroomFromRoster(rows, hrNow)
	if got := hr["uuid:u-gem7"]; got <= 1 {
		t.Fatalf("bucket best score = %v, want > 1 (any offerable dir gives the bucket room)", got)
	}
}

// TestHeadroomFromRosterEmptyIsNil ensures an empty/irrelevant roster yields a nil signal, so
// the pure planner falls back to stable-by-name rather than a spurious all-zero headroom order.
func TestHeadroomFromRosterEmptyIsNil(t *testing.T) {
	if hr := headroomFromRoster(nil, hrNow); hr != nil {
		t.Fatalf("nil roster -> %v, want nil signal", hr)
	}
	if hr := headroomFromRoster([]fleetaccounts.Account{{Product: "opencode", AccountUUID: hrStr("u-glm")}}, hrNow); hr != nil {
		t.Fatalf("no-claude roster -> %v, want nil signal", hr)
	}
}

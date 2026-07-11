package fleetaccounts

import (
	"testing"
	"time"
)

// snNow anchors the undated time-only reset forms deterministically.
var snNow = time.Date(2026, time.June, 30, 10, 0, 0, 0, time.UTC)

// TestResetSoonnessOrdersSoonerHigher pins the core property the walled-tier tie-break relies
// on: a nearer future reset scores higher than a farther one, both strictly inside [0,1).
func TestResetSoonnessOrdersSoonerHigher(t *testing.T) {
	soon, ok1 := ResetSoonness("11am", snNow) // +1h
	late, ok2 := ResetSoonness("3pm", snNow)  // +5h
	if !ok1 || !ok2 {
		t.Fatalf("both future resets must parse: 11am ok=%v, 3pm ok=%v", ok1, ok2)
	}
	if !(soon > late) {
		t.Fatalf("nearer reset must score higher: 11am=%v, 3pm=%v", soon, late)
	}
	for _, tc := range []struct {
		name string
		v    float64
	}{{"11am", soon}, {"3pm", late}} {
		if tc.v < 0 || tc.v >= 1 {
			t.Fatalf("%s soonness = %v, want in [0,1)", tc.name, tc.v)
		}
	}
}

// TestResetSoonnessAtNowIsNearOne checks a reset essentially at now scores near the top of the
// band (about to free up).
func TestResetSoonnessAtNowIsNearOne(t *testing.T) {
	// 10am == now; the daily-reset slack rolls a just-passed time-only reset to tomorrow, so
	// use a couple minutes ahead to stay "today" and near now.
	v, ok := ResetSoonness("10:01am", snNow)
	if !ok {
		t.Fatal("a near-future reset must parse")
	}
	if v < 0.9 {
		t.Fatalf("a reset ~1min out should score near 1, got %v", v)
	}
}

// TestResetSoonnessUnparseableOrExpired checks the ok=false paths: empty, garbage, and an
// already-expired dated reset all report no soonness (the account is not waiting on it).
func TestResetSoonnessUnparseableOrExpired(t *testing.T) {
	for _, s := range []string{"", "whenever", "next tuesday"} {
		if v, ok := ResetSoonness(s, snNow); ok {
			t.Fatalf("ResetSoonness(%q) = (%v,true), want ok=false", s, v)
		}
	}
	// A dated reset comfortably in the past (last month) is expired -> no soonness.
	if v, ok := ResetSoonness("May 1, 3pm", snNow); ok {
		t.Fatalf("expired dated reset -> (%v,true), want ok=false", v)
	}
}

// TestResetSoonnessFarFutureIsZeroButOk checks a dated reset beyond the soonness horizon is
// still a valid future reset (ok=true) but scores 0 — future, yet no sooner than the horizon.
func TestResetSoonnessFarFutureIsZeroButOk(t *testing.T) {
	v, ok := ResetSoonness("Aug 1, 3pm", snNow) // ~32 days out, past the 24h horizon
	if !ok {
		t.Fatal("a far-future dated reset must still parse as future")
	}
	if v != 0 {
		t.Fatalf("far-future reset soonness = %v, want 0 (beyond horizon)", v)
	}
}

// TestResetParsesWeeklyAtAndWeekdayForms pins parity with fleet_accounts._reset_is_future on
// the DATED weekly reset shapes Claude actually emits: an "at" separator and an optional
// leading weekday token ("Mon Jun 25 at 1pm"). Before the weekday-strip + "at"-layout fix these
// parsed as nil (unknown), which throttleIsActive treats fail-closed as a still-active weekly
// cap — walling a healthy seat indefinitely and suppressing its fresh-OK probes. now is well
// before the reset so every form must read as a live future weekly.
func TestResetParsesWeeklyAtAndWeekdayForms(t *testing.T) {
	now := time.Date(2026, time.June, 20, 10, 0, 0, 0, time.UTC)
	future := []string{
		"Jun 25 at 1pm",
		"Jun 25 at 1:30pm",
		"Mon Jun 25 at 1pm",
		"Mon Jun 25 at 1:30pm",
		"Jun 25, 1pm",
		"Jun 25 at 1pm (America/Los_Angeles)",
	}
	for _, f := range future {
		if _, ok := resetTime(f, now); !ok {
			t.Errorf("resetTime(%q) not parsed; a real weekly form must parse", f)
		}
		if r := resetIsFuture(f, now); r == nil || !*r {
			t.Errorf("resetIsFuture(%q) = %v, want future — an unparsed weekly walls the seat fail-closed", f, r)
		}
	}
	// A narrative form we do not model stays unknown (nil), matching Python's None.
	if r := resetIsFuture("resets Monday", now); r != nil {
		t.Errorf("resetIsFuture(%q) = %v, want nil (unknown)", "resets Monday", r)
	}
}

// TestResetIsFutureUnchanged guards the refactor: resetIsFuture now reads the shared resetTime
// core, and must still report future/expired/unknown exactly as before.
func TestResetIsFutureUnchanged(t *testing.T) {
	if r := resetIsFuture("3pm", snNow); r == nil || !*r {
		t.Fatalf("3pm (5h ahead) should be future, got %v", r)
	}
	if r := resetIsFuture("", snNow); r != nil {
		t.Fatalf("empty reset should be unknown (nil), got %v", r)
	}
	if r := resetIsFuture("May 1, 3pm", snNow); r == nil || *r {
		t.Fatalf("May 1 (expired dated) should be past (false), got %v", r)
	}
}

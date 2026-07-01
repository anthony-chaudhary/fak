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

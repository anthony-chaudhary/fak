package seatpark

import "testing"

// The kernel's load-bearing promise: a no-seat task neither bursts (unbounded
// immediate retry) nor parks forever. These tests pin the READY -> PARKED ->
// EXHAUSTED progression, the geometric backoff with cap, and the fail-toward-READY
// invariants (a first encounter and a missing clock never silently stall).

func TestDecide_FirstEncounterIsReadyNotBurst(t *testing.T) {
	// Parks==0: no prior park, so the first launch attempt is allowed (READY).
	d := Decide(Input{TaskID: "t1", Parks: 0, NowUnix: 1000})
	if d.Status != StatusReady {
		t.Fatalf("first encounter status = %q, want %q", d.Status, StatusReady)
	}
	if !d.ShouldAttempt() {
		t.Fatal("first encounter should attempt")
	}
	if d.BackoffSeconds != 0 || d.NextRetryUnix != 0 {
		t.Fatalf("never-parked task should carry no window: backoff=%d next=%d", d.BackoffSeconds, d.NextRetryUnix)
	}
}

func TestDecide_ParkedWhileInsideWindow(t *testing.T) {
	// Parked once at t=1000, default base 30s => window closes at 1030; at t=1010 -> PARKED.
	d := Decide(Input{TaskID: "t1", Parks: 1, LastParkUnix: 1000, NowUnix: 1010})
	if d.Status != StatusParked {
		t.Fatalf("status = %q, want %q (inside window)", d.Status, StatusParked)
	}
	if d.ShouldAttempt() {
		t.Fatal("a parked task must not attempt this tick")
	}
	if !d.Retryable() {
		t.Fatal("a parked task is still retryable")
	}
	if d.NextRetryUnix != 1030 {
		t.Fatalf("NextRetryUnix = %d, want 1030 (1000 + 30s base)", d.NextRetryUnix)
	}
}

func TestDecide_ReadyOnceWindowElapsed(t *testing.T) {
	// Same park, but now past the window boundary => READY to re-attempt.
	d := Decide(Input{TaskID: "t1", Parks: 1, LastParkUnix: 1000, NowUnix: 1031})
	if d.Status != StatusReady {
		t.Fatalf("status = %q, want %q (window elapsed)", d.Status, StatusReady)
	}
}

func TestDecide_WindowBoundaryIsInclusiveReady(t *testing.T) {
	// now == NextRetryUnix is NOT before the boundary => READY (strict <).
	d := Decide(Input{Parks: 1, LastParkUnix: 1000, NowUnix: 1030})
	if d.Status != StatusReady {
		t.Fatalf("status = %q at the exact boundary, want %q", d.Status, StatusReady)
	}
}

func TestDecide_GeometricBackoffWithCap(t *testing.T) {
	// Default base 30, factor 2, cap 300: 30, 60, 120, 240, then capped at 300.
	want := map[int]int64{1: 30, 2: 60, 3: 120, 4: 240, 5: 300, 6: 300, 10: 300}
	for parks, w := range want {
		got := backoffSeconds(Policy{}.withDefaults(), parks)
		if got != w {
			t.Errorf("backoff at %d parks = %d, want %d", parks, got, w)
		}
	}
}

func TestDecide_ExhaustedAtBudget(t *testing.T) {
	// Default MaxParks 5: at 5 parks the task is EXHAUSTED even if a window is open.
	d := Decide(Input{TaskID: "t1", Parks: 5, LastParkUnix: 1000, NowUnix: 1001})
	if d.Status != StatusExhausted {
		t.Fatalf("status = %q at the budget, want %q", d.Status, StatusExhausted)
	}
	if d.Retryable() {
		t.Fatal("an exhausted task must not be retryable")
	}
	if d.ShouldAttempt() {
		t.Fatal("an exhausted task must not attempt")
	}
}

func TestDecide_ExhaustedOverridesOpenWindow(t *testing.T) {
	// The hard budget stop wins over an otherwise-open backoff window.
	d := Decide(Input{Parks: 6, LastParkUnix: 1000, NowUnix: 1000})
	if d.Status != StatusExhausted {
		t.Fatalf("status = %q, want %q (budget overrides window)", d.Status, StatusExhausted)
	}
}

func TestDecide_NoClockNeverStallsSilently(t *testing.T) {
	// NowUnix==0: the caller supplies no clock, so a parked task must fail toward
	// READY (report the window it WOULD wait, but never PARK without a "now").
	d := Decide(Input{Parks: 2, LastParkUnix: 1000, NowUnix: 0})
	if d.Status != StatusReady {
		t.Fatalf("status = %q with no clock, want %q", d.Status, StatusReady)
	}
	if d.NextRetryUnix != 1060 {
		t.Fatalf("NextRetryUnix = %d, want 1060 (informational window still reported)", d.NextRetryUnix)
	}
}

func TestDecide_CustomPolicyOverrides(t *testing.T) {
	p := Policy{MaxParks: 2, BaseSeconds: 10, Factor: 3, CapSeconds: 1000}
	// park 1 -> 10s, park 2 -> 30s; MaxParks 2 => 2 parks is EXHAUSTED.
	if got := backoffSeconds(p.withDefaults(), 2); got != 30 {
		t.Fatalf("custom backoff at 2 parks = %d, want 30", got)
	}
	d := Decide(Input{Parks: 2, LastParkUnix: 100, NowUnix: 101, Policy: p})
	if d.Status != StatusExhausted {
		t.Fatalf("status = %q, want %q (custom MaxParks 2)", d.Status, StatusExhausted)
	}
	if d.MaxParks != 2 {
		t.Fatalf("effective MaxParks = %d, want 2", d.MaxParks)
	}
}

func TestPolicy_ZeroValueUsesDocumentedDefaults(t *testing.T) {
	p := Policy{}.withDefaults()
	if p.MaxParks != DefaultMaxParks || p.BaseSeconds != DefaultBaseSeconds || p.Factor != DefaultFactor || p.CapSeconds != DefaultCapSeconds {
		t.Fatalf("zero policy defaults = %+v, want the documented constants", p)
	}
}

func TestPolicy_SubTwoFactorFallsBackSoBackoffActuallyGrows(t *testing.T) {
	// A factor < 2 would not back off; it must fall back to the default.
	p := Policy{Factor: 1}.withDefaults()
	if p.Factor != DefaultFactor {
		t.Fatalf("factor = %d, want fallback to %d", p.Factor, DefaultFactor)
	}
}

func TestSourcedCapMatchesRateLimitPrecedent(t *testing.T) {
	// The cap is anchored to attemptbudget's rate-limit window (5m). Pin it so a
	// silent edit that unmoors the two is caught.
	if DefaultCapSeconds != 5*60 {
		t.Errorf("DefaultCapSeconds = %d, want 300 (attemptbudget FailureClassRateLimit precedent)", DefaultCapSeconds)
	}
}

func TestStatusClosedSet(t *testing.T) {
	for _, s := range []Status{StatusReady, StatusParked, StatusExhausted} {
		if !s.Valid() {
			t.Errorf("%q should be a known Status", s)
		}
	}
	if Status("SEAT_BOGUS").Valid() {
		t.Error("an unknown token must not validate as a Status")
	}
}

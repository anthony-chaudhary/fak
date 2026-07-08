package relay

import "testing"

// TestNoProgressEscapeTripsAfterKEmptyLegs is the #1893 done-condition witness: K
// consecutive legs with no verified progress trip RELAY_NO_PROGRESS and halt the relay.
func TestNoProgressEscapeTripsAfterKEmptyLegs(t *testing.T) {
	const k = 3
	e := NoProgressEscape{MaxEmptyLegs: k}

	// The first k-1 empty legs accumulate but do not yet halt.
	for i := 1; i < k; i++ {
		halt, reason := e.ObserveLeg(verifiedSteps(0))
		if halt {
			t.Fatalf("leg %d: halted early (empty run %d < K=%d)", i, e.EmptyRun(), k)
		}
		if reason != "" {
			t.Fatalf("leg %d: unexpected reason %q before the escape trips", i, reason)
		}
		if got := e.EmptyRun(); got != i {
			t.Fatalf("leg %d: EmptyRun() = %d, want %d", i, got, i)
		}
	}

	// The K-th consecutive empty leg trips the escape.
	halt, reason := e.ObserveLeg(verifiedSteps(0))
	if !halt {
		t.Fatalf("the K=%d-th empty leg must halt the relay", k)
	}
	if reason != ReasonNoProgress {
		t.Fatalf("halt reason = %q, want %q", reason, ReasonNoProgress)
	}
	if ReasonNoProgress != "RELAY_NO_PROGRESS" {
		t.Fatalf("ReasonNoProgress = %q, want the closed vocabulary token RELAY_NO_PROGRESS", ReasonNoProgress)
	}
}

// TestNoProgressResetsOnVerifiedMovement pins that a leg which advances the verified
// progress cursor resets the empty run, so only CONSECUTIVE empty legs count.
func TestNoProgressResetsOnVerifiedMovement(t *testing.T) {
	e := NoProgressEscape{MaxEmptyLegs: 2}

	if halt, _ := e.ObserveLeg(verifiedSteps(0)); halt { // empty run = 1
		t.Fatalf("first empty leg must not halt with K=2")
	}
	// A verified advance (1 new step) resets the counter before it reaches K.
	if halt, _ := e.ObserveLeg(verifiedSteps(1)); halt {
		t.Fatalf("a verified-progress leg must not halt")
	}
	if got := e.EmptyRun(); got != 0 {
		t.Fatalf("EmptyRun() = %d, want 0 after verified movement", got)
	}
	// Now two fresh empty legs are needed again to trip.
	if halt, _ := e.ObserveLeg(verifiedSteps(1)); halt { // no advance past high-water 1
		t.Fatalf("second empty run leg 1 must not halt yet")
	}
	halt, reason := e.ObserveLeg(verifiedSteps(1)) // still no advance -> empty run reaches 2
	if !halt || reason != ReasonNoProgress {
		t.Fatalf("two consecutive non-advancing legs after the reset must halt with RELAY_NO_PROGRESS; got halt=%v reason=%q", halt, reason)
	}
}

// TestNoProgressUnknownCountsAsEmpty pins the fail-closed rule: an unverifiable read
// counts as an empty leg, so a stream of unknown legs still trips the escape.
func TestNoProgressUnknownCountsAsEmpty(t *testing.T) {
	e := NoProgressEscape{MaxEmptyLegs: 2}
	if halt, _ := e.ObserveLeg(unknownProgress); halt {
		t.Fatalf("first unknown leg must not halt with K=2")
	}
	halt, reason := e.ObserveLeg(unknownProgress)
	if !halt || reason != ReasonNoProgress {
		t.Fatalf("consecutive unverifiable legs must trip RELAY_NO_PROGRESS (fail closed); got halt=%v reason=%q", halt, reason)
	}
}

// TestNoProgressDisabledNeverHalts pins that the zero/unset policy never trips:
// MaxEmptyLegs <= 0 disables the escape regardless of how many empty legs pass.
func TestNoProgressDisabledNeverHalts(t *testing.T) {
	for _, k := range []int{0, -1} {
		e := NoProgressEscape{MaxEmptyLegs: k}
		for i := 0; i < 5; i++ {
			if halt, reason := e.ObserveLeg(verifiedSteps(0)); halt || reason != "" {
				t.Fatalf("MaxEmptyLegs=%d: a disabled escape must never halt (leg %d)", k, i)
			}
		}
	}
}

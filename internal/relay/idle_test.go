package relay

import "testing"

// fakeWitness is a WatchWitness whose verdict is fixed — the injected durable-witness stand-in
// that lets the idle fold be tested without a live dos_verify / CI checker.
type fakeWitness struct{ verdict WatchVerdict }

func (f fakeWitness) CheckInvariant() WatchVerdict { return f.verdict }

// obsFrom builds an IdleObservation the way a caller would: it re-checks the invariant through
// the injected WatchWitness and pairs the verdict with the durable pending-work read.
func obsFrom(w WatchWitness, pending int, pendingKnown bool) IdleObservation {
	return IdleObservation{Invariant: w.CheckInvariant(), PendingAdmitted: pending, PendingKnown: pendingKnown}
}

// TestWatchGoalIdleParksInsteadOfNoProgress is the #4143 done-condition witness: a watch relay
// whose invariant holds against ground truth with zero admitted pending work is driven well past
// K no-progress legs and parks with RELAY_IDLE_PARKED every time — it NEVER trips
// RELAY_NO_PROGRESS, and the underlying empty-run counter never advances.
func TestWatchGoalIdleParksInsteadOfNoProgress(t *testing.T) {
	e := &IdleAwareEscape{Escape: NoProgressEscape{MaxEmptyLegs: 3}}
	holds := fakeWitness{WatchHolds}
	for leg := 0; leg < 7; leg++ { // 7 > K=3: a naive empty counter would have tripped at leg 3
		out := e.ObserveLeg(unknownProgress, obsFrom(holds, 0, true))
		if out.Halt {
			t.Fatalf("leg %d: a proven-idle watch relay must never halt with RELAY_NO_PROGRESS", leg)
		}
		if !out.Parked || out.Reason != ReasonIdleParked {
			t.Fatalf("leg %d: want park with %q, got parked=%v reason=%q", leg, ReasonIdleParked, out.Parked, out.Reason)
		}
		if !e.Parked() {
			t.Fatalf("leg %d: Parked() must report the idle park", leg)
		}
	}
	if got := e.Escape.EmptyRun(); got != 0 {
		t.Fatalf("idle parks must not advance the no-progress counter; empty run = %d", got)
	}
}

// TestWatchGoalIdleStuckStillTrips proves the terminator only makes the escape MORE conservative:
// a relay that is NOT proven idle (its invariant is violated — there is real work) is counted as
// an empty leg exactly as before and trips RELAY_NO_PROGRESS at K.
func TestWatchGoalIdleStuckStillTrips(t *testing.T) {
	e := &IdleAwareEscape{Escape: NoProgressEscape{MaxEmptyLegs: 3}}
	stuck := obsFrom(fakeWitness{WatchViolated}, 0, true) // invariant broken -> not idle
	for leg := 1; leg <= 3; leg++ {
		out := e.ObserveLeg(unknownProgress, stuck)
		if out.Parked {
			t.Fatalf("leg %d: a stuck relay must never park idle", leg)
		}
		if leg < 3 && out.Halt {
			t.Fatalf("leg %d: escape tripped early", leg)
		}
		if leg == 3 {
			if !out.Halt || out.Reason != ReasonNoProgress {
				t.Fatalf("leg 3: want halt with %q, got halt=%v reason=%q", ReasonNoProgress, out.Halt, out.Reason)
			}
		}
	}
}

// TestWatchGoalIdleFailsClosed pins the idle predicate's fail-closed edges: idle is true ONLY
// when the invariant holds AND the ledger positively confirms zero admitted pending work. An
// unknown invariant, an unread pending count, any positive pending work, or a violated invariant
// each make the leg NOT idle, so it is counted toward the no-progress escape.
func TestWatchGoalIdleFailsClosed(t *testing.T) {
	cases := []struct {
		name     string
		verdict  WatchVerdict
		pending  int
		known    bool
		wantIdle bool
	}{
		{"holds_zero_pending_known", WatchHolds, 0, true, true},
		{"holds_but_pending_work", WatchHolds, 2, true, false},
		{"holds_but_pending_unread", WatchHolds, 0, false, false},
		{"invariant_unknown", WatchUnknown, 0, true, false},
		{"invariant_violated", WatchViolated, 0, true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			e := &IdleAwareEscape{Escape: NoProgressEscape{MaxEmptyLegs: 1}}
			out := e.ObserveLeg(unknownProgress, obsFrom(fakeWitness{c.verdict}, c.pending, c.known))
			if out.Parked != c.wantIdle {
				t.Fatalf("parked=%v want %v", out.Parked, c.wantIdle)
			}
			// With MaxEmptyLegs=1, a non-idle leg trips immediately; an idle leg never counts.
			if c.wantIdle && out.Halt {
				t.Fatalf("a proven-idle leg must not trip the escape")
			}
			if !c.wantIdle && (!out.Halt || out.Reason != ReasonNoProgress) {
				t.Fatalf("a non-idle leg must count toward RELAY_NO_PROGRESS; halt=%v reason=%q", out.Halt, out.Reason)
			}
		})
	}
}

// TestWatchGoalIdleProgressBeatsPark proves verified forward progress takes priority over an idle
// park: a leg that advances the verified cursor is delegated to the escape (which resets the
// empty run and raises the high-water mark), never swallowed as a park — even when the idle
// observation would otherwise qualify.
func TestWatchGoalIdleProgressBeatsPark(t *testing.T) {
	e := &IdleAwareEscape{Escape: NoProgressEscape{MaxEmptyLegs: 2}}
	holdsIdle := obsFrom(fakeWitness{WatchHolds}, 0, true)

	// Two empty (non-idle) legs bring the escape to the brink.
	stuck := obsFrom(fakeWitness{WatchViolated}, 0, true)
	e.ObserveLeg(unknownProgress, stuck) // empty run = 1
	// A leg that makes verified forward progress AND looks idle must reset, not park.
	out := e.ObserveLeg(verifiedSteps(1), holdsIdle)
	if out.Parked {
		t.Fatal("a progress-advancing leg must not be swallowed as an idle park")
	}
	if out.Halt {
		t.Fatal("a progress-advancing leg must not halt")
	}
	if got := e.Escape.EmptyRun(); got != 0 {
		t.Fatalf("verified progress must reset the empty run; got %d", got)
	}
}

// TestWatchGoalIdleReArmsOnWitnessedEvent pins the ExternalEvent re-arm trigger: a parked watch
// relay re-arms only on a WITNESSED flip of external ground truth, and fails closed on a missing
// witness (an empty parked baseline or an empty current observation never re-arms).
func TestWatchGoalIdleReArmsOnWitnessedEvent(t *testing.T) {
	cases := []struct {
		name        string
		parked, obs string
		wantRearm   bool
	}{
		{"unchanged_no_rearm", "sha-aaa", "sha-aaa", false},
		{"flipped_rearm", "sha-aaa", "sha-bbb", true},
		{"no_baseline_fail_closed", "", "sha-bbb", false},
		{"no_observation_fail_closed", "sha-aaa", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ev := ExternalEvent{Kind: "git_ref", ParkedToken: c.parked, ObservedToken: c.obs}
			if got := ev.Rearmed(); got != c.wantRearm {
				t.Fatalf("Rearmed()=%v want %v", got, c.wantRearm)
			}
		})
	}
}

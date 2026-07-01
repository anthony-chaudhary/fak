package recall

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestStaleFactAsCurrentDetected is the bite test for issue #1594: a bounded fact
// whose stated validity has lapsed, and that the caller marks Required (load-bearing
// current state), must be refused — GuardAgainstStaleFact returns a typed
// *StaleFactError (never a bare string, never the recalled value) instead of letting
// the stale value reach action context. This is the fixture the done condition names:
// "a stale-current fixture returns a typed fault instead of allowing the recalled
// value into action context."
func TestStaleFactAsCurrentDetected(t *testing.T) {
	p := Page{
		Step:       0,
		Durability: durabilityBounded,
		ValidTo:    100,
	}
	decision, err := GuardAgainstStaleFact(p, StaleFactCheck{AsOf: 150, Required: true})

	if err == nil {
		t.Fatalf("expired bounded fact marked Required must be refused, got nil error (decision=%+v)", decision)
	}
	var sfErr *StaleFactError
	if !errors.As(err, &sfErr) {
		t.Fatalf("error must be a typed *StaleFactError, got %T: %v", err, err)
	}
	if !errors.Is(err, ErrStaleFactAsCurrent) {
		t.Fatalf("error must wrap ErrStaleFactAsCurrent via errors.Is, got %v", err)
	}
	if decision.Outcome != StaleFactExpiredDeny {
		t.Fatalf("expired+Required bounded fact: want outcome %q, got %q (reason=%s)", StaleFactExpiredDeny, decision.Outcome, decision.Reason)
	}
	if decision.Reason == "" {
		t.Fatalf("decision must carry an operator-readable reason, got empty")
	}
}

// TestStaleFactAsCurrentTurnScopedDetected proves the SECOND stale-as-current shape
// this issue closes: a turn-scoped fact (no explicit ValidTo was ever stamped — the
// "it's 3pm" case from CONTEXT-IS-NOT-MEMORY.md) recalled in a LATER turn and marked
// Required. validityGate (recall.go) only ever checked the bounded class; a
// turn/session-scoped fact with no stated expiry previously had no staleness check at
// all once past ValidTo-less durability classes. DetectStaleFact closes that gap.
func TestStaleFactAsCurrentTurnScopedDetected(t *testing.T) {
	p := Page{
		Step:       0, // recorded at turn/step 0
		Durability: ctxmmu.DurabilityTurn,
	}
	decision, err := GuardAgainstStaleFact(p, StaleFactCheck{AsOf: 5, Required: true})

	if err == nil {
		t.Fatalf("turn-scoped fact recalled at a later tick and marked Required must be refused, got nil (decision=%+v)", decision)
	}
	if decision.Outcome != StaleFactExpiredMustQuery {
		t.Fatalf("expired+Required turn-scoped fact: want outcome %q, got %q", StaleFactExpiredMustQuery, decision.Outcome)
	}
}

// TestStaleFactFreshNotFlagged proves the counterpart: a durable fact, and a bounded
// fact still inside its validity window, both pass GuardAgainstStaleFact cleanly (nil
// error, StaleFactFresh) — the gate must not cry wolf on genuinely current facts, or
// callers would learn to route around it.
func TestStaleFactFreshNotFlagged(t *testing.T) {
	cases := []struct {
		name string
		p    Page
		chk  StaleFactCheck
	}{
		{
			name: "durable fact never expires",
			p:    Page{Step: 0, Durability: ctxmmu.DurabilityDurable},
			chk:  StaleFactCheck{AsOf: 10_000, Required: true},
		},
		{
			name: "bounded fact still inside its window",
			p:    Page{Step: 0, Durability: durabilityBounded, ValidTo: 100},
			chk:  StaleFactCheck{AsOf: 50, Required: true},
		},
		{
			name: "turn-scoped fact recalled within the same turn",
			p:    Page{Step: 5, Durability: ctxmmu.DurabilityTurn},
			chk:  StaleFactCheck{AsOf: 5, Required: true},
		},
		{
			name: "no current tick supplied never expires (mirrors validityGate's asOf<=0 no-op)",
			p:    Page{Step: 0, Durability: ctxmmu.DurabilityTurn},
			chk:  StaleFactCheck{AsOf: 0, Required: true},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			decision, err := GuardAgainstStaleFact(tc.p, tc.chk)
			if err != nil {
				t.Fatalf("fresh fact must not be refused: %v", err)
			}
			if decision.Outcome != StaleFactFresh {
				t.Fatalf("want outcome %q, got %q (reason=%s)", StaleFactFresh, decision.Outcome, decision.Reason)
			}
		})
	}
}

// TestStaleFactNotRequiredNeedsRefreshOnly proves the non-Required asymmetry
// GuardAgainstStaleFact borrows from ctxplan.DecidePageFault: a stale fact the caller
// does NOT mark Required is not hard-gated (a background aside is not being smuggled
// into action context) — it still yields a typed StaleFactExpiredNeedsRefresh
// decision so the caller can proactively refresh, but the error is nil.
func TestStaleFactNotRequiredNeedsRefreshOnly(t *testing.T) {
	p := Page{Step: 0, Durability: durabilityBounded, ValidTo: 100}
	decision, err := GuardAgainstStaleFact(p, StaleFactCheck{AsOf: 150, Required: false})
	if err != nil {
		t.Fatalf("non-required stale fact must not be hard-gated, got error: %v", err)
	}
	if decision.Outcome != StaleFactExpiredNeedsRefresh {
		t.Fatalf("want outcome %q, got %q", StaleFactExpiredNeedsRefresh, decision.Outcome)
	}
}

// TestStaleFactOutcomeClosedVocabulary pins the closed-vocabulary membership
// contract, mirroring ctxplan's TestValidPageFaultOutcome-style pin: every outcome
// DetectStaleFact can produce is a member, and a foreign/garbage value is not.
func TestStaleFactOutcomeClosedVocabulary(t *testing.T) {
	members := []StaleFactOutcome{
		StaleFactFresh, StaleFactExpiredNeedsRefresh, StaleFactExpiredMustQuery, StaleFactExpiredDeny,
	}
	for _, m := range members {
		if !ValidStaleFactOutcome(m) {
			t.Fatalf("%q must be a member of the closed vocabulary", m)
		}
		if got := m.String(); got != string(m) {
			t.Fatalf("String() of a valid member must round-trip, got %q", got)
		}
	}
	if ValidStaleFactOutcome(StaleFactOutcome("garbage")) {
		t.Fatalf("garbage value must not be a member of the closed vocabulary")
	}
	if got := StaleFactOutcome("garbage").String(); got != "unknown(garbage)" {
		t.Fatalf("unknown value String() = %q, want the fail-closed unknown(...) form", got)
	}
	if got := StaleFactOutcome("").String(); got != "(unset)" {
		t.Fatalf("empty value String() = %q, want (unset)", got)
	}
}

// TestStaleFactReplayable proves DetectStaleFact is a pure function: the same
// (Page, StaleFactCheck) input always reproduces the identical decision, which is
// what makes a caller's stale-fact refusal auditable/replayable rather than a
// one-off judgment call (mirrors ctxplan.TestPageFaultDecisionIsReplayable's intent).
func TestStaleFactReplayable(t *testing.T) {
	p := Page{Step: 2, Durability: durabilityBounded, ValidTo: 30}
	chk := StaleFactCheck{AsOf: 40, Required: true}
	first := DetectStaleFact(p, chk)
	for i := 0; i < 5; i++ {
		got := DetectStaleFact(p, chk)
		if got != first {
			t.Fatalf("DetectStaleFact is not pure: run %d = %+v, want %+v", i, got, first)
		}
	}
}

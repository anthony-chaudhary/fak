package dispatchtick

import (
	"strings"
	"testing"
	"time"
)

func TestUsageCapAdvisoryArmsWhenCapsCoverMajority(t *testing.T) {
	// 6 usage-capped accounts of 8: clears the default floor of 3 AND covers >= half the
	// fleet, so the advisory arms.
	u := UsageCapAdvisory{Capped: 6, Accounts: 8, FreeSeats: 2}
	if !u.Armed() {
		t.Fatalf("Armed() = false, want true (6 capped of 8, >= floor 3, >= half)")
	}
	note := u.Note()
	if note == nil {
		t.Fatalf("Note() = nil for an armed census")
	}
	if note["signal"] != UsageCapExhaustionSignal {
		t.Errorf("signal = %v, want %q", note["signal"], UsageCapExhaustionSignal)
	}
	if note["advisory_only"] != true {
		t.Errorf("advisory_only = %v, want true", note["advisory_only"])
	}
	if note["capped_accounts"] != 6 || note["total_accounts"] != 8 || note["free_seats"] != 2 {
		t.Errorf("census = %v/%v free=%v, want 6/8 free 2", note["capped_accounts"], note["total_accounts"], note["free_seats"])
	}
	hint, _ := note["hint"].(string)
	// The hint must name the mislabel trap so a reader does not confuse it with the
	// rate_budget backoff, and must state it blocks nothing.
	if !strings.Contains(hint, "rate_limit") || !strings.Contains(strings.ToLower(hint), "not blocked") {
		t.Errorf("hint missing mislabel/advisory language: %q", hint)
	}
}

func TestUsageCapAdvisoryDoesNotArmOnHealthyFleet(t *testing.T) {
	// 3 capped accounts clear the floor, but they are a small minority of a 20-account
	// fleet -- ample headroom, so a spawn is fine and the advisory must stay silent.
	u := UsageCapAdvisory{Capped: 3, Accounts: 20, FreeSeats: 40}
	if u.Armed() {
		t.Fatalf("Armed() = true, want false (3 capped of 20: caps are a minority)")
	}
	if u.Note() != nil {
		t.Fatalf("Note() = non-nil for an unarmed census (must omit the field)")
	}
}

func TestUsageCapAdvisoryBelowFloorAbstains(t *testing.T) {
	// 2 capped accounts of 2: caps cover the whole fleet, but 2 < the default floor of 3,
	// so a tiny fleet rotating through its reset is treated as churn, not exhaustion.
	u := UsageCapAdvisory{Capped: 2, Accounts: 2, FreeSeats: 0}
	if u.Armed() {
		t.Fatalf("Armed() = true, want false (2 capped < floor 3)")
	}
}

func TestUsageCapAdvisoryCustomThreshold(t *testing.T) {
	// An overlaid threshold (shell FAK_USAGECAP_ADVISORY_MIN) raises the floor: 4 capped
	// accounts no longer arm under a floor of 5.
	u := UsageCapAdvisory{Capped: 4, Accounts: 4, FreeSeats: 0, Threshold: 5}
	if u.Armed() {
		t.Fatalf("Armed() = true, want false (4 capped < custom floor 5)")
	}
	u.Threshold = 4
	if !u.Armed() {
		t.Fatalf("Armed() = false, want true (4 capped >= custom floor 4, whole fleet)")
	}
}

func TestUsageCapAdvisoryZeroCensusNotArmed(t *testing.T) {
	// The zero value (no accounts, no caps) must never arm -- a caller that wires nothing
	// stays byte-identical.
	if (UsageCapAdvisory{}).Armed() {
		t.Fatalf("zero-value Armed() = true, want false")
	}
}

func TestUsageCapAdvisoryNoteRendersResetHorizon(t *testing.T) {
	now := time.Date(2026, 7, 7, 15, 5, 0, 0, time.UTC)
	reset := time.Date(2026, 7, 7, 15, 35, 0, 0, time.UTC)
	u := UsageCapAdvisory{Capped: 6, Accounts: 7, FreeSeats: 1, EarliestReset: reset, Now: now}
	note := u.Note()
	if note["earliest_reset"] != "2026-07-07T15:35:00Z" {
		t.Errorf("earliest_reset = %v, want RFC3339 15:35Z", note["earliest_reset"])
	}
	hint, _ := note["hint"].(string)
	if !strings.Contains(hint, "2026-07-07T15:35:00Z") || !strings.Contains(hint, "30m") {
		t.Errorf("hint missing reset instant / remaining: %q", hint)
	}
}

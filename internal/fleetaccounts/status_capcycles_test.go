package fleetaccounts

import (
	"testing"
	"time"
)

// The Phase-3 cap-disambiguation cycles, verified end-to-end through computeRuntimeStatus:
// the ledger observation must reach both DisambiguateCap seams and flip the seat's status.
// The cap math itself is unit-tested with an injected clock in capstate_test.go; these
// tests prove the WIRING (derive -> disambiguate -> hold/reopen) against a real ledger.

// TestLedgerAgingReopensStaleWeekly drives the aging valve at the carried-throttle seam: a
// seat whose only probe evidence is an 8-day-old LIMIT, still carrying an unparseable weekly
// and no daily leg, has outlived any real weekly window and must reopen. Fully deterministic
// — the episode age is read from the ledger timestamp, not the wall clock.
func TestLedgerAgingReopensStaleWeekly(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	// One blocked probe 8 days ago: too stale to be a fresh verdict, so the fold falls to the
	// carried throttle, and its timestamp is the episode's first-seen.
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "LIMIT",
		time.Now().UTC().Add(-8*24*time.Hour), `"weekly":"sometime never"`))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"weekly": "sometime never"},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("an 8-day-old weekly episode must age out and reopen, got %+v", st)
	}
}

// TestLedgerAgingHoldsYoungWeekly is the age-threshold control: the same shape but only
// 3 days into the episode still holds — aging must not fire early.
func TestLedgerAgingHoldsYoungWeekly(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "LIMIT",
		time.Now().UTC().Add(-3*24*time.Hour), `"weekly":"sometime never"`))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"weekly": "sometime never"},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("a 3-day-old weekly episode is still within the window and must hold, got %+v", st)
	}
}

// TestLedgerOKStreakOverridesStaleWeekly drives the probe-override cycle at the fresh-OK
// seam: two consecutive OK probes past a passed daily reset overturn a stale/unparseable
// weekly the seat has demonstrably outgrown, so a fresh OK finally reopens it.
func TestLedgerOKStreakOverridesStaleWeekly(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	now := time.Now().UTC()
	writeProbeLedger(t, rd,
		probeLine(t, ".claude-a", "OK", now.Add(-12*time.Minute), ""),
		probeLine(t, ".claude-a", "OK", now.Add(-3*time.Minute), ""), // latest is fresh -> led.Available
	)
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{
			"reset":  futureResetStr(-30 * time.Minute), // provably-passed daily reset
			"weekly": "sometime never",                  // unparseable -> fail-closed without the streak
		},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("a 2-OK streak past the daily reset must overturn the stale weekly, got %+v", st)
	}
	if st.StatusSource != "probe-ledger" {
		t.Fatalf("status_source = %q, want probe-ledger", st.StatusSource)
	}
}

// TestLedgerSingleOKHoldsUnparseableWeekly is the streak-threshold control at the same seam:
// a lone fresh OK (streak 1) is not enough to overturn the fail-closed weekly, so the seat
// stays walled. This is what keeps a single flukey probe from reopening a real cap.
func TestLedgerSingleOKHoldsUnparseableWeekly(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	now := time.Now().UTC()
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", now.Add(-3*time.Minute), ""))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{
			"reset":  futureResetStr(-30 * time.Minute),
			"weekly": "sometime never",
		},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("a lone fresh OK must not overturn the fail-closed weekly, got %+v", st)
	}
}

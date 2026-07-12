package fleetaccounts

import (
	"testing"
	"time"
)

// TestDeriveCapObservationOKStreak counts consecutive OK verdicts at the tail and stops at
// the first non-OK, ignoring OKs that precede an intervening block.
func TestDeriveCapObservationOKStreak(t *testing.T) {
	rd := t.TempDir()
	base := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)
	// chronological: OK, LIMIT, OK, OK  -> tail streak is 2 (the pre-block OK does not count)
	writeProbeLedger(t, rd,
		probeLine(t, ".claude-a", "OK", base.Add(-40*time.Minute), ""),
		probeLine(t, ".claude-a", "LIMIT", base.Add(-30*time.Minute), `"weekly":"sometime never"`),
		probeLine(t, ".claude-a", "OK", base.Add(-20*time.Minute), ""),
		probeLine(t, ".claude-a", "OK", base.Add(-10*time.Minute), ""),
	)
	obs := deriveCapObservation(".claude-a", rd)
	if obs.OKStreak != 2 {
		t.Fatalf("OKStreak = %d, want 2", obs.OKStreak)
	}
	if obs.HasFirstSeen {
		t.Fatalf("tail is OK, so no live episode: HasFirstSeen=%v FirstSeen=%v", obs.HasFirstSeen, obs.FirstSeen)
	}
}

// TestDeriveCapObservationEpisodeStart sets FirstSeen to the first entry of the trailing
// blocked run when the seat is currently blocked, and reports no OK streak.
func TestDeriveCapObservationEpisodeStart(t *testing.T) {
	rd := t.TempDir()
	base := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)
	episodeStart := base.Add(-3 * time.Hour)
	// OK, then a contiguous block episode: LIMIT, LIMIT (still blocked at the tail).
	writeProbeLedger(t, rd,
		probeLine(t, ".claude-a", "OK", base.Add(-5*time.Hour), ""),
		probeLine(t, ".claude-a", "LIMIT", episodeStart, `"weekly":"sometime never"`),
		probeLine(t, ".claude-a", "LIMIT", base.Add(-1*time.Hour), `"weekly":"sometime never"`),
	)
	obs := deriveCapObservation(".claude-a", rd)
	if obs.OKStreak != 0 {
		t.Fatalf("OKStreak = %d, want 0 (tail is blocked)", obs.OKStreak)
	}
	if !obs.HasFirstSeen || !obs.FirstSeen.Equal(episodeStart) {
		t.Fatalf("FirstSeen = %v (has=%v), want %v (episode start, not the pre-block OK)", obs.FirstSeen, obs.HasFirstSeen, episodeStart)
	}
}

// TestDeriveCapObservationNoHistory: an account the prober never touched yields the zero
// observation, which keeps DisambiguateCap on its legacy path.
func TestDeriveCapObservationNoHistory(t *testing.T) {
	rd := t.TempDir()
	writeProbeLedger(t, rd, probeLine(t, ".claude-other", "OK", time.Now().UTC(), ""))
	obs := deriveCapObservation(".claude-a", rd)
	if obs != (CapObservation{}) {
		t.Fatalf("no history should be the zero observation, got %+v", obs)
	}
}

// TestDeriveCapObservationFeedsDisambiguation is the end-to-end Phase-2 seam: a derived
// observation, handed to DisambiguateCap with a matching registry throttle, drives the
// override — proving the bridge produces exactly what the core consumes. (Phase 3 wires
// this pairing into computeRuntimeStatus; here we assert the two halves compose.)
func TestDeriveCapObservationFeedsDisambiguation(t *testing.T) {
	rd := t.TempDir()
	base := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd,
		probeLine(t, ".claude-a", "OK", base.Add(-20*time.Minute), ""),
		probeLine(t, ".claude-a", "OK", base.Add(-10*time.Minute), ""),
	)
	obs := deriveCapObservation(".claude-a", rd) // OKStreak 2, no episode
	// registry carries a passed daily reset + an unparseable weekly -> the streak overturns it.
	thr := map[string]any{"reset": base.Add(-4 * time.Hour).Format("Jan 2, 3:04pm"), "weekly": "sometime never"}
	cs := DisambiguateCap(thr, obs, base, DefaultCapPolicy())
	if cs.OverriddenBy != 2 || cs.WeeklyActive || cs.Active {
		t.Fatalf("derived streak should overturn the stale weekly: %+v", cs)
	}
}

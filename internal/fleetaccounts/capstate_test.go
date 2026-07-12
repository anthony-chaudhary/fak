package fleetaccounts

import (
	"testing"
	"time"
)

// capFixedNow is a mid-year, boundary-free instant so the dated reset strings the core
// parses ("Jun 13, 3:04pm") round-trip deterministically regardless of the wall clock.
var capFixedNow = time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)

// capResetAt renders a reset string resetTime parses for capFixedNow + d.
func capResetAt(d time.Duration) string {
	return capFixedNow.Add(d).Format("Jan 2, 3:04pm")
}

// TestDisambiguateCapZeroObsMatchesLegacy pins the base contract: with a zero CapObservation
// the core reproduces throttleIsActive / weeklyThrottleIsActive exactly, including the
// fail-closed rule that an unparseable/absent reset stays active.
func TestDisambiguateCapZeroObsMatchesLegacy(t *testing.T) {
	pol := DefaultCapPolicy()
	cases := []struct {
		name       string
		thr        map[string]any
		wantActive bool
		wantWeekly bool
		wantKind   CapKind
	}{
		{"future weekly + future daily", map[string]any{"reset": capResetAt(4 * time.Hour), "weekly": capResetAt(72 * time.Hour)}, true, true, CapWeekly},
		{"past weekly + future daily", map[string]any{"reset": capResetAt(4 * time.Hour), "weekly": capResetAt(-48 * time.Hour)}, true, false, CapDaily},
		{"past weekly + past daily", map[string]any{"reset": capResetAt(-4 * time.Hour), "weekly": capResetAt(-48 * time.Hour)}, false, false, CapNone},
		{"unparseable weekly holds fail-closed", map[string]any{"reset": capResetAt(-4 * time.Hour), "weekly": "sometime never"}, true, true, CapWeeklyUnknown},
		{"no weekly + future daily", map[string]any{"reset": capResetAt(4 * time.Hour)}, true, false, CapDaily},
		{"no weekly + past daily", map[string]any{"reset": capResetAt(-4 * time.Hour)}, false, false, CapNone},
		{"empty map is active fail-closed", map[string]any{}, true, false, CapDaily},
		{"no weekly + unparseable daily active fail-closed", map[string]any{"reset": "whenever"}, true, false, CapDaily},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cs := DisambiguateCap(tc.thr, CapObservation{}, capFixedNow, pol)
			if cs.Active != tc.wantActive || cs.WeeklyActive != tc.wantWeekly {
				t.Fatalf("Active=%v WeeklyActive=%v, want %v/%v", cs.Active, cs.WeeklyActive, tc.wantActive, tc.wantWeekly)
			}
			if cs.Kind != tc.wantKind {
				t.Fatalf("Kind=%v, want %v", cs.Kind, tc.wantKind)
			}
			if cs.AgedOut || cs.OverriddenBy != 0 {
				t.Fatalf("zero-obs must not age/override: AgedOut=%v OverriddenBy=%d", cs.AgedOut, cs.OverriddenBy)
			}
		})
	}
}

// TestDisambiguateCapAgingReleasesStaleWeekly: an episode older than WeeklyMaxAge releases a
// held weekly (it has outlived any real weekly window); the daily leg then decides, and an
// unknown daily no longer walls the seat.
func TestDisambiguateCapAgingReleasesStaleWeekly(t *testing.T) {
	pol := DefaultCapPolicy()
	old := CapObservation{FirstSeen: capFixedNow.Add(-8 * 24 * time.Hour), HasFirstSeen: true}
	young := CapObservation{FirstSeen: capFixedNow.Add(-6 * 24 * time.Hour), HasFirstSeen: true}

	// Unparseable weekly, no live daily -> aged out frees the seat entirely.
	cs := DisambiguateCap(map[string]any{"weekly": "sometime never"}, old, capFixedNow, pol)
	if !cs.AgedOut || cs.WeeklyActive || cs.Active {
		t.Fatalf("stale unparseable weekly should age out and free: %+v", cs)
	}
	if cs.Kind != CapNone {
		t.Fatalf("aged-out weekly with no daily: Kind=%v want CapNone", cs.Kind)
	}

	// Unparseable weekly but a genuinely future daily reset -> weekly ages out, daily holds.
	cs = DisambiguateCap(map[string]any{"reset": capResetAt(4 * time.Hour), "weekly": "sometime never"}, old, capFixedNow, pol)
	if !cs.AgedOut || cs.WeeklyActive || !cs.Active {
		t.Fatalf("aged weekly with future daily should hold on the daily leg: %+v", cs)
	}
	if cs.Kind != CapDaily {
		t.Fatalf("aged weekly + future daily: Kind=%v want CapDaily", cs.Kind)
	}

	// Younger than WeeklyMaxAge -> the hold still stands.
	cs = DisambiguateCap(map[string]any{"weekly": "sometime never"}, young, capFixedNow, pol)
	if cs.AgedOut || !cs.WeeklyActive || !cs.Active {
		t.Fatalf("weekly younger than max-age must still hold: %+v", cs)
	}
}

// TestDisambiguateCapProbeOverride: a run of fresh-OK probes past a passed daily reset
// overturns a stale weekly hold; a shorter streak, or a still-future daily, does not.
func TestDisambiguateCapProbeOverride(t *testing.T) {
	pol := DefaultCapPolicy() // OverrideStreak 2
	pastDaily := capResetAt(-4 * time.Hour)

	// Streak >= 2 past a passed daily reset -> overturned.
	cs := DisambiguateCap(map[string]any{"reset": pastDaily, "weekly": "sometime never"},
		CapObservation{OKStreak: 2}, capFixedNow, pol)
	if cs.OverriddenBy != 2 || cs.WeeklyActive || cs.Active {
		t.Fatalf("streak 2 past a passed daily should overturn the hold: %+v", cs)
	}

	// Streak of 1 is not enough -> the hold stands.
	cs = DisambiguateCap(map[string]any{"reset": pastDaily, "weekly": "sometime never"},
		CapObservation{OKStreak: 1}, capFixedNow, pol)
	if cs.OverriddenBy != 0 || !cs.WeeklyActive || !cs.Active {
		t.Fatalf("streak 1 must not overturn the hold: %+v", cs)
	}

	// A genuinely future weekly with no passed daily reset holds despite a long streak:
	// the override requires provably-past daily evidence, so a real weekly is never cut short.
	cs = DisambiguateCap(map[string]any{"reset": capResetAt(4 * time.Hour), "weekly": capResetAt(72 * time.Hour)},
		CapObservation{OKStreak: 5}, capFixedNow, pol)
	if cs.OverriddenBy != 0 || !cs.WeeklyActive || !cs.Active {
		t.Fatalf("a live weekly with a future daily must hold regardless of streak: %+v", cs)
	}

	// Aging takes precedence over override (override is gated on !AgedOut).
	old := CapObservation{FirstSeen: capFixedNow.Add(-8 * 24 * time.Hour), HasFirstSeen: true, OKStreak: 2}
	cs = DisambiguateCap(map[string]any{"reset": pastDaily, "weekly": "sometime never"}, old, capFixedNow, pol)
	if !cs.AgedOut || cs.OverriddenBy != 0 {
		t.Fatalf("aging should win over override: %+v", cs)
	}
}

// TestCapStateBlockReasonAndFreeUp pins the reason composition and the weekly-first free-up.
func TestCapStateBlockReasonAndFreeUp(t *testing.T) {
	cs := DisambiguateCap(map[string]any{"reset": "3pm", "weekly": "Mon 1pm"}, CapObservation{}, capFixedNow, DefaultCapPolicy())
	if cs.BlockReason != "usage limit; resets 3pm; weekly Mon 1pm" {
		t.Fatalf("block reason = %q", cs.BlockReason)
	}
	if cs.EffectiveFreeUp() != "Mon 1pm" {
		t.Fatalf("free-up should be weekly-first, got %q", cs.EffectiveFreeUp())
	}
	if got := (CapState{Reset: "3pm"}).EffectiveFreeUp(); got != "3pm" {
		t.Fatalf("no weekly -> daily reset, got %q", got)
	}
}

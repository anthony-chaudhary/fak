package cadencereport

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestLedgerLivenessDarkToLive is the revival witness: a ledger whose newest tick
// is older than DarkMultiple cadences reads DARK (the cadence loop loopfleet folds),
// and appending a fresh tick flips the SAME classifier to LIVE. This proves the
// dark->live transition from the producing side, against the same boundary
// loopfleet.foldCadence/classify applies.
func TestLedgerLivenessDarkToLive(t *testing.T) {
	// `now` matches the live diagnosis: the on-disk ledger's only row is stamped
	// 2026-06-27, three days before today, so it is past the 48h dark window.
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)

	// The stale ledger exactly as it sits on disk (one row, ~3 days old).
	staleLedger := `{"schema":"fak-cadence-ledger/1","date":"2026-06-27","commit":"9a10d29","generated_at":"2026-06-27T04:42:05Z","verdict":"OK","scores_debt":1295}`
	dark := LivenessOfLedger(staleLedger, now)
	if dark.State != loopmgr.HealthDark || !dark.Dark || dark.Live {
		t.Fatalf("a ~3d-old ledger must read DARK, got %+v", dark)
	}
	// Sanity: the age the verdict read is past the dark window (2 daily cadences).
	if dark.AgeSeconds <= 2*CadenceLivenessSeconds {
		t.Fatalf("dark age %ds should exceed the 2-cadence dark window %ds", dark.AgeSeconds, 2*CadenceLivenessSeconds)
	}

	// Record a fresh cadence tick: append a row stamped now. This is exactly what a
	// live `fak cadence --append-history` run writes (GeneratedAt = now).
	freshRow := LedgerRow{
		Schema:      LedgerSchema,
		Date:        now.Format("2006-01-02"),
		Commit:      "feedface",
		GeneratedAt: now.Format(time.RFC3339),
		Verdict:     "OK",
	}
	line, err := AppendLedgerLine(freshRow)
	if err != nil {
		t.Fatalf("AppendLedgerLine: %v", err)
	}
	revivedLedger := staleLedger + "\n" + line

	live := LivenessOfLedger(revivedLedger, now)
	if live.State != loopmgr.HealthLive || !live.Live || live.Dark {
		t.Fatalf("after a fresh tick the ledger must read LIVE, got %+v", live)
	}
	if live.AgeSeconds != 0 {
		t.Fatalf("a same-instant fresh tick should read age 0, got %ds", live.AgeSeconds)
	}

	// The transition is the point: the SAME ledger content, read by the SAME
	// classifier, went dark -> live solely because a current-stamped row landed.
	if dark.State == live.State {
		t.Fatalf("dark and live states must differ to witness the transition (both %q)", dark.State)
	}
}

// TestLivenessBoundariesMatchLoopfleet pins every branch of the classifier to the
// same live/stale/dark boundary loopfleet.classify uses (age<=cadence live,
// <=DarkMultiple*cadence stale, beyond dark; never-ticked dark), so the self-witness
// can never drift from the cross-ledger fold it mirrors.
func TestLivenessBoundariesMatchLoopfleet(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	cadence := CadenceLivenessSeconds
	at := func(secsAgo int64) int64 { return now.UnixNano() - secsAgo*int64(time.Second) }

	cases := []struct {
		name     string
		lastTick int64
		want     loopmgr.HealthState
	}{
		{"never ticked is dark", 0, loopmgr.HealthDark},
		{"within cadence is live", at(cadence / 2), loopmgr.HealthLive},
		{"exactly at cadence is live", at(cadence), loopmgr.HealthLive},
		{"just past cadence is stale", at(cadence + 1), loopmgr.HealthStale},
		{"exactly at dark boundary is stale", at(2 * cadence), loopmgr.HealthStale},
		{"past dark window is dark", at(2*cadence + 1), loopmgr.HealthDark},
		{"future tick clamps to live", now.UnixNano() + int64(time.Hour), loopmgr.HealthLive},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := livenessFromTick(c.lastTick, cadence, now, loopmgr.DefaultDarkMultiple)
			if got.State != c.want {
				t.Errorf("livenessFromTick state = %q, want %q", got.State, c.want)
			}
		})
	}
}

// TestLatestTickFoldsMaxNotLastLine proves the witness folds the NEWEST GeneratedAt
// across rows (as loopfleet does), so a fresh tick revives the loop even if a
// stale-stamped row is appended after it, and a garbled GeneratedAt is skipped.
func TestLatestTickFoldsMaxNotLastLine(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	rows := []LedgerRow{
		{Date: "2026-06-30", GeneratedAt: now.Format(time.RFC3339)}, // fresh
		{Date: "2026-06-27", GeneratedAt: "2026-06-27T04:42:05Z"},   // older, appended last
		{Date: "2026-06-29", GeneratedAt: "not-a-timestamp"},        // garbled, skipped
	}
	l := LedgerLiveness(rows, now)
	if l.State != loopmgr.HealthLive {
		t.Fatalf("newest tick should drive LIVE regardless of row order, got %+v", l)
	}
}

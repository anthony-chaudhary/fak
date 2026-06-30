package cadencereport

// liveness.go is the cadence loop's SELF-WITNESS: given the ledger this package
// produces, is the most-recent cadence tick fresh enough that the cross-ledger
// loopfleet fold (internal/loopfleet) will classify the cadence loop LIVE rather
// than DARK?
//
// The package already WRITES the durable liveness signal — every LedgerRow carries
// a GeneratedAt stamp, and loopfleet.foldCadence reads exactly that field as the
// loop's last tick, then classifies now-lastTick against a daily cadence and the
// dark multiple. But the package that OWNS the ledger had no way to READ that
// liveness back: it could record a tick and still not answer "did this revive the
// loop, or is the newest row already stale-to-dark?". That asymmetry is how the
// cadence loop went DARK silently — a tick is appended, but nothing on the
// producing side asserts the ledger now reads live.
//
// Liveness closes that loop. It MIRRORS loopfleet's contract exactly (the daily
// cadence horizon, loopmgr's HealthState vocabulary and DefaultDarkMultiple), so a
// row this package judges LIVE is the same row loopfleet judges live, and a
// dark->live transition is provable from the producing side by a focused test —
// not only observable after the fact in the cross-ledger pane.

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// CadenceLivenessSeconds is the interval a cadence tick is expected to land
// within. It is the SAME daily horizon loopfleet's cadence adapter applies
// (loopfleet.dailyCadence = 24h), pinned here so this package's self-witness and
// the cross-ledger fold draw the live/stale/dark line at the identical age.
const CadenceLivenessSeconds int64 = 24 * 60 * 60

// Liveness is the cadence loop's own read of its ledger freshness: the most-recent
// tick, its age against the cadence horizon, and the derived HealthState — the same
// vocabulary loopfleet uses, so "live" here means "live there".
type Liveness struct {
	// State is the derived verdict (live/stale/dark/unknown), loopmgr's vocabulary.
	State loopmgr.HealthState `json:"state"`
	// Dark is the surfaced boolean a scheduler gates on: true iff State == dark.
	Dark bool `json:"dark"`
	// Live is the surfaced boolean a post-tick assertion checks: true iff State == live.
	Live bool `json:"live"`
	// LastTickUnixNano is the newest ledger row's GeneratedAt as unix-nanos, 0 when
	// the ledger has no row carrying a parseable GeneratedAt (never ticked).
	LastTickUnixNano int64 `json:"last_tick_unix_nano,omitempty"`
	// AgeSeconds is now-lastTick in whole seconds; 0 when the loop has never ticked
	// (Dark by the never-ticked rule, not by age).
	AgeSeconds int64 `json:"age_seconds,omitempty"`
	// CadenceSeconds is the horizon the verdict compared the age against.
	CadenceSeconds int64 `json:"cadence_seconds"`
}

// LedgerLiveness classifies the freshness of the most-recent cadence tick on a
// parsed ledger, against `now`. It is PURE (the clock is supplied, nothing is read
// or mutated) so the dark->live transition is provable without touching disk.
//
// It reads the liveness signal the same way loopfleet.foldCadence does — the newest
// row's GeneratedAt is the last tick — and classifies it with the same rule:
//   - no row with a parseable GeneratedAt: DARK (ledgered/expected, never observed firing).
//   - tick age <= cadence:                  LIVE.
//   - cadence < age <= DarkMultiple*cadence: STALE (slipping).
//   - age > DarkMultiple*cadence:            DARK (gone quiet past its cadence).
func LedgerLiveness(rows []LedgerRow, now time.Time) Liveness {
	return livenessFromTick(latestTickUnixNano(rows), CadenceLivenessSeconds, now, loopmgr.DefaultDarkMultiple)
}

// LivenessOfLedger parses a raw JSONL ledger and classifies its freshness. It is
// the disk-reading convenience over LedgerLiveness, sharing ParseLedger so a
// hand-edited or partly-garbled ledger degrades to "the rows that parsed" rather
// than crashing the witness.
func LivenessOfLedger(content string, now time.Time) Liveness {
	return LedgerLiveness(ParseLedger(content), now)
}

// latestTickUnixNano returns the newest parseable GeneratedAt across the rows as
// unix-nanos (0 when no row carries one). loopfleet folds the max tick, not the
// last line, so a row order shuffle can't change the verdict; this mirrors that.
func latestTickUnixNano(rows []LedgerRow) int64 {
	var max int64
	for _, r := range rows {
		t, err := time.Parse(time.RFC3339, r.GeneratedAt)
		if err != nil {
			continue
		}
		if ns := t.UTC().UnixNano(); ns > max {
			max = ns
		}
	}
	return max
}

// livenessFromTick is the pure classifier shared by LedgerLiveness and its tests.
// It mirrors loopfleet.classify (which itself mirrors loopmgr.deriveState) so the
// three draw the dark line identically.
func livenessFromTick(lastTickUnixNano, cadenceSeconds int64, now time.Time, darkMultiple int64) Liveness {
	if darkMultiple < 1 {
		darkMultiple = loopmgr.DefaultDarkMultiple
	}
	l := Liveness{LastTickUnixNano: lastTickUnixNano, CadenceSeconds: cadenceSeconds}
	if lastTickUnixNano <= 0 {
		l.State = loopmgr.HealthDark
		l.Dark = true
		return l
	}
	ageNanos := now.UTC().UnixNano() - lastTickUnixNano
	if ageNanos < 0 {
		ageNanos = 0
	}
	l.AgeSeconds = ageNanos / int64(time.Second)
	if cadenceSeconds <= 0 {
		l.State = loopmgr.HealthUnknown
		return l
	}
	cadenceNanos := cadenceSeconds * int64(time.Second)
	darkNanos := darkMultiple * cadenceNanos
	switch {
	case ageNanos <= cadenceNanos:
		l.State = loopmgr.HealthLive
	case ageNanos <= darkNanos:
		l.State = loopmgr.HealthStale
	default:
		l.State = loopmgr.HealthDark
	}
	l.Dark = l.State == loopmgr.HealthDark
	l.Live = l.State == loopmgr.HealthLive
	return l
}

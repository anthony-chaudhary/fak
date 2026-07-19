package loopmgr

import (
	"testing"
	"time"
)

// The OS-scheduler rung (#4989). The ledger plane reads liveness from ONE input —
// the last appended row vs cadence — so a loop whose OS task fires successfully on
// cadence but appends no row on that tick reads DARK despite being demonstrably
// alive at the OS layer. These tests pin the promotion AND, more importantly, every
// fail-closed path: the rung's one unacceptable failure is fabricating liveness for
// a loop that really is dead.

// gardenCase builds the named real-world instance from the issue: loop
// `garden-stale-work-tick`, registered hourly, whose ledger last ticked ~15h ago
// (far past DarkMultiple*cadence -> DARK) because its OS task runs the read-only
// `fak garden --check`, which never calls witnessGardenTick.
func gardenCase(t *testing.T) (Status, Registry, time.Time) {
	t.Helper()
	now := time.Unix(1_000_000, 0).UTC()
	secAgo := func(s int64) int64 { return now.Add(-time.Duration(s) * time.Second).UnixNano() }

	st := Summarize([]Event{
		tick(gardenLoopID, EventStart, StatusRunning, secAgo(54_060)),
		tick(gardenLoopID, EventEnd, StatusClaimedDone, secAgo(54_000)), // ~15h stale
	}, now)

	reg := Registry{Jobs: map[string]Job{}}
	if err := reg.Put(Job{
		Schedule: Schedule{JobID: gardenLoopID, IntervalSeconds: 3600, MissedRun: MissedSkip},
		State:    JobArmed,
	}, now); err != nil {
		t.Fatalf("registry.Put: %v", err)
	}
	return st, reg, now
}

const gardenLoopID = "garden-stale-work-tick"

func rowFor(t *testing.T, rep HealthReport, id string) HealthRow {
	t.Helper()
	for _, r := range rep.Rows {
		if r.LoopID == id {
			return r
		}
	}
	t.Fatalf("no row for loop %q in %+v", id, rep.Rows)
	return HealthRow{}
}

// TestFoldHealthWithOS_FiredButNoLedgerRow is the headline failure class: the loop
// the ledger calls DARK, whose mapped OS task fired 0x0 ten minutes ago (well within
// its hourly cadence), must be surfaced as fired-but-no-ledger-row — not as a loop
// that is not running at all.
func TestFoldHealthWithOS_FiredButNoLedgerRow(t *testing.T) {
	st, reg, now := gardenCase(t)

	// Ledger plane alone: this is the false-DARK the issue reports.
	base := rowFor(t, FoldHealth(st, reg, now, HealthThresholds{}), gardenLoopID)
	if base.State != HealthDark || !base.Dark {
		t.Fatalf("precondition: ledger-only fold must read DARK, got state=%q dark=%v", base.State, base.Dark)
	}
	if base.OSFiredNoLedgerRow {
		t.Fatal("ledger-only fold must not claim an OS witness it never read")
	}

	rep := FoldHealthWithOS(st, reg, now, HealthThresholds{}, map[string]OSTaskInfo{
		gardenLoopID: {
			TaskLabel:       "FleetStaleWorkGarden",
			Fired:           true, // LastTaskResult 0x0
			LastRunUnixNano: now.Add(-10 * time.Minute).UnixNano(),
		},
	})
	row := rowFor(t, rep, gardenLoopID)

	if !row.OSFiredNoLedgerRow {
		t.Fatal("a ledger-dark loop whose mapped OS task fired 0x0 within cadence must read fired-but-no-ledger-row, not false-DARK")
	}
	if row.OSTaskLabel != "FleetStaleWorkGarden" {
		t.Errorf("OSTaskLabel = %q, want the corroborating task label", row.OSTaskLabel)
	}
	if row.OSLastRunUnixNano != now.Add(-10*time.Minute).UnixNano() {
		t.Errorf("OSLastRunUnixNano = %d, want the task's last-run time carried for the reader", row.OSLastRunUnixNano)
	}
	// The verdict sits ALONGSIDE liveness and must not overload it: the ledger gap is
	// a real fact and a --json consumer gating on Dark keeps its current meaning.
	if row.State != HealthDark || !row.Dark {
		t.Errorf("state/dark = %q/%v, want them left at dark/true (the rung must not overload the liveness verdict)", row.State, row.Dark)
	}
	if rep.Rollup.OSFiredNoLedgerRow != 1 {
		t.Errorf("rollup.OSFiredNoLedgerRow = %d, want 1", rep.Rollup.OSFiredNoLedgerRow)
	}
	if rep.Rollup.Dark != 1 {
		t.Errorf("rollup.Dark = %d, want 1 — the row is still dark in the ledger; the OS tally is a subset, not a sibling", rep.Rollup.Dark)
	}
}

// TestFoldHealthWithOS_FailsClosed pins the whole refusal surface. Every one of these
// must leave the loop DARK: an OS witness that is absent, unhealthy, unplaceable in
// time, or itself stale is NOT evidence the loop ran.
func TestFoldHealthWithOS_FailsClosed(t *testing.T) {
	st, reg, now := gardenCase(t)

	cases := []struct {
		name    string
		witness map[string]OSTaskInfo
	}{
		{
			name:    "no OS witness at all (nil map) — absence never fabricates liveness",
			witness: nil,
		},
		{
			name:    "loop has no mapped task — an unmapped loop is out of scope",
			witness: map[string]OSTaskInfo{"some-other-loop": {TaskLabel: "FleetOther", Fired: true, LastRunUnixNano: now.Add(-1 * time.Minute).UnixNano()}},
		},
		{
			name: "task ran but its result was not 0x0 — a failing task is not a witness",
			witness: map[string]OSTaskInfo{gardenLoopID: {
				TaskLabel: "FleetStaleWorkGarden", Fired: false,
				LastRunUnixNano: now.Add(-10 * time.Minute).UnixNano(),
			}},
		},
		{
			name: "task fired 0x0 but LastRunTime is unreadable — cannot prove it was WITHIN cadence",
			witness: map[string]OSTaskInfo{gardenLoopID: {
				TaskLabel: "FleetStaleWorkGarden", Fired: true, LastRunUnixNano: 0,
			}},
		},
		{
			name: "task fired 0x0 but its own last run is past cadence — the OS task stopped firing too",
			witness: map[string]OSTaskInfo{gardenLoopID: {
				TaskLabel: "FleetStaleWorkGarden", Fired: true,
				LastRunUnixNano: now.Add(-3 * time.Hour).UnixNano(), // cadence is 1h
			}},
		},
		{
			name: "task fired 0x0 but its last run is stamped in the FUTURE — a run that has not happened cannot corroborate a tick",
			witness: map[string]OSTaskInfo{gardenLoopID: {
				TaskLabel: "FleetStaleWorkGarden", Fired: true,
				LastRunUnixNano: now.Add(1 * time.Hour).UnixNano(), // ahead of now: was clamped to age 0 pre-fix, fabricating liveness
			}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := FoldHealthWithOS(st, reg, now, HealthThresholds{}, tc.witness)
			row := rowFor(t, rep, gardenLoopID)
			if row.OSFiredNoLedgerRow {
				t.Errorf("OSFiredNoLedgerRow = true, want false — this witness must not promote the loop out of DARK")
			}
			if row.State != HealthDark || !row.Dark {
				t.Errorf("state/dark = %q/%v, want dark/true (fail-closed)", row.State, row.Dark)
			}
			if row.OSTaskLabel != "" || row.OSLastRunUnixNano != 0 {
				t.Errorf("uncorroborated row carried OS evidence (%q/%d), want none", row.OSTaskLabel, row.OSLastRunUnixNano)
			}
			if rep.Rollup.OSFiredNoLedgerRow != 0 {
				t.Errorf("rollup.OSFiredNoLedgerRow = %d, want 0", rep.Rollup.OSFiredNoLedgerRow)
			}
		})
	}
}

// TestFoldHealthWithOS_OnlyOverlaysDark proves the rung is scoped to the state it
// exists to correct. A LIVE loop with a fresh OS witness must be untouched — the
// verdict is an explanation of a dark ledger, never a decoration on a healthy row.
func TestFoldHealthWithOS_OnlyOverlaysDark(t *testing.T) {
	now := time.Unix(1_000_000, 0).UTC()
	st := Summarize([]Event{
		tick(gardenLoopID, EventStart, StatusRunning, now.Add(-40*time.Second).UnixNano()),
		tick(gardenLoopID, EventEnd, StatusClaimedDone, now.Add(-30*time.Second).UnixNano()),
	}, now)
	reg := Registry{Jobs: map[string]Job{}}
	if err := reg.Put(Job{
		Schedule: Schedule{JobID: gardenLoopID, IntervalSeconds: 3600, MissedRun: MissedSkip},
		State:    JobArmed,
	}, now); err != nil {
		t.Fatalf("registry.Put: %v", err)
	}

	row := rowFor(t, FoldHealthWithOS(st, reg, now, HealthThresholds{}, map[string]OSTaskInfo{
		gardenLoopID: {TaskLabel: "FleetStaleWorkGarden", Fired: true, LastRunUnixNano: now.Add(-1 * time.Minute).UnixNano()},
	}), gardenLoopID)

	if row.State != HealthLive {
		t.Fatalf("state = %q, want live (precondition)", row.State)
	}
	if row.OSFiredNoLedgerRow || row.OSTaskLabel != "" {
		t.Error("the OS rung must only overlay a DARK row; a live loop must be left untouched")
	}
}

// TestFoldHealth_MatchesNilOSWitness pins the compatibility contract that keeps the
// existing ledger-only callers (internal/loopscore, `fak loop health` with the rung
// off) byte-identical: FoldHealth is exactly FoldHealthWithOS with no witness.
func TestFoldHealth_MatchesNilOSWitness(t *testing.T) {
	st, reg, now := gardenCase(t)
	if got, want := FoldHealthWithOS(st, reg, now, HealthThresholds{}, nil), FoldHealth(st, reg, now, HealthThresholds{}); !reportsEqual(got, want) {
		t.Fatalf("FoldHealthWithOS(nil) != FoldHealth:\n got=%+v\nwant=%+v", got, want)
	}
}

func reportsEqual(a, b HealthReport) bool {
	if a.Schema != b.Schema || a.TSUnixNano != b.TSUnixNano || len(a.Rows) != len(b.Rows) || a.Rollup != b.Rollup {
		return false
	}
	for i := range a.Rows {
		if a.Rows[i] != b.Rows[i] {
			return false
		}
	}
	return true
}

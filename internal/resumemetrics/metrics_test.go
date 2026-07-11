package resumemetrics

import "testing"

// TestReset_Floor pins the known-zero floor every other test starts from: after Reset the
// process-global registry reports no activity at all.
func TestReset_Floor(t *testing.T) {
	Reset()
	if Active() {
		t.Fatal("Active() true immediately after Reset")
	}
	got := Read()
	want := Snapshot{}
	if got.Ticks != want.Ticks || got.ProgressWitnessed != want.ProgressWitnessed ||
		got.Actions != nil || got.AutohealResults != nil || got.MonitorStatus != nil || got.HealthRollup != "" {
		t.Fatalf("Read() after Reset not a clean floor: %+v", got)
	}
}

// TestRecorders_EachMovesItsCounter is the #3803 acceptance in miniature: every recorder moves
// its OWN named counter and only that one, and the moved value is authoritative in-process (no
// ledger involved).
func TestRecorders_EachMovesItsCounter(t *testing.T) {
	Reset()

	Tick()
	Tick()
	RecordAction("launch")
	RecordAction("launch")
	RecordAction("skip")
	RecordAutohealResult("restarted")
	SetMonitorStatus("resume_watchdog", "green")
	SetHealthRollup("degraded")
	ProgressWitnessed()

	if !Active() {
		t.Fatal("Active() false after recording activity")
	}
	got := Read()

	if got.Ticks != 2 {
		t.Errorf("Ticks = %d, want 2", got.Ticks)
	}
	if got.ProgressWitnessed != 1 {
		t.Errorf("ProgressWitnessed = %d, want 1", got.ProgressWitnessed)
	}
	if got.Actions["launch"] != 2 || got.Actions["skip"] != 1 {
		t.Errorf("Actions = %v, want launch:2 skip:1", got.Actions)
	}
	if got.AutohealResults["restarted"] != 1 {
		t.Errorf("AutohealResults = %v, want restarted:1", got.AutohealResults)
	}
	if got.MonitorStatus["resume_watchdog"] != "green" {
		t.Errorf("MonitorStatus = %v, want resume_watchdog:green", got.MonitorStatus)
	}
	if got.HealthRollup != "degraded" {
		t.Errorf("HealthRollup = %q, want degraded", got.HealthRollup)
	}
}

// TestDriveCarry is the #4138 acceptance gate: the drive-carry outcome counter splits a
// watchdog relaunch that RESTORED its carried budget from one that RESET to a fresh cap, the
// split is authoritative in-process, and recording either outcome makes the surface Active.
func TestDriveCarry(t *testing.T) {
	Reset()
	if Read().DriveCarry != nil {
		t.Fatalf("DriveCarry not nil at the clean floor: %v", Read().DriveCarry)
	}

	RecordDriveCarry("restored")
	RecordDriveCarry("reset")

	if !Active() {
		t.Fatal("Active() false after recording a drive-carry outcome")
	}
	got := Read()
	if got.DriveCarry["restored"] != 1 || got.DriveCarry["reset"] != 1 {
		t.Errorf("DriveCarry = %v, want restored:1 reset:1", got.DriveCarry)
	}

	// Norm folds a blank outcome to the nameable "unknown" bucket, never a phantom key.
	RecordDriveCarry("  RESTORED ")
	RecordDriveCarry("")
	got = Read()
	if got.DriveCarry["restored"] != 2 {
		t.Errorf("DriveCarry[restored] = %d, want 2 after a padded+uppercased repeat", got.DriveCarry["restored"])
	}
	if got.DriveCarry["unknown"] != 1 {
		t.Errorf("DriveCarry[unknown] = %d, want 1 for the empty outcome", got.DriveCarry["unknown"])
	}

	// Reset returns the bucket map to the nil floor, no cross-test bleed.
	Reset()
	if Read().DriveCarry != nil {
		t.Errorf("DriveCarry not nil after Reset: %v", Read().DriveCarry)
	}
}

// TestNorm_ClosedTokens pins the token folding: verdicts are trimmed+lowercased and an empty
// token never creates an unnameable bucket — it folds to "unknown".
func TestNorm_ClosedTokens(t *testing.T) {
	Reset()
	RecordAction("  LAUNCH ")
	RecordAction("")
	SetMonitorStatus("", "  RED ")
	SetHealthRollup("")

	got := Read()
	if got.Actions["launch"] != 1 {
		t.Errorf("expected normalized 'launch':1, got %v", got.Actions)
	}
	if got.Actions["unknown"] != 1 {
		t.Errorf("expected empty action folded to 'unknown':1, got %v", got.Actions)
	}
	if got.MonitorStatus["unknown"] != "red" {
		t.Errorf("expected empty monitor name folded to 'unknown' with 'red', got %v", got.MonitorStatus)
	}
	if got.HealthRollup != "unknown" {
		t.Errorf("expected empty rollup folded to 'unknown', got %q", got.HealthRollup)
	}
}

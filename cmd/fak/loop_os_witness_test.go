package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// The OS-scheduler rung end-to-end through the real `fak loop health` surface
// (#4989), witnessed on the named instance the operator reported: loop
// `garden-stale-work-tick` <-> task `FleetStaleWorkGarden`.
//
// These are RENDER witnesses on purpose. The defect is a word an operator reads in a
// pane — "dark-loop", meaning "not ticking anywhere" — printed for a loop that is
// demonstrably firing at the OS layer. So the proof captures the bytes the surface
// emits and asserts the wrong word is gone and the right one is there; a unit test
// on the fold alone would never touch the thing that was actually wrong.

// gardenHealthFixture writes a ledger whose garden loop last ticked in 1970 (so it is
// DARK against any real `now`) plus an hourly registry entry for it — the shape the
// real box is in when `fak garden --check` fires hourly without ever calling
// witnessGardenTick.
func gardenHealthFixture(t *testing.T) (ledger, registry string) {
	t.Helper()
	dir := t.TempDir()
	ledger = filepath.Join(dir, "loops.jsonl")
	registry = filepath.Join(dir, "loop-registry.json")

	appendLoopTestEventAt(t, ledger, loopmgr.Event{
		LoopID: gardenTickLoopID,
		Kind:   loopmgr.EventFire,
	}, int64(time.Second))

	reg := loopmgr.Registry{Jobs: map[string]loopmgr.Job{}}
	if err := reg.Put(loopmgr.Job{
		Schedule: loopmgr.Schedule{
			JobID:           gardenTickLoopID,
			IntervalSeconds: gardenTickIntervalSeconds,
			MissedRun:       loopmgr.MissedSkip,
		},
		State: loopmgr.JobArmed,
	}, time.Unix(0, 0).UTC()); err != nil {
		t.Fatalf("registry.Put: %v", err)
	}
	if err := loopmgr.SaveRegistry(registry, reg); err != nil {
		t.Fatalf("SaveRegistry: %v", err)
	}
	return ledger, registry
}

// schedSnapshot writes a captured schedscan probe payload for FleetStaleWorkGarden
// with the given last result + last-run age — the same JSON shape the live
// PowerShell probe emits, which is why --sched-from can stand in for it off-Windows.
func schedSnapshot(t *testing.T, lastResult int64, ranAgo time.Duration) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "sched.json")
	body := fmt.Sprintf(`[{
	  "TaskName": %q,
	  "TaskPath": "\\",
	  "State": "Ready",
	  "LogonType": "S4U",
	  "LastRunTime": %q,
	  "LastTaskResult": %d,
	  "NextRunTime": %q,
	  "NumberOfMissedRuns": 0
	}]`, gardenTickTaskLabel,
		time.Now().UTC().Add(-ranAgo).Format(time.RFC3339Nano), lastResult,
		time.Now().UTC().Add(time.Hour).Format(time.RFC3339Nano))
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write sched snapshot: %v", err)
	}
	return path
}

func runLoopHealthCapture(t *testing.T, argv ...string) string {
	t.Helper()
	var stdout, stderr bytes.Buffer
	if code := runLoop(&stdout, &stderr, append([]string{"health"}, argv...)); code != 0 {
		t.Fatalf("loop health code=%d stderr=%s", code, stderr.String())
	}
	return stdout.String()
}

// TestLoopHealthRendersOSLiveLedgerDarkNotFalseDark is the headline witness: with the
// OS rung on, the garden loop's mapped task shows LastTaskResult 0x0 ten minutes ago
// — inside its hourly cadence — so the pane must stop calling it dark-loop.
func TestLoopHealthRendersOSLiveLedgerDarkNotFalseDark(t *testing.T) {
	ledger, registry := gardenHealthFixture(t)

	// Precondition — the reported defect, captured before the rung is engaged.
	base := runLoopHealthCapture(t, "--ledger", ledger, "--registry", registry)
	if !strings.Contains(base, "dark-loop") {
		t.Fatalf("precondition: the ledger-only pane must render the false-DARK this rung corrects:\n%s", base)
	}

	out := runLoopHealthCapture(t, "--ledger", ledger, "--registry", registry,
		"--sched-from", schedSnapshot(t, 0x0, 10*time.Minute))

	if !strings.Contains(out, "os-live-ledger-dark") {
		t.Fatalf("pane must surface the fired-but-no-ledger-row state for an OS-alive loop:\n%s", out)
	}
	if strings.Contains(out, "dark-loop") {
		t.Fatalf("pane still calls an OS-alive loop dark-loop — the false-DARK is not fixed:\n%s", out)
	}
	if !strings.Contains(out, "os-live-ledger-dark=1") {
		t.Fatalf("summary must tally the OS-corroborated loops:\n%s", out)
	}
}

// TestLoopHealthOSRungFailsClosedInPane pins the fail-closed half at the SURFACE: a
// task that did not report 0x0, or whose last run is unreadable/absent, must leave
// the operator looking at dark-loop. Fabricating liveness here would be worse than
// the false-DARK it replaces.
func TestLoopHealthOSRungFailsClosedInPane(t *testing.T) {
	ledger, registry := gardenHealthFixture(t)

	cases := []struct {
		name string
		snap string
	}{
		{
			name: "last run was refused (0x800710E0) — a failing task is not a liveness witness",
			snap: schedSnapshot(t, 0x800710E0, 10*time.Minute),
		},
		{
			name: "task action exited non-zero (0x1)",
			snap: schedSnapshot(t, 0x1, 10*time.Minute),
		},
		{
			name: "fired 0x0 but its own last run is past cadence — the task stopped firing too",
			snap: schedSnapshot(t, 0x0, 5*time.Hour),
		},
		{
			name: "0x41300 'ready to run at its next scheduled time' is a STATUS, not a run",
			snap: schedSnapshot(t, 0x41300, 10*time.Minute),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := runLoopHealthCapture(t, "--ledger", ledger, "--registry", registry, "--sched-from", tc.snap)
			if strings.Contains(out, "os-live-ledger-dark") {
				t.Fatalf("rung promoted a loop on a witness that proves nothing:\n%s", out)
			}
			if !strings.Contains(out, "dark-loop") {
				t.Fatalf("loop must stay dark-loop when the OS witness fails closed:\n%s", out)
			}
		})
	}
}

// TestLoopHealthOSRungDegradesOnUnreadableSnapshot: losing the OS input must cost the
// re-description, never the pane. The fold still renders from the ledger plane.
func TestLoopHealthOSRungDegradesOnUnreadableSnapshot(t *testing.T) {
	ledger, registry := gardenHealthFixture(t)
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"health", "--ledger", ledger, "--registry", registry,
		"--sched-from", filepath.Join(t.TempDir(), "does-not-exist.json")})
	if code != 0 {
		t.Fatalf("an unreadable OS snapshot must degrade, not fail the fold: code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "dark-loop") {
		t.Fatalf("ledger-plane verdict must survive a missing OS witness:\n%s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "OS-task rung skipped") {
		t.Errorf("a skipped rung must be surfaced, never silent: stderr=%s", stderr.String())
	}
}

// TestLoopHealthOSRungJSONCarriesEvidence: the --json contract carries the verdict and
// the evidence behind it, so a machine consumer sees WHICH task corroborated, and sees
// dark stay true (the rung sits alongside the liveness verdict, it does not overload it).
func TestLoopHealthOSRungJSONCarriesEvidence(t *testing.T) {
	ledger, registry := gardenHealthFixture(t)
	out := runLoopHealthCapture(t, "--ledger", ledger, "--registry", registry,
		"--sched-from", schedSnapshot(t, 0x0, 10*time.Minute), "--json")

	var rep loopmgr.HealthReport
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("unmarshal health: %v\n%s", err, out)
	}
	if len(rep.Rows) != 1 {
		t.Fatalf("rows = %+v, want the single garden row", rep.Rows)
	}
	row := rep.Rows[0]
	if !row.OSFiredNoLedgerRow {
		t.Error("os_fired_no_ledger_row must be true for the OS-corroborated loop")
	}
	if row.OSTaskLabel != gardenTickTaskLabel {
		t.Errorf("os_task_label = %q, want %q", row.OSTaskLabel, gardenTickTaskLabel)
	}
	if row.OSLastRunUnixNano == 0 {
		t.Error("os_last_run_unix_nano must carry the corroborating run time")
	}
	if !row.Dark || row.State != loopmgr.HealthDark {
		t.Error("dark/state must stay true/dark — the rung must not overload the liveness verdict a consumer already gates on")
	}
	if rep.Rollup.OSFiredNoLedgerRow != 1 {
		t.Errorf("rollup.os_fired_no_ledger_row = %d, want 1", rep.Rollup.OSFiredNoLedgerRow)
	}
}

// TestLoopOSTaskMapBindsGardenIdentity pins the explicit identity map. The join is
// keyed by this table and nothing else — no name-similarity inference — so the map
// itself is the contract worth a test.
func TestLoopOSTaskMapBindsGardenIdentity(t *testing.T) {
	if got := loopOSTaskMap()[gardenTickLoopID]; got != gardenTickTaskLabel {
		t.Fatalf("loopOSTaskMap()[%q] = %q, want %q", gardenTickLoopID, got, gardenTickTaskLabel)
	}
}

// TestLoopOSWitnessesDecode pins the schedscan->witness decode, including the two
// traps: an unmapped task must never produce a witness, and 0x0 must be the ONLY
// result that sets Fired.
func TestLoopOSWitnessesDecode(t *testing.T) {
	ran := time.Now().UTC().Add(-time.Minute).Format(time.RFC3339Nano)
	rows := []schedScanTaskInfo{
		{TaskName: gardenTickTaskLabel, LastTaskResult: 0x0, LastRunTime: ran},
		{TaskName: "FleetSomethingElse", LastTaskResult: 0x0, LastRunTime: ran},
	}
	got := loopOSWitnesses(rows, loopOSTaskMap())

	if len(got) != 1 {
		t.Fatalf("witnesses = %+v, want only the mapped garden loop (an unmapped task is out of scope)", got)
	}
	w := got[gardenTickLoopID]
	if !w.Fired || w.TaskLabel != gardenTickTaskLabel || w.LastRunUnixNano == 0 {
		t.Fatalf("garden witness = %+v, want a fired 0x0 witness with a parsed last-run time", w)
	}

	// 0x41300 decodes to schedscan severity "ok", but it means "ready to run at its
	// next scheduled time" — no run happened. Fired must key on the code, not severity.
	ready := loopOSWitnesses([]schedScanTaskInfo{
		{TaskName: gardenTickTaskLabel, LastTaskResult: 0x41300, LastRunTime: ran},
	}, loopOSTaskMap())
	if ready[gardenTickLoopID].Fired {
		t.Error("0x41300 (ready to run) must not set Fired — it is a status, not a completed run")
	}

	// An unparseable LastRunTime must leave the witness unplaceable in time (0), which
	// the fold refuses to corroborate, rather than defaulting to a 1970 timestamp.
	bad := loopOSWitnesses([]schedScanTaskInfo{
		{TaskName: gardenTickTaskLabel, LastTaskResult: 0x0, LastRunTime: "not-a-time"},
	}, loopOSTaskMap())
	if bad[gardenTickLoopID].LastRunUnixNano != 0 {
		t.Error("an unparseable LastRunTime must yield 0 (unplaceable), never a fabricated instant")
	}
}

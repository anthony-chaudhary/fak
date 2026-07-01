package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestWorkerHeartbeatEventStartedVsNeverStarted proves the pure event builder maps
// Classification.Started() onto the two Reason values `fak dispatch audit
// --heartbeat` promises: STARTED (running) vs NEVER_STARTED (failed) — the witness
// distinction issue #1782 asks for, without any I/O.
func TestWorkerHeartbeatEventStartedVsNeverStarted(t *testing.T) {
	started := dispatchaudit.Classify(dispatchaudit.Worker{Log: "resolve-1.log", CommitSHA: "abc1234"}, dispatchaudit.DefaultThresholds())
	ev := workerHeartbeatEvent(started)
	if ev.Kind != loopmgr.EventHeartbeat || ev.Reason != "STARTED" || ev.Status != loopmgr.StatusRunning {
		t.Fatalf("started event = %+v, want kind=heartbeat reason=STARTED status=running", ev)
	}

	neverStarted := dispatchaudit.Classify(dispatchaudit.Worker{Log: "resolve-2.log", LogSizeKnown: true, LogBytes: 0, PIDAlive: false}, dispatchaudit.DefaultThresholds())
	ev = workerHeartbeatEvent(neverStarted)
	if ev.Kind != loopmgr.EventHeartbeat || ev.Reason != "NEVER_STARTED" || ev.Status != loopmgr.StatusFailed {
		t.Fatalf("never-started event = %+v, want kind=heartbeat reason=NEVER_STARTED status=failed", ev)
	}
}

// TestRunDispatchAuditHeartbeatRecordsStartedRow is the end-to-end witness: a real
// .dispatch-runs/ fixture with one shipped worker (real evidence it ran) and one
// dead-PID/zero-byte-log worker (no evidence it ever ran) goes through the CLI's
// opt-in --heartbeat flag, and the loop ledger comes out with exactly one STARTED
// and one NEVER_STARTED row.
func TestRunDispatchAuditHeartbeatRecordsStartedRow(t *testing.T) {
	runsDir := t.TempDir()
	writeHeartbeatFixture(t, runsDir, "resolve-100-20260628-105439.log",
		"# fak-spawn 20260628-105439 issue=100 lane=tools backend=claude argv0=claude\n"+
			"working...\n"+
			"✅ Commit created: `b68ead49` - implements the thing (closes #100)\n")
	writeHeartbeatFixture(t, runsDir, "resolve-100-20260628-105439.backend", "claude")
	// A dead-PID, zero-byte log: no structural evidence this worker ever started.
	writeHeartbeatFixture(t, runsDir, "resolve-200-20260628-110000.log", "")

	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr strings.Builder
	code := runDispatchAudit(&stdout, &stderr, []string{"--runs-dir", runsDir, "--heartbeat", "--ledger", ledger})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}

	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load ledger: %v", err)
	}
	var started, neverStarted int
	for _, ev := range events {
		if ev.Kind != loopmgr.EventHeartbeat {
			continue
		}
		switch ev.Reason {
		case "STARTED":
			started++
		case "NEVER_STARTED":
			neverStarted++
		}
	}
	if started != 1 {
		t.Fatalf("started heartbeats = %d, want 1 (events: %+v)", started, events)
	}
	if neverStarted != 1 {
		t.Fatalf("never-started heartbeats = %d, want 1 (events: %+v)", neverStarted, events)
	}
}

func writeHeartbeatFixture(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture %s: %v", name, err)
	}
}

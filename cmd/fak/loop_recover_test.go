package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/looprecover"
)

// runLoopRecoverAt drives the recover core directly and returns stdout, stderr, exit code.
func runLoopRecoverAt(argv ...string) (string, string, int) {
	var out, errb bytes.Buffer
	code := runLoopRecover(&out, &errb, argv)
	return out.String(), errb.String(), code
}

// appendEv writes one valid (hash-chained) ledger event at a fixed clock.
func appendEv(t *testing.T, path, runID string, kind loopmgr.EventKind, status loopmgr.RunStatus, atSec int64) {
	t.Helper()
	ev := loopmgr.Event{LoopID: "issue-resolve-dispatch", RunID: runID, Kind: kind, Status: status, Summary: "#" + runID}
	if _, err := loopmgr.Append(path, ev, loopmgr.WithClock(func() time.Time { return time.Unix(atSec, 0).UTC() })); err != nil {
		t.Fatalf("append %s/%s: %v", runID, kind, err)
	}
}

// recoverLedger builds a ledger with one run of each recovery-relevant shape and returns its
// path. now is the test clock; stale window in the tests is 30 min (1800s).
func recoverLedger(t *testing.T, now int64) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendEv(t, path, "r-orphan", loopmgr.EventStart, "", now-5000)                             // started, silent 83m -> orphaned
	appendEv(t, path, "r-live", loopmgr.EventStart, "", now-300)                                // started, recent -> running
	appendEv(t, path, "r-witness", loopmgr.EventStart, "", now-4000)                            // started ...
	appendEv(t, path, "r-witness", loopmgr.EventWitness, loopmgr.StatusWitnessedDone, now-3900) // ... witnessed -> complete
	appendEv(t, path, "r-unwit", loopmgr.EventStart, "", now-4000)                              // started ...
	appendEv(t, path, "r-unwit", loopmgr.EventEnd, "", now-3900)                                // ... ended, no witness -> unwitnessed
	return path
}

// TestLoopRecoverWorklist is the goal in one test: the recover fold flags the orphaned and
// unwitnessed runs (and only those), leaving the witnessed run complete and the recent run
// running.
func TestLoopRecoverWorklist(t *testing.T) {
	const now = 2_000_000
	path := recoverLedger(t, now)
	out, errb, code := runLoopRecoverAt("--ledger", path, "--now", "2000000", "--stale-min", "30", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var res looprecover.Result
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if res.OrphanedCount != 1 || res.UnwitnessedCount != 1 || res.CompleteCount != 1 || res.RunningCount != 1 {
		t.Errorf("counts = orphaned %d unwit %d complete %d running %d, want 1/1/1/1",
			res.OrphanedCount, res.UnwitnessedCount, res.CompleteCount, res.RunningCount)
	}
	// The worklist is orphaned-first then unwitnessed; both, and only both, are recovery candidates.
	want := []string{"r-orphan", "r-unwit"}
	if strings.Join(res.Recover, ",") != strings.Join(want, ",") {
		t.Errorf("recover = %v, want %v", res.Recover, want)
	}
}

// TestLoopRecoverTable: the human table names the worklist and the recover summary.
func TestLoopRecoverTable(t *testing.T) {
	const now = 2_000_000
	path := recoverLedger(t, now)
	out, _, code := runLoopRecoverAt("--ledger", path, "--now", "2000000", "--stale-min", "30")
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	for _, want := range []string{"orphaned", "unwitnessed", "r-orphan", "recover 2 run(s)"} {
		if !strings.Contains(out, want) {
			t.Errorf("table missing %q:\n%s", want, out)
		}
	}
	// The default (non --all) view does not list the healthy runs.
	if strings.Contains(out, "r-live") {
		t.Errorf("default view should not list the running run:\n%s", out)
	}
}

// TestLoopRecoverWiredIntoLoop: `fak loop recover` routes through the loop switch.
func TestLoopRecoverWiredIntoLoop(t *testing.T) {
	const now = 2_000_000
	path := recoverLedger(t, now)
	var out, errb bytes.Buffer
	if code := runLoop(&out, &errb, []string{"recover", "--ledger", path, "--now", "2000000", "--json"}); code != 0 {
		t.Fatalf("fak loop recover routed exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(out.String(), `"recover"`) {
		t.Errorf("routed output missing recover worklist:\n%s", out.String())
	}
}

// TestLoopRecoverStaleTuning: a wider stale window keeps the silent run running; a narrow one
// orphans it.
func TestLoopRecoverStaleTuning(t *testing.T) {
	const now = 2_000_000
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendEv(t, path, "r", loopmgr.EventStart, "", now-1200) // silent 20 min
	wide, _, _ := runLoopRecoverAt("--ledger", path, "--now", "2000000", "--stale-min", "30", "--json")
	narrow, _, _ := runLoopRecoverAt("--ledger", path, "--now", "2000000", "--stale-min", "10", "--json")
	var w, n looprecover.Result
	_ = json.Unmarshal([]byte(wide), &w)
	_ = json.Unmarshal([]byte(narrow), &n)
	if w.OrphanedCount != 0 {
		t.Errorf("30-min window: orphaned %d, want 0 (silent only 20 min)", w.OrphanedCount)
	}
	if n.OrphanedCount != 1 {
		t.Errorf("10-min window: orphaned %d, want 1 (silent 20 min exceeds it)", n.OrphanedCount)
	}
}

// TestLoopRecoverErrors: a missing ledger is the empty (no-runs) case, not an error — an
// operator running recover before any dispatch should see "no runs", exit 0; a stray arg is a
// usage error (2).
func TestLoopRecoverMissingLedgerIsEmpty(t *testing.T) {
	out, _, code := runLoopRecoverAt("--ledger", filepath.Join(t.TempDir(), "nope.jsonl"))
	if code != 0 {
		t.Errorf("missing ledger: exit = %d, want 0 (empty, not an error)", code)
	}
	if !strings.Contains(out, "no runs to recover") {
		t.Errorf("missing ledger should render the empty case:\n%s", out)
	}
}

func TestLoopRecoverStrayArg(t *testing.T) {
	if _, _, code := runLoopRecoverAt("stray"); code != 2 {
		t.Errorf("unexpected arg: exit = %d, want 2", code)
	}
}

// TestLoopRecoverForkedLedgerSurvives pins the separation-of-concerns fix: a forked seq chain
// (a concurrent double-append to the append-only audit log) must NOT take recover down. The
// tolerant LoadPrefix recovers the valid prefix, the break is surfaced as a finding on stderr
// (logging), the worklist is planned from the prefix (default working behavior), and the audit
// log is never rewritten — exit 0, not 1 (the strict loader's behavior before the fix).
func TestLoopRecoverForkedLedgerSurvives(t *testing.T) {
	const now = 2_000_000
	path := recoverLedger(t, now) // 6 valid hash-chained events
	// Append a raw line that collides on seq (the real-world fork: two loops appended
	// concurrently and both got the same seq). This breaks the strict chain at this line.
	forkLine := `{"schema":"fak.loop-event.v1","seq":1,"loop_id":"x","kind":"fire","prev_hash":"deadbeef","hash":"00"}` + "\n"
	if err := appendRawLine(t, path, forkLine); err != nil {
		t.Fatalf("append fork line: %v", err)
	}

	out, errb, code := runLoopRecoverAt("--ledger", path, "--now", "2000000", "--stale-min", "30", "--json")
	if code != 0 {
		t.Fatalf("forked ledger: exit = %d, want 0 (recover must survive a fork; stderr: %s)", code, errb)
	}
	if !strings.Contains(errb, "integrity break") {
		t.Errorf("forked ledger: stderr should report the integrity break as a finding, got: %s", errb)
	}
	if !strings.Contains(errb, "audit log is left intact") {
		t.Errorf("forked ledger: stderr should state the audit log is not rewritten, got: %s", errb)
	}
	// The worklist is still planned from the recovered prefix (the 6 valid events).
	var res looprecover.Result
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if res.OrphanedCount != 1 || res.UnwitnessedCount != 1 {
		t.Errorf("forked ledger recovered prefix counts = orphaned %d unwit %d, want 1/1", res.OrphanedCount, res.UnwitnessedCount)
	}

	// The control-pane envelope folds the break into the reason and stays ok:true (advisory).
	cp, _, cpCode := runLoopRecoverAt("--ledger", path, "--now", "2000000", "--stale-min", "30", "--control-pane")
	if cpCode != 0 {
		t.Fatalf("control-pane on forked ledger: exit = %d, want 0", cpCode)
	}
	var env map[string]any
	if err := json.Unmarshal([]byte(cp), &env); err != nil {
		t.Fatalf("control-pane output is not valid JSON: %v\n%s", err, cp)
	}
	if env["ok"] != true {
		t.Errorf("control-pane ok = %v, want true (a found orphan / a fork is advisory, not a red)", env["ok"])
	}
	if env["ledger_broken"] != true {
		t.Errorf("control-pane ledger_broken = %v, want true", env["ledger_broken"])
	}
	if reason, _ := env["reason"].(string); !strings.Contains(reason, "integrity break") {
		t.Errorf("control-pane reason should fold the integrity break, got: %v", env["reason"])
	}
}

// appendRawLine appends a raw (possibly chain-breaking) line to a ledger file, bypassing
// loopmgr.Append's hash-chaining — the only way to synthesize a forked ledger in a test.
func appendRawLine(t *testing.T, path, line string) error {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line)
	return err
}

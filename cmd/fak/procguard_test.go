package main

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// TestLoadLeaseLiveness witnesses the dead-owner reaper's production lease lookup:
// a recently-active run reads ALIVE, a run silent past the stale window reads
// DEAD (orphaned owner), a witnessed/terminal run reads DEAD (owner gone), and a
// run id the ledger never recorded reads DEAD (registry row absent). This is the
// LeaseAlive that keys procguard.ClassifyDeadOwnerOrphans in `fak process-guard
// report --reap-dead-owner`. Issue #3596.
func TestLoadLeaseLiveness(t *testing.T) {
	now := time.Now().Unix()
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendEv(t, path, "r-live", loopmgr.EventStart, "", now-60)                              // recent -> running -> alive
	appendEv(t, path, "r-dead", loopmgr.EventStart, "", now-5000)                            // silent past stale -> orphaned -> dead
	appendEv(t, path, "r-done", loopmgr.EventStart, "", now-4000)                            // started ...
	appendEv(t, path, "r-done", loopmgr.EventWitness, loopmgr.StatusWitnessedDone, now-3900) // ... witnessed -> complete -> dead

	leaseAlive, note := loadLeaseLiveness(path)
	if note != "" {
		t.Fatalf("readable ledger must not skip the mode: %s", note)
	}
	if leaseAlive == nil {
		t.Fatal("readable ledger must yield a lease lookup")
	}
	if !leaseAlive("r-live") {
		t.Fatalf("a recently-active run must read ALIVE")
	}
	if leaseAlive("r-dead") {
		t.Fatalf("a run silent past the stale window must read DEAD")
	}
	if leaseAlive("r-done") {
		t.Fatalf("a witnessed/terminal run must read DEAD (owner gone)")
	}
	if leaseAlive("r-never-seen") {
		t.Fatalf("a run id absent from the ledger must read DEAD (registry row absent)")
	}
}

// TestLoadLeaseLivenessFailClosed: an unreadable ledger path must SKIP the mode
// (return a nil lookup + a note), never treat every owner as dead.
func TestLoadLeaseLivenessFailClosed(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist", "loops.jsonl")
	leaseAlive, note := loadLeaseLiveness(missing)
	if leaseAlive != nil {
		t.Fatalf("an unreadable ledger must yield a nil lookup (skip the mode), got non-nil")
	}
	if note == "" {
		t.Fatalf("an unreadable ledger must carry a skip note for the operator")
	}
}

func TestClassifyHungDispatchOrphans(t *testing.T) {
	age := 3600
	idleCPU := 0.1
	activeCPU := 8.0
	threads := 12
	handles := 80
	ws := 130
	parent := 10
	deadOwner := 999
	relations := []procguard.Proc{
		{PID: parent, Name: "fak"},
		{PID: 101, PPID: &deadOwner, Name: "opencode", Cmdline: `opencode run "Resolve GitHub issue #3868"`, AgeSec: &age},
		{PID: 102, PPID: &deadOwner, Name: "opencode", Cmdline: `opencode run "Resolve GitHub issue #3869"`, AgeSec: &age},
		{PID: 103, PPID: &deadOwner, Name: "opencode", Cmdline: "opencode run ordinary-task", AgeSec: &age},
		{PID: 104, PPID: &deadOwner, Name: "opencode", Cmdline: `opencode run "Resolve GitHub issue #3870"`, AgeSec: &age},
		{PID: 105, PPID: &parent, Name: "opencode", Cmdline: `opencode run "Resolve GitHub issue #3871"`, AgeSec: &age},
	}
	samples := []procguard.Proc{
		{PID: 101, Threads: &threads, Handles: &handles, WSMB: &ws, CPUPct: &idleCPU},
		{PID: 102, Threads: &threads, Handles: &handles, WSMB: &ws, CPUPct: &idleCPU},
		{PID: 103, Threads: &threads, Handles: &handles, WSMB: &ws, CPUPct: &idleCPU},
		{PID: 104, Threads: &threads, Handles: &handles, WSMB: &ws, CPUPct: &activeCPU},
		{PID: 105, Threads: &threads, Handles: &handles, WSMB: &ws, CPUPct: &idleCPU},
	}

	rows := classifyHungDispatchOrphans(
		relations,
		samples,
		procguard.NewRelationTopology(relations),
		map[int]bool{102: true},
		1800,
		nil,
		nil,
	)
	if len(rows) != 1 || rows[0].PID != 101 {
		t.Fatalf("only the idle marker worker with dead owner and no lease should flag: %+v", rows)
	}
	if rows[0].Kind != hungDispatchOrphanKind {
		t.Fatalf("kind = %q, want %q", rows[0].Kind, hungDispatchOrphanKind)
	}
	if rows[0].CPUPct == nil || *rows[0].CPUPct != idleCPU {
		t.Fatalf("idle CPU witness missing from finding: %+v", rows[0])
	}

	killed := []int{}
	report := procguard.Build(nil, procguard.Options{
		OrphanRows: rows,
		Enact:      false,
		Killer: func(pid int) (bool, string) {
			killed = append(killed, pid)
			return true, "killed"
		},
	})
	if len(killed) != 0 || report.Flagged[0].Action != "report" {
		t.Fatalf("report-only mode must flag without killing: killed=%v payload=%+v", killed, report)
	}

	enacted := procguard.Build(nil, procguard.Options{
		OrphanRows: rows,
		Enact:      true,
		Killer: func(pid int) (bool, string) {
			killed = append(killed, pid)
			return true, "killed"
		},
	})
	if len(killed) != 1 || killed[0] != 101 || enacted.Flagged[0].Action != "killed" {
		t.Fatalf("--enact must tree-kill the classified orphan: killed=%v payload=%+v", killed, enacted)
	}
}

func TestVerifiedProcGuardKillerRejectsSurvivingRoot(t *testing.T) {
	killer := verifiedTreeReaper(
		func(int) (bool, string) {
			return true, "terminated 2 process(es) via native process tree"
		},
		func(int) bool { return true },
		func(time.Duration) {},
	)
	ok, detail := killer(101)
	if ok {
		t.Fatalf("a surviving root must turn a reported kill success into failure: %s", detail)
	}
	if !strings.Contains(detail, "elevation required") {
		t.Fatalf("failure must name the operator remedy, got %q", detail)
	}

	payload := procguard.Build(nil, procguard.Options{
		OrphanRows: []procguard.Finding{{
			PID: 101, Name: "opencode", Kind: hungDispatchOrphanKind,
			Reasons: []string{"hung dispatch orphan"},
		}},
		Enact:  true,
		Killer: killer,
	})
	if payload.OK || len(payload.Enacted) != 1 || payload.Enacted[0].OK || payload.Flagged[0].Action != "kill-failed" {
		t.Fatalf("a surviving session-0 root must surface as a failed enact verdict: %+v", payload)
	}
}

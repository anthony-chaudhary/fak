package looporphanscan

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/looporphan"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func ptr(i int) *int { return &i }

func supByPID(sv []looporphan.Supervisor, pid int) (looporphan.Supervisor, bool) {
	for _, s := range sv {
		if s.PID == pid {
			return s, true
		}
	}
	return looporphan.Supervisor{}, false
}

func TestExtractSupervisors_ParentLivenessAndLane(t *testing.T) {
	cfg := DefaultConfig()
	procs := []procguard.Proc{
		// A live owning driver.
		{PID: 1, Cmdline: "fak guard serve", Start: "s1"},
		// Supervisor with a live, recognizable parent -> ParentAlive, lane parsed.
		{PID: 100, PPID: ptr(1), Cmdline: "dos loop --lane auth", Start: "s100"},
		// Supervisor whose parent PID is absent from the census -> ParentDead.
		{PID: 200, PPID: ptr(999), Cmdline: "dos loop --region billing", Start: "s200"},
		// Supervisor whose parent is present but unrecognizable -> ParentUnknown.
		{PID: 300, PPID: ptr(400), Cmdline: "fak loop drive --lane=web", Start: "s300"},
		{PID: 400, PPID: ptr(1), Cmdline: "explorer.exe", Start: "s400"},
		// Not a supervisor -> skipped.
		{PID: 500, PPID: ptr(1), Cmdline: "claude -p /dispatch", Start: "s500"},
	}
	sv := ExtractSupervisors(procs, cfg)
	if len(sv) != 3 {
		t.Fatalf("want 3 supervisors, got %d: %+v", len(sv), sv)
	}

	if s, _ := supByPID(sv, 100); s.Parent != looporphan.ParentAlive || s.Lane != "auth" {
		t.Errorf("pid100: want ParentAlive/lane=auth, got %s/%q", s.Parent, s.Lane)
	}
	if s, _ := supByPID(sv, 200); s.Parent != looporphan.ParentDead || s.Lane != "billing" {
		t.Errorf("pid200: want ParentDead/lane=billing, got %s/%q", s.Parent, s.Lane)
	}
	if s, _ := supByPID(sv, 300); s.Parent != looporphan.ParentUnknown || s.Lane != "web" {
		t.Errorf("pid300: want ParentUnknown/lane=web (--flag=value form), got %s/%q", s.Parent, s.Lane)
	}
}

func TestExtractSupervisors_SubtreeWorkerCount(t *testing.T) {
	cfg := DefaultConfig()
	procs := []procguard.Proc{
		{PID: 100, PPID: ptr(1), Cmdline: "dos loop --lane auth", Start: "s100"},
		// direct child worker
		{PID: 101, PPID: ptr(100), Cmdline: "claude -p /work", Start: "s101"},
		// grandchild worker (transitive subtree)
		{PID: 102, PPID: ptr(101), Cmdline: "fak c compute", Start: "s102"},
		// unrelated worker under a different tree -> not counted
		{PID: 900, PPID: ptr(1), Cmdline: "claude -p /elsewhere", Start: "s900"},
		// idle sibling supervisor, no workers
		{PID: 200, PPID: ptr(999), Cmdline: "dos loop --lane other", Start: "s200"},
	}
	sv := ExtractSupervisors(procs, cfg)
	if s, _ := supByPID(sv, 100); s.LiveWorkers != 2 {
		t.Errorf("pid100: want 2 subtree workers, got %d", s.LiveWorkers)
	}
	if s, _ := supByPID(sv, 200); s.LiveWorkers != 0 {
		t.Errorf("pid200: want 0 subtree workers, got %d", s.LiveWorkers)
	}
}

func TestScan_KeepLiveReapIdleDuplicate(t *testing.T) {
	cfg := DefaultConfig()
	procs := []procguard.Proc{
		// two detached supervisors on the same lane, parents gone
		{PID: 100, PPID: ptr(999), Cmdline: "dos loop --lane auth", Start: "s100"},
		{PID: 200, PPID: ptr(998), Cmdline: "dos loop --lane auth", Start: "s200"},
		// a live worker under 100 only
		{PID: 101, PPID: ptr(100), Cmdline: "claude -p /work", Start: "s101"},
	}
	rep := Scan(procs, cfg)
	if rep.Keep != 1 || rep.Reap != 1 {
		t.Fatalf("want keep=1 reap=1, got keep=%d reap=%d (%+v)", rep.Keep, rep.Reap, rep.Verdicts)
	}
	// the reaped one must be the idle 200, never the live 100
	pids := rep.ReapPIDs()
	if len(pids) != 1 || pids[0] != 200 {
		t.Fatalf("want reap [200], got %v", pids)
	}
}

func TestConfirmReap_Fence(t *testing.T) {
	// same process (start matches, running) -> confirmed
	if !ConfirmReap(100, "s100", func(pid int) (string, bool) { return "s100", true }) {
		t.Error("matching start + running should confirm reap")
	}
	// PID recycled (start changed) -> refused
	if ConfirmReap(100, "s100", func(pid int) (string, bool) { return "sZZZ", true }) {
		t.Error("changed start (PID reuse) must refuse reap")
	}
	// vanished (not running) -> refused
	if ConfirmReap(100, "s100", func(pid int) (string, bool) { return "", false }) {
		t.Error("not-running PID must refuse reap")
	}
}

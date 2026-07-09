package main

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/looporphan"
	"github.com/anthony-chaudhary/fak/internal/looprecover"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func mkProc(pid, ppid int, name, cmdline, start string) procguard.Proc {
	p := procguard.Proc{PID: pid, Name: name, Cmdline: cmdline, Start: start}
	if ppid != 0 {
		p.PPID = procguard.IntPtr(ppid)
	}
	return p
}

// A mixed live-ish process table: three loop supervisors (one orphan-idle, one
// attached-idle, one parenting a live worker) plus the non-supervisor session and
// worker processes they relate to.
func reapFixtureProcs() []procguard.Proc {
	return []procguard.Proc{
		mkProc(500, 400, "pwsh", "pwsh -NoProfile -Command ...", "s500"),         // alive session, parents 1001
		mkProc(1000, 999, "fak", "fak loop drive --lane auth", "s1000"),          // ppid 999 absent -> orphan, idle -> REAP
		mkProc(1001, 500, "fak", "fak loop drive --lane build", "s1001"),         // ppid 500 alive, idle -> KEEP attached
		mkProc(1002, 999, "fak", "fak dos loop --lane data", "s1002"),            // ppid 999 absent, but parents a worker -> KEEP live
		mkProc(1003, 1002, "claude", "claude -p --lane data 'do the thing'", ""), // the live worker under 1002
	}
}

// verdictFor returns the verdict for a pid from a Report (zero value if absent).
func verdictFor(rep looporphan.Report, pid int) looporphan.Verdict {
	for _, v := range rep.Verdicts {
		if v.PID == pid {
			return v
		}
	}
	return looporphan.Verdict{}
}

func TestLoopReapCensus_FoldsSupervisorsWithLivenessAndWorkers(t *testing.T) {
	census := loopReapCensus(reapFixtureProcs(), defaultLoopSupervisorMarkers, defaultLoopWorkerMarkers)

	// Only the three supervisors are folded; the session and worker are not.
	if len(census) != 3 {
		t.Fatalf("want 3 supervisors folded, got %d: %+v", len(census), census)
	}
	by := map[int]looporphan.Supervisor{}
	for _, s := range census {
		by[s.PID] = s
	}

	if s := by[1000]; s.Lane != "auth" || s.Parent != looporphan.ParentDead || s.LiveWorkers != 0 || s.Start == "" {
		t.Fatalf("pid 1000 (orphan idle): got lane=%q parent=%q workers=%d start=%q",
			s.Lane, s.Parent, s.LiveWorkers, s.Start)
	}
	if s := by[1001]; s.Lane != "build" || s.Parent != looporphan.ParentAlive || s.LiveWorkers != 0 {
		t.Fatalf("pid 1001 (attached idle): got lane=%q parent=%q workers=%d", s.Lane, s.Parent, s.LiveWorkers)
	}
	if s := by[1002]; s.Lane != "data" || s.LiveWorkers != 1 {
		t.Fatalf("pid 1002 (live work): got lane=%q parent=%q workers=%d", s.Lane, s.Parent, s.LiveWorkers)
	}
}

func TestLoopReapCensus_PlanKeepsLiveAndAttachedReapsOrphan(t *testing.T) {
	census := loopReapCensus(reapFixtureProcs(), defaultLoopSupervisorMarkers, defaultLoopWorkerMarkers)
	rep := looporphan.Plan(census, looporphan.DefaultConfig())

	if v := verdictFor(rep, 1000); v.Action != looporphan.REAP || v.Reason != looporphan.ReasonOrphanIdle {
		t.Fatalf("pid 1000: want REAP/%s, got %s/%s", looporphan.ReasonOrphanIdle, v.Action, v.Reason)
	}
	if v := verdictFor(rep, 1001); v.Action != looporphan.KEEP || v.Reason != looporphan.ReasonKeepAttached {
		t.Fatalf("pid 1001: want KEEP/%s, got %s/%s", looporphan.ReasonKeepAttached, v.Action, v.Reason)
	}
	if v := verdictFor(rep, 1002); v.Action != looporphan.KEEP || v.Reason != looporphan.ReasonKeepLiveWork {
		t.Fatalf("pid 1002: want KEEP/%s, got %s/%s", looporphan.ReasonKeepLiveWork, v.Action, v.Reason)
	}
	if rep.Keep != 2 || rep.Reap != 1 {
		t.Fatalf("counts: want keep=2 reap=1, got keep=%d reap=%d", rep.Keep, rep.Reap)
	}
}

func TestParseLoopLane(t *testing.T) {
	cases := map[string]string{
		"fak loop drive --lane auth":        "auth",
		"fak loop drive --loop RS7 --json":  "RS7",
		"fak loop drive --goal=GOAL.md":     "GOAL.md",
		"fak dos loop --region east --json": "east",
		"fak loop drive":                    "",
		"fak loop drive --lane":             "", // dangling flag, no value
		"fak loop drive --lane --json":      "", // next token is a flag, not a value
		"fak loop drive --lane auth --json": "auth",
	}
	for cmd, want := range cases {
		if got := parseLoopLane(cmd); got != want {
			t.Errorf("parseLoopLane(%q) = %q, want %q", cmd, got, want)
		}
	}
}

// withReapSeams swaps the collector and killer seams for the duration of a test.
func withReapSeams(t *testing.T, procs []procguard.Proc, killed *[]int) {
	t.Helper()
	prevCollect, prevKill := loopReapCollectRelations, loopReapKill
	t.Cleanup(func() { loopReapCollectRelations, loopReapKill = prevCollect, prevKill })
	loopReapCollectRelations = func() ([]procguard.Proc, string) { return procs, "" }
	loopReapKill = func(pid int) (bool, string) {
		*killed = append(*killed, pid)
		return true, "reaped by test stub"
	}
}

func TestRunLoopReap_ReportModeExits3AndKillsNothing(t *testing.T) {
	var killed []int
	withReapSeams(t, reapFixtureProcs(), &killed)

	var out, errb bytes.Buffer
	rc := runLoopReap(&out, &errb, nil)

	if rc != 3 {
		t.Fatalf("report mode with a reap-eligible supervisor must exit 3, got %d (stderr=%q)", rc, errb.String())
	}
	if len(killed) != 0 {
		t.Fatalf("report mode must not kill anything, killed=%v", killed)
	}
	s := out.String()
	for _, want := range []string{"supervisors=3", "keep=2 reap=1", "REAP", "LOOP_ORPHAN_IDLE", "--reap"} {
		if !strings.Contains(s, want) {
			t.Fatalf("report output missing %q:\n%s", want, s)
		}
	}
}

func TestRunLoopReap_ReapModeKillsOnlyTheReapSet(t *testing.T) {
	var killed []int
	withReapSeams(t, reapFixtureProcs(), &killed)

	var out, errb bytes.Buffer
	rc := runLoopReap(&out, &errb, []string{"--reap"})

	if rc != 0 {
		t.Fatalf("--reap must exit 0 after enacting, got %d (stderr=%q)", rc, errb.String())
	}
	if len(killed) != 1 || killed[0] != 1000 {
		t.Fatalf("--reap must tree-kill only the orphan-idle supervisor [1000], killed=%v", killed)
	}
	// The load-bearing safety invariant: a supervisor parenting live work is never killed.
	for _, pid := range killed {
		if pid == 1002 {
			t.Fatalf("SAFETY VIOLATION: killed pid 1002, which parents a live worker")
		}
	}
	if !strings.Contains(out.String(), "ENACTED") {
		t.Fatalf("enacted output must carry the ENACTED banner:\n%s", out.String())
	}
}

func TestRunLoopReap_JSONShape(t *testing.T) {
	var killed []int
	withReapSeams(t, reapFixtureProcs(), &killed)

	var out, errb bytes.Buffer
	runLoopReap(&out, &errb, []string{"--json"})

	var rep loopReapReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("--json output is not valid JSON: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak-loop-reap/1" {
		t.Fatalf("schema = %q, want fak-loop-reap/1", rep.Schema)
	}
	if rep.Supervisors != 3 || rep.Keep != 2 || rep.Reap != 1 || rep.Enacted {
		t.Fatalf("json counts: %+v", rep)
	}
	if len(rep.Verdicts) != 3 {
		t.Fatalf("want 3 verdict rows, got %d", len(rep.Verdicts))
	}
	pids := []int{}
	for _, v := range rep.Verdicts {
		pids = append(pids, v.PID)
	}
	sort.Ints(pids)
	if want := []int{1000, 1001, 1002}; !intsEqual(pids, want) {
		t.Fatalf("verdict pids = %v, want %v", pids, want)
	}
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestRunLoopReap_KillTimeFenceRefusesReusedPID proves the shell's kill-time fence:
// when a REAP pid's start-time identity changes between the plan and the kill (PID
// reuse), the kill is refused - so a recycled PID is never the process that dies,
// even though the plan said REAP.
func TestRunLoopReap_KillTimeFenceRefusesReusedPID(t *testing.T) {
	prevCollect, prevKill := loopReapCollectRelations, loopReapKill
	t.Cleanup(func() { loopReapCollectRelations, loopReapKill = prevCollect, prevKill })

	// The plan (1st collect) sees pid 1000 with start "s1000"; the kill-time rescan
	// (2nd collect) sees pid 1000 with a DIFFERENT start - the PID was reused.
	call := 0
	loopReapCollectRelations = func() ([]procguard.Proc, string) {
		call++
		procs := reapFixtureProcs()
		if call >= 2 {
			for i := range procs {
				if procs[i].PID == 1000 {
					procs[i].Start = "s1000-REUSED"
				}
			}
		}
		return procs, ""
	}
	var killed []int
	loopReapKill = func(pid int) (bool, string) { killed = append(killed, pid); return true, "stub" }

	var out, errb bytes.Buffer
	rc := runLoopReap(&out, &errb, []string{"--reap"})

	if rc != 0 {
		t.Fatalf("--reap should exit 0, got %d (stderr=%q)", rc, errb.String())
	}
	if len(killed) != 0 {
		t.Fatalf("kill-time fence must refuse the reused PID; killed=%v", killed)
	}
	if !strings.Contains(out.String(), "PID reused or vanished") {
		t.Fatalf("output should explain the reuse refusal:\n%s", out.String())
	}
}

// TestRunLoopReap_KillTimeFenceRefusesUnfenced proves that even --allow-unfenced
// (which lets the core surface a REAP for a fence-less supervisor in the report)
// cannot land an unverifiable kill: with no start-time fence the pid's identity
// cannot be re-checked, so the kill site refuses it.
func TestRunLoopReap_KillTimeFenceRefusesUnfenced(t *testing.T) {
	prevCollect, prevKill := loopReapCollectRelations, loopReapKill
	t.Cleanup(func() { loopReapCollectRelations, loopReapKill = prevCollect, prevKill })

	procs := reapFixtureProcs()
	for i := range procs {
		if procs[i].PID == 1000 {
			procs[i].Start = "" // fence-less
		}
	}
	loopReapCollectRelations = func() ([]procguard.Proc, string) { return procs, "" }
	var killed []int
	loopReapKill = func(pid int) (bool, string) { killed = append(killed, pid); return true, "stub" }

	var out, errb bytes.Buffer
	rc := runLoopReap(&out, &errb, []string{"--reap", "--allow-unfenced"})

	if rc != 0 {
		t.Fatalf("--reap should exit 0, got %d (stderr=%q)", rc, errb.String())
	}
	if len(killed) != 0 {
		t.Fatalf("fence-less kill must be refused even with --allow-unfenced; killed=%v", killed)
	}
	if !strings.Contains(out.String(), "no start-time fence") {
		t.Fatalf("output should explain the fence-less refusal:\n%s", out.String())
	}
}

// TestLoopParentState_SupervisorAtPid1IsAlive locks the container-pid-1 guard: a
// loop whose parent is a recognized supervisor running AS pid 1 (a containerized
// entrypoint that drives loops) is ParentAlive, not mistaken for orphaned-to-init.
// The check runs before the pid-1 rule and is OS-independent, so it holds on POSIX
// where the blanket ppid==1 rule would otherwise flip it to ParentDead.
func TestLoopParentState_SupervisorAtPid1IsAlive(t *testing.T) {
	byPID := map[int]procguard.Proc{
		1: {PID: 1, Name: "fak", Cmdline: "fak loop drive --lane root", Start: "s1"},
	}
	live := looprecover.Liveness(func(pid int) (string, bool) {
		if p, ok := byPID[pid]; ok {
			return p.Start, true
		}
		return "", false
	})
	if got := loopParentState(1, byPID, defaultLoopSupervisorMarkers, live); got != looporphan.ParentAlive {
		t.Fatalf("supervisor driving loops at pid 1: want ParentAlive, got %s", got)
	}
	// A present, live, non-supervisor parent is still judged alive by the probe.
	byPID[7] = procguard.Proc{PID: 7, Name: "pwsh", Cmdline: "pwsh -Command run", Start: "s7"}
	if got := loopParentState(7, byPID, defaultLoopSupervisorMarkers, live); got != looporphan.ParentAlive {
		t.Fatalf("live non-supervisor parent: want ParentAlive, got %s", got)
	}
	// An absent parent is ParentDead (gone), independent of the pid-1 rule.
	if got := loopParentState(4242, byPID, defaultLoopSupervisorMarkers, live); got != looporphan.ParentDead {
		t.Fatalf("absent parent: want ParentDead, got %s", got)
	}
}

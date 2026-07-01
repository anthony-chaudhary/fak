package fleetmon

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func staleSet(r JanitorResult) map[int]ChildCommand {
	m := map[int]ChildCommand{}
	for _, c := range r.Stale {
		m[c.RootPID] = c
	}
	return m
}

func TestJanitorSimpleStaleVsLegitTest(t *testing.T) {
	now := time.Now()
	rootStart := tsAt(now, 60) // worker root started an hour ago
	procs := []procguard.Proc{
		proc(100, 1, "claude", "claude -p /dos-dispatch-loop", 3600, rootStart),
		// A simple `ls` alive 6 minutes: stale (> 5m simple ceiling).
		proc(200, 100, "ls", "ls -la /c/work/fak", 360, tsAt(now, 6)),
		// A `go test` alive 6 minutes: NOT stale (< 10m test ceiling) — a legit witness.
		proc(201, 100, "go", "go test ./internal/...", 360, tsAt(now, 6)),
		// A `go test` alive 12 minutes: stale (> 10m test ceiling) — wedged.
		proc(202, 100, "go", "go test ./...", 720, tsAt(now, 12)),
	}
	res := EvaluateJanitor(JanitorInput{
		Procs:   procs,
		Workers: []WorkerRoot{{PID: 100, Session: "issue-1", Start: rootStart}},
		Now:     now,
	})
	stale := staleSet(res)
	if _, ok := stale[200]; !ok {
		t.Error("a 6-minute `ls` must be stale (simple-shell 5m ceiling)")
	}
	if _, ok := stale[201]; ok {
		t.Error("a 6-minute `go test` must NOT be stale (test 10m ceiling) — a legit long witness")
	}
	if _, ok := stale[202]; !ok {
		t.Error("a 12-minute `go test` must be stale (test 10m ceiling)")
	}
}

func TestJanitorNeverKillsWorkerRoot(t *testing.T) {
	now := time.Now()
	rootStart := tsAt(now, 60)
	procs := []procguard.Proc{proc(100, 1, "claude", "claude -p", 3600, rootStart)}
	res := EvaluateJanitor(JanitorInput{Procs: procs, Workers: []WorkerRoot{{PID: 100, Start: rootStart}}, Now: now})
	if len(res.Stale) != 0 {
		t.Fatalf("a lone worker root must never be a kill target, got %+v", res.Stale)
	}
}

func TestJanitorProtectsMCPServer(t *testing.T) {
	now := time.Now()
	rootStart := tsAt(now, 60)
	procs := []procguard.Proc{
		proc(100, 1, "claude", "claude -p", 3600, rootStart),
		// A DOS MCP server child, old — must be protected, never reaped.
		proc(300, 100, "python", "python -m dos_mcp.server", 3000, tsAt(now, 50)),
	}
	res := EvaluateJanitor(JanitorInput{Procs: procs, Workers: []WorkerRoot{{PID: 100, Start: rootStart}}, Now: now})
	if _, ok := staleSet(res)[300]; ok {
		t.Fatal("a DOS MCP server must never be in the stale/kill set")
	}
	// It should surface as protected (old enough to be stale-by-age but spared).
	var found bool
	for _, c := range res.Protected {
		if c.RootPID == 300 {
			found = true
		}
	}
	if !found {
		t.Error("the old MCP child should be reported as stale-but-protected")
	}
}

// TestJanitorPIDReuseStaleParentProcessId is the reuse fence: an old process whose
// ParentProcessId points at a pid the OS later recycled into the worker root. Its
// start time PREDATES the worker root, so it is not this worker's child and must
// never be reaped as one — even though it is far past the simple-shell ceiling.
func TestJanitorPIDReuseStaleParentProcessId(t *testing.T) {
	now := time.Now()
	rootStart := tsAt(now, 5) // the worker root is YOUNG (started 5 min ago)
	procs := []procguard.Proc{
		proc(100, 1, "claude", "claude -p", 300, rootStart),
		// pid 200 claims ppid 100 but started 40 min ago — BEFORE the worker root.
		// pid 100 was reused: 200 is an unrelated old process, not the worker's child.
		proc(200, 100, "cat", "cat bigfile", 2400, tsAt(now, 40)),
	}
	res := EvaluateJanitor(JanitorInput{Procs: procs, Workers: []WorkerRoot{{PID: 100, Start: rootStart}}, Now: now})
	if _, ok := staleSet(res)[200]; ok {
		t.Fatal("a process that PREDATES the worker root (stale PPID / PID reuse) must never be reaped")
	}
	var protectedForReuse bool
	for _, c := range res.Children {
		if c.RootPID == 200 && c.Protected {
			protectedForReuse = true
		}
	}
	if !protectedForReuse {
		t.Error("the predating process should be marked protected")
	}
}

func TestJanitorProtectsAncestorOfWorker(t *testing.T) {
	now := time.Now()
	launcherStart := tsAt(now, 90)
	rootStart := tsAt(now, 60)
	// A launcher shell (pid 50) is the PARENT of the worker root (pid 100). Even if
	// the launcher looks like a stale idle shell, killing it would take the worker
	// down, so it must be protected. Here we make pid 50 also a "child" scan target
	// by giving another worker rooted at 51 whose subtree includes 50's tree — but
	// the simplest witness: 50 is an ancestor, so it's protected wherever it appears.
	procs := []procguard.Proc{
		proc(40, 1, "explorer", "explorer", 9000, tsAt(now, 150)),
		proc(50, 40, "pwsh", "pwsh -c launch", 5400, launcherStart),
		proc(100, 50, "claude", "claude -p", 3600, rootStart),
		// a stale simple child directly under the ancestor shell
		proc(60, 50, "ls", "ls", 4000, tsAt(now, 66)),
	}
	// Two workers: the real one (100) and a sibling whose root is the launcher's
	// parent, so the launcher (50) is walked as a child AND is an ancestor of 100.
	res := EvaluateJanitor(JanitorInput{
		Procs:   procs,
		Workers: []WorkerRoot{{PID: 100, Start: rootStart}, {PID: 40, Start: tsAt(now, 150)}},
		Now:     now,
	})
	for _, c := range res.Stale {
		if c.RootPID == 50 {
			t.Fatal("an ancestor of a worker root must never be reaped (it would stop the worker)")
		}
	}
}

func TestJanitorLongWitnessNeverStaleOnAge(t *testing.T) {
	now := time.Now()
	rootStart := tsAt(now, 120)
	procs := []procguard.Proc{
		proc(100, 1, "claude", "claude -p", 7200, rootStart),
		// `fak serve` running 90 minutes — a long witness, never stale on age alone.
		proc(400, 100, "fak", "fak serve --port 8080", 5400, tsAt(now, 90)),
	}
	res := EvaluateJanitor(JanitorInput{Procs: procs, Workers: []WorkerRoot{{PID: 100, Start: rootStart}}, Now: now})
	if _, ok := staleSet(res)[400]; ok {
		t.Fatal("a long-witness process (fak serve) must never be stale on age alone")
	}
}

func TestJanitorBroadScanStale(t *testing.T) {
	now := time.Now()
	rootStart := tsAt(now, 60)
	procs := []procguard.Proc{
		proc(100, 1, "claude", "claude -p", 3600, rootStart),
		proc(210, 100, "rg", "rg --files", 400, tsAt(now, 6)),
	}
	res := EvaluateJanitor(JanitorInput{Procs: procs, Workers: []WorkerRoot{{PID: 100, Start: rootStart}}, Now: now})
	if _, ok := staleSet(res)[210]; !ok {
		t.Fatal("a 6-minute broad scan must be stale (5m broad-scan ceiling)")
	}
}

func TestJanitorTreePIDsCollectsDescendants(t *testing.T) {
	now := time.Now()
	rootStart := tsAt(now, 60)
	procs := []procguard.Proc{
		proc(100, 1, "claude", "claude -p", 3600, rootStart),
		proc(500, 100, "bash", "bash -lc 'go test ./...'", 900, tsAt(now, 15)),
		proc(501, 500, "go", "go test ./...", 880, tsAt(now, 14)),
		proc(502, 501, "compile", "compile", 800, tsAt(now, 13)),
	}
	res := EvaluateJanitor(JanitorInput{Procs: procs, Workers: []WorkerRoot{{PID: 100, Start: rootStart}}, Now: now})
	stale := staleSet(res)
	c, ok := stale[500]
	if !ok {
		t.Fatal("the wedged test subtree root should be stale")
	}
	if len(c.TreePIDs) != 3 {
		t.Fatalf("tree PIDs should include the subtree {500,501,502}, got %v", c.TreePIDs)
	}
}

func TestJanitorNextActionMessages(t *testing.T) {
	now := time.Now()
	// Nothing stale.
	res := EvaluateJanitor(JanitorInput{Procs: nil, Workers: nil, Now: now})
	if res.NextAction == "" {
		t.Fatal("next action should never be empty")
	}
}

func TestJanitorDecisionRecord(t *testing.T) {
	now := time.Now()
	c := ChildCommand{RootPID: 200, Name: "ls", Command: "ls -la", Class: CmdSimpleShell, AgeSec: 400, WorkerPID: 100, Session: "issue-1", Issue: 1, TreePIDs: []int{200}, Reason: "stale"}
	d := NewJanitorDecision(c, JanitorActionTerminated, "", now)
	if d.Schema != JanitorLedgerSchema {
		t.Fatalf("schema not stamped: %q", d.Schema)
	}
	if d.Action != JanitorActionTerminated || d.RootPID != 200 || d.WorkerPID != 100 {
		t.Fatalf("decision fields wrong: %+v", d)
	}
	if len(d.TreePIDs) != 1 || d.AgeSec != 400 || d.Command != "ls -la" {
		t.Fatalf("decision must carry command/age/pids: %+v", d)
	}
	line, err := AppendJanitorDecisionLine(d)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, `"action":"terminated"`) || !strings.Contains(line, `"root_pid":200`) {
		t.Fatalf("decision line missing fields: %s", line)
	}
}

package main

import (
	"fmt"
	"os/exec"
	"runtime"
	"sync"
	"testing"
	"time"
)

// #2989: a budget restart must be single-child. stopGuardChild previously escalated to
// child.Process.Kill(), which reaps ONLY the immediate PID and leaves the child's descendant
// subtree (the node runtime + MCP/tool subprocesses a `claude` spawns) orphaned. Across two
// budget restarts that stranded three live claude children under one guard parent, which then
// poisoned dispatch preflight (unattributed_live=3 -> REFUSE_NO_SEAT). The fix routes the
// escalation through guardChildTreeKill (= procguard.KillPID), a process-TREE kill.
//
// These tests exercise the escalation decision directly: they fail against the old bare-Kill
// path (which never called a tree reaper) and pass once the grace-window escalation hands the
// child's PID to the tree killer.

// spawnLongLivedChild starts a real process so child.Process.Pid is a live, valid PID the
// reaper stub can assert on and terminate. It is a plain ~30s sleep; the test does NOT feed
// child.Wait() into the grace `select` (see below), so the helper's signal disposition is
// irrelevant — this keeps the witness deterministic across platforms.
func spawnLongLivedChild(t *testing.T) *exec.Cmd {
	t.Helper()
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/c", "ping -n 30 127.0.0.1 >NUL")
	} else {
		cmd = exec.Command("sh", "-c", "sleep 30")
	}
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot spawn a long-lived child on this platform: %v", err)
	}
	return cmd
}

func TestStopGuardChildTreeKillsOnGraceTimeout(t *testing.T) {
	child := spawnLongLivedChild(t)
	// The wait channel is held by the test rather than fed from child.Wait(): we are modelling a
	// child that does NOT exit on the polite os.Interrupt, so the grace branch must run. Feeding
	// a real child.Wait() would couple the assertion to the helper's SIGINT disposition, which
	// varies by platform/shell. The reaper stub sends on wait (as the real child.Wait()
	// goroutine would once procguard.KillPID reaps the tree), unblocking the trailing <-wait.
	wait := make(chan error, 1)

	prev := guardChildTreeKill
	t.Cleanup(func() {
		guardChildTreeKill = prev
		_ = child.Process.Kill()
	})

	var mu sync.Mutex
	var killedPID int
	guardChildTreeKill = func(pid int) (bool, string) {
		mu.Lock()
		killedPID = pid
		mu.Unlock()
		_ = child.Process.Kill()
		wait <- nil
		return true, "reaped by test stub"
	}

	// The child never signals wait within the grace window, so stopGuardChild must escalate.
	stopGuardChild(child, wait, 50*time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	if killedPID == 0 {
		t.Fatal("grace window expired but the tree reaper was never invoked — a bare " +
			"child.Kill() would orphan the descendant subtree (#2989)")
	}
	if killedPID != child.Process.Pid {
		t.Fatalf("tree reaper killed pid %d, want the child's pid %d", killedPID, child.Process.Pid)
	}
}

func TestStopGuardChildNoTreeKillWhenChildExitsInGrace(t *testing.T) {
	// A child that exits cleanly within the grace window must be reaped by its own exit, NOT by
	// a tree kill — the destructive escalation is reserved for the stuck case.
	child := execCommandExit(0)
	if err := child.Start(); err != nil {
		t.Skipf("cannot spawn a short-lived child: %v", err)
	}
	wait := make(chan error, 1)
	go func() { wait <- child.Wait() }()

	prev := guardChildTreeKill
	t.Cleanup(func() { guardChildTreeKill = prev })

	var called bool
	guardChildTreeKill = func(pid int) (bool, string) {
		called = true
		return true, fmt.Sprintf("unexpected kill of %d", pid)
	}

	// Generous grace so the fast-exiting child is observed via <-wait first.
	stopGuardChild(child, wait, 5*time.Second)

	if called {
		t.Fatal("tree reaper invoked for a child that exited within grace — escalation must " +
			"fire only when the graceful interrupt is ignored")
	}
}

func TestStopGuardChildNilSafe(t *testing.T) {
	// Defensive: a nil child (or one that never started) must be a no-op, never a reaper call.
	prev := guardChildTreeKill
	t.Cleanup(func() { guardChildTreeKill = prev })
	guardChildTreeKill = func(int) (bool, string) {
		t.Fatal("tree reaper must not run for a nil child")
		return false, ""
	}
	stopGuardChild(nil, nil, time.Second)
	stopGuardChild(&exec.Cmd{}, nil, time.Second)
}

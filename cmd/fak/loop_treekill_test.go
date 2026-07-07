package main

import (
	"os/exec"
	"sync"
	"testing"
	"time"
)

// #3101 (sibling of #2989): every fak loop / loop drive kill path funnels through the single
// execLoopCommand.Kill() sink. It previously called c.cmd.Process.Kill(), which reaps ONLY the
// immediate PID and orphans the wrapped agent's descendant subtree (the node runtime + MCP/tool
// subprocesses a `claude`/`codex` spawns). On a `fak loop drive --deadline` loop that re-spawns
// the agent every turn, deadline expiry is the normal outcome, so every timed-out turn stranded
// a live subtree — the same orphan-accumulation that poisoned dispatch preflight in #2989
// (unattributed_live -> REFUSE_NO_SEAT). The fix routes the sink through loopTreeKill
// (= procguard.KillPID), a process-TREE kill.
//
// These tests fail against the old bare-Kill path (which never called a tree reaper) and pass
// once the sink hands the child's PID to the tree killer. spawnLongLivedChild is shared with
// guard_treekill_test.go (same package).

func TestExecLoopCommandKillTreeKills(t *testing.T) {
	child := spawnLongLivedChild(t)

	prev := loopTreeKill
	t.Cleanup(func() {
		loopTreeKill = prev
		_ = child.Process.Kill()
	})

	var mu sync.Mutex
	var killedPID int
	loopTreeKill = func(pid int) (bool, string) {
		mu.Lock()
		killedPID = pid
		mu.Unlock()
		_ = child.Process.Kill()
		return true, "reaped by test stub"
	}

	if err := (execLoopCommand{cmd: child}).Kill(); err != nil {
		t.Fatalf("execLoopCommand.Kill() returned error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if killedPID == 0 {
		t.Fatal("Kill() never invoked the tree reaper — a bare child.Process.Kill() would " +
			"orphan the wrapped agent's descendant subtree (#3101)")
	}
	if killedPID != child.Process.Pid {
		t.Fatalf("tree reaper killed pid %d, want the child's pid %d", killedPID, child.Process.Pid)
	}
}

func TestWaitLoopDriveCommandDeadlineTreeKills(t *testing.T) {
	// The hot path: waitLoopDriveCommand hits cmd.Kill() on deadline expiry — the normal outcome
	// of `fak loop drive --deadline`. Drive a real execLoopCommand through an already-spent
	// deadline and assert the wrapped child's PID is handed to the tree reaper, not bare-killed.
	child := spawnLongLivedChild(t)

	prev := loopTreeKill
	t.Cleanup(func() {
		loopTreeKill = prev
		_ = child.Process.Kill()
	})

	var mu sync.Mutex
	var killedPID int
	loopTreeKill = func(pid int) (bool, string) {
		mu.Lock()
		killedPID = pid
		mu.Unlock()
		_ = child.Process.Kill() // unblocks the background cmd.Wait() the real tree kill would
		return true, "reaped by test stub"
	}

	rc, timedOut, err := waitLoopDriveCommand(execLoopCommand{cmd: child}, time.Now().Add(-time.Second))
	if !timedOut || rc != 3 || err == nil {
		t.Fatalf("spent deadline must report a timed-out kill: got rc=%d timedOut=%v err=%v", rc, timedOut, err)
	}

	mu.Lock()
	defer mu.Unlock()
	if killedPID != child.Process.Pid {
		t.Fatalf("deadline-expiry kill reaped pid %d, want the child's pid %d (a bare kill "+
			"would orphan the subtree, #3101)", killedPID, child.Process.Pid)
	}
}

func TestExecLoopCommandKillNilSafe(t *testing.T) {
	// Defensive: a nil command (or one that never started) must be a no-op, never a reaper call.
	prev := loopTreeKill
	t.Cleanup(func() { loopTreeKill = prev })
	loopTreeKill = func(int) (bool, string) {
		t.Fatal("tree reaper must not run for a nil/unstarted command")
		return false, ""
	}
	if err := (execLoopCommand{}).Kill(); err != nil {
		t.Fatalf("nil-command Kill() returned error: %v", err)
	}
	if err := (execLoopCommand{cmd: &exec.Cmd{}}).Kill(); err != nil {
		t.Fatalf("unstarted-command Kill() returned error: %v", err)
	}
}

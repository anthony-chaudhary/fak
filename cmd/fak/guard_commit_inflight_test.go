package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestGuardCommitInFlight(t *testing.T) {
	t.Run("IndexLock", TestProbeGuardCommitInFlight_IndexLock)
	t.Run("Worktree", TestProbeGuardCommitInFlight_Worktree)
	t.Run("ProcessTree", TestFindActiveCommitProcessInTree)
	t.Run("GracePolling", TestGuardCommitGracePolling)
}

func TestProbeGuardCommitInFlight_IndexLock(t *testing.T) {
	dir := t.TempDir()
	gitDir := filepath.Join(dir, ".git")
	if err := os.MkdirAll(gitDir, 0755); err != nil {
		t.Fatalf("failed to create .git dir: %v", err)
	}

	lockPath := filepath.Join(gitDir, "index.lock")
	if err := os.WriteFile(lockPath, []byte(""), 0644); err != nil {
		t.Fatalf("failed to create index.lock: %v", err)
	}

	// Fresh index.lock (< 5m old) -> InFlight = true, Source = "index_lock"
	res := probeGuardCommitInFlightDetailed(0, dir)
	if !res.InFlight || res.Source != "index_lock" {
		t.Fatalf("expected InFlight=true, Source=index_lock; got %+v", res)
	}

	// Stale index.lock (> 5m old) -> InFlight = false
	staleTime := time.Now().Add(-10 * time.Minute)
	if err := os.Chtimes(lockPath, staleTime, staleTime); err != nil {
		t.Fatalf("failed to set mtime: %v", err)
	}

	resStale := probeGuardCommitInFlightDetailed(0, dir)
	if resStale.InFlight {
		t.Fatalf("expected stale index.lock to be ignored, got %+v", resStale)
	}
}

func TestProbeGuardCommitInFlight_Worktree(t *testing.T) {
	dir := t.TempDir()
	targetGitDir := filepath.Join(dir, "repo_git_target")
	if err := os.MkdirAll(targetGitDir, 0755); err != nil {
		t.Fatalf("failed to create target gitdir: %v", err)
	}

	// Create .git file with gitdir: <target>
	gitFilePath := filepath.Join(dir, ".git")
	if err := os.WriteFile(gitFilePath, []byte("gitdir: "+targetGitDir+"\n"), 0644); err != nil {
		t.Fatalf("failed to write .git file: %v", err)
	}

	// Put index.lock in target
	lockPath := filepath.Join(targetGitDir, "index.lock")
	if err := os.WriteFile(lockPath, []byte(""), 0644); err != nil {
		t.Fatalf("failed to create index.lock: %v", err)
	}

	res := probeGuardCommitInFlightDetailed(0, dir)
	if !res.InFlight || res.Source != "index_lock" {
		t.Fatalf("expected InFlight=true via worktree indirection, got %+v", res)
	}

	// Test relative gitdir path
	relDir := t.TempDir()
	relTarget := filepath.Join(relDir, "common", "worktrees", "wt1")
	if err := os.MkdirAll(relTarget, 0755); err != nil {
		t.Fatalf("failed to create relative target: %v", err)
	}
	relGitFile := filepath.Join(relDir, ".git")
	if err := os.WriteFile(relGitFile, []byte("gitdir: common/worktrees/wt1\n"), 0644); err != nil {
		t.Fatalf("failed to write relative .git file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(relTarget, "index.lock"), []byte(""), 0644); err != nil {
		t.Fatalf("failed to create rel index.lock: %v", err)
	}
	resRel := probeGuardCommitInFlightDetailed(0, relDir)
	if !resRel.InFlight || resRel.Source != "index_lock" {
		t.Fatalf("expected InFlight=true via relative worktree indirection, got %+v", resRel)
	}
}

func TestFindActiveCommitProcessInTree(t *testing.T) {
	ppidRoot := 1
	ppid1000 := 1000
	ppid1001 := 1001
	ppidUnrelated := 1

	mockProcs := []procguard.Proc{
		{PID: 1000, PPID: &ppidRoot, Name: "claude", Cmdline: "claude"},
		{PID: 1001, PPID: &ppid1000, Name: "bash", Cmdline: "/bin/bash"},
		{PID: 1002, PPID: &ppid1001, Name: "git", Cmdline: "git commit -m test"},
		{PID: 2000, PPID: &ppidUnrelated, Name: "git", Cmdline: "git commit -m other"},
	}

	origCollector := guardProcCollector
	defer func() { guardProcCollector = origCollector }()
	guardProcCollector = func() ([]procguard.Proc, string) {
		return mockProcs, ""
	}

	// Root PID 1000 (Claude) -> Child 1001 (Bash) -> Grandchild 1002 (git commit)
	// PID 1002 should be found
	p, found := findActiveCommitProcessInTree(1000)
	if !found {
		t.Fatalf("expected commit process 1002 to be found")
	}
	if p.PID != 1002 {
		t.Fatalf("expected PID 1002, got %d", p.PID)
	}

	// Unrelated process (PID 2000 not descendant of 1000) is ignored when looking at 1000
	// If 1002 is not a commit process, 1000 has no commit process
	mockProcsNonCommit := []procguard.Proc{
		{PID: 1000, PPID: &ppidRoot, Name: "claude", Cmdline: "claude"},
		{PID: 1001, PPID: &ppid1000, Name: "bash", Cmdline: "/bin/bash"},
		{PID: 1002, PPID: &ppid1001, Name: "git", Cmdline: "git status"},
		{PID: 2000, PPID: &ppidUnrelated, Name: "git", Cmdline: "git commit -m other"},
	}
	guardProcCollector = func() ([]procguard.Proc, string) {
		return mockProcsNonCommit, ""
	}

	_, foundUnrelated := findActiveCommitProcessInTree(1000)
	if foundUnrelated {
		t.Fatalf("expected unrelated process 2000 to be ignored, but found a commit process")
	}

	// Also verify fak sync / fak sweep / fak commit
	ppid3000 := 3000
	mockFakProcs := []procguard.Proc{
		{PID: 3000, PPID: &ppidRoot, Name: "agent", Cmdline: "agent"},
		{PID: 3001, PPID: &ppid3000, Name: "fak.exe", Cmdline: "fak sync push"},
	}
	guardProcCollector = func() ([]procguard.Proc, string) {
		return mockFakProcs, ""
	}
	pFak, foundFak := findActiveCommitProcessInTree(3000)
	if !foundFak || pFak.PID != 3001 {
		t.Fatalf("expected fak process 3001 found, got found=%v, pid=%d", foundFak, pFak.PID)
	}

	// Invalid rootPID
	if _, foundInvalid := findActiveCommitProcessInTree(0); foundInvalid {
		t.Fatalf("expected PID 0 to return false")
	}
	if _, foundNegative := findActiveCommitProcessInTree(-1); foundNegative {
		t.Fatalf("expected PID -1 to return false")
	}
}

func TestGuardCommitGracePolling(t *testing.T) {
	origInFlight := isGuardCommitInFlight
	defer func() { isGuardCommitInFlight = origInFlight }()

	// 1. Tests simulated commit in-flight for 2 polls then completing within grace window.
	calls := 0
	isGuardCommitInFlight = func(pid int, root string) (bool, string) {
		calls++
		if calls <= 2 {
			return true, "in-flight commit"
		}
		return false, ""
	}

	outcome, err := pollGuardCommitGrace(1000, t.TempDir(), 2*time.Second, 10*time.Millisecond, nil)
	if outcome != guardCommitGraceCompleted || err != nil {
		t.Fatalf("expected completed within grace, got outcome=%v, err=%v", outcome, err)
	}
	if calls < 3 {
		t.Fatalf("expected at least 3 polls, got %d", calls)
	}

	// 2. Tests simulated hung commit exceeding grace window.
	isGuardCommitInFlight = func(pid int, root string) (bool, string) {
		return true, "hung commit process"
	}

	outcomeHung, errHung := pollGuardCommitGrace(1000, t.TempDir(), 40*time.Millisecond, 10*time.Millisecond, nil)
	if outcomeHung != guardCommitGraceExpired || errHung != nil {
		t.Fatalf("expected grace expired, got outcome=%v, err=%v", outcomeHung, errHung)
	}

	// 3. Child exits during grace window
	waitCh := make(chan error, 1)
	expectedErr := errors.New("child crashed")
	waitCh <- expectedErr
	outcomeExit, errExit := pollGuardCommitGrace(1000, t.TempDir(), 2*time.Second, 10*time.Millisecond, waitCh)
	if outcomeExit != guardCommitGraceChildExited || errExit != expectedErr {
		t.Fatalf("expected child exited with error, got outcome=%v, err=%v", outcomeExit, errExit)
	}
}

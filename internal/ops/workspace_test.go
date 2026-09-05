package ops

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestWorkspaceManagerSweep(t *testing.T) {
	tmp := t.TempDir()
	gitDir := filepath.Join(tmp, ".git")
	_ = os.MkdirAll(gitDir, 0o755)

	// Create a dead-PID lock
	deadLock := filepath.Join(gitDir, "fak-commit.lock")
	_ = os.WriteFile(deadLock, []byte("99999999"), 0o644)

	// Create a live-PID lock
	liveLock := filepath.Join(gitDir, "index.lock")
	_ = os.WriteFile(liveLock, []byte("12345"), 0o644)

	wm := NewWorkspaceManager(tmp)
	wm.ProcessAlive = func(pid int) bool {
		return pid == 12345
	}

	// Dry run
	dryRes, err := wm.SweepLocksAndWorktrees(context.Background(), true)
	if err != nil {
		t.Fatalf("dry run error: %v", err)
	}
	if len(dryRes.LocksEvicted) != 1 {
		t.Errorf("expected 1 lock evicted in dry run, got %d: %v", len(dryRes.LocksEvicted), dryRes.LocksEvicted)
	}

	// Apply
	res, err := wm.SweepLocksAndWorktrees(context.Background(), false)
	if err != nil {
		t.Fatalf("apply error: %v", err)
	}
	if len(res.LocksEvicted) != 1 || res.LocksEvicted[0] != "fak-commit.lock" {
		t.Errorf("expected fak-commit.lock evicted, got %v", res.LocksEvicted)
	}

	// Verify dead lock is removed and live lock is kept
	if _, err := os.Stat(deadLock); !os.IsNotExist(err) {
		t.Errorf("expected dead lock to be removed, stat err: %v", err)
	}
	if _, err := os.Stat(liveLock); err != nil {
		t.Errorf("expected live lock to remain, stat err: %v", err)
	}
}

func TestSweepPreservesActiveWorkerWorktrees(t *testing.T) {
	repo := t.TempDir()
	runGit := func(dir string, args ...string) {
		t.Helper()
		cmdArgs := append([]string{"-c", "user.name=Test", "-c", "user.email=test@example.com"}, args...)
		cmd := exec.Command("git", cmdArgs...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v failed in %s: %v\n%s", args, dir, err, out)
		}
	}

	runGit(repo, "init", "-q")
	runGit(repo, "commit", "-q", "--allow-empty", "-m", "initial commit")

	scratchDir := filepath.Join(repo, "_scratch")
	if err := os.MkdirAll(scratchDir, 0o755); err != nil {
		t.Fatal(err)
	}

	wtLivePID := filepath.Join(scratchDir, workerworktree.DirName("cmd", "livepid01"))
	wtLiveLease := filepath.Join(scratchDir, workerworktree.DirName("docs", "livelease01"))
	wtLiveStamp := filepath.Join(scratchDir, workerworktree.DirName("abi", "livestamp01"))
	wtDead := filepath.Join(scratchDir, workerworktree.DirName("ops", "dead01"))

	runGit(repo, "worktree", "add", "--detach", wtLivePID)
	runGit(repo, "worktree", "add", "--detach", wtLiveLease)
	runGit(repo, "worktree", "add", "--detach", wtLiveStamp)
	runGit(repo, "worktree", "add", "--detach", wtDead)

	oldTime := time.Now().Add(-2 * time.Hour)

	// 1. wtLivePID: active PID 12345 in lease.json
	if err := workerworktree.WriteWorkerLease(wtLivePID, workerworktree.WorkerLease{
		PID:         12345,
		CreatedAt:   oldTime,
		HeartbeatTS: time.Now(),
	}); err != nil {
		t.Fatalf("WriteWorkerLease wtLivePID: %v", err)
	}

	// 2. wtLiveLease: dead PID in lease.json but active leaseref on lane "docs"
	if err := workerworktree.WriteWorkerLease(wtLiveLease, workerworktree.WorkerLease{
		PID:         99999999,
		CreatedAt:   oldTime,
		HeartbeatTS: oldTime,
	}); err != nil {
		t.Fatalf("WriteWorkerLease wtLiveLease: %v", err)
	}
	if _, err := leaseref.NewInDir(repo).Acquire(context.Background(), leaseref.Record{
		ID:         "resolve-docs",
		TTLSeconds: 3600,
		AcquiredAt: time.Now().Unix(),
	}); err != nil {
		t.Fatalf("leaseref Acquire: %v", err)
	}

	// 3. wtLiveStamp: dead PID in lease.json but active PID 12345 in owner stamp sidecar
	if err := workerworktree.WriteWorkerLease(wtLiveStamp, workerworktree.WorkerLease{
		PID:         99999999,
		CreatedAt:   oldTime,
		HeartbeatTS: oldTime,
	}); err != nil {
		t.Fatalf("WriteWorkerLease wtLiveStamp: %v", err)
	}
	stampPath := workerworktree.OwnerStampPath(wtLiveStamp)
	if err := os.MkdirAll(filepath.Dir(stampPath), 0o755); err != nil {
		t.Fatal(err)
	}
	stampData := []byte(`{"pid":12345,"schema":"fak-worker-owner/1"}`)
	if err := os.WriteFile(stampPath, stampData, 0o600); err != nil {
		t.Fatal(err)
	}

	// 4. wtDead: dead PID, stale heartbeat, no active lease on lane "ops"
	if err := workerworktree.WriteWorkerLease(wtDead, workerworktree.WorkerLease{
		PID:         99999999,
		CreatedAt:   oldTime,
		HeartbeatTS: oldTime,
	}); err != nil {
		t.Fatalf("WriteWorkerLease wtDead: %v", err)
	}

	// Age all worktrees past the 30-minute floor
	for _, p := range []string{wtLivePID, wtLiveLease, wtLiveStamp, wtDead} {
		if err := os.Chtimes(p, oldTime, oldTime); err != nil {
			t.Fatalf("chtimes %s: %v", p, err)
		}
	}

	wm := NewWorkspaceManager(repo)
	wm.ProcessAlive = func(pid int) bool {
		return pid == 12345
	}

	res, err := wm.SweepLocksAndWorktrees(context.Background(), false)
	if err != nil {
		t.Fatalf("SweepLocksAndWorktrees: %v", err)
	}

	// Live PID worktree must be preserved
	if _, err := os.Stat(wtLivePID); err != nil {
		t.Errorf("expected live PID worker worktree %s to be preserved, stat err: %v", wtLivePID, err)
	}

	// Live lease worktree must be preserved
	if _, err := os.Stat(wtLiveLease); err != nil {
		t.Errorf("expected live lease worker worktree %s to be preserved, stat err: %v", wtLiveLease, err)
	}

	// Live owner stamp worktree must be preserved
	if _, err := os.Stat(wtLiveStamp); err != nil {
		t.Errorf("expected live owner stamp worker worktree %s to be preserved, stat err: %v", wtLiveStamp, err)
	}

	// Dead worktree must be reaped
	if _, err := os.Stat(wtDead); !os.IsNotExist(err) {
		t.Errorf("expected dead worker worktree %s to be reaped, stat err: %v", wtDead, err)
	}

	// Verify res.WorktreesPruned contains wtDead base name
	deadBase := filepath.Base(wtDead)
	prunedFound := false
	for _, p := range res.WorktreesPruned {
		if p == deadBase {
			prunedFound = true
			break
		}
	}
	if !prunedFound {
		t.Errorf("expected %s in res.WorktreesPruned: %v", deadBase, res.WorktreesPruned)
	}
}
